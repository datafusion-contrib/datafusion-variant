use std::sync::Arc;

use arrow::array::{Array, BooleanArray, cast::AsArray};
use arrow_schema::{DataType, Field};
use datafusion::common::{exec_err, utils::take_function_args};
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};
use datafusion::scalar::ScalarValue;
use parquet_variant_compute::{VariantArray, VariantType};

/// Returns true only for a present Variant null; SQL NULL returns false.
///
/// Like Spark, this inspects the top-level value, not nulls inside a container,
/// and does not perform full validation of the Variant's contents.
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct IsVariantNullUdf {
    signature: Signature,
}

impl Default for IsVariantNullUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(1), Volatility::Immutable),
        }
    }
}

fn validate_input(field: &Field) -> Result<()> {
    // Spark accepts an untyped NULL, but not a NULL of an unrelated SQL type.
    if field.data_type() != &DataType::Null && field.try_extension_type::<VariantType>().is_err() {
        return exec_err!("is_variant_null expects a Variant or untyped NULL argument");
    }
    Ok(())
}

fn is_variant_null_at(array: &VariantArray, index: usize) -> Result<bool> {
    if array.is_null(index) {
        return Ok(false);
    }
    // A non-null shredded value is a primitive, object or list, never Variant null.
    // Avoid materializing it: borrowed Variant access does not support all containers.
    // NullArray has no physical validity bitmap, so is_valid alone is insufficient.
    if array
        .typed_value_field()
        .is_some_and(|v| v.data_type() != &DataType::Null && v.is_valid(index))
    {
        return Ok(false);
    }
    let Some(value) = array.value_field().filter(|v| v.is_valid(index)) else {
        // A present top-level Variant with neither representation populated
        // falls back to Variant null. Missing extracted fields must instead
        // be marked SQL NULL in the outer validity bitmap.
        return Ok(true);
    };
    let bytes = match value.data_type() {
        DataType::Binary => value.as_binary::<i32>().value(index),
        DataType::LargeBinary => value.as_binary::<i64>().value(index),
        DataType::BinaryView => value.as_binary_view().value(index),
        _ => return exec_err!("is_variant_null expects a binary Variant value column"),
    };
    // Spark's isVariantNull uses the same header check and rejects an empty payload.
    // Do not decode metadata or recursively validate an otherwise non-null value.
    match bytes.first() {
        Some(header) => Ok(*header == 0),
        None => exec_err!("is_variant_null received a malformed Variant: empty value"),
    }
}

impl ScalarUDFImpl for IsVariantNullUdf {
    fn name(&self) -> &str {
        "is_variant_null"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        let [field] = take_function_args(self.name(), args.arg_fields)?;
        validate_input(field)?;
        Ok(Arc::new(Field::new(self.name(), DataType::Boolean, false)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [field] = take_function_args(self.name(), &args.arg_fields)?;
        let [arg] = take_function_args(self.name(), &args.args)?;
        validate_input(field)?;

        // Evaluate scalars as one-row arrays; array inputs retain their length.
        let array = arg.to_array(1)?;
        if matches!(arg, ColumnarValue::Scalar(_)) {
            let result = if array.data_type() == &DataType::Null {
                false
            } else {
                is_variant_null_at(&VariantArray::try_new(array.as_ref())?, 0)?
            };
            return Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(result))));
        }

        let result = if array.data_type() == &DataType::Null {
            BooleanArray::from(vec![false; array.len()])
        } else {
            let array = VariantArray::try_new(array.as_ref())?;
            let values = (0..array.len())
                .map(|i| is_variant_null_at(&array, i))
                .collect::<Result<Vec<_>>>()?;
            BooleanArray::from(values)
        };

        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        ArrayRef, BinaryArray, BinaryViewArray, LargeBinaryArray, NullArray, StringArray,
        StructArray,
    };
    use arrow::buffer::NullBuffer;
    use parquet_variant_compute::{json_to_variant, shred_variant};

    fn invoke(array: StructArray, scalar: bool) -> Result<ColumnarValue> {
        let udf = IsVariantNullUdf::default();
        let field = Arc::new(
            Field::new("v", array.data_type().clone(), true).with_extension_type(VariantType),
        );
        let fields = vec![field];
        let return_field = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &fields,
            scalar_arguments: &[None],
        })?;
        assert_eq!(return_field.data_type(), &DataType::Boolean);
        assert!(!return_field.is_nullable());
        let number_rows = array.len();
        let input = if scalar {
            ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(array)))
        } else {
            ColumnarValue::Array(Arc::new(array))
        };
        udf.invoke_with_args(ScalarFunctionArgs {
            args: vec![input],
            arg_fields: fields,
            return_field,
            number_rows,
            config_options: Default::default(),
        })
    }

    fn assert_results(array: &StructArray, expected: &[bool]) {
        let ColumnarValue::Array(actual) = invoke(array.clone(), false).unwrap() else {
            panic!("expected array output");
        };
        let actual = actual.as_boolean();
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual.null_count(), 0);
        assert_eq!(actual.values().iter().collect::<Vec<_>>(), expected);
        for (i, expected) in expected.iter().enumerate() {
            let actual = invoke(array.slice(i, 1), true).unwrap();
            assert!(
                matches!(actual, ColumnarValue::Scalar(ScalarValue::Boolean(Some(v))) if v == *expected)
            );
        }
    }

    #[test]
    fn sliced_and_empty_variant_layouts() {
        // SQL tests cover the value matrix. Here, force a nonzero array offset
        // and exercise scalar views of each physical layout, including SQL NULL.
        let json: ArrayRef = Arc::new(StringArray::from(vec![
            Some(r#""padding""#),
            None,
            Some("null"),
            Some("0"),
            Some(r#"{"a":null}"#),
            Some("[null]"),
        ]));
        let array = json_to_variant(&json).unwrap();
        let expected = [false, true, false, false, false];
        let mut layouts = vec![array.clone()];
        for data_type in [
            DataType::Int64,
            DataType::Struct(vec![Field::new("a", DataType::Int64, true)].into()),
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
        ] {
            layouts.push(shred_variant(&array, &data_type).unwrap());
        }
        for layout in layouts {
            assert_results(layout.slice(1, expected.len()).inner(), &expected);
            assert_results(layout.slice(3, 0).inner(), &[]);
        }
    }

    fn raw_variant(values: ArrayRef, nulls: Option<NullBuffer>) -> StructArray {
        // Deliberately invalid metadata: like Spark's predicate, this function
        // needs only the value header, not a decoded metadata dictionary.
        let metadata: ArrayRef =
            Arc::new(BinaryViewArray::from(vec![&[1, 2, 3][..]; values.len()]));
        StructArray::new(
            vec![
                Field::new("metadata", DataType::BinaryView, false),
                Field::new("value", values.data_type().clone(), true),
            ]
            .into(),
            vec![metadata, values],
            nulls,
        )
    }

    #[test]
    fn header_check_supports_binary_layouts_and_ignores_sql_null_payloads() {
        // A nonzero header is false, even without the rest of that value's body.
        // An empty payload under a SQL-null row must never be inspected.
        let bytes = vec![&[0][..], &[12][..], &[][..]];
        let layouts: Vec<ArrayRef> = vec![
            Arc::new(BinaryArray::from(bytes.clone())),
            Arc::new(LargeBinaryArray::from(bytes.clone())),
            Arc::new(BinaryViewArray::from(bytes)),
        ];
        for values in layouts {
            let array = raw_variant(values, Some(NullBuffer::from(vec![true, true, false])));
            assert_results(&array, &[true, false, false]);
        }
    }

    #[test]
    fn missing_row_and_present_null_fallback_are_distinct() {
        // Identical absent value columns: the outer validity distinguishes a
        // missing extracted row from the required-value Variant-null fallback.
        let values = Arc::new(BinaryViewArray::from(vec![None::<&[u8]>, None]));
        let array = raw_variant(values, Some(NullBuffer::from(vec![false, true])));
        assert_results(&array, &[false, true]);
    }

    #[test]
    fn null_typed_value_uses_encoded_value_or_null_fallback() {
        let mut fields = vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("typed_value", DataType::Null, true),
        ];
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(BinaryViewArray::from(vec![&[1, 0, 0][..]; 4])),
            Arc::new(NullArray::new(4)),
        ];
        let nulls = Some(NullBuffer::from(vec![true, true, true, false]));
        let array = StructArray::new(fields.clone().into(), columns.clone(), nulls.clone());
        assert_results(&array, &[true, true, true, false]);

        // A NullArray must not hide an encoded value or bypass SQL validity.
        fields.push(Field::new("value", DataType::BinaryView, true));
        columns.push(Arc::new(BinaryViewArray::from(vec![
            None,
            Some(&[0][..]),
            Some(&[12][..]),
            Some(&[][..]),
        ])));
        let array = StructArray::new(fields.into(), columns, nulls);
        assert_results(&array, &[true, true, false, false]);
    }

    #[test]
    fn empty_valid_payload_returns_error_like_spark() {
        // Port of Spark VariantExpressionSuite's "is_variant_null invalid input".
        let array = raw_variant(Arc::new(BinaryViewArray::from(vec![&[][..]])), None);
        for scalar in [false, true] {
            let error = invoke(array.clone(), scalar).unwrap_err().to_string();
            assert!(error.contains("malformed Variant: empty value"), "{error}");
        }
    }
}
