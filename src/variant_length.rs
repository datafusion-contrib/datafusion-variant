use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Array};
use arrow_schema::DataType;
use datafusion::common::exec_datafusion_err;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use datafusion::scalar::ScalarValue;
use parquet_variant::Variant;
use parquet_variant_compute::VariantArray;

use crate::shared::{path_from_scalar, try_field_as_variant_array};

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantLengthUdf {
    signature: Signature,
}

impl Default for VariantLengthUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(2), Volatility::Immutable),
        }
    }
}

/// Returns the number of elements in an Array or Object variant at the given path.
/// Returns NULL if the variant at the path is not an Array or Object.
fn variant_length(variant: Option<Variant<'_, '_>>) -> Option<u64> {
    let variant = variant?;
    match variant {
        Variant::List(list) => Some(list.len() as u64),
        Variant::Object(obj) => Some(obj.len() as u64),
        _ => None,
    }
}

impl ScalarUDFImpl for VariantLengthUdf {
    fn name(&self) -> &str {
        "variant_length"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (variant_arg, path_arg) = match args.args.as_slice() {
            [variant_arg, path_arg] => (variant_arg, path_arg),
            _ => return datafusion::common::exec_err!("expected 2 arguments"),
        };

        let variant_field = args
            .arg_fields
            .first()
            .ok_or_else(|| exec_datafusion_err!("expected argument field"))?;

        try_field_as_variant_array(variant_field.as_ref())?;

        match (variant_arg, path_arg) {
            (ColumnarValue::Array(variant_array), ColumnarValue::Scalar(path_scalar)) => {
                let path = path_from_scalar(path_scalar)?;
                let variant_array = VariantArray::try_new(variant_array.as_ref())?;
                let values = variant_array
                    .iter()
                    .map(|variant| {
                        let at_path = variant.as_ref().and_then(|v| v.get_path(&path));
                        variant_length(at_path)
                    })
                    .collect::<Vec<_>>();

                Ok(ColumnarValue::Array(
                    Arc::new(UInt64Array::from(values)) as ArrayRef
                ))
            }
            (ColumnarValue::Scalar(scalar_variant), ColumnarValue::Scalar(path_scalar)) => {
                let ScalarValue::Struct(variant_array) = scalar_variant else {
                    return datafusion::common::exec_err!("expected struct array");
                };

                let path = path_from_scalar(path_scalar)?;
                let variant_array = VariantArray::try_new(variant_array.as_ref())?;
                let variant = variant_array.iter().next().flatten();
                let at_path = variant.as_ref().and_then(|v| v.get_path(&path));
                let value = variant_length(at_path);

                Ok(ColumnarValue::Scalar(ScalarValue::UInt64(value)))
            }
            _ => datafusion::common::exec_err!(
                "unsupported argument combination: variant_length requires a scalar path"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, ArrayRef, UInt64Array};
    use arrow_schema::{DataType, Field, Fields};
    use parquet_variant_compute::VariantType;

    use crate::shared::{build_variant_array_from_json_array, variant_scalar_from_json};

    use super::*;

    fn arg_fields() -> Vec<Arc<Field>> {
        vec![
            Arc::new(
                Field::new("input", DataType::Struct(Fields::empty()), true)
                    .with_extension_type(VariantType),
            ),
            Arc::new(Field::new("path", DataType::Utf8, true)),
        ]
    }

    #[test]
    fn test_scalar_array_length() {
        let udf = VariantLengthUdf::default();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Scalar(variant_scalar_from_json(serde_json::json!([1, 2, 3]))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("".to_string()))),
            ],
            return_field: Arc::new(Field::new("result", DataType::UInt64, true)),
            arg_fields: arg_fields(),
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::UInt64(Some(value))) = result else {
            panic!("expected u64 scalar")
        };
        assert_eq!(value, 3);
    }

    #[test]
    fn test_scalar_object_length() {
        let udf = VariantLengthUdf::default();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Scalar(variant_scalar_from_json(
                    serde_json::json!({"a": 1, "b": 2}),
                )),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("".to_string()))),
            ],
            return_field: Arc::new(Field::new("result", DataType::UInt64, true)),
            arg_fields: arg_fields(),
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::UInt64(Some(value))) = result else {
            panic!("expected u64 scalar")
        };
        assert_eq!(value, 2);
    }

    #[test]
    fn test_scalar_nested_path() {
        let udf = VariantLengthUdf::default();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Scalar(variant_scalar_from_json(
                    serde_json::json!({"a": [1, 2, 3, 4]}),
                )),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("a".to_string()))),
            ],
            return_field: Arc::new(Field::new("result", DataType::UInt64, true)),
            arg_fields: arg_fields(),
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::UInt64(Some(value))) = result else {
            panic!("expected u64 scalar")
        };
        assert_eq!(value, 4);
    }

    #[test]
    fn test_scalar_primitive_returns_null() {
        let udf = VariantLengthUdf::default();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Scalar(variant_scalar_from_json(serde_json::json!(42))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("".to_string()))),
            ],
            return_field: Arc::new(Field::new("result", DataType::UInt64, true)),
            arg_fields: arg_fields(),
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::UInt64(value)) = result else {
            panic!("expected u64 scalar")
        };
        assert_eq!(value, None);
    }

    #[test]
    fn test_array_variants() {
        let udf = VariantLengthUdf::default();
        let input = build_variant_array_from_json_array(&[
            Some(serde_json::json!([1, 2])),
            Some(serde_json::json!({"a": 1, "b": 2, "c": 3})),
            Some(serde_json::json!(42)),
            None,
        ]);
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(arrow::array::StructArray::from(input)) as ArrayRef),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("".to_string()))),
            ],
            return_field: Arc::new(Field::new("result", DataType::UInt64, true)),
            arg_fields: arg_fields(),
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Array(values) = result else {
            panic!("expected array")
        };

        let values = values.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(
            values.into_iter().collect::<Vec<_>>(),
            vec![Some(2), Some(3), None, None]
        );
    }
}
