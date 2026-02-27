use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, StructArray};
use arrow_schema::{DataType, Field};
use datafusion::{
    common::exec_err,
    error::Result,
    logical_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
        TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet_variant::Variant;
use parquet_variant_compute::{VariantArray, VariantArrayBuilder, cast_to_variant};

use crate::shared::{try_parse_binary_columnar, try_parse_binary_scalar};

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct CastToVariantUdf {
    signature: Signature,
}

impl Default for CastToVariantUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::VariadicAny, Volatility::Immutable),
        }
    }
}

impl CastToVariantUdf {
    fn canonical_variant_data_type() -> DataType {
        VariantArrayBuilder::new(0).build().data_type().clone()
    }

    fn canonical_return_field(name: &str) -> Field {
        VariantArrayBuilder::new(0)
            .build()
            .field(name.to_string())
            .with_nullable(true)
    }

    fn append_variant_or_null(
        builder: &mut VariantArrayBuilder,
        metadata: Option<&[u8]>,
        value: Option<&[u8]>,
    ) -> Result<()> {
        match (metadata, value) {
            (Some(m), Some(v)) if !m.is_empty() && !v.is_empty() => {
                builder.append_variant(Variant::try_new(m, v)?);
            }
            _ => builder.append_null(),
        }

        Ok(())
    }

    fn from_metadata_value(
        metadata_argument: &ColumnarValue,
        variant_argument: &ColumnarValue,
    ) -> Result<ColumnarValue> {
        let out = match (metadata_argument, variant_argument) {
            (ColumnarValue::Array(metadata_array), ColumnarValue::Array(value_array)) => {
                if metadata_array.len() != value_array.len() {
                    return exec_err!(
                        "expected metadata array to be of same length as variant array"
                    );
                }

                let metadata_array = try_parse_binary_columnar(metadata_array)?;
                let value_array = try_parse_binary_columnar(value_array)?;

                let mut builder = VariantArrayBuilder::new(metadata_array.len());
                for (m, v) in metadata_array.into_iter().zip(value_array) {
                    Self::append_variant_or_null(&mut builder, m, v)?;
                }
                let out: StructArray = builder.build().into();

                ColumnarValue::Array(Arc::new(out) as ArrayRef)
            }
            (ColumnarValue::Scalar(metadata_value), ColumnarValue::Array(value_array)) => {
                let metadata = try_parse_binary_scalar(metadata_value)?;
                let value_array = try_parse_binary_columnar(value_array)?;

                let mut builder = VariantArrayBuilder::new(value_array.len());
                for v in value_array {
                    Self::append_variant_or_null(&mut builder, metadata.map(|m| m.as_slice()), v)?;
                }
                let arr: StructArray = builder.build().into();

                ColumnarValue::Array(Arc::new(arr) as ArrayRef)
            }
            (ColumnarValue::Scalar(metadata_value), ColumnarValue::Scalar(value_scalar)) => {
                let metadata = try_parse_binary_scalar(metadata_value)?;
                let value = try_parse_binary_scalar(value_scalar)?;

                match (metadata, value) {
                    (Some(m), Some(v)) if !m.is_empty() && !v.is_empty() => {
                        let mut b = VariantArrayBuilder::new(1);
                        b.append_variant(Variant::try_new(m.as_slice(), v.as_slice())?);
                        let arr: StructArray = b.build().into();

                        ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(arr)))
                    }
                    _ => ColumnarValue::Scalar(ScalarValue::Null),
                }
            }
            (ColumnarValue::Array(metadata_array), ColumnarValue::Scalar(value_scalar)) => {
                let metadata_array = try_parse_binary_columnar(metadata_array)?;
                let value = try_parse_binary_scalar(value_scalar)?;

                let mut builder = VariantArrayBuilder::new(metadata_array.len());
                for m in metadata_array {
                    Self::append_variant_or_null(&mut builder, m, value.map(|v| v.as_slice()))?;
                }
                let arr: StructArray = builder.build().into();

                ColumnarValue::Array(Arc::new(arr))
            }
        };

        Ok(out)
    }

    fn from_array(array: &ArrayRef) -> Result<ColumnarValue> {
        // If the array is already a Variant array, pass it through unchanged
        if let Some(struct_array) = array.as_struct_opt()
            && VariantArray::try_new(struct_array).is_ok()
        {
            return Ok(ColumnarValue::Array(Arc::clone(array)));
        }

        let variant_array = cast_to_variant(array.as_ref())?;
        let struct_array: StructArray = variant_array.into();

        Ok(ColumnarValue::Array(Arc::new(struct_array)))
    }

    fn from_scalar_value(scalar_value: &ScalarValue) -> Result<ColumnarValue> {
        if let ScalarValue::Struct(struct_array) = scalar_value
            && VariantArray::try_new(struct_array.as_ref()).is_ok()
        {
            return Ok(ColumnarValue::Scalar(ScalarValue::Struct(
                struct_array.clone(),
            )));
        }

        let array = scalar_value.to_array_of_size(1)?;
        let variant_array = cast_to_variant(array.as_ref())?;
        let struct_array: StructArray = variant_array.into();

        Ok(ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
            struct_array,
        ))))
    }
}

impl ScalarUDFImpl for CastToVariantUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "cast_to_variant"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(Self::canonical_variant_data_type())
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> Result<Arc<Field>> {
        Ok(Arc::new(Self::canonical_return_field(self.name())))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match args.args.as_slice() {
            [metadata_value, variant_value] => {
                Self::from_metadata_value(metadata_value, variant_value)
            }
            [ColumnarValue::Scalar(scalar_value)] => Self::from_scalar_value(scalar_value),
            [ColumnarValue::Array(array)] => Self::from_array(array),
            _ => exec_err!("unrecognized argument"),
        }
    }
}

#[cfg(test)]
mod tests {

    use arrow::array::{FixedSizeBinaryBuilder, Int32Array, StringArray, StringViewArray};
    use arrow_schema::Fields;
    use parquet_variant::Variant;
    use parquet_variant_compute::{VariantArray, VariantType};

    use crate::shared::{build_variant_array_from_json, build_variant_array_from_json_array};

    use super::*;

    #[test]
    fn test_scalar_float64() {
        let udf = CastToVariantUdf::default();

        let arg_field = Arc::new(Field::new("input", DataType::Float64, true));
        let return_field = Arc::new(Field::new(
            "res",
            udf.return_type(&[DataType::Float64]).unwrap(),
            true,
        ));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(ScalarValue::Float64(Some(3.25)))],
            return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let res = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Struct(variant_array)) = res else {
            panic!("expected struct scalar")
        };

        let variant_array = VariantArray::try_new(variant_array.as_ref()).unwrap();

        assert_eq!(variant_array.value(0), Variant::Double(3.25));
    }

    #[test]
    fn test_array_int32() {
        let udf = CastToVariantUdf::default();

        let arg_field = Arc::new(Field::new("input", DataType::Int32, true));
        let return_field = Arc::new(Field::new(
            "res",
            udf.return_type(&[DataType::Int32]).unwrap(),
            true,
        ));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(Int32Array::from(vec![
                Some(1),
                None,
                Some(-5),
            ])) as ArrayRef)],
            return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let res = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = res else {
            panic!("expected array output")
        };

        let variant_array = VariantArray::try_new(arr.as_ref()).unwrap();

        assert_eq!(variant_array.value(0), Variant::Int32(1));
        assert!(variant_array.is_null(1));
        assert_eq!(variant_array.value(2), Variant::Int32(-5));
    }

    #[test]
    fn test_return_field_extension_type() {
        let udf = CastToVariantUdf::default();

        let arg_field = Arc::new(Field::new("input", DataType::Utf8, true));

        let return_field = udf
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: std::slice::from_ref(&arg_field),
                scalar_arguments: &[None],
            })
            .unwrap();

        assert!(matches!(return_field.extension_type(), VariantType));
        assert_eq!(
            return_field.data_type(),
            &DataType::Struct(Fields::from(vec![
                Field::new("metadata", DataType::BinaryView, false),
                Field::new("value", DataType::BinaryView, false),
            ]))
        );
    }

    #[test]
    fn test_scalar_binary_views() {
        let expected_variant_array = build_variant_array_from_json(&serde_json::json!({
            "name": "norm",
        }));

        let (input_metadata, input_value) = {
            let metadata = expected_variant_array.metadata_field().value(0);
            let value = expected_variant_array.value_field().unwrap().value(0);

            (metadata, value)
        };

        let udf = CastToVariantUdf::default();

        let metadata_field = Arc::new(Field::new("metadata", DataType::BinaryView, true));
        let variant_field = Arc::new(Field::new("value", DataType::BinaryView, true));

        let return_field = Arc::new(Field::new(
            "res",
            udf.return_type(&[DataType::BinaryView, DataType::BinaryView])
                .unwrap(),
            true,
        ));

        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Scalar(ScalarValue::BinaryView(Some(input_metadata.to_vec()))),
                ColumnarValue::Scalar(ScalarValue::BinaryView(Some(input_value.to_vec()))),
            ],
            return_field,
            arg_fields: vec![metadata_field, variant_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let res = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Struct(variant_array)) = res else {
            panic!("expected scalar value struct array")
        };

        let variant_array = VariantArray::try_new(variant_array.as_ref()).unwrap();

        assert_eq!(&variant_array, &expected_variant_array);
    }

    #[test]
    fn test_array_string() {
        let udf = CastToVariantUdf::default();

        let arg_field = Arc::new(Field::new("input", DataType::Utf8, true));
        let return_field = Arc::new(Field::new(
            "res",
            udf.return_type(&[DataType::Utf8]).unwrap(),
            true,
        ));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(StringArray::from(vec![
                Some("abcdefghijklmnop"),
                None,
                Some("hello world"),
            ])) as ArrayRef)],
            return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let res = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = res else {
            panic!("expected array output")
        };

        let variant_array = VariantArray::try_new(arr.as_ref()).unwrap();

        assert_eq!(variant_array.value(0), Variant::from("abcdefghijklmnop"));
        assert!(variant_array.is_null(1));
        assert_eq!(variant_array.value(2), Variant::from("hello world"));
    }

    #[test]
    fn test_fixed_size_binary_uuid_like() {
        let udf = CastToVariantUdf::default();

        let arg_field = Arc::new(Field::new("input", DataType::FixedSizeBinary(16), true));
        let return_field = Arc::new(Field::new(
            "res",
            udf.return_type(&[DataType::FixedSizeBinary(16)]).unwrap(),
            true,
        ));

        let mut builder = FixedSizeBinaryBuilder::with_capacity(3, 16);
        builder.append_value([1u8; 16]).unwrap();
        builder.append_null();
        builder.append_value([2u8; 16]).unwrap();
        let array = builder.finish();

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(array) as ArrayRef)],
            return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let res = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = res else {
            panic!("expected array output")
        };

        let variant_array = VariantArray::try_new(arr.as_ref()).unwrap();

        assert_eq!(variant_array.value(0), Variant::Binary(&[1u8; 16]));
        assert!(variant_array.is_null(1));
        assert_eq!(variant_array.value(2), Variant::Binary(&[2u8; 16]));
    }

    #[test]
    fn test_array_binary_views() {
        let expected_variant_array = build_variant_array_from_json_array(&[
            Some(serde_json::json!({
                "name": "norm",
            })),
            None,
            None,
            Some(serde_json::json!({
                "id": 1,
                "parent_id": 0,
                "child_ids": [2, 3, 4, 5]
            })),
        ]);

        let (input_metadata_array, input_value_array) = {
            let metadata = expected_variant_array.metadata_field().clone();
            let value = expected_variant_array.value_field().unwrap().clone();

            (metadata, value)
        };

        let udf = CastToVariantUdf::default();

        let metadata_field = Arc::new(Field::new("metadata", DataType::BinaryView, true));
        let variant_field = Arc::new(Field::new("value", DataType::BinaryView, true));

        let return_field = Arc::new(Field::new(
            "res",
            udf.return_type(&[DataType::BinaryView, DataType::BinaryView])
                .unwrap(),
            true,
        ));

        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(input_metadata_array) as ArrayRef),
                ColumnarValue::Array(Arc::new(input_value_array) as ArrayRef),
            ],
            return_field,
            arg_fields: vec![metadata_field, variant_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let res = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(variant_array) = res else {
            panic!("expected scalar value struct array")
        };

        let variant_array = VariantArray::try_new(variant_array.as_ref()).unwrap();

        assert_eq!(&variant_array, &expected_variant_array);
    }

    #[test]
    fn test_array_string_view() {
        let udf = CastToVariantUdf::default();

        let arg_field = Arc::new(Field::new("input", DataType::Utf8View, true));
        let return_field = Arc::new(Field::new(
            "res",
            udf.return_type(&[DataType::Utf8View]).unwrap(),
            true,
        ));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(StringViewArray::from(vec![
                Some("short"),
                None,
                Some("another"),
            ])) as ArrayRef)],
            return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let res = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = res else {
            panic!("expected array output")
        };

        let variant_array = VariantArray::try_new(arr.as_ref()).unwrap();

        assert_eq!(variant_array.value(0), Variant::from("short"));
        assert!(variant_array.is_null(1));
        assert_eq!(variant_array.value(2), Variant::from("another"));
    }
}
