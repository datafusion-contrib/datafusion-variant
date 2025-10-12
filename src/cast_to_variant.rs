use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StructArray};
use arrow_schema::{DataType, Field, Fields};
use datafusion::{
    common::exec_err,
    error::Result,
    logical_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet_variant::Variant;
use parquet_variant_compute::VariantArrayBuilder;

use crate::shared::{try_field_as_binary, try_parse_binary_columnar, try_parse_binary_scalar};

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct CastToVariantUdf {
    signature: Signature,
}

impl Default for CastToVariantUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(2), Volatility::Immutable),
        }
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
        Ok(DataType::Struct(Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, true),
        ])))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match args.arg_fields.as_slice() {
            [metadata_field, variant_field] => {
                try_field_as_binary(metadata_field.as_ref())?;
                try_field_as_binary(variant_field.as_ref())?;
            }
            _ => {
                // right now, let's only support (BinaryViewArray, BinaryViewArray)
                // but I don't see why we couldn't call cast_to_variant(string_column) -> VariantArray...
                return exec_err!("unsupported, expected 2 arguments");
            }
        }

        let [metadata_argument, variant_argument] = args.args.as_slice() else {
            return exec_err!("expected 2 arguments");
        };

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
                    match (m, v) {
                        (Some(m), Some(v)) if !m.is_empty() && !v.is_empty() => {
                            builder.append_variant(Variant::try_new(m, v)?)
                        }
                        _ => builder.append_null(),
                    }
                }

                let out: StructArray = builder.build().into();

                ColumnarValue::Array(Arc::new(out) as ArrayRef)
            }
            (ColumnarValue::Scalar(metadata_value), ColumnarValue::Array(value_array)) => {
                let metadata = try_parse_binary_scalar(metadata_value)?;
                let value_array = try_parse_binary_columnar(value_array)?;

                let mut builder = VariantArrayBuilder::new(value_array.len());

                for v in value_array {
                    match (metadata, v) {
                        (Some(m), Some(v)) if !m.is_empty() && !v.is_empty() => {
                            builder.append_variant(Variant::try_new(m.as_slice(), v)?);
                        }
                        _ => builder.append_null(),
                    }
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

                let mut b = VariantArrayBuilder::new(metadata_array.len());

                for m in metadata_array {
                    match (m, value) {
                        (Some(m), Some(v)) if !m.is_empty() && !v.is_empty() => {
                            b.append_variant(Variant::try_new(m, v.as_slice())?)
                        }
                        _ => b.append_null(),
                    }
                }

                let arr: StructArray = b.build().into();

                ColumnarValue::Array(Arc::new(arr))
            }
        };

        Ok(out)
    }
}

#[cfg(test)]
mod tests {

    use parquet_variant_compute::VariantArray;

    use crate::shared::{
        are_variant_arrays_equal, build_variant_array_from_json,
        build_variant_array_from_json_array,
    };

    use super::*;

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

        assert!(are_variant_arrays_equal(
            &variant_array,
            &expected_variant_array
        ));
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

        assert!(are_variant_arrays_equal(
            &variant_array,
            &expected_variant_array
        ));
    }
}
