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
                        (Some(m), Some(v)) => builder.append_variant(Variant::try_new(m, v)?),
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
                        (Some(m), Some(v)) => {
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
                    (Some(m), Some(v)) => {
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
                        (Some(m), Some(v)) => b.append_variant(Variant::try_new(m, v.as_slice())?),
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
