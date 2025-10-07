// https://docs.databricks.com/gcp/en/sql/language-manual/functions/parse_json

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryViewArray, LargeStringArray, StringArray, StringViewArray,
};
use arrow_schema::{DataType, Field};
use datafusion::{
    common::{exec_datafusion_err, exec_err},
    error::Result as DataFusionResult,
    logical_expr::{ColumnarValue, ScalarUDFImpl, Signature, TypeSignature},
    scalar::ScalarValue,
};
use parquet_variant::VariantBuilder;
use parquet_variant_json::JsonToVariant as JsonToVariantExt;

use crate::extension_type::VariantExtensionType;

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct JsonToVariantUDF {
    signature: Signature,
}

impl Default for JsonToVariantUDF {
    fn default() -> Self {
        Self {
            signature: Signature::new(
                TypeSignature::Uniform(
                    1,
                    vec![DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View],
                ),
                datafusion::logical_expr::Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for JsonToVariantUDF {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "json_to_variant"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
        Ok(DataType::BinaryView)
    }

    fn return_field_from_args(
        &self,
        args: datafusion::logical_expr::ReturnFieldArgs,
    ) -> datafusion::error::Result<Arc<Field>> {
        let arg_fields = args.arg_fields;

        if arg_fields.len() != 1 {
            return exec_err!(
                "Incorrect number of arguments for string_to_uuid. Expected json_to_variant(...)"
            );
        }

        match args.arg_fields[0].data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {}
            _ => return exec_err!("Incorrect data type for json_to_variant, expected "),
        }

        let is_nullable = args.arg_fields[0].is_nullable();

        Ok(Arc::new(
            Field::new(self.name(), DataType::BinaryView, is_nullable)
                .with_extension_type(VariantExtensionType),
        ))
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> DataFusionResult<datafusion::logical_expr::ColumnarValue> {
        let arg = args
            .args
            .first()
            .ok_or_else(|| exec_datafusion_err!("empty argument, expected 1 argument"))?;

        let out = match arg {
            ColumnarValue::Scalar(scalar_value) => {
                let out = if let Some(json_str) = match scalar_value {
                    ScalarValue::Utf8(json)
                    | ScalarValue::Utf8View(json)
                    | ScalarValue::LargeUtf8(json) => json,
                    unsupported => {
                        return exec_err!("Unsupported data type {}", unsupported.data_type());
                    }
                } {
                    let mut variant_builder = VariantBuilder::new();
                    variant_builder
                        .append_json(json_str)
                        .map_err(|e| exec_datafusion_err!("Failed to parse JSON: {}", e))?;

                    let (_metadata, value) = variant_builder.finish();

                    Some(value)
                } else {
                    None
                };

                ColumnarValue::Scalar(ScalarValue::BinaryView(out))
            }
            ColumnarValue::Array(arr) => match arr.data_type() {
                DataType::Utf8 => ColumnarValue::Array(from_utf8_arr(arr)?),
                DataType::LargeUtf8 => ColumnarValue::Array(from_large_utf8_arr(arr)?),
                DataType::Utf8View => ColumnarValue::Array(from_utf8view_arr(arr)?),
                _ => return exec_err!("Invalid data type {}", arr.data_type()),
            },
        };

        Ok(out)
    }
}

macro_rules! define_from_string_array {
    ($fn_name:ident, $array_type:ty) => {
        pub(crate) fn $fn_name(arr: &ArrayRef) -> DataFusionResult<ArrayRef> {
            let arr = arr
                .as_any()
                .downcast_ref::<$array_type>()
                .ok_or(exec_datafusion_err!(
                    "Unable to downcast array as expected by type."
                ))?;

            let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(arr.len());

            for v in arr {
                out.push(if let Some(json_str) = v {
                    let mut variant_builder = VariantBuilder::new();
                    variant_builder
                        .append_json(json_str)
                        .map_err(|e| exec_datafusion_err!("Failed to parse JSON: {}", e))?;

                    // how do we encode metadata?
                    let (_m, v) = variant_builder.finish();

                    Some(v)
                } else {
                    None
                });
            }

            let out_ref = out
                .iter()
                .map(|opt| opt.as_ref().map(|v| v.as_slice()))
                .collect::<Vec<_>>();

            Ok(Arc::new(BinaryViewArray::from(out_ref)))
        }
    };
}

define_from_string_array!(from_utf8_arr, StringArray);
define_from_string_array!(from_utf8view_arr, StringViewArray);
define_from_string_array!(from_large_utf8_arr, LargeStringArray);

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::ScalarFunctionArgs;

    #[test]
    fn test_json_to_variant_udf_scalar_none() {
        let udf = JsonToVariantUDF::default();

        let json_input = ScalarValue::Utf8(None);

        let return_field = Arc::new(Field::new("result", DataType::BinaryView, true));
        let arg_field = Arc::new(Field::new("input", DataType::Utf8, true));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(json_input)],
            return_field: return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        match result {
            ColumnarValue::Scalar(ScalarValue::BinaryView(None)) => {}
            _ => panic!("Expected null BinaryView result"),
        }
    }

    #[test]
    fn test_json_to_variant_udf_scalar_null() {
        let udf = JsonToVariantUDF::default();

        let json_input = ScalarValue::Utf8(Some("null".into()));

        let return_field = Arc::new(Field::new("result", DataType::BinaryView, true));
        let arg_field = Arc::new(Field::new("input", DataType::Utf8, true));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(json_input)],
            return_field: return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        let (_expected_m, expected_v) = {
            let mut expected_variant = VariantBuilder::new();
            expected_variant.append_value(());
            expected_variant.finish()
        };

        match result {
            ColumnarValue::Scalar(ScalarValue::BinaryView(Some(bytes))) => {
                assert_eq!(bytes, expected_v);
            }
            _ => panic!("Expected non-null BinaryView result"),
        }
    }

    #[test]
    fn test_json_to_variant_udf_scalar() {
        let udf = JsonToVariantUDF::default();

        let json_input =
            ScalarValue::Utf8(Some(r#"{"key": 123, "data": [4, 5, "str"]}"#.to_string()));

        let return_field = Arc::new(Field::new("result", DataType::BinaryView, true));
        let arg_field = Arc::new(Field::new("input", DataType::Utf8, true));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(json_input)],
            return_field: return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        match result {
            ColumnarValue::Scalar(ScalarValue::BinaryView(Some(bytes))) => {
                assert!(!bytes.is_empty(), "Expected non-empty variant bytes");
            }
            _ => panic!("Expected scalar BinaryView result"),
        }
    }

    #[test]
    fn test_json_to_variant_udf_scalar_number() {
        let udf = JsonToVariantUDF::default();

        let json_input = ScalarValue::Utf8(Some("123".to_string()));

        let return_field = Arc::new(Field::new("result", DataType::BinaryView, true));
        let arg_field = Arc::new(Field::new("input", DataType::Utf8, true));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(json_input)],
            return_field: return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        let (_expected_m, expected_v) = {
            let mut expected_variant = VariantBuilder::new();
            expected_variant.append_value(123_u8);
            expected_variant.finish()
        };

        match result {
            ColumnarValue::Scalar(ScalarValue::BinaryView(Some(bytes))) => {
                assert!(!bytes.is_empty(), "Expected non-empty variant bytes");

                assert_eq!(bytes, expected_v);
            }
            _ => panic!("Expected scalar BinaryView result"),
        }
    }
}
