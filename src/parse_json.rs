// https://docs.databricks.com/gcp/en/sql/language-manual/functions/parse_json

use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields};
use datafusion::{
    common::exec_err,
    error::Result,
    logical_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    },
    scalar::ScalarValue,
};
use parquet_variant_compute::{VariantArrayBuilder, VariantType, json_to_variant};
use parquet_variant_json::JsonToVariant as JsonToVariantExt;

use crate::shared::try_parse_string_scalar;

/// Returns a Variant from a JSON string
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct ParseJsonUdf {
    signature: Signature,
}

impl Default for ParseJsonUdf {
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

impl ScalarUDFImpl for ParseJsonUdf {
    fn name(&self) -> &str {
        "parse_json"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Struct(Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, false),
        ])))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        let [input] = args.arg_fields else {
            return exec_err!("expected 1 argument");
        };
        // Arrow preserves input SQL nulls; invalid JSON errors instead of adding nulls.
        let data_type = self.return_type(std::slice::from_ref(input.data_type()))?;
        Ok(Arc::new(
            Field::new(self.name(), data_type, input.is_nullable())
                .with_extension_type(VariantType),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [input] = args.args.as_slice() else {
            return exec_err!("expected 1 argument");
        };
        match input {
            ColumnarValue::Scalar(scalar_value) => {
                // Use the same parser as the array kernel without converting the input to an array.
                let json_str = try_parse_string_scalar(scalar_value)?;
                let mut builder = VariantArrayBuilder::new(1);
                match json_str {
                    Some(json_str) => builder.append_json(json_str.as_str())?,
                    None => builder.append_null(),
                }
                Ok(ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
                    builder.build().into_inner(),
                ))))
            }
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(Arc::new(
                json_to_variant(array)?.into_inner(),
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_and_array_parsing_agree() -> Result<()> {
        let udf = ParseJsonUdf::default();
        for input in [
            None,
            Some("null"),
            Some("123"),
            Some(r#"{"items":[true,null,"\uD83D\uDE05"]}"#),
            Some("9223372036854775809"),
            Some(r#"{"a":1,"a":2}"#),
            Some("{"),
            Some(r#""\uD835""#),
        ] {
            for scalar in [
                ScalarValue::Utf8(input.map(str::to_owned)),
                ScalarValue::LargeUtf8(input.map(str::to_owned)),
                ScalarValue::Utf8View(input.map(str::to_owned)),
            ] {
                let field = Arc::new(Field::new("input", scalar.data_type(), true));
                let return_field = udf.return_field_from_args(ReturnFieldArgs {
                    arg_fields: std::slice::from_ref(&field),
                    scalar_arguments: &[],
                })?;
                let invoke = |input| {
                    udf.invoke_with_args(ScalarFunctionArgs {
                        args: vec![input],
                        arg_fields: vec![Arc::clone(&field)],
                        return_field: Arc::clone(&return_field),
                        number_rows: 1,
                        config_options: Default::default(),
                    })
                };
                let array_result = invoke(ColumnarValue::Array(scalar.to_array()?));
                let scalar_result = invoke(ColumnarValue::Scalar(scalar));
                match (scalar_result, array_result) {
                    (Ok(ColumnarValue::Scalar(scalar)), Ok(ColumnarValue::Array(array))) => {
                        assert_eq!(scalar.to_array()?.to_data(), array.to_data());
                    }
                    (Err(scalar), Err(array)) => {
                        assert_eq!(scalar.to_string(), array.to_string());
                    }
                    results => panic!("scalar/array mismatch for {input:?}: {results:?}"),
                }
            }
        }
        Ok(())
    }
}
