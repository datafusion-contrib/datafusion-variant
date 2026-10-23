// https://docs.databricks.com/gcp/en/sql/language-manual/functions/parse_json

use std::sync::Arc;

use arrow::compute::CastOptions;
use arrow_schema::{DataType, Field};
use datafusion::{
    common::exec_err,
    error::Result,
    logical_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet_variant_compute::{VariantArrayBuilder, json_to_variant};
use parquet_variant_json::JsonToVariant as JsonToVariantExt;

use crate::shared::{try_parse_string_columnar, try_parse_string_scalar};
use crate::variant_type::variant_field;

/// Parses JSON, returning an error for invalid JSON values.
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct ParseJsonUdf {
    signature: Signature,
}

impl Default for ParseJsonUdf {
    fn default() -> Self {
        Self {
            signature: Signature::uniform(
                1,
                vec![DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View],
                Volatility::Immutable,
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
        Ok(variant_field(self.name()).data_type().clone())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        Ok(Arc::new(return_field_for_parse_json(self.name(), args)?))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_parse_json(
            args,
            CastOptions {
                safe: false,
                ..Default::default()
            },
        )
    }
}

/// Parses JSON, returning SQL NULL for invalid JSON values.
/// Invalid argument types and counts still return errors.
/// Accepts the same inputs as [`ParseJsonUdf`].
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct TryParseJsonUdf {
    signature: Signature,
}

impl Default for TryParseJsonUdf {
    fn default() -> Self {
        Self {
            signature: Signature::uniform(
                1,
                vec![DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for TryParseJsonUdf {
    fn name(&self) -> &str {
        "try_parse_json"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(variant_field(self.name()).data_type().clone())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        Ok(Arc::new(
            return_field_for_parse_json(self.name(), args)?.with_nullable(true),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_parse_json(
            args,
            CastOptions {
                safe: true,
                ..Default::default()
            },
        )
    }
}

fn return_field_for_parse_json(name: &str, args: ReturnFieldArgs) -> Result<Field> {
    let [input] = args.arg_fields else {
        return exec_err!("expected 1 argument");
    };
    // Strict parsing preserves input nullability; try parsing can add SQL nulls.
    Ok(variant_field(name).with_nullable(input.is_nullable()))
}

#[inline]
fn append_json(
    builder: &mut VariantArrayBuilder,
    json: Option<&str>,
    cast_options: &CastOptions<'_>,
) -> Result<()> {
    let Some(json) = json else {
        builder.append_null();
        return Ok(());
    };
    // JSON parsing precedes mutation; unfinished nested builders roll back on error.
    if let Err(error) = builder.append_json(json) {
        if !cast_options.safe {
            return Err(error.into());
        }
        builder.append_null();
    }
    Ok(())
}

fn invoke_parse_json(
    args: ScalarFunctionArgs,
    cast_options: CastOptions<'_>,
) -> Result<ColumnarValue> {
    let [input] = args.args.as_slice() else {
        return exec_err!("expected 1 argument");
    };
    match input {
        ColumnarValue::Scalar(scalar_value) => {
            let json = try_parse_string_scalar(scalar_value)?;
            let mut builder = VariantArrayBuilder::new(1);
            append_json(&mut builder, json.map(String::as_str), &cast_options)?;
            Ok(ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
                builder.build().into_inner(),
            ))))
        }
        ColumnarValue::Array(array) if !cast_options.safe => Ok(ColumnarValue::Array(Arc::new(
            json_to_variant(array)?.into_inner(),
        ))),
        ColumnarValue::Array(array) => {
            // TODO: Delegate to json_to_variant once it supports CastOptions:
            // https://github.com/apache/arrow-rs/issues/11775
            let values = try_parse_string_columnar(array)?;
            let mut builder = VariantArrayBuilder::new(values.len());
            for json in values {
                append_json(&mut builder, json, &cast_options)?;
            }
            Ok(ColumnarValue::Array(Arc::new(builder.build().into_inner())))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_parsing_rejects_non_string_inputs() -> Result<()> {
        let udf = TryParseJsonUdf::default();
        let scalar = ScalarValue::Int32(Some(1));
        for input in [
            ColumnarValue::Scalar(scalar.clone()),
            ColumnarValue::Array(scalar.to_array()?),
        ] {
            let args = ScalarFunctionArgs {
                args: vec![input],
                arg_fields: vec![Arc::new(Field::new("input", DataType::Int32, false))],
                return_field: Arc::new(variant_field("result")),
                number_rows: 1,
                config_options: Default::default(),
            };
            assert!(matches!(
                udf.invoke_with_args(args),
                Err(datafusion::error::DataFusionError::Execution(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn scalar_and_array_parsing_agree() -> Result<()> {
        let strict = ParseJsonUdf::default();
        let try_parse = TryParseJsonUdf::default();
        for udf in [&strict as &dyn ScalarUDFImpl, &try_parse] {
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
        }
        Ok(())
    }
}
