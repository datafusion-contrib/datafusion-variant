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
use datafusion_doc::Documentation;
use datafusion_macros::user_doc;
use parquet_variant_compute::{VariantArrayBuilder, json_to_variant};
use parquet_variant_json::JsonToVariant as JsonToVariantExt;

use crate::shared::{try_parse_string_columnar, try_parse_string_scalar};
use crate::variant_type::variant_field;

/// Parses JSON, returning an error for invalid JSON values.
#[user_doc(
    doc_section(label = "Variant Functions"),
    description = "Parses a JSON string into a Variant value. Invalid JSON raises an error.
SQL NULL returns SQL NULL; the JSON text `null` produces a present Variant null.

Uses Arrow's JSON parser. Fractional JSON numbers become DOUBLE; decimal precision
and duplicate-key handling have known differences from Spark (see [#71](https://github.com/datafusion-contrib/datafusion-variant/issues/71)).",
    syntax_example = "parse_json(json)",
    sql_example = r#"```sql
> SELECT to_json(parse_json('{"a":1}')) AS json;
+---------+
| json    |
+---------+
| {"a":1} |
+---------+
> SELECT parse_json(NULL) IS NULL AS sql_null, is_variant_null(parse_json('null')) AS variant_null;
+----------+--------------+
| sql_null | variant_null |
+----------+--------------+
| true     | true         |
+----------+--------------+
```"#,
    argument(
        name = "json",
        description = "String expression containing a JSON value."
    ),
    related_udf(name = "try_parse_json"),
    related_udf(name = "to_json")
)]
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
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

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
#[user_doc(
    doc_section(label = "Variant Functions"),
    description = "Parses a JSON string into a Variant value, returning SQL NULL for invalid JSON.
In a column, a malformed value produces NULL for that row. SQL NULL returns SQL NULL;
the JSON text `null` produces a present Variant null. Invalid argument types or counts
still raise errors. Shares the parser limitations of `parse_json`.",
    syntax_example = "try_parse_json(json)",
    sql_example = r#"```sql
> SELECT to_json(try_parse_json('{"a":1}')) AS json;
+---------+
| json    |
+---------+
| {"a":1} |
+---------+
> SELECT try_parse_json('{') IS NULL AS invalid_json, is_variant_null(try_parse_json('null')) AS variant_null;
+--------------+--------------+
| invalid_json | variant_null |
+--------------+--------------+
| true         | true         |
+--------------+--------------+
```"#,
    argument(
        name = "json",
        description = "String expression containing a JSON value."
    ),
    related_udf(name = "parse_json"),
    related_udf(name = "to_json")
)]
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
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

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
