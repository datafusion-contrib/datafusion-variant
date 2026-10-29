// SPDX-License-Identifier: Apache-2.0

// https://docs.databricks.com/gcp/en/sql/language-manual/functions/to_json

use std::sync::Arc;

use arrow::array::StringViewArray;
use arrow_schema::DataType;
use datafusion::{
    common::{exec_datafusion_err, exec_err},
    error::Result,
    logical_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use datafusion_doc::Documentation;
use datafusion_macros::user_doc;
use parquet_variant_compute::{VariantArray, variant_to_json};
use parquet_variant_json::VariantToJson;

use crate::shared::try_field_as_variant_array;

/// Serializes a Variant value as JSON, preserving SQL NULL.
#[user_doc(
    doc_section(label = "Variant Functions"),
    description = "Serializes a Variant value to a JSON string. SQL NULL returns SQL NULL;
Variant null becomes the string `null`. Strings retain their JSON quotes.
This function accepts Variant values only. The optional second argument is currently
ignored; serialization options are not implemented.",
    syntax_example = "to_json(value[, options])",
    sql_example = r#"```sql
> SELECT to_json(parse_json('"hello"')) AS json;
+---------+
| json    |
+---------+
| "hello" |
+---------+
> SELECT to_json(parse_json('null')) AS json, to_json(parse_json(NULL)) IS NULL AS sql_null;
+------+----------+
| json | sql_null |
+------+----------+
| null | true     |
+------+----------+
```"#,
    argument(name = "value", description = "Variant expression to serialize."),
    argument(
        name = "options",
        description = "Optional argument, currently ignored."
    ),
    related_udf(name = "parse_json"),
    related_udf(name = "try_parse_json")
)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantToJsonUdf {
    signature: Signature,
}

impl Default for VariantToJsonUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(
                TypeSignature::OneOf(vec![TypeSignature::Any(1), TypeSignature::Any(2)]),
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for VariantToJsonUdf {
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn name(&self) -> &str {
        "to_json"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8View)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let field = args
            .arg_fields
            .first()
            .ok_or_else(|| exec_datafusion_err!("empty argument, expected 1 argument"))?;

        try_field_as_variant_array(field.as_ref())?;

        // TODO: Parse and validate the optional options map instead of ignoring it.
        // Spark's Variant serializer uses its timeZone option when formatting timestamps.
        // First add time-zone options to Arrow's VariantToJson scalar serializer and
        // variant_to_json batch kernel, including timestamps nested in objects/arrays;
        // then wire the UDF option and session time zone through both execution paths.
        let arg = args
            .args
            .first()
            .ok_or_else(|| exec_datafusion_err!("empty argument, expected 1 argument"))?;

        let out = match arg {
            ColumnarValue::Scalar(scalar) => {
                let ScalarValue::Struct(variant_array) = scalar else {
                    return exec_err!("Unsupported data type: {}", scalar.data_type());
                };

                let variant_array = VariantArray::try_new(variant_array.as_ref())?;
                let json = if variant_array.is_null(0) {
                    None
                } else {
                    Some(variant_array.value(0).to_json_string()?)
                };

                ColumnarValue::Scalar(ScalarValue::Utf8View(json))
            }
            ColumnarValue::Array(input) => {
                let json = variant_to_json(input)?;
                // Preserve the UDF's Utf8View output while sharing Arrow's JSON
                // buffer for non-inline strings.
                ColumnarValue::Array(Arc::new(StringViewArray::from(&json)))
            }
        };

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::{Field, Fields};
    use parquet_variant_compute::VariantType;
    use serde_json::Value;

    use crate::shared::build_variant_array_from_json;

    use super::*;

    #[test]
    fn test_scalar_primitive() {
        let expected_json = serde_json::json!("norm");
        let input = build_variant_array_from_json(&expected_json);

        let variant_input = ScalarValue::Struct(Arc::new(input.into()));

        let udf = VariantToJsonUdf::default();
        let return_field = Arc::new(Field::new("result", DataType::Utf8View, true));
        let arg_field = Arc::new(
            Field::new("input", DataType::Struct(Fields::empty()), true)
                .with_extension_type(VariantType),
        );

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(variant_input)],
            return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(j))) = result else {
            panic!("expected valid json string")
        };

        assert_eq!(j.as_str(), r#""norm""#);
    }

    #[test]
    fn test_to_json_udf_scalar_complex() {
        let expected_json = serde_json::json!({
            "name": "norm",
            "age": 50,
            "list": [false, true, ()]
        });

        let input = build_variant_array_from_json(&expected_json);

        let variant_input = ScalarValue::Struct(Arc::new(input.into()));

        let udf = VariantToJsonUdf::default();

        let return_field = Arc::new(Field::new("result", DataType::Utf8View, true));
        let arg_field = Arc::new(
            Field::new("input", DataType::Struct(Fields::empty()), true)
                .with_extension_type(VariantType),
        );

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(variant_input)],
            return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(j))) = result else {
            panic!("expected valid json string")
        };

        let json: Value = serde_json::from_str(j.as_str()).unwrap();
        assert_eq!(json, expected_json);
    }
}
