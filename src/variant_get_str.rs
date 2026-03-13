use std::sync::Arc;

use arrow::array::{ArrayRef, StringViewArray};
use arrow_schema::DataType;
use datafusion::{
    error::Result,
    logical_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet_variant::Variant;
use parquet_variant_json::VariantToJson;

use crate::shared::invoke_variant_get_typed;

/// Extracts a string value from a Variant by path.
///
/// `variant_get_str(variant, path)` returns the value at `path` as a UTF8 string.
/// - String values are returned as-is (no JSON quotes)
/// - Non-string values (numbers, booleans, objects, arrays) are JSON-serialized
/// - Returns NULL if the path does not exist
///
/// This is similar to PostgreSQL's `->>` operator.
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantGetStrUdf {
    signature: Signature,
}

impl Default for VariantGetStrUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(2), Volatility::Immutable),
        }
    }
}

fn scalar_from_string(value: Option<String>) -> ScalarValue {
    ScalarValue::Utf8View(value)
}

fn string_array_from_values(values: Vec<Option<String>>) -> ArrayRef {
    let out: StringViewArray = values.into_iter().collect();
    Arc::new(out)
}

fn extract_string(value: Variant<'_, '_>) -> Result<Option<String>> {
    if let Some(s) = value.as_string() {
        Ok(Some(s.to_string()))
    } else {
        // If the path resolves to a non-string variant, return its JSON string.
        Ok(Some(value.to_json_string()?))
    }
}

impl ScalarUDFImpl for VariantGetStrUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_get_str"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8View)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_variant_get_typed(
            args,
            scalar_from_string,
            string_array_from_values,
            extract_string,
        )
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, ArrayRef, StringViewArray};
    use datafusion::logical_expr::ColumnarValue;
    use datafusion::scalar::ScalarValue;

    use crate::shared::{
        build_variant_get_args, standard_variant_get_arg_fields, variant_array_from_json_rows,
        variant_scalar_from_json,
    };

    use super::*;

    #[test]
    fn test_scalar_string_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("name".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        assert_eq!(s, "norm");
    }

    #[test]
    fn test_scalar_numeric_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("age".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        assert_eq!(s, "50");
    }

    #[test]
    fn test_scalar_missing_path() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"name": "norm"}));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("missing".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(None)) = result else {
            panic!("expected NULL Utf8View scalar");
        };
    }

    #[test]
    fn test_scalar_nested_object() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "obj": {"a": 1, "b": 2}
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("obj".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        let json: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(json, serde_json::json!({"a": 1, "b": 2}));
    }

    #[test]
    fn test_scalar_boolean_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"flag": true}));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("flag".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        assert_eq!(s, "true");
    }

    #[test]
    fn test_scalar_null_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"key": null}));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("key".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        assert_eq!(s, "null");
    }

    #[test]
    fn test_array_variant_scalar_path() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "age": 30}),
            serde_json::json!({"name": "bob", "age": 40}),
            serde_json::json!({"age": 50}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("name".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let str_arr = arr.as_any().downcast_ref::<StringViewArray>().unwrap();
        assert_eq!(str_arr.len(), 3);
        assert_eq!(str_arr.value(0), "alice");
        assert_eq!(str_arr.value(1), "bob");
        assert!(str_arr.is_null(2));
    }

    #[test]
    fn test_array_variant_array_paths() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "age": 30}),
            serde_json::json!({"name": "bob", "age": 40}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);
        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec!["name", "age"]));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Array(path_array),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let str_arr = arr.as_any().downcast_ref::<StringViewArray>().unwrap();
        assert_eq!(str_arr.len(), 2);
        assert_eq!(str_arr.value(0), "alice");
        assert_eq!(str_arr.value(1), "40");
    }

    #[test]
    fn test_array_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "list": [1, 2, 3]
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("list".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        let json: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(json, serde_json::json!([1, 2, 3]));
    }
}
