use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray};
use arrow_schema::DataType;
use datafusion::{
    error::Result,
    logical_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet_variant::Variant;

use crate::shared::invoke_variant_get_typed;

/// Extracts a bool value from a Variant by path.
///
/// `variant_get_bool(variant, path)` returns the value at `path` as a Boolean.
/// Boolean values are returned as-is
//  Non-boolean values are returned as null
/// - Returns NULL if the path does not exist

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantGetBoolUdf {
    signature: Signature,
}

impl Default for VariantGetBoolUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(2), Volatility::Immutable),
        }
    }
}

fn scalar_from_bool(value: Option<bool>) -> ScalarValue {
    ScalarValue::Boolean(value)
}

fn bool_array_from_values(values: Vec<Option<bool>>) -> ArrayRef {
    Arc::new(values.into_iter().collect::<BooleanArray>())
}

fn extract_bool(value: Variant<'_, '_>) -> Result<Option<bool>> {
    Ok(value.as_boolean())
}

impl ScalarUDFImpl for VariantGetBoolUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_get_bool"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_variant_get_typed(args, scalar_from_bool, bool_array_from_values, extract_bool)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, ArrayRef, BooleanArray, StringViewArray};
    use datafusion::logical_expr::ColumnarValue;
    use datafusion::scalar::ScalarValue;

    use crate::shared::{
        build_variant_get_args, standard_variant_get_arg_fields, variant_array_from_json_rows,
        variant_scalar_from_json,
    };

    use super::*;

    #[test]
    fn test_scalar_bool_true() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50,
            "active": true,
        }));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("active".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Boolean(Some(s))) = result else {
            panic!("expected Boolean scalar");
        };

        assert!(s);
    }

    #[test]
    fn test_scalar_bool_false() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50,
            "active": false,
        }));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("active".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Boolean(Some(s))) = result else {
            panic!("expected Boolean scalar");
        };

        assert!(!s);
    }

    #[test]
    fn test_scalar_missing_path() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"name": "norm"}));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("missing".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Boolean(None)) = result else {
            panic!("expected NULL Boolean scalar");
        };
    }

    #[test]
    fn test_scalar_nested_object() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "obj": {"a": 1, "flag": true}
        }));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("obj.flag".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Boolean(Some(s))) = result else {
            panic!("expected Boolean scalar");
        };

        assert!(s);
    }

    #[test]
    fn test_scalar_null_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"key": null}));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("key".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Boolean(None)) = result else {
            panic!("expected Boolean scalar");
        };
    }

    #[test]
    fn test_array_variant_scalar_path() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "age": 30, "active" : true}),
            serde_json::json!({"name": "bob", "age": 40, "active" : false}),
            serde_json::json!({"age": 50}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("active".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let bool_arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(bool_arr.len(), 3);
        assert_eq!(bool_arr.value(0), true);
        assert_eq!(bool_arr.value(1), false);
        assert!(bool_arr.is_null(2));
    }

    #[test]
    fn test_array_variant_array_paths() {
        // one variant, multiple paths extracted
        // "active" → true, "deleted" → false, "missing" → NULL
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "active": true,
            "deleted": false
        }));

        let path_array: ArrayRef =
            Arc::new(StringViewArray::from(vec!["active", "deleted", "missing"]));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Array(path_array),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let bool_arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(bool_arr.len(), 3);
        assert_eq!(bool_arr.value(0), true);
        assert_eq!(bool_arr.value(1), false);
        assert!(bool_arr.is_null(2)); // missing path → NULL
    }

    #[test]
    fn test_array_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "list": [true, false, true]
        }));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("list".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        // expect NULL for arrays
        let ColumnarValue::Scalar(ScalarValue::Boolean(None)) = result else {
            panic!("expected NULL boolean, got {:?}", result);
        };
    }
}
