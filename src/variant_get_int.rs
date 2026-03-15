use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array};
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

/// Extracts an integer value from a Variant by path.
///
/// `variant_get_int(variant, path)` returns the value at `path` as an `INT64`.
/// - Integer values are returned as-is (with widening when needed)
/// - Non-integer values return NULL
/// - Returns NULL if the path does not exist
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantGetIntUdf {
    signature: Signature,
}

impl Default for VariantGetIntUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(2), Volatility::Immutable),
        }
    }
}

fn scalar_from_int(value: Option<i64>) -> ScalarValue {
    ScalarValue::Int64(value)
}

fn int_array_from_values(values: Vec<Option<i64>>) -> ArrayRef {
    Arc::new(values.into_iter().collect::<Int64Array>())
}

fn extract_int(value: Variant<'_, '_>) -> Result<Option<i64>> {
    Ok(value.as_int64())
}

impl ScalarUDFImpl for VariantGetIntUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_get_int"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_variant_get_typed(args, scalar_from_int, int_array_from_values, extract_int)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, ArrayRef, Int64Array, StringViewArray};
    use datafusion::logical_expr::ColumnarValue;
    use datafusion::scalar::ScalarValue;

    use crate::shared::{
        build_variant_get_args, standard_variant_get_arg_fields, variant_array_from_json_rows,
        variant_scalar_from_json,
    };

    use super::*;

    #[test]
    fn test_scalar_integer_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50
        }));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("age".to_string()))),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Int64(Some(v))) = result else {
            panic!("expected Int64 scalar");
        };

        assert_eq!(v, 50);
    }

    #[test]
    fn test_scalar_non_integer_value_returns_null() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50.5
        }));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("name".to_string()))),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Int64(None)) = result else {
            panic!("expected NULL Int64 scalar");
        };
    }

    #[test]
    fn test_scalar_float_value_returns_null() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "price": 10.5
        }));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("price".to_string()))),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Int64(None)) = result else {
            panic!("expected NULL Int64 scalar");
        };
    }

    #[test]
    fn test_scalar_missing_path() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"name": "norm"}));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("missing".to_string()))),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Int64(None)) = result else {
            panic!("expected NULL Int64 scalar");
        };
    }

    #[test]
    fn test_array_variant_scalar_path() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "age": 30}),
            serde_json::json!({"name": "bob", "age": 40}),
            serde_json::json!({"name": "charlie"}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("age".to_string()))),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let int_arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_arr.len(), 3);
        assert_eq!(int_arr.value(0), 30);
        assert_eq!(int_arr.value(1), 40);
        assert!(int_arr.is_null(2));
    }

    #[test]
    fn test_array_variant_array_paths() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "age": 30}),
            serde_json::json!({"name": "bob", "age": 40}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);
        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec!["age", "name"]));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Array(path_array),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let int_arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_arr.len(), 2);
        assert_eq!(int_arr.value(0), 30);
        assert!(int_arr.is_null(1));
    }

    #[test]
    fn test_scalar_variant_array_paths() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "alice",
            "age": 30
        }));

        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec!["age", "name", "missing"]));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Array(path_array),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let int_arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_arr.len(), 3);
        assert_eq!(int_arr.value(0), 30);
        assert!(int_arr.is_null(1));
        assert!(int_arr.is_null(2));
    }
}
