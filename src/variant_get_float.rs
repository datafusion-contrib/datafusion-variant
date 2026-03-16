use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array};
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

/// Extracts a floating-point value from a Variant by path.
///
/// `variant_get_float(variant, path)` returns the value at `path` as a `FLOAT64`.
/// - Float values are returned as-is
/// - Integer values are returned as `FLOAT64` (large values may lose precision)
/// - Non-numeric values return NULL
/// - Returns NULL if the path does not exist
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantGetFloatUdf {
    signature: Signature,
}

impl Default for VariantGetFloatUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(2), Volatility::Immutable),
        }
    }
}

fn scalar_from_float(value: Option<f64>) -> ScalarValue {
    ScalarValue::Float64(value)
}

fn float_array_from_values(values: Vec<Option<f64>>) -> ArrayRef {
    Arc::new(values.into_iter().collect::<Float64Array>())
}

fn extract_float(value: Variant<'_, '_>) -> Result<Option<f64>> {
    Ok(value
        .as_f64()
        .or_else(|| value.as_int64().map(|int| int as f64)))
}

impl ScalarUDFImpl for VariantGetFloatUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_get_float"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_variant_get_typed(
            args,
            scalar_from_float,
            float_array_from_values,
            extract_float,
        )
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, ArrayRef, Float64Array, StringViewArray};
    use datafusion::logical_expr::ColumnarValue;
    use datafusion::scalar::ScalarValue;

    use crate::shared::{
        build_variant_get_args, standard_variant_get_arg_fields, variant_array_from_json_rows,
        variant_scalar_from_json,
    };

    use super::*;

    #[test]
    fn test_scalar_float_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "price": 50.5
        }));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("price".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Float64(Some(v))) = result else {
            panic!("expected Float64 scalar");
        };

        assert_eq!(v, 50.5);
    }

    #[test]
    fn test_scalar_integer_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50
        }));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("age".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Float64(Some(v))) = result else {
            panic!("expected Float64 scalar");
        };

        assert_eq!(v, 50.0);
    }

    #[test]
    fn test_scalar_large_integer_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "n": 9007199254740993_i64
        }));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("n".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Float64(Some(v))) = result else {
            panic!("expected Float64 scalar");
        };

        // `f64` cannot exactly represent all i64 values; this mirrors json_get_float behavior.
        assert_eq!(v, 9_007_199_254_740_992.0);
    }

    #[test]
    fn test_scalar_non_numeric_value_returns_null() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50.5
        }));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("name".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Float64(None)) = result else {
            panic!("expected NULL Float64 scalar");
        };
    }

    #[test]
    fn test_scalar_missing_path() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"name": "norm"}));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("missing".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Float64(None)) = result else {
            panic!("expected NULL Float64 scalar");
        };
    }

    #[test]
    fn test_array_variant_scalar_path() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "price": 30.25}),
            serde_json::json!({"name": "bob", "price": 40}),
            serde_json::json!({"name": "charlie"}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("price".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let float_arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(float_arr.len(), 3);
        assert_eq!(float_arr.value(0), 30.25);
        assert_eq!(float_arr.value(1), 40.0);
        assert!(float_arr.is_null(2));
    }

    #[test]
    fn test_array_variant_array_paths() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "price": 30.25}),
            serde_json::json!({"name": "bob", "price": 40}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);
        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec!["price", "name"]));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Array(path_array),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let float_arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(float_arr.len(), 2);
        assert_eq!(float_arr.value(0), 30.25);
        assert!(float_arr.is_null(1));
    }

    #[test]
    fn test_scalar_variant_array_paths() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "alice",
            "price": 30.25,
            "count": 3
        }));

        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec![
            "price", "count", "name", "missing",
        ]));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Array(path_array),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let float_arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(float_arr.len(), 4);
        assert_eq!(float_arr.value(0), 30.25);
        assert_eq!(float_arr.value(1), 3.0);
        assert!(float_arr.is_null(2));
        assert!(float_arr.is_null(3));
    }
}
