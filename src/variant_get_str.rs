use std::sync::Arc;

use arrow::array::StringViewArray;
use arrow_schema::DataType;
use datafusion::{
    common::{exec_datafusion_err, exec_err},
    error::{DataFusionError, Result},
    logical_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet_variant::VariantPath;
use parquet_variant_compute::VariantArray;
use parquet_variant_json::VariantToJson;

use crate::shared::{
    try_field_as_variant_array, try_parse_string_columnar, try_parse_string_scalar,
};

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
        let (variant_arg, path_arg) = match args.args.as_slice() {
            [variant_arg, path_arg] => (variant_arg, path_arg),
            _ => return exec_err!("expected 2 arguments"),
        };

        let variant_field = args
            .arg_fields
            .first()
            .ok_or_else(|| exec_datafusion_err!("expected argument field"))?;

        try_field_as_variant_array(variant_field.as_ref())?;

        let out = match (variant_arg, path_arg) {
            (ColumnarValue::Array(variant_array), ColumnarValue::Scalar(path_scalar)) => {
                let path = try_parse_string_scalar(path_scalar)?
                    .map(|s| s.as_str())
                    .unwrap_or_default();

                let variant_array = VariantArray::try_new(variant_array.as_ref())?;
                let out = variant_array_get_str(&variant_array, path)?;

                ColumnarValue::Array(Arc::new(out))
            }
            (ColumnarValue::Scalar(scalar_variant), ColumnarValue::Scalar(path_scalar)) => {
                let ScalarValue::Struct(variant_array) = scalar_variant else {
                    return exec_err!("expected struct array");
                };

                let path = try_parse_string_scalar(path_scalar)?
                    .map(|s| s.as_str())
                    .unwrap_or_default();

                let variant_array = VariantArray::try_new(variant_array.as_ref())?;
                let result = variant_get_str_single(&variant_array, 0, path)?;

                ColumnarValue::Scalar(ScalarValue::Utf8View(result))
            }
            (ColumnarValue::Array(variant_array), ColumnarValue::Array(paths)) => {
                if variant_array.len() != paths.len() {
                    return exec_err!("expected variant array and paths to be of same length");
                }

                let paths = try_parse_string_columnar(paths)?;
                let variant_array = VariantArray::try_new(variant_array.as_ref())?;

                let results: Vec<Option<String>> = (0..variant_array.len())
                    .map(|i| {
                        let path = paths[i].unwrap_or_default();
                        variant_get_str_single(&variant_array, i, path)
                    })
                    .collect::<Result<_>>()?;

                let out: StringViewArray = results.into_iter().collect();
                ColumnarValue::Array(Arc::new(out))
            }
            (ColumnarValue::Scalar(scalar_variant), ColumnarValue::Array(paths)) => {
                let ScalarValue::Struct(variant_array) = scalar_variant else {
                    return exec_err!("expected struct array");
                };

                let variant_array = VariantArray::try_new(variant_array.as_ref())?;
                let paths = try_parse_string_columnar(paths)?;

                let results: Vec<Option<String>> = paths
                    .iter()
                    .map(|path| {
                        let path = path.unwrap_or_default();
                        variant_get_str_single(&variant_array, 0, path)
                    })
                    .collect::<Result<_>>()?;

                let out: StringViewArray = results.into_iter().collect();
                ColumnarValue::Array(Arc::new(out))
            }
        };

        Ok(out)
    }
}

fn variant_get_str_single(
    variant_array: &VariantArray,
    index: usize,
    path: &str,
) -> Result<Option<String>> {
    let Some(variant) = variant_array.iter().nth(index).flatten() else {
        return Ok(None);
    };

    let variant_path = VariantPath::from(path);
    let Some(value) = variant.get_path(&variant_path) else {
        return Ok(None);
    };

    if let Some(s) = value.as_string() {
        Ok(Some(s.to_string()))
    } else {
        // if the path resolves to a non-string variant, return its JSON string
        Ok(Some(value.to_json_string()?))
    }
}

fn variant_array_get_str(variant_array: &VariantArray, path: &str) -> Result<StringViewArray> {
    let variant_path = VariantPath::from(path);

    let results: Vec<Option<String>> = variant_array
        .iter()
        .map(|maybe_variant| {
            let Some(variant) = maybe_variant else {
                return Ok(None);
            };

            let Some(value) = variant.get_path(&variant_path) else {
                return Ok(None);
            };

            if let Some(s) = value.as_string() {
                Ok(Some(s.to_string()))
            } else {
                Ok(Some(value.to_json_string()?))
            }
        })
        .collect::<Result<_, DataFusionError>>()?;

    Ok(results.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, ArrayRef, StructArray};
    use arrow_schema::{Field, Fields};
    use parquet_variant_compute::{VariantArrayBuilder, VariantType};
    use parquet_variant_json::JsonToVariant;

    use super::*;

    fn variant_scalar_from_json(json: serde_json::Value) -> ScalarValue {
        let mut builder = VariantArrayBuilder::new(1);
        builder.append_json(json.to_string().as_str()).unwrap();
        ScalarValue::Struct(Arc::new(builder.build().into()))
    }

    fn variant_array_from_json_rows(json_rows: &[serde_json::Value]) -> ArrayRef {
        let mut builder = VariantArrayBuilder::new(json_rows.len());
        for value in json_rows {
            builder.append_json(value.to_string().as_str()).unwrap();
        }
        let variant_array: StructArray = builder.build().into();
        Arc::new(variant_array) as ArrayRef
    }

    fn standard_arg_fields() -> Vec<Arc<Field>> {
        vec![
            Arc::new(
                Field::new("input", DataType::Struct(Fields::empty()), true)
                    .with_extension_type(VariantType),
            ),
            Arc::new(Field::new("path", DataType::Utf8, true)),
        ]
    }

    fn build_args(
        variant_input: ColumnarValue,
        path: ColumnarValue,
        arg_fields: Vec<Arc<Field>>,
    ) -> ScalarFunctionArgs {
        ScalarFunctionArgs {
            args: vec![variant_input, path],
            return_field: Arc::new(Field::new("result", DataType::Utf8View, true)),
            arg_fields,
            number_rows: Default::default(),
            config_options: Default::default(),
        }
    }

    #[test]
    fn test_scalar_string_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("name".to_string()))),
            standard_arg_fields(),
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
        let args = build_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("age".to_string()))),
            standard_arg_fields(),
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
        let args = build_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("missing".to_string()))),
            standard_arg_fields(),
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
        let args = build_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("obj".to_string()))),
            standard_arg_fields(),
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
        let args = build_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("flag".to_string()))),
            standard_arg_fields(),
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
        let args = build_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("key".to_string()))),
            standard_arg_fields(),
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
        let args = build_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("name".to_string()))),
            standard_arg_fields(),
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
        let args = build_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Array(path_array),
            standard_arg_fields(),
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
        let args = build_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("list".to_string()))),
            standard_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        let json: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(json, serde_json::json!([1, 2, 3]));
    }
}
