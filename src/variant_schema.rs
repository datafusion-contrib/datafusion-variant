use std::collections::BTreeMap;

use arrow::array::AsArray;
use arrow_schema::DataType;
use datafusion::{
    common::{exec_datafusion_err, exec_err},
    error::Result,
    logical_expr::{ColumnarValue, ScalarUDFImpl, Signature, TypeSignature, Volatility},
    scalar::ScalarValue,
};
use parquet_variant::Variant;
use parquet_variant_compute::VariantArray;

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantSchemaUDF {
    signature: Signature,
}

impl Default for VariantSchemaUDF {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::VariadicAny, Volatility::Immutable),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum VariantSchema {
    Primitive(String),
    Object(BTreeMap<String, VariantSchema>),
    Array(Box<VariantSchema>),
    Variant,
}

fn schema_from_variant(v: &Variant) -> VariantSchema {
    match v {
        Variant::Object(obj) => {
            let mut fields = BTreeMap::new();
            for (k, v) in obj.iter() {
                fields.insert(k.to_string(), schema_from_variant(&v));
            }
            VariantSchema::Object(fields)
        }
        Variant::List(list) => {
            let mut schemas: Vec<VariantSchema> =
                list.iter().map(|v| schema_from_variant(v)).collect();

                schemas.sort();
                schemas.dedup();

                if schemas.len() == 1 {
                    VariantSchema::Array(Box::new(schemas.pop().unwrap()))
                } else {
                    VariantSchema::Array(Box::new(VariantSchema::Variant))
                }
                }
                // primitives
                _ => VariantSchema::Primitive(variant_schema_str(v))
        }
    }

fn variant_schema_str<'m, 'v>(v: &Variant<'m, 'v>) -> String {
    match v {
        Variant::Null => "NULL".to_string(),
        Variant::Int8(_) => "INT(8, SIGNED)".to_string(),
        Variant::Int16(_) => "INT(16, SIGNED)".to_string(),
        Variant::Int32(_) => "INT(32, SIGNED)".to_string(),
        Variant::Int64(_) => "INT(64, SIGNED)".to_string(),
        Variant::Float(_) => "FLOAT".to_string(),
        Variant::Double(_) => "DOUBLE".to_string(),
        Variant::Decimal4(d) => {
            format!("DECIMAL({}, {})", d.integer().to_string().len(), d.scale())
        }
        Variant::Decimal8(d) => {
            format!("DECIMAL({}, {})", d.integer().to_string().len(), d.scale())
        }
        Variant::Decimal16(d) => {
            format!("DECIMAL({}, {})", d.integer().to_string().len(), d.scale())
        }
        Variant::BooleanTrue | Variant::BooleanFalse => "BOOLEAN".to_string(),
        Variant::String(_) | Variant::ShortString(_) => "STRING".to_string(),
        Variant::Binary(_) => "BINARY".to_string(),
        Variant::Date(_) => "DATE".to_string(),
        Variant::Time(_) => "TIME".to_string(),
        Variant::TimestampMicros(_) => "TIMESTAMP(isAdjustedToUTC=true, MICROS)".to_string(),
        Variant::TimestampNtzMicros(_) => "TIMESTAMP(isAdjustedToUTC=false, MICROS)".to_string(),
        Variant::TimestampNanos(_) => "TIMESTAMP(isAdjustedToUTC=true, NANOS)".to_string(),
        Variant::TimestampNtzNanos(_) => "TIMESTAMP(isAdjustedToUTC=false, NANOS)".to_string(),
        Variant::Uuid(_) => "UUID".to_string(),

        Variant::Object(obj) => {
            let fields: Vec<String> = obj
                .iter()
                .map(|(k, v)| format!("{k}: {}", variant_schema_str(&v)))
                .collect();
            format!("OBJECT<{}>", fields.join(", "))
        }

        Variant::List(list) => {
            let mut item_types: Vec<String> = list.iter().map(|v| variant_schema_str(&v)).collect();
            item_types.sort();
            item_types.dedup();
            let array_type = if item_types.len() == 1 {
                item_types[0].clone()
            } else {
                "VARIANT".to_string()
            };
            format!("ARRAY<{array_type}>")
        }
    }
}

fn infer_variant_schema(variant: &ColumnarValue) -> Result<ColumnarValue> {
    match variant {
        ColumnarValue::Scalar(scalar) => {
            let ScalarValue::Struct(struct_array) = scalar else {
                return exec_err!("Unsupported data type: {}", scalar.data_type());
            };
            let variant_array = VariantArray::try_new(struct_array.as_ref())?;
            let v = variant_array.value(0);
            let schema_str = variant_schema_str(&v);
            Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(Some(
                schema_str,
            ))))
        }
        ColumnarValue::Array(arr) => {
            let variant_array =
                VariantArray::try_new(arr.as_struct()).expect("Expect VariantArray");
            let mut item_types: Vec<String> = variant_array
                .iter()
                .filter_map(|v| v.as_ref().map(|v| variant_schema_str(v)))
                .collect();
            item_types.sort();
            item_types.dedup();
            let array_type = if item_types.len() == 1 {
                item_types[0].clone()
            } else {
                "VARIANT".to_string()
            };
            Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(Some(format!(
                "ARRAY<{array_type}>"
            )))))
        }
    }
}

impl ScalarUDFImpl for VariantSchemaUDF {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_schema"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let arg = args
            .args
            .first()
            .ok_or_else(|| exec_datafusion_err!("empty argument, expected 1 argument"))?;
        infer_variant_schema(arg)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::StructArray;
    use arrow_schema::{DataType, Field, Fields};
    use chrono::{DateTime, NaiveDate, NaiveTime};
    use datafusion::{
        logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl},
        scalar::ScalarValue,
    };
    use parquet_variant::{Variant, VariantDecimal4};
    use parquet_variant_compute::{VariantArray, VariantType};
    use std::sync::Arc;

    use crate::{VariantSchemaUDF, shared::{build_variant_array_from_json, build_variant_array_from_json_array}};

    fn build_scalar_udf_args(struct_array: StructArray) -> ScalarFunctionArgs {
        let return_field = Arc::new(Field::new("result", DataType::Utf8View, true));
        let arg_field = Arc::new(
            Field::new("input", DataType::Struct(Fields::empty()), true)
                .with_extension_type(VariantType),
        );
        ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
                struct_array,
            )))],
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            return_field,
            config_options: Default::default(),
        }
    }

    fn build_array_udf_args(struct_array: StructArray) -> ScalarFunctionArgs {
        let return_field = Arc::new(Field::new("result", DataType::Utf8View, true));
        let arg_field = Arc::new(
            Field::new("input", DataType::Struct(Fields::empty()), true)
                .with_extension_type(VariantType),
        );
        ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
                struct_array,
            )))],
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            return_field,
            config_options: Default::default(),
        }
    }

    #[test]
    fn test_get_single_typed_null_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant = Variant::Null;
        let variant_array = VariantArray::from_iter(vec![variant]);
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "NULL")
    }

    #[test]
    fn test_get_single_typed_int32_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant = Variant::from(1234i32);
        let variant_array = VariantArray::from_iter(vec![variant]);
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "INT(32, SIGNED)")
    }

    #[test]
    fn test_get_single_typed_date_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant = Variant::from(NaiveDate::from_ymd_opt(1990, 1, 1).expect("Expect NaiveDate"));
        let variant_array = VariantArray::from_iter(vec![variant]);
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "DATE")
    }

    #[test]
    fn test_get_single_typed_timestamp_micro_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant =
            Variant::from(DateTime::from_timestamp(1431648000, 0).expect("Expect TimeStamp"));
        let variant_array = VariantArray::from_iter(vec![variant]);
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "TIMESTAMP(isAdjustedToUTC=true, MICROS)")
    }

    #[test]
    fn test_get_single_typed_decimal_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant = Variant::Decimal4(VariantDecimal4::try_new(1234, 1).expect("Expect decimal"));
        let variant_array = VariantArray::from_iter(vec![variant]);
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "DECIMAL(4, 1)")
    }

    #[test]
    fn test_get_single_typed_float_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant = Variant::from(123.4f32);
        let variant_array = VariantArray::from_iter(vec![variant]);
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "FLOAT")
    }

    #[test]
    fn test_get_single_typed_double_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant = Variant::from(123.4f64);
        let variant_array = VariantArray::from_iter(vec![variant]);
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "DOUBLE")
    }

    #[test]
    fn test_get_single_typed_bool_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant = Variant::BooleanTrue;
        let variant_array = VariantArray::from_iter(vec![variant]);
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "BOOLEAN")
    }

    #[test]
    fn test_get_single_typed_binary_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant = Variant::Binary(&[1u8, 2, 3]);
        let variant_array = VariantArray::from_iter(vec![variant]);
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "BINARY")
    }

    #[test]
    fn test_get_single_typed_string_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant = Variant::from("foo");
        let variant_array = VariantArray::from_iter(vec![variant]);
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "STRING")
    }

    #[test]
    fn test_get_single_typed_time_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant = Variant::from(NaiveTime::from_hms_opt(0, 0, 0).expect("Expect NaiveTime"));
        let variant_array = VariantArray::from_iter(vec![variant]);
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "TIME")
    }

    #[test]
    fn test_get_single_struct_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant_array = build_variant_array_from_json(&serde_json::json!({
            "key": 123, "data": [4, 5]
        }));
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(
            schema,
            "OBJECT<data: ARRAY<INT(8, SIGNED)>, key: INT(8, SIGNED)>"
        )
    }

    #[test]
    fn test_get_single_struct_variant_conflicting_schema() {
        let udf = VariantSchemaUDF::default();
        let variant_array = build_variant_array_from_json(&serde_json::json!({
            "data": [{"a":"a"}, 5]
        }));
        let struct_array = variant_array.into_inner();
        let args = build_scalar_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "OBJECT<data: ARRAY<VARIANT>>")
    }

    #[test]
    fn test_get_array_variant_schema() {
        let udf = VariantSchemaUDF::default();
        let variant_array = build_variant_array_from_json_array(&[Some(serde_json::json!({"foo": "bar", "wing": {"ding": "dong"}})), None, Some(serde_json::json!({"wing": 123}))]);
        let struct_array = variant_array.into_inner();
        let args = build_array_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "OBJECT<data: ARRAY<VARIANT>>")
    }
}
