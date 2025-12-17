use std::collections::BTreeMap;

use arrow::array::AsArray;
use arrow_schema::{DataType, TimeUnit};
use datafusion::{
    common::exec_err,
    error::{DataFusionError, Result},
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

/// For schema_from_variant Schema representation
/// there are 4 possible types available depending on the Variant
/// For ColumnarValue::Scalar:
/// Primitive -> Just the corresponding SQL data type based on Variant type
/// Array -> Should be an <ARRAY<Type>> given that type is the same, if not <ARRAY<Variant>>
/// Object -> Each object field should keep track of its type <OBJECT<foo: STRING, bar: INT(8, SIGNED)>>
///
/// For ColumnarValue::Array we get the type for each individual Variant value and compare it with the rest.
/// - If one of the values is <VARIANT> type => we can early terminate and call everything <VARIANT>
///
/// For different Variant types we will use different ways of keeping track of types between the rows:
/// - If the outer/inner type differs, we call it <VARIANT> and early terminate this level.
/// - Primitive -> create a set to store Primitive types and if the set.len() > 1, we call everything <VARIANT>
/// - Array -> array type is flat, so the same implementation as Primitive should work for this Array type
/// - Object -> is the difficult one in this case. Different rows can include or exlude certain fields, and we
///   need to keep track of each field between different rows separately. To keep it the fields sorted we will use
///   a BTree with fields as keys and set of types as values. We will add values to each field's set and if set.len()
///   \> 1, we call this field <VARIANT> and we no longer need to keep track of it.
///    
///
/// Later we should also implement Databricks' coersion into a similiar type:
/// > "The schema of each VARIANT value is merged together by field name. When two fields with the same name have
/// > a different type across records, Databricks uses the least common type. When no such type exists, the type
/// > is derived as a VARIANT. For example, INT and DOUBLE become DOUBLE, while TIMESTAMP and STRING become VARIANT." \
/// > https://docs.databricks.com/gcp/en/sql/language-manual/functions/schema_of_variant_agg
///
#[derive(Debug, PartialEq, Eq, Clone)]
enum VariantSchema {
    Primitive(PrimitiveType),
    Array(Box<VariantSchema>),
    Object(BTreeMap<String, VariantSchema>),
    Variant,
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum PrimitiveType {
    Int { bits: u8, signed: bool },
    Float,
    Double,
    Decimal { precision: u8, scale: u8 },
    Boolean,
    String,
    Binary,
    Date,
    Time,
    Timestamp { utc: bool, unit: TimeUnit },
    Uuid,
    Null,
}

/// This function extracts the schema from a single Variant scalar
fn schema_from_variant(v: &Variant) -> VariantSchema {
    match v {
        Variant::Object(obj) => {
            let fields = obj
                .iter()
                .map(|(k, v)| (k.to_string(), schema_from_variant(&v)))
                .collect();

            VariantSchema::Object(fields)
        }

        Variant::List(list) => {
            let inner = list
                .iter()
                .map(|v| schema_from_variant(&v))
                .reduce(merge_variant_schema)
                .unwrap_or(VariantSchema::Variant);

            VariantSchema::Array(Box::new(inner))
        }
        // primitives
        _ => VariantSchema::Primitive(primitive_from_variant(v)),
    }
}

fn schema_to_string(schema: &VariantSchema) -> String {
    match schema {
        VariantSchema::Primitive(s) => primitive_to_string(s),

        VariantSchema::Variant => "VARIANT".to_string(),

        VariantSchema::Array(inner) => {
            format!("ARRAY<{}>", schema_to_string(inner))
        }

        VariantSchema::Object(fields) => {
            let parts: Vec<String> = fields
                .iter()
                .map(|(k, v)| format!("{k}: {}", schema_to_string(v)))
                .collect();
            format!("OBJECT<{}>", parts.join(", "))
        }
    }
}

fn primitive_to_string(p: &PrimitiveType) -> String {
    match p {
        PrimitiveType::Int { bits, signed } => format!(
            "INT({bits}, {})",
            if *signed { "SIGNED" } else { "UNSIGNED" }
        ),
        PrimitiveType::Float => "FLOAT".to_string(),
        PrimitiveType::Double => "DOUBLE".to_string(),
        PrimitiveType::Decimal { precision, scale } => format!("DECIMAL({precision}, {scale})"),
        PrimitiveType::Boolean => "BOOLEAN".to_string(),
        PrimitiveType::String => "STRING".to_string(),
        PrimitiveType::Binary => "BINARY".to_string(),
        PrimitiveType::Date => "DATE".to_string(),
        PrimitiveType::Time => "TIME".to_string(),
        PrimitiveType::Timestamp { utc, unit } => {
            format!("TIMESTAMP(isAdjustedToUTC={utc}, {unit:?})")
        }
        PrimitiveType::Uuid => "UUID".to_string(),
        PrimitiveType::Null => "NULL".to_string(),
    }
}

fn primitive_from_variant<'m, 'v>(v: &Variant<'m, 'v>) -> PrimitiveType {
    match v {
        Variant::Null => PrimitiveType::Null,
        Variant::Int8(_) => PrimitiveType::Int {
            bits: 8,
            signed: true,
        },
        Variant::Int16(_) => PrimitiveType::Int {
            bits: 16,
            signed: true,
        },
        Variant::Int32(_) => PrimitiveType::Int {
            bits: 32,
            signed: true,
        },
        Variant::Int64(_) => PrimitiveType::Int {
            bits: 64,
            signed: true,
        },
        Variant::Float(_) => PrimitiveType::Float,
        Variant::Double(_) => PrimitiveType::Double,
        Variant::Decimal4(d) => PrimitiveType::Decimal {
            precision: d.integer().to_string().len() as u8,
            scale: d.scale(),
        },
        Variant::Decimal8(d) => PrimitiveType::Decimal {
            precision: d.integer().to_string().len() as u8,
            scale: d.scale(),
        },
        Variant::Decimal16(d) => PrimitiveType::Decimal {
            precision: d.integer().to_string().len() as u8,
            scale: d.scale(),
        },
        Variant::BooleanTrue | Variant::BooleanFalse => PrimitiveType::Boolean,
        Variant::String(_) | Variant::ShortString(_) => PrimitiveType::String,
        Variant::Binary(_) => PrimitiveType::Binary,
        Variant::Date(_) => PrimitiveType::Date,
        Variant::Time(_) => PrimitiveType::Time,
        Variant::TimestampMicros(_) => PrimitiveType::Timestamp {
            utc: true,
            unit: TimeUnit::Microsecond,
        },
        Variant::TimestampNtzMicros(_) => PrimitiveType::Timestamp {
            utc: false,
            unit: TimeUnit::Microsecond,
        },
        Variant::TimestampNanos(_) => PrimitiveType::Timestamp {
            utc: true,
            unit: TimeUnit::Nanosecond,
        },
        Variant::TimestampNtzNanos(_) => PrimitiveType::Timestamp {
            utc: false,
            unit: TimeUnit::Nanosecond,
        },
        Variant::Uuid(_) => PrimitiveType::Uuid,
        _ => unreachable!("Should be only applied to Primitive Variant"),
    }
}

fn merge_primitives(a: PrimitiveType, b: PrimitiveType) -> Option<PrimitiveType> {
    use PrimitiveType::*;

    match (a, b) {
        // null handling
        (Null, x) | (x, Null) => Some(x),
        // normal case
        (x, y) if x == y => Some(x),
        // numeric widening
        (Int { .. }, Double) | (Double, Int { .. }) => Some(Double),
        (Int { .. }, Float) | (Float, Int { .. }) => Some(Float),
        (Float, Double) | (Double, Float) => Some(Double),

        // decimal rules (simplified)
        (
            Decimal {
                precision: p1,
                scale: s1,
            },
            Decimal {
                precision: p2,
                scale: s2,
            },
        ) => Some(Decimal {
            precision: p1.max(p2),
            scale: s1.max(s2),
        }),

        _ => None,
    }
}

fn merge_variant_schema(a: VariantSchema, b: VariantSchema) -> VariantSchema {
    use VariantSchema::*;

    match (a, b) {
        (Variant, _) | (_, Variant) => Variant,

        (Primitive(p1), Primitive(p2)) => {
            merge_primitives(p1, p2).map(Primitive).unwrap_or(Variant)
        }

        (Array(a), Array(b)) => Array(Box::new(merge_variant_schema(*a, *b))),

        (Object(mut a), Object(b)) => {
            for (k, v_b) in b {
                a.entry(k)
                    .and_modify(|v_a| *v_a = merge_variant_schema(v_a.clone(), v_b.clone()))
                    .or_insert(v_b);
            }
            Object(a)
        }

        _ => Variant,
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

            let schema = schema_from_variant(&v);
            let schema_str = schema_to_string(&schema);

            Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(Some(
                schema_str,
            ))))
        }
        ColumnarValue::Array(arr) => {
            let variant_array =
                VariantArray::try_new(arr.as_struct()).expect("Expect VariantArray");

            let final_schema = variant_array
                .iter()
                .flatten()
                .map(|v| schema_from_variant(&v))
                .reduce(merge_variant_schema)
                .unwrap_or(VariantSchema::Variant);

            Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(Some(
                schema_to_string(&final_schema),
            ))))
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
        let arg = args.args.first().ok_or_else(|| {
            DataFusionError::Execution("empty argument, expected 1 argument".to_string())
        })?;
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

    use crate::{
        VariantSchemaUDF,
        shared::{build_variant_array_from_json, build_variant_array_from_json_array},
    };

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
        assert_eq!(schema, "TIMESTAMP(isAdjustedToUTC=true, Microsecond)")
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
        let variant_array = build_variant_array_from_json_array(&[
            Some(serde_json::json!({"foo": "bar", "wing": {"ding": "dong"}})),
            None,
            Some(serde_json::json!({"wing": {"ding": "man"}})),
        ]);
        let struct_array = variant_array.into_inner();
        let args = build_array_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "OBJECT<foo: STRING, wing: OBJECT<ding: STRING>>")
    }

    #[test]
    fn test_get_array_variant_conflicting_schema() {
        let udf = VariantSchemaUDF::default();
        let variant_array = build_variant_array_from_json_array(&[
            Some(serde_json::json!({"foo": "bar", "wing": {"ding": "dong"}})),
            None,
            Some(serde_json::json!({"wing": 123})),
        ]);
        let struct_array = variant_array.into_inner();
        let args = build_array_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "OBJECT<foo: STRING, wing: VARIANT>")
    }
}
