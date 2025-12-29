use arrow_schema::{DataType, TimeUnit};
use datafusion::{
    common::exec_err,
    error::{DataFusionError, Result},
    logical_expr::{ColumnarValue, ScalarUDFImpl, Signature, TypeSignature, Volatility},
    scalar::ScalarValue,
};
use parquet_variant::Variant;
use parquet_variant_compute::VariantArray;
use std::collections::BTreeMap;

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

/// Schema inference rules for VARIANT values.
///
/// The inferred schema can be one of four logical forms:
/// - Primitive: a concrete SQL / Arrow data type
/// - Array: ARRAY<inner>, where `inner` is the merged element schema
/// - Object: OBJECT<field: schema, ...>, merged field-wise by name
/// - Variant: fallback when no common schema can be determined
///
/// ## Scalar input
/// When the input is a single VARIANT value:
/// - Primitive values map directly to their corresponding data type
/// - Arrays infer a common element schema across all elements
/// - Objects infer schemas per field recursively
/// - Mixed or incompatible types resolve to VARIANT
///
/// ## Array input
/// When the input is an array of VARIANT values:
/// - Each element is inferred independently
/// - Schemas are merged across rows
/// - If any merge step resolves to VARIANT, inference short-circuits
///
/// ## Merge rules
/// - If outer (or inner) kinds differ, the result is VARIANT
/// - Primitive types are merged using widening / least-common-type rules
/// - Arrays merge by merging their element schemas
/// - Objects merge field-by-field:
///   - Missing fields are allowed
///   - Field schemas are merged independently
///   - A field becomes VARIANT if its values are incompatible
///
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum VariantSchema {
    Primitive(DataType),
    Array(Box<VariantSchema>),
    Object(BTreeMap<String, VariantSchema>),
    Variant,
}

/// This function extracts the schema from a single Variant scalar
pub fn schema_from_variant(v: &Variant) -> VariantSchema {
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
                .unwrap_or(VariantSchema::Primitive(DataType::Null));

            VariantSchema::Array(Box::new(inner))
        }
        _ => VariantSchema::Primitive(primitive_from_variant(v)),
    }
}

/// This helper function is used to calculate decimal precision
/// for [primitive_from_variant] decimal Variants conversion
fn decimal_precision<T: Into<i128>>(val: T) -> u8 {
    let mut n = val.into();
    if n == 0 {
        return 1;
    }
    if n < 0 {
        n = -n
    }

    let mut digits = 0;
    while n != 0 {
        digits += 1;
        n /= 10;
    }
    digits
}

/// This function is used to extract datatype from a primitive Variant
fn primitive_from_variant<'m, 'v>(v: &Variant<'m, 'v>) -> DataType {
    match v {
        Variant::Null => DataType::Null,
        Variant::Int8(_) => DataType::Int8,
        Variant::Int16(_) => DataType::Int16,
        Variant::Int32(_) => DataType::Int32,
        Variant::Int64(_) => DataType::Int64,
        Variant::Float(_) => DataType::Float32,
        Variant::Double(_) => DataType::Float64,
        Variant::Decimal4(d) => {
            DataType::Decimal32(decimal_precision(d.integer()), d.scale() as i8)
        }
        Variant::Decimal8(d) => {
            DataType::Decimal64(decimal_precision(d.integer()), d.scale() as i8)
        }
        Variant::Decimal16(d) => {
            DataType::Decimal128(decimal_precision(d.integer()), d.scale() as i8)
        }
        Variant::BooleanTrue | Variant::BooleanFalse => DataType::Boolean,
        Variant::String(_) | Variant::ShortString(_) | Variant::Uuid(_) => DataType::Utf8,
        Variant::Binary(_) => DataType::Binary,
        Variant::Date(_) => DataType::Date32,
        Variant::Time(_) => DataType::Time64(TimeUnit::Microsecond),
        Variant::TimestampMicros(_) => {
            DataType::Timestamp(TimeUnit::Microsecond, Some("utc".into()))
        }
        Variant::TimestampNtzMicros(_) => DataType::Timestamp(TimeUnit::Microsecond, None),
        Variant::TimestampNanos(_) => DataType::Timestamp(TimeUnit::Nanosecond, Some("utc".into())),
        Variant::TimestampNtzNanos(_) => DataType::Timestamp(TimeUnit::Nanosecond, None),
        _ => unreachable!("Should be only applied to Primitive Variant, not Object or List"),
    }
}

/// This function is used to merge types between schemas
/// and coerce them into a common type when possible if types
/// are different
///
/// Todo: needs more work on type coercing
/// - add decimal coercion rules
fn merge_primitives(a: DataType, b: DataType) -> Option<DataType> {
    use DataType::*;

    match (a, b) {
        (x, y) if x == y => Some(x),
        // numeric widening
        // docs.databricks.com/aws/en/sql/language-manual/sql-ref-datatype-rules#type-precedence-list
        // For least common type resolution FLOAT is skipped to avoid loss of precision.
        (Int8 | Int16 | Int32 | Int64 | Float32, Float64)
        | (Float64, Int8 | Int16 | Int32 | Int64 | Float32) => Some(Float64),
        (Int8 | Int16 | Int32, Int64) | (Int64, Int8 | Int16 | Int32) => Some(Int64),
        (Int8 | Int16, Int32) | (Int32, Int8 | Int16) => Some(Int32),
        (Date32, Timestamp(tu, tz)) | (Timestamp(tu, tz), Date32) => Some(Timestamp(tu, tz)),

        _ => None,
    }
}

/// Merges two inferred Variant schemas into a common schema.
/// Returns VARIANT if no common schema can be determined.
pub fn merge_variant_schema(a: VariantSchema, b: VariantSchema) -> VariantSchema {
    use VariantSchema::*;

    match (a, b) {
        (Variant, _) | (_, Variant) => Variant,

        (Primitive(DataType::Null), x) | (x, Primitive(DataType::Null)) => x,

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

/// Prints schema in a presentable manner
pub fn print_schema(schema: &VariantSchema) -> String {
    match schema {
        VariantSchema::Primitive(s) => format!("{s}"),

        VariantSchema::Variant => "VARIANT".to_string(),

        VariantSchema::Array(inner) => {
            format!("ARRAY<{}>", print_schema(inner))
        }

        VariantSchema::Object(fields) => {
            let parts: Vec<String> = fields
                .iter()
                .map(|(k, v)| format!("{k}: {}", print_schema(v)))
                .collect();
            format!("OBJECT<{}>", parts.join(", "))
        }
    }
}

/// Final function used to retrieve the schema from a single Variant
fn infer_variant_schema(variant: &ColumnarValue) -> Result<ColumnarValue> {
    if let ColumnarValue::Scalar(scalar) = variant {
        let ScalarValue::Struct(struct_array) = scalar else {
            return exec_err!("Unsupported data type: {}", scalar.data_type());
        };
        let variant_array = VariantArray::try_new(struct_array.as_ref())?;
        let v = variant_array.value(0);

        let schema = schema_from_variant(&v);

        Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(Some(
            print_schema(&schema),
        ))))
    } else {
        exec_err!("Expected a ScalarValue, got: {:?}", variant)
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
        Ok(DataType::Utf8View)
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

    use crate::{VariantSchemaUDF, shared::build_variant_array_from_json};

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
        assert_eq!(schema, "Null")
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
        assert_eq!(schema, "Int32")
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
        assert_eq!(schema, "Date32")
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
        assert_eq!(schema, "Timestamp(µs, \"utc\")")
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
        assert_eq!(schema, "Decimal32(4, 1)")
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
        assert_eq!(schema, "Float32")
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
        assert_eq!(schema, "Float64")
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
        assert_eq!(schema, "Boolean")
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
        assert_eq!(schema, "Binary")
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
        assert_eq!(schema, "Utf8")
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
        assert_eq!(schema, "Time64(µs)")
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
        assert_eq!(schema, "OBJECT<data: ARRAY<Int8>, key: Int8>")
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
}
