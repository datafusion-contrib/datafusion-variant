use std::{ops::Deref, sync::Arc};

use arrow::array::AsArray;
use arrow_schema::{DataType, Field, Fields, TimeUnit};
use datafusion::{
    common::exec_err,
    error::{DataFusionError, Result},
    logical_expr::{ColumnarValue, ScalarUDFImpl, Signature, TypeSignature, Volatility},
    scalar::ScalarValue,
};
use parquet_variant::Variant;
use parquet_variant_compute::{VariantArray, VariantType};

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
/// Variant
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
/// This function extracts the schema from a single Variant scalar
fn schema_from_variant(v: &Variant) -> DataType {
    match v {
        Variant::Object(obj) => {
            let fields = obj
                .iter()
                .map(|(k, v)| Field::new(k.to_string(), schema_from_variant(&v), true))
                .collect();

            DataType::Struct(fields)
        }

        Variant::List(list) => {
            let inner = list
                .iter()
                .map(|v| Field::new("", schema_from_variant(&v), true))
                .reduce(merge_fields)
                .unwrap_or(
                    Field::new("item", DataType::Binary, true).with_extension_type(VariantType),
                );

            DataType::List(Arc::new(inner))
        }
        // primitives
        _ => primitive_from_variant(v),
    }
}

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
        _ => unreachable!("Should be only applied to Primitive Variant"),
    }
}

// Todo: needs more work on type coercing
fn merge_datatypes(a: DataType, b: DataType) -> DataType {
    use DataType::*;

    match (a, b) {
        // null handling
        (Null, x) | (x, Null) => x.clone(),
        // normal case
        (x, y) if x == y => x.clone(),
        // numeric widening
        // docs.databricks.com/aws/en/sql/language-manual/sql-ref-datatype-rules#type-precedence-list
        // For least common type resolution FLOAT is skipped to avoid loss of precision.
        (Int8 | Int16 | Int32 | Int64 | Float32, Float64)
        | (Float64, Int8 | Int16 | Int32 | Int64 | Float32) => Float64,
        (Int8 | Int16 | Int32, Int64) | (Int64, Int8 | Int16 | Int32) => Int64,
        (Int8 | Int16, Int32) | (Int32, Int8 | Int16) => Int32,

        (Date32, Timestamp(tu, tz)) | (Timestamp(tu, tz), Date32) => Timestamp(tu, tz),

        // // decimal rules (simplified)
        // (
        //     Decimal {
        //         precision: p1,
        //         scale: s1,
        //     },
        //     Decimal {
        //         precision: p2,
        //         scale: s2,
        //     },
        // ) => Some(Decimal {
        //     precision: p1.max(p2),
        //     scale: s1.max(s2),
        // }),
        (List(a), List(b)) => {
            DataType::List(Arc::new(merge_fields(a.deref().clone(), b.deref().clone())))
        }

        (Struct(a), Struct(b)) => {
            // Step 1: extract Fields into Vec<Field>
            let mut merged_fields: Vec<Field> = a
                .iter() // iterates over &Arc<Field>
                .map(|f| f.as_ref().clone()) // clone Field out of Arc
                .collect();

            // Step 2: merge b_fields
            for b_field in b.iter() {
                if let Some(existing) = merged_fields
                    .iter_mut()
                    .find(|f| f.name() == b_field.name())
                {
                    *existing = merge_fields(existing.clone(), b_field.deref().clone());
                } else {
                    merged_fields.push((**b_field).clone()); // clone b_field Field
                }
            }

            // Step 3: build new Struct
            DataType::Struct(Fields::from(merged_fields))
        }
        _ => unreachable!("the cases above should cover everything"),
    }
}

fn merge_fields(a: Field, b: Field) -> Field {
    if a.extension_type_name() == Some("VARIANT") && b.extension_type_name() == Some("VARIANT") {
        return Field::new("merged_field", DataType::Binary, true).with_extension_type(VariantType);
    }
    let merged_type = merge_datatypes(a.data_type().clone(), b.data_type().clone());

    Field::new(a.name(), merged_type, a.is_nullable() || b.is_nullable())
}

fn infer_variant_schema(variant: &ColumnarValue) -> Result<ColumnarValue> {
    match variant {
        ColumnarValue::Scalar(scalar) => {
            let ScalarValue::Struct(struct_array) = scalar else {
                return exec_err!("Unsupported data type: {}", scalar.data_type());
            };
            let variant_array = VariantArray::try_new(struct_array.as_ref())?;
            let v = variant_array.value(0);

            let data_type = schema_from_variant(&v);

            Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(Some(format!(
                "{data_type}"
            )))))
        }
        ColumnarValue::Array(arr) => {
            let variant_array =
                VariantArray::try_new(arr.as_struct()).expect("Expect VariantArray");

            let final_schema = variant_array
                .iter()
                .flatten()
                .map(|v| schema_from_variant(&v))
                .reduce(merge_datatypes);

            Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(Some(format!(
                "{final_schema:?}"
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
            args: vec![ColumnarValue::Array(Arc::new(struct_array))],
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
        // assert_eq!(schema, "OBJECT<foo: Utf8, wing: OBJECT<ding: Utf8>>")
        assert_eq!(schema, "Some(Struct([Field { name: \"foo\", data_type: Utf8, nullable: true }, Field { name: \"wing\", data_type: Struct([Field { name: \"ding\", data_type: Utf8, nullable: true }]), nullable: true }]))")
    }

    #[test]
    fn test_get_array_variant_conflicting_schema() {
        let udf = VariantSchemaUDF::default();
        let variant_array = build_variant_array_from_json_array(&[
            Some(serde_json::json!({"foo": "bar", "wing": {"ding": "dong"}})),
            // None,
            Some(serde_json::json!({"wing": 123})),
        ]);
        let struct_array = variant_array.into_inner();
        let args = build_array_udf_args(struct_array);
        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
            panic!()
        };
        assert_eq!(schema, "OBJECT<foo: Utf8, wing: VARIANT>")
    }
}
