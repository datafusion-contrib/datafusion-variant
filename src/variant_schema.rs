use arrow::array::{ArrayRef, StringViewArray};
use arrow_schema::{DataType, TimeUnit};
use datafusion::{
    common::exec_err,
    error::{DataFusionError, Result},
    logical_expr::{ColumnarValue, ScalarUDFImpl, Signature, TypeSignature, Volatility},
    scalar::ScalarValue,
};
use parquet_variant::Variant;
use parquet_variant_compute::VariantArray;
use serde_json::Value;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::Arc;

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantSchemaUDF {
    signature: Signature,
}

impl Default for VariantSchemaUDF {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(1), Volatility::Immutable),
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
        Ok(DataType::Utf8View)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let arg = &args.args[0];
        infer_variant_schema(arg)
    }
}

/// Infers a schema description for one VARIANT value.
///
/// The inferred schema can be one of four logical forms:
/// - Primitive: a concrete SQL / Arrow data type
/// - Array: `ARRAY<inner>`, where `inner` is merged across elements in that array value
/// - Object: `OBJECT<field: schema, ...>`, merged recursively per field
/// - Variant: fallback when no common inner schema can be determined
///
/// Execution semantics:
/// - Scalar input: infer one schema string for that value.
/// - Columnar input: infer one schema string per row (vectorized row-wise behavior).
/// - This function does not merge schemas across rows.
///
/// Merge rules (within one VARIANT value only):
/// - If outer (or inner) kinds differ, the result is `VARIANT`
/// - Primitive types are merged using widening / least-common-type rules
/// - Arrays merge by merging their element schemas
/// - Objects merge field-by-field; missing fields are allowed
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum VariantSchema {
    Primitive(DataType),
    Array(Box<VariantSchema>),
    Object(BTreeMap<String, VariantSchema>),
    Variant,
}

impl VariantSchema {
    pub fn to_state_bytes(&self) -> Vec<u8> {
        self.to_state_string().into_bytes()
    }

    pub fn to_state_string(&self) -> String {
        schema_to_json(self).to_string()
    }

    pub fn from_state_bytes(bytes: &[u8]) -> Result<Self> {
        let state = std::str::from_utf8(bytes).map_err(|e| {
            DataFusionError::Execution(format!("invalid variant_schema utf8 state: {e}"))
        })?;
        Self::from_state_str(state)
    }

    pub fn from_state_str(state: &str) -> Result<Self> {
        let value = serde_json::from_str::<Value>(state).map_err(|e| {
            DataFusionError::Execution(format!("invalid variant_schema json state: {e}"))
        })?;
        Self::try_from(&value)
    }
}

impl From<&Variant<'_, '_>> for VariantSchema {
    fn from(value: &Variant) -> Self {
        match value {
            Variant::Object(obj) => {
                let fields = obj
                    .iter()
                    .map(|(k, v)| (k.to_string(), Self::from(&v)))
                    .collect();

                VariantSchema::Object(fields)
            }
            Variant::List(list) => {
                let inner = list
                    .iter()
                    .map(|v| Self::from(&v))
                    .try_fold(VariantSchema::Primitive(DataType::Null), |acc, next| {
                        let merged = merge_variant_schema(acc, next);
                        if merged == VariantSchema::Variant {
                            Err(merged)
                        } else {
                            Ok(merged)
                        }
                    })
                    .unwrap_or_else(|schema| schema);

                VariantSchema::Array(Box::new(inner))
            }
            _ => VariantSchema::Primitive(primitive_from_variant(value)),
        }
    }
}

impl TryFrom<&Value> for VariantSchema {
    type Error = DataFusionError;

    fn try_from(value: &Value) -> std::result::Result<Self, Self::Error> {
        let obj = match value {
            Value::Object(obj) => obj,
            _ => return exec_err!("invalid variant_schema state: expected object"),
        };

        let kind = match obj.get("kind") {
            Some(Value::String(v)) => v.as_str(),
            _ => return exec_err!("invalid variant_schema state: missing or invalid `kind`"),
        };

        match kind {
            "primitive" => {
                let dtype_str = match obj.get("dtype") {
                    Some(Value::String(v)) => v,
                    _ => {
                        return exec_err!(
                            "invalid variant_schema primitive state: missing or invalid `dtype`"
                        );
                    }
                };

                let dtype = match dtype_str.parse::<DataType>() {
                    Ok(v) => v,
                    Err(e) => return exec_err!("invalid variant_schema datatype state: {e}"),
                };
                Ok(VariantSchema::Primitive(dtype))
            }
            "array" => {
                let inner = match obj.get("inner") {
                    Some(v) => v,
                    None => {
                        return exec_err!("invalid variant_schema array state: missing `inner`");
                    }
                };
                Ok(VariantSchema::Array(Box::new(Self::try_from(inner)?)))
            }
            "object" => {
                let fields_obj = match obj.get("fields") {
                    Some(Value::Object(v)) => v,
                    _ => {
                        return exec_err!(
                            "invalid variant_schema object state: missing or invalid `fields`"
                        );
                    }
                };

                let mut fields = BTreeMap::new();
                for (field_name, field_value) in fields_obj {
                    fields.insert(field_name.clone(), Self::try_from(field_value)?);
                }

                Ok(VariantSchema::Object(fields))
            }
            "variant" => Ok(VariantSchema::Variant),
            other => exec_err!("invalid variant_schema state kind: {other}"),
        }
    }
}

fn schema_to_json(schema: &VariantSchema) -> Value {
    match schema {
        VariantSchema::Primitive(dtype) => serde_json::json!({
            "kind": "primitive",
            "dtype": dtype.to_string()
        }),
        VariantSchema::Array(inner) => serde_json::json!({
            "kind": "array",
            "inner": schema_to_json(inner)
        }),
        VariantSchema::Object(fields) => {
            let fields_json = fields
                .iter()
                .map(|(k, v)| (k.clone(), schema_to_json(v)))
                .collect::<serde_json::Map<String, Value>>();

            serde_json::json!({
                "kind": "object",
                "fields": fields_json
            })
        }
        VariantSchema::Variant => serde_json::json!({ "kind": "variant" }),
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
/// docs.databricks.com/aws/en/sql/language-manual/sql-ref-datatype-rules#type-precedence-list
fn merge_primitives(a: DataType, b: DataType) -> Option<DataType> {
    match (a, b) {
        (x, y) if x == y => Some(x),
        _ => None,
    }
}

/// Merges two inferred Variant schemas into a common schema.
/// Returns VARIANT if no common schema can be determined.
pub fn merge_variant_schema(a: VariantSchema, b: VariantSchema) -> VariantSchema {
    let mut merged = a;
    merge_variant_schema_from(&mut merged, &b);
    merged
}

pub fn merge_variant_schema_from(target: &mut VariantSchema, incoming: &VariantSchema) {
    use VariantSchema::*;

    if matches!(target, Variant) || matches!(incoming, Variant) {
        *target = Variant;
        return;
    }

    if matches!(incoming, Primitive(DataType::Null)) {
        return;
    }

    if matches!(target, Primitive(DataType::Null)) {
        *target = incoming.clone();
        return;
    }

    match incoming {
        Primitive(p2) => {
            if let Primitive(p1) = target {
                let merged = merge_primitives(p1.clone(), p2.clone())
                    .map(Primitive)
                    .unwrap_or(Variant);
                *target = merged;
            } else {
                *target = Variant;
            }
        }
        Array(b) => {
            if let Array(a) = target {
                merge_variant_schema_from(a.as_mut(), b.as_ref());
            } else {
                *target = Variant;
            }
        }
        Object(b) => {
            if let Object(a) = target {
                for (k, v_b) in b {
                    match a.entry(k.clone()) {
                        Entry::Occupied(mut occ) => merge_variant_schema_from(occ.get_mut(), v_b),
                        Entry::Vacant(vac) => {
                            vac.insert(v_b.clone());
                        }
                    }
                }
            } else {
                *target = Variant;
            }
        }
        Variant => {
            *target = Variant;
        }
    }
}

pub fn merge_variant_schema_into(target: &mut VariantSchema, incoming: VariantSchema) {
    merge_variant_schema_from(target, &incoming);
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

/// Retrieve schema text from a VARIANT scalar or array (row-wise for arrays).
fn infer_variant_schema(variant: &ColumnarValue) -> Result<ColumnarValue> {
    match variant {
        ColumnarValue::Scalar(scalar) => {
            let ScalarValue::Struct(struct_array) = scalar else {
                return exec_err!("Unsupported data type: {}", scalar.data_type());
            };

            let variant_array = VariantArray::try_new(struct_array.as_ref())?;
            let v = variant_array.value(0);
            let schema = VariantSchema::from(&v);

            Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(Some(
                print_schema(&schema),
            ))))
        }
        ColumnarValue::Array(array) => {
            let variant_array = VariantArray::try_new(array.as_ref())?;
            let out = variant_array
                .iter()
                .map(|v| v.map(|v| print_schema(&VariantSchema::from(&v))))
                .collect::<Vec<_>>();

            let out: StringViewArray = out.into();
            Ok(ColumnarValue::Array(Arc::new(out) as ArrayRef))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{VariantSchema, print_schema};
    use arrow_schema::DataType;
    use std::collections::BTreeMap;

    #[test]
    fn state_round_trip_uses_utf8_json() {
        let mut fields = BTreeMap::new();
        fields.insert(
            "a:key,with>delims".to_string(),
            VariantSchema::Array(Box::new(VariantSchema::Primitive(DataType::Int64))),
        );

        let schema = VariantSchema::Object(fields);
        let bytes = schema.to_state_bytes();

        let text = std::str::from_utf8(&bytes).expect("state should be utf8");
        assert!(text.contains("\"kind\":\"object\""));
        assert!(text.contains("\"fields\""));

        let decoded = VariantSchema::from_state_bytes(&bytes).expect("round-trip decode");
        assert_eq!(decoded, schema);
        assert_eq!(print_schema(&decoded), print_schema(&schema));
    }
}
