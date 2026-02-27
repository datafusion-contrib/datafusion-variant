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
use std::collections::BTreeMap;
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
/// - This function does not merge schemas across rows. For cross-row/group merge use
///   `variant_schema_agg`.
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
        let mut out = Vec::new();
        encode_variant_schema(self, &mut out);
        out
    }

    pub fn from_state_bytes(bytes: &[u8]) -> Result<Self> {
        let mut offset = 0usize;
        let decoded = decode_variant_schema(bytes, &mut offset)?;
        if offset != bytes.len() {
            return exec_err!("invalid variant_schema_agg state: trailing bytes");
        }
        Ok(decoded)
    }
}

fn encode_len_prefixed_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn read_u8(input: &[u8], offset: &mut usize) -> Result<u8> {
    let Some(v) = input.get(*offset) else {
        return exec_err!("invalid variant_schema_agg state: missing tag");
    };
    *offset += 1;
    Ok(*v)
}

fn read_u32(input: &[u8], offset: &mut usize) -> Result<u32> {
    let Some(raw) = input.get(*offset..(*offset + 4)) else {
        return exec_err!("invalid variant_schema_agg state: missing u32");
    };
    *offset += 4;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn read_len_prefixed_bytes<'a>(input: &'a [u8], offset: &mut usize) -> Result<&'a [u8]> {
    let len = read_u32(input, offset)? as usize;
    let Some(raw) = input.get(*offset..(*offset + len)) else {
        return exec_err!("invalid variant_schema_agg state: truncated payload");
    };
    *offset += len;
    Ok(raw)
}

fn encode_variant_schema(schema: &VariantSchema, out: &mut Vec<u8>) {
    match schema {
        VariantSchema::Primitive(dtype) => {
            out.push(0);
            encode_len_prefixed_bytes(out, dtype.to_string().as_bytes());
        }
        VariantSchema::Array(inner) => {
            out.push(1);
            encode_variant_schema(inner, out);
        }
        VariantSchema::Object(fields) => {
            out.push(2);
            out.extend_from_slice(&(fields.len() as u32).to_le_bytes());
            for (key, value) in fields {
                encode_len_prefixed_bytes(out, key.as_bytes());
                encode_variant_schema(value, out);
            }
        }
        VariantSchema::Variant => out.push(3),
    }
}

fn decode_variant_schema(input: &[u8], offset: &mut usize) -> Result<VariantSchema> {
    match read_u8(input, offset)? {
        0 => {
            let raw = read_len_prefixed_bytes(input, offset)?;
            let dtype_str = match std::str::from_utf8(raw) {
                Ok(v) => v,
                Err(e) => return exec_err!("invalid variant_schema_agg state: {e}"),
            };
            let dtype = match dtype_str.parse::<DataType>() {
                Ok(v) => v,
                Err(e) => return exec_err!("invalid variant_schema_agg datatype state: {e}"),
            };
            Ok(VariantSchema::Primitive(dtype))
        }
        1 => Ok(VariantSchema::Array(Box::new(decode_variant_schema(
            input, offset,
        )?))),
        2 => {
            let count = read_u32(input, offset)? as usize;
            let mut fields = BTreeMap::new();
            for _ in 0..count {
                let key_raw = read_len_prefixed_bytes(input, offset)?;
                let key = match std::str::from_utf8(key_raw) {
                    Ok(v) => v.to_string(),
                    Err(e) => return exec_err!("invalid variant_schema_agg field key: {e}"),
                };
                let value = decode_variant_schema(input, offset)?;
                fields.insert(key, value);
            }
            Ok(VariantSchema::Object(fields))
        }
        3 => Ok(VariantSchema::Variant),
        tag => exec_err!("invalid variant_schema_agg state tag: {tag}"),
    }
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
fn merge_decimal_types(p1: u8, s1: i8, p2: u8, s2: i8) -> Option<DataType> {
    const DECIMAL128_MAX_PRECISION: i16 = 38;

    // Decimal scale is non-negative in Arrow logical types.
    if s1 < 0 || s2 < 0 {
        return None;
    }

    let scale = s1.max(s2);
    let int_digits_1 = p1 as i16 - s1 as i16;
    let int_digits_2 = p2 as i16 - s2 as i16;
    let int_digits = int_digits_1.max(int_digits_2);
    let precision = int_digits + scale as i16;
    let precision = precision.max(1);

    // Decimal128 max precision in Arrow.
    if precision > DECIMAL128_MAX_PRECISION {
        return None;
    }

    Some(DataType::Decimal128(precision as u8, scale))
}

fn merge_int_and_decimal(int_min_precision: u8, p: u8, s: i8) -> Option<DataType> {
    merge_decimal_types(int_min_precision, 0, p, s)
}

fn merge_primitives(a: DataType, b: DataType) -> Option<DataType> {
    use DataType::*;
    const MIN_DECIMAL_PRECISION_FOR_INT8_INT16_INT32: u8 = 10;
    const MIN_DECIMAL_PRECISION_FOR_INT64: u8 = 20;

    match (a, b) {
        (x, y) if x == y => Some(x),
        // numeric widening
        // docs.databricks.com/aws/en/sql/language-manual/sql-ref-datatype-rules#type-precedence-list
        // For least common type resolution FLOAT is skipped to avoid loss of precision.
        (Int8 | Int16 | Int32 | Int64 | Float32, Float64)
        | (Float64, Int8 | Int16 | Int32 | Int64 | Float32) => Some(Float64),
        (Int8 | Int16 | Int32, Int64) | (Int64, Int8 | Int16 | Int32) => Some(Int64),
        (Int8 | Int16, Int32) | (Int32, Int8 | Int16) => Some(Int32),
        (Int8, Int16) | (Int16, Int8) => Some(Int16),
        // Keep precision safety over float32 when mixing integral + float32.
        (Int8 | Int16 | Int32 | Int64, Float32) | (Float32, Int8 | Int16 | Int32 | Int64) => {
            Some(Float64)
        }
        (Timestamp(tu1, tz1), Timestamp(tu2, tz2)) => {
            if tz1 != tz2 {
                None
            } else {
                let merged_tu =
                    if matches!(tu1, TimeUnit::Nanosecond) || matches!(tu2, TimeUnit::Nanosecond) {
                        TimeUnit::Nanosecond
                    } else {
                        TimeUnit::Microsecond
                    };
                Some(Timestamp(merged_tu, tz1))
            }
        }
        // Databricks precedence list promotes DATE -> TIMESTAMP.
        // Preserve the timestamp timezone annotation when present.
        (Date32, Timestamp(tu, tz)) | (Timestamp(tu, tz), Date32) => Some(Timestamp(tu, tz)),
        (Decimal32(p1, s1), Decimal32(p2, s2))
        | (Decimal32(p1, s1), Decimal64(p2, s2))
        | (Decimal32(p1, s1), Decimal128(p2, s2))
        | (Decimal64(p1, s1), Decimal32(p2, s2))
        | (Decimal64(p1, s1), Decimal64(p2, s2))
        | (Decimal64(p1, s1), Decimal128(p2, s2))
        | (Decimal128(p1, s1), Decimal32(p2, s2))
        | (Decimal128(p1, s1), Decimal64(p2, s2))
        | (Decimal128(p1, s1), Decimal128(p2, s2)) => merge_decimal_types(p1, s1, p2, s2),
        (Int8, Decimal32(p, s))
        | (Int8, Decimal64(p, s))
        | (Int8, Decimal128(p, s))
        | (Decimal32(p, s), Int8)
        | (Decimal64(p, s), Int8)
        | (Decimal128(p, s), Int8) => {
            merge_int_and_decimal(MIN_DECIMAL_PRECISION_FOR_INT8_INT16_INT32, p, s)
        }
        (Int16, Decimal32(p, s))
        | (Int16, Decimal64(p, s))
        | (Int16, Decimal128(p, s))
        | (Decimal32(p, s), Int16)
        | (Decimal64(p, s), Int16)
        | (Decimal128(p, s), Int16) => {
            merge_int_and_decimal(MIN_DECIMAL_PRECISION_FOR_INT8_INT16_INT32, p, s)
        }
        (Int32, Decimal32(p, s))
        | (Int32, Decimal64(p, s))
        | (Int32, Decimal128(p, s))
        | (Decimal32(p, s), Int32)
        | (Decimal64(p, s), Int32)
        | (Decimal128(p, s), Int32) => {
            merge_int_and_decimal(MIN_DECIMAL_PRECISION_FOR_INT8_INT16_INT32, p, s)
        }
        (Int64, Decimal32(p, s))
        | (Int64, Decimal64(p, s))
        | (Int64, Decimal128(p, s))
        | (Decimal32(p, s), Int64)
        | (Decimal64(p, s), Int64)
        | (Decimal128(p, s), Int64) => {
            merge_int_and_decimal(MIN_DECIMAL_PRECISION_FOR_INT64, p, s)
        }
        // Prefer floating fallback when mixing decimals with floating point values.
        (Decimal32(_, _) | Decimal64(_, _) | Decimal128(_, _), Float32 | Float64)
        | (Float32 | Float64, Decimal32(_, _) | Decimal64(_, _) | Decimal128(_, _)) => {
            Some(Float64)
        }

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

/// Retrieve schema text from a VARIANT scalar or array (row-wise for arrays).
fn infer_variant_schema(variant: &ColumnarValue) -> Result<ColumnarValue> {
    match variant {
        ColumnarValue::Scalar(scalar) => {
            let ScalarValue::Struct(struct_array) = scalar else {
                return exec_err!("Unsupported data type: {}", scalar.data_type());
            };

            let variant_array = VariantArray::try_new(struct_array.as_ref())?;
            let v = variant_array.value(0);
            let schema = schema_from_variant(&v);

            Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(Some(
                print_schema(&schema),
            ))))
        }
        ColumnarValue::Array(array) => {
            let variant_array = VariantArray::try_new(array.as_ref())?;
            let out = variant_array
                .iter()
                .map(|v| v.map(|v| print_schema(&schema_from_variant(&v))))
                .collect::<Vec<_>>();

            let out: StringViewArray = out.into();
            Ok(ColumnarValue::Array(Arc::new(out) as ArrayRef))
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
        let arg = args.args.first().ok_or_else(|| {
            DataFusionError::Execution("empty argument, expected 1 argument".to_string())
        })?;
        infer_variant_schema(arg)
    }
}
