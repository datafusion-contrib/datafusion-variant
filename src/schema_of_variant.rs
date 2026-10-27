use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, StringViewArray, StringViewBuilder};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::{
    common::{
        Result, ScalarValue, exec_datafusion_err, exec_err, internal_datafusion_err,
        utils::take_function_args,
    },
    functions::utils::make_scalar_function,
    logical_expr::{
        Accumulator, AggregateUDFImpl, ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs,
        ScalarUDFImpl, Signature, TypeSignature, Volatility, function::AccumulatorArgs,
        utils::AggregateOrderSensitivity,
    },
    sql::sqlparser::ast::{
        ArrayElemTypeDef, DataType as SqlType, ExactNumberInfo, Ident, ObjectName,
        StructBracketKind, StructField, TimezoneInfo,
    },
};
use datafusion_doc::Documentation;
use datafusion_macros::user_doc;
use parquet_variant::Variant;
use parquet_variant_compute::{VariantArray, unshred_variant};
use serde::Deserialize;

use crate::{shared::try_parse_binary_columnar, variant_type::is_variant};

/// Infer a SQL schema for each Variant value, using Spark's schema_of_variant rules.
/// SQL NULL propagates; a present Variant null has schema VOID.
#[user_doc(
    doc_section(label = "Variant Functions"),
    description = "Returns a SQL schema string for a Variant value. SQL NULL returns SQL NULL;
Variant null has schema `VOID`. Object fields are sorted by name. Array element
schemas are merged, widening compatible types and using `VARIANT` for incompatible types.

Fractional numbers parsed by `parse_json` have schema `DOUBLE`, unlike Spark's
decimal inference. Use `CAST(decimal_value AS VARIANT)` to preserve decimal types.",
    syntax_example = "schema_of_variant(value)",
    sql_example = r#"```sql
> SELECT schema_of_variant(parse_json('null')) AS schema;
+--------+
| schema |
+--------+
| VOID   |
+--------+
> SELECT schema_of_variant(parse_json('[{"b":true,"a":0}]')) AS schema;
+--------------------------------------+
| schema                               |
+--------------------------------------+
| ARRAY<OBJECT<a: BIGINT, b: BOOLEAN>> |
+--------------------------------------+
```"#,
    argument(
        name = "value",
        description = "Variant expression whose schema is inferred, or an untyped SQL NULL."
    )
)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct SchemaOfVariantUdf {
    signature: Signature,
}

impl Default for SchemaOfVariantUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(1), Volatility::Immutable),
        }
    }
}

fn validate_input(field: &Field, name: &str) -> Result<()> {
    if field.data_type() != &DataType::Null && !is_variant(field) {
        return exec_err!("{name} expects a Variant or untyped NULL argument");
    }
    Ok(())
}

impl ScalarUDFImpl for SchemaOfVariantUdf {
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn name(&self) -> &str {
        "schema_of_variant"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8View)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        let [field] = take_function_args(self.name(), args.arg_fields)?;
        validate_input(field, self.name())?;
        Ok(Arc::new(Field::new(
            self.name(),
            DataType::Utf8View,
            field.is_nullable(),
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [field] = take_function_args(self.name(), &args.arg_fields)?;
        validate_input(field, self.name())?;
        make_scalar_function(
            |arrays| {
                let [input] = take_function_args(self.name(), arrays)?;
                schema_array(input.as_ref())
            },
            vec![],
        )(&args.args)
    }
}

/// Infer and merge schemas across rows. Empty and all-null groups return VOID.
#[user_doc(
    doc_section(label = "Variant Functions"),
    description = "Returns a SQL schema string by merging the schemas of Variant values across rows.
SQL NULL values are ignored. Empty input, all-SQL-NULL input, and all-Variant-null
input return `VOID`. Compatible types are widened; incompatible types become `VARIANT`.
Object fields are merged by name and sorted in the output. Array element schemas
are merged using the same rules.

Fractional numbers parsed by `parse_json` contribute `DOUBLE`, unlike Spark's
decimal inference. Use `CAST(decimal_value AS VARIANT)` to preserve decimal types.",
    syntax_example = "schema_of_variant_agg(value)",
    sql_example = r#"```sql
> SELECT schema_of_variant_agg(parse_json(j)) AS schema FROM (VALUES ('1'), ('2'), ('3')) AS t(j);
+--------+
| schema |
+--------+
| BIGINT |
+--------+
> SELECT schema_of_variant_agg(parse_json(j)) AS schema FROM (VALUES ('{"a":1}'), ('{"b":true}'), ('{"c":1.23}')) AS t(j);
+------------------------------------------+
| schema                                   |
+------------------------------------------+
| OBJECT<a: BIGINT, b: BOOLEAN, c: DOUBLE> |
+------------------------------------------+
```"#,
    argument(
        name = "value",
        description = "Variant expression to aggregate, or an untyped SQL NULL."
    )
)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct SchemaOfVariantAggUdf {
    signature: Signature,
}

impl Default for SchemaOfVariantAggUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(1), Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for SchemaOfVariantAggUdf {
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn name(&self) -> &str {
        "schema_of_variant_agg"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8View)
    }

    fn return_field(&self, fields: &[FieldRef]) -> Result<FieldRef> {
        let [field] = take_function_args(self.name(), fields)?;
        validate_input(field, self.name())?;
        Ok(Arc::new(Field::new(self.name(), DataType::Utf8View, false)))
    }

    fn is_nullable(&self) -> bool {
        false
    }

    fn order_sensitivity(&self) -> AggregateOrderSensitivity {
        AggregateOrderSensitivity::Insensitive
    }

    fn default_value(&self, _: &DataType) -> Result<ScalarValue> {
        Ok(ScalarValue::Utf8View(Some("VOID".into())))
    }

    fn accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let [field] = take_function_args(self.name(), args.expr_fields)?;
        validate_input(field, self.name())?;
        Ok(Box::new(SchemaAccumulator::default()))
    }
}

#[derive(Debug)]
struct SchemaAccumulator {
    schema: SqlType,
}

impl Default for SchemaAccumulator {
    fn default() -> Self {
        Self {
            schema: named_type("VOID"),
        }
    }
}

impl SchemaAccumulator {
    fn merge(&mut self, schema: SqlType) {
        // Move out the current schema without allocating a temporary named type.
        let current = std::mem::replace(&mut self.schema, SqlType::Unspecified);
        self.schema = merge_schema(current, schema);
    }
}

impl Accumulator for SchemaAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let [input] = take_function_args("schema_of_variant_agg", values)?;
        if input.data_type() == &DataType::Null {
            return Ok(());
        }
        let variants = unshred_variant(&VariantArray::try_new(input.as_ref())?)?;
        let metadata = try_parse_binary_columnar(variants.metadata_column())?;
        let encoded = try_parse_binary_columnar(variants.value_column())?;
        for (i, metadata) in metadata.into_iter().enumerate() {
            if !variants.is_null(i) {
                let value = match encoded[i] {
                    Some(bytes) => Variant::try_new(
                        metadata.ok_or_else(|| {
                            exec_datafusion_err!("schema_of_variant_agg received missing metadata")
                        })?,
                        bytes,
                    )?,
                    None => Variant::Null,
                };
                self.merge(infer_schema(&value));
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::Utf8View(Some(print_schema(&self.schema))))
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        // Serialize the AST, not the display schema: OBJECT is Spark output syntax,
        // and field names and types must survive partial aggregation losslessly.
        Ok(vec![ScalarValue::Utf8View(Some(
            serde_json::to_string(&self.schema)
                .map_err(|e| internal_datafusion_err!("Cannot serialize Variant schema: {e}"))?,
        ))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let [state] = take_function_args("schema_of_variant_agg state", states)?;
        for value in state.as_string_view().iter().flatten() {
            // These states are generated internally from already-validated Variant
            // values. The AST's JSON representation nests much deeper than the input.
            let mut decoder = serde_json::Deserializer::from_str(value);
            decoder.disable_recursion_limit();
            self.merge(
                SqlType::deserialize(&mut decoder).map_err(|e| {
                    internal_datafusion_err!("Cannot deserialize Variant schema: {e}")
                })?,
            );
        }
        Ok(())
    }

    fn size(&self) -> usize {
        size_of_val(self) + schema_heap_size(&self.schema)
    }
}

// Account for heap allocations in the subset of SQL types produced by infer_schema.
fn schema_heap_size(schema: &SqlType) -> usize {
    match schema {
        SqlType::Array(ArrayElemTypeDef::AngleBracket(element)) => {
            size_of_val(element.as_ref()) + schema_heap_size(element)
        }
        SqlType::Struct(fields, _) => {
            fields.capacity() * size_of::<StructField>()
                + fields
                    .iter()
                    .map(|field| {
                        field
                            .field_name
                            .as_ref()
                            .map_or(0, |name| name.value.capacity())
                            + schema_heap_size(&field.field_type)
                    })
                    .sum::<usize>()
        }
        SqlType::Custom(name, _) => {
            name.0.capacity() * size_of::<datafusion::sql::sqlparser::ast::ObjectNamePart>()
                + name
                    .0
                    .iter()
                    .map(|part| part.as_ident().map_or(0, |name| name.value.capacity()))
                    .sum::<usize>()
        }
        _ => 0,
    }
}

fn schema_array(input: &dyn Array) -> Result<ArrayRef> {
    if input.data_type() == &DataType::Null {
        return Ok(Arc::new(StringViewArray::new_null(input.len())));
    }

    // Arrow handles shredded layouts and cheaply clones already-unshredded arrays.
    let variants = unshred_variant(&VariantArray::try_new(input)?)?;
    // try_value uses the panicking Variant::new for binary values. Use the
    // fully validating decoder until Arrow provides a fallible equivalent.
    let metadata = try_parse_binary_columnar(variants.metadata_column())?;
    let encoded = try_parse_binary_columnar(variants.value_column())?;
    let mut output = StringViewBuilder::with_capacity(variants.len());
    for (i, metadata) in metadata.into_iter().enumerate() {
        if variants.is_null(i) {
            output.append_null();
            continue;
        }

        let value = match encoded[i] {
            Some(bytes) => Variant::try_new(
                metadata.ok_or_else(|| {
                    exec_datafusion_err!("schema_of_variant received missing metadata")
                })?,
                bytes,
            )?,
            None => Variant::Null,
        };
        output.append_value(print_schema(&infer_schema(&value)));
    }
    Ok(Arc::new(output.finish()))
}

// Use SQL types directly, with ordinary named types for Spark's VOID and VARIANT.
// Scalar inference and aggregate partial states share this representation.
fn named_type(name: &str) -> SqlType {
    SqlType::Custom(ObjectName::from(vec![Ident::new(name)]), vec![])
}

fn array_type(element: SqlType) -> SqlType {
    SqlType::Array(ArrayElemTypeDef::AngleBracket(Box::new(element)))
}

fn decimal_type(integer: i128, scale: u8) -> SqlType {
    let digits = integer.unsigned_abs().checked_ilog10().map_or(1, |n| n + 1);
    decimal(u64::from(digits).max(u64::from(scale)), i64::from(scale))
}

fn decimal(precision: u64, scale: i64) -> SqlType {
    SqlType::Decimal(ExactNumberInfo::PrecisionAndScale(precision, scale))
}

// The caller validates the complete Variant before traversal.
fn infer_schema(value: &Variant<'_, '_>) -> SqlType {
    match value {
        Variant::Null => named_type("VOID"),
        Variant::BooleanTrue | Variant::BooleanFalse => SqlType::Boolean,
        Variant::Int8(_) | Variant::Int16(_) | Variant::Int32(_) | Variant::Int64(_) => {
            SqlType::BigInt(None)
        }
        Variant::Float(_) => SqlType::Float(ExactNumberInfo::None),
        Variant::Double(_) => SqlType::Double(ExactNumberInfo::None),
        Variant::Decimal4(d) => decimal_type(d.integer().into(), d.scale()),
        Variant::Decimal8(d) => decimal_type(d.integer().into(), d.scale()),
        Variant::Decimal16(d) => decimal_type(d.integer(), d.scale()),
        Variant::String(_) | Variant::ShortString(_) => SqlType::String(None),
        Variant::Binary(_) => SqlType::Binary(None),
        Variant::Date(_) => SqlType::Date,
        Variant::Time(_) => SqlType::Time(None, TimezoneInfo::None),
        Variant::TimestampMicros(_) | Variant::TimestampNanos(_) => {
            SqlType::Timestamp(None, TimezoneInfo::None)
        }
        Variant::TimestampNtzMicros(_) | Variant::TimestampNtzNanos(_) => {
            SqlType::TimestampNtz(None)
        }
        Variant::Uuid(_) => SqlType::Uuid,
        Variant::List(list) => {
            let mut element = named_type("VOID");
            for value in list.iter() {
                element = merge_schema(element, infer_schema(&value));
            }
            array_type(element)
        }
        Variant::Object(object) => {
            let mut fields = object
                .iter()
                .map(|(name, value)| StructField {
                    field_name: Some(Ident::new(name)),
                    field_type: infer_schema(&value),
                    options: None,
                })
                .collect::<Vec<_>>();
            fields.sort();
            SqlType::Struct(fields, StructBracketKind::AngleBrackets)
        }
    }
}

fn field_name(field: &StructField) -> &str {
    &field
        .field_name
        .as_ref()
        .expect("inferred object fields are named")
        .value
}

/// Spark JsonInferSchema.compatibleType with VARIANT as the incompatible-type fallback.
fn merge_schema(a: SqlType, b: SqlType) -> SqlType {
    use SqlType::*;
    if a == b || b == named_type("VOID") {
        return a;
    }
    if a == named_type("VOID") {
        return b;
    }
    match (a, b) {
        (Array(ArrayElemTypeDef::AngleBracket(a)), Array(ArrayElemTypeDef::AngleBracket(b))) => {
            array_type(merge_schema(*a, *b))
        }
        (Struct(mut a, _), Struct(b, _)) => {
            for field in b {
                if let Some(existing) = a.iter_mut().find(|f| field_name(f) == field_name(&field)) {
                    existing.field_type =
                        merge_schema(existing.field_type.clone(), field.field_type);
                } else {
                    a.push(field);
                }
            }
            a.sort();
            Struct(a, StructBracketKind::AngleBrackets)
        }
        (
            Decimal(ExactNumberInfo::PrecisionAndScale(p1, s1)),
            Decimal(ExactNumberInfo::PrecisionAndScale(p2, s2)),
        ) => {
            let scale = s1.max(s2);
            // Inferred Variant decimals have nonnegative scales and at most 38 digits.
            let precision = (p1 as i64 - s1).max(p2 as i64 - s2) + scale;
            if precision > 38 {
                Double(ExactNumberInfo::None)
            } else {
                decimal(precision as u64, scale)
            }
        }
        (a @ Decimal(_), BigInt(_)) | (BigInt(_), a @ Decimal(_)) => {
            merge_schema(a, decimal(20, 0))
        }
        (Double(_), BigInt(_) | Float(_) | Decimal(_))
        | (BigInt(_) | Float(_) | Decimal(_), Double(_))
        | (Float(_), Decimal(_))
        | (Decimal(_), Float(_)) => Double(ExactNumberInfo::None),
        (Float(_), BigInt(_)) | (BigInt(_), Float(_)) => Float(ExactNumberInfo::None),
        (Date, b @ (Timestamp(..) | TimestampNtz(_)))
        | (b @ (Timestamp(..) | TimestampNtz(_)), Date) => b,
        (a @ Timestamp(..), TimestampNtz(_)) | (TimestampNtz(_), a @ Timestamp(..)) => a,
        _ => named_type("VARIANT"),
    }
}

fn print_schema(schema: &SqlType) -> String {
    match schema {
        SqlType::Struct(fields, _) => {
            let fields = fields
                .iter()
                .map(|field| {
                    let name = field_name(field);
                    let plain = name
                        .as_bytes()
                        .first()
                        .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
                        && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_');
                    let name = if plain {
                        name.to_owned()
                    } else {
                        Ident::with_quote('`', name).to_string()
                    };
                    format!("{name}: {}", print_schema(&field.field_type))
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("OBJECT<{fields}>")
        }
        SqlType::Array(ArrayElemTypeDef::AngleBracket(element)) => {
            format!("ARRAY<{}>", print_schema(element))
        }
        _ => schema.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Array, ArrayRef, AsArray, BinaryArray, BinaryViewArray, LargeBinaryArray, StringArray,
        StructArray,
    };
    use datafusion::scalar::ScalarValue;
    use parquet_variant_compute::{VariantType, json_to_variant, shred_variant};

    fn invoke(array: StructArray, scalar: bool) -> Result<ColumnarValue> {
        let udf = SchemaOfVariantUdf::default();
        let field = Arc::new(
            Field::new("v", array.data_type().clone(), true).with_extension_type(VariantType),
        );
        let number_rows = array.len();
        let arg = if scalar {
            ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(array)))
        } else {
            ColumnarValue::Array(Arc::new(array))
        };
        let return_field = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: std::slice::from_ref(&field),
            scalar_arguments: &[None],
        })?;
        udf.invoke_with_args(ScalarFunctionArgs {
            args: vec![arg],
            arg_fields: vec![field],
            return_field,
            number_rows,
            config_options: Default::default(),
        })
    }

    #[tokio::test]
    async fn aggregate_across_partitions_and_batches() {
        use arrow::{array::Int32Array, datatypes::Schema, record_batch::RecordBatch};
        use datafusion::{
            datasource::MemTable,
            logical_expr::AggregateUDF,
            physical_plan::displayable,
            prelude::{SessionConfig, SessionContext},
        };

        let ctx = SessionContext::new_with_config(
            SessionConfig::new()
                .with_target_partitions(3)
                .with_batch_size(1),
        );
        ctx.register_udaf(AggregateUDF::new_from_impl(SchemaOfVariantAggUdf::default()));
        let mut partitions = Vec::new();
        let mut schema = None;
        for value in [Some(r#"{"a":1}"#), Some(r#"{"b":true}"#), None] {
            let json: ArrayRef = Arc::new(StringArray::from(vec![value, None]));
            let variants = json_to_variant(&json).unwrap();
            let variants = shred_variant(
                &variants,
                &DataType::Struct(vec![Field::new("a", DataType::Int64, true)].into()),
            )
            .unwrap();
            let batch_schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                variants.field("v"),
            ]));
            let batch = RecordBatch::try_new(
                batch_schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![0, 1])),
                    Arc::new(variants.into_inner()),
                ],
            )
            .unwrap();
            partitions.push(vec![batch.slice(0, 1), batch.slice(1, 1)]);
            schema = Some(batch_schema);
        }
        ctx.register_table(
            "inputs",
            Arc::new(MemTable::try_new(schema.unwrap(), partitions).unwrap()),
        )
        .unwrap();
        for grouped in [false, true] {
            let query = if grouped {
                "SELECT schema_of_variant_agg(v) FROM inputs GROUP BY id ORDER BY id"
            } else {
                "SELECT schema_of_variant_agg(v) FROM inputs"
            };
            let df = ctx.sql(query).await.unwrap();
            assert!(!df.schema().field(0).is_nullable());
            let plan = df.clone().create_physical_plan().await.unwrap();
            let display = displayable(plan.as_ref()).indent(false).to_string();
            assert!(display.contains("mode=Partial"), "{display}");
            assert!(display.contains("mode=Final"), "{display}");
            let batches = df.collect().await.unwrap();
            let actual: Vec<_> = batches
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(0)
                        .as_string_view()
                        .iter()
                        .map(|s| s.unwrap().to_owned())
                })
                .collect();
            let expected = if grouped {
                vec!["OBJECT<a: BIGINT, b: BOOLEAN>", "VOID"]
            } else {
                vec!["OBJECT<a: BIGINT, b: BOOLEAN>"]
            };
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn aggregate_deep_state_roundtrip() {
        // Regression: serializing the SQL AST adds enough nesting to exceed
        // serde_json's default recursion limit for an otherwise valid schema.
        let mut schema = decimal(38, 17);
        for _ in 0..32 {
            schema = SqlType::Struct(
                vec![StructField {
                    field_name: Some(Ident::new("a`\"b")),
                    field_type: array_type(schema),
                    options: None,
                }],
                StructBracketKind::AngleBrackets,
            );
        }
        let mut partial = SchemaAccumulator {
            schema: schema.clone(),
        };
        let mut merged = SchemaAccumulator::default();
        merged
            .merge_batch(&[ScalarValue::iter_to_array(partial.state().unwrap()).unwrap()])
            .unwrap();
        assert_eq!(merged.schema, schema);
    }

    #[test]
    fn sliced_empty_and_shredded_inputs() {
        let json: ArrayRef = Arc::new(StringArray::from(vec![
            Some("0"),
            None,
            Some("null"),
            Some("{}"),
            Some(r#"{"a":1,"b":true}"#),
            Some("[]"),
            Some("[1,null,128]"),
            Some(r#"[{"a":1},{"b":true}]"#),
        ]));
        let original = json_to_variant(&json).unwrap();
        let object = DataType::Struct(vec![Field::new("a", DataType::Int64, true)].into());
        let mut layouts = vec![original.clone()];
        for data_type in [
            DataType::Int64,
            object.clone(),
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            DataType::List(Arc::new(Field::new("item", object, true))),
        ] {
            layouts.push(shred_variant(&original, &data_type).unwrap());
        }
        let expected = [
            None,
            Some("VOID"),
            Some("OBJECT<>"),
            Some("OBJECT<a: BIGINT, b: BOOLEAN>"),
            Some("ARRAY<VOID>"),
            Some("ARRAY<BIGINT>"),
            Some("ARRAY<OBJECT<a: BIGINT, b: BOOLEAN>>"),
        ];
        for layout in layouts {
            let sliced = layout.slice(1, expected.len());
            let ColumnarValue::Array(actual) = invoke(sliced.inner().clone(), false).unwrap()
            else {
                panic!("expected array");
            };
            assert_eq!(actual.as_string_view().iter().collect::<Vec<_>>(), expected);
            for (i, expected) in expected.iter().enumerate() {
                let ColumnarValue::Scalar(actual) =
                    invoke(sliced.slice(i, 1).into_inner(), true).unwrap()
                else {
                    panic!("expected scalar");
                };
                assert_eq!(actual, ScalarValue::Utf8View(expected.map(str::to_owned)));
            }
            let ColumnarValue::Array(empty) =
                invoke(layout.slice(2, 0).into_inner(), false).unwrap()
            else {
                panic!("expected array");
            };
            assert!(empty.is_empty());
        }
    }

    #[test]
    fn binary_layouts_and_unknown_primitive() {
        // Spark VariantExpressionSuite: unknown primitive id 25. Build the raw
        // storage directly to exercise this UDF's malformed-input handling.
        for bytes in [&[0][..], &[25 << 2][..]] {
            let values: Vec<ArrayRef> = vec![
                Arc::new(BinaryArray::from(vec![bytes])),
                Arc::new(LargeBinaryArray::from(vec![bytes])),
                Arc::new(BinaryViewArray::from(vec![bytes])),
            ];
            for value in values {
                let array = StructArray::new(
                    vec![
                        Field::new("metadata", DataType::BinaryView, false),
                        Field::new("value", value.data_type().clone(), false),
                    ]
                    .into(),
                    vec![Arc::new(BinaryViewArray::from(vec![&[1, 0, 0][..]])), value],
                    None,
                );
                for scalar in [false, true] {
                    let result = invoke(array.clone(), scalar);
                    if bytes[0] == 0 {
                        assert!(result.is_ok(), "{result:?}");
                    } else {
                        assert!(
                            result
                                .unwrap_err()
                                .to_string()
                                .contains("unknown primitive")
                        );
                    }
                }
            }
        }
    }
}
