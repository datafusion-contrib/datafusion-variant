# Function behavior and SQL compatibility

See the [function index](../README.md#functions) for SQL examples.

## Registration

The [quickstart](../README.md#2-register-variant) uses `register_all` for functions
and field access. Explicit SQL `VARIANT` type names also require
`VariantTypePlanner` when constructing the session:

```rust
use std::sync::Arc;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_variant::{VariantTypePlanner, register_all};

let state = SessionStateBuilder::new()
    .with_default_features()
    .with_type_planner(Arc::new(VariantTypePlanner))
    .build();
let mut ctx = SessionContext::new_with_state(state);
register_all(&mut ctx)?;
```

For selective registration, wrap scalar implementations with
`ScalarUDF::new_from_impl` and aggregates with `AggregateUDF::new_from_impl`,
then pass them to `ctx.register_udf` or `ctx.register_udaf`. Use
`register_variant_get_functions` for the getters and casts from VARIANT,
`register_variant_conversion_functions` for casts to VARIANT and `to_variant_object`, and
`ctx.register_expr_planner(Arc::new(VariantExprPlanner))` for `:` field access.
Use `register_variant_explode_functions` for the explode table functions.

## Function behavior

`variant_explode(input)` emits immediate array elements or object fields in
encoded order as `pos` (zero-based INT), `key` (VARCHAR, NULL for arrays), and
`value` (VARIANT). Scalars, empty containers, and SQL/Variant null emit no rows;
`variant_explode_outer` emits one all-NULL row instead. Child Variant nulls stay
Variant values. See [SQL examples](../tests/test_files/variant_explode.slt).

Correlated calls support comma, CROSS, and INNER joins; correlated LEFT joins
and `WITH ORDINALITY` are unsupported. All output columns are marked nullable.

`is_valid_variant(v)` validates values using [Arrow's rules](../src/is_valid_variant.rs).
SQL NULL propagates; Variant null is valid.

`try_parse_json` accepts the same inputs as `parse_json`, but returns SQL NULL
for invalid JSON values. Valid JSON null remains a Variant null.
Both functions currently share Arrow's JSON
parsing limitations, including decimal precision and duplicate-key handling.

`to_json` serializes Variant values and is registered with
`VariantToJsonUdf::default()`.

`CAST(value AS VARIANT)` converts scalars and arrays. `TRY_CAST` returns SQL NULL
for failed conversions. Structs and string-keyed maps require
`to_variant_object(value)`, which also accepts arrays. Existing Variant values
retain their storage through identity casts.

Conversion currently omits null object fields, accepts duplicate object keys,
and does not support nested Variant inputs. These gaps are covered by skipped
expectations in the [conversion tests](../tests/test_files/spark_variant_conversions.slt).

`schema_of_variant(v)` returns a SQL schema for each value, for example
`OBJECT<a: BIGINT, b: ARRAY<STRING>>`. It merges types within arrays, preserves
SQL NULL, and reports `VOID` for Variant null. It follows Spark's
`schema_of_variant` inference rules. Arrow's JSON parser produces `DOUBLE`
for fractional JSON numbers; use `CAST(decimal_value AS VARIANT)` to retain
decimal precision and scale.

`schema_of_variant_agg(v)` merges schemas across rows, skipping SQL NULLs.
Empty and all-null groups return `VOID`. It shares the scalar function's
inference rules.

`variant_get_json(v, 'a')` extracts a value and serializes it as JSON, including
quotes around strings. It retains a direct extraction path that avoids the
intermediate Variant array in `to_json(variant_get(v, 'a'))`.

`variant_get` and `try_variant_get` accept an optional **SQL type string literal**,
such as `BIGINT`, `VARCHAR`, `DECIMAL(12,2)`, or `ARRAY<INT>`. Types follow
DataFusion SQL semantics exclusively: `INT8` (including `Int8`) means BIGINT;
Arrow-only names such as `Int64` and `Utf8View` are rejected. `variant_get`
raises an error on failed conversions; `try_variant_get` returns NULL for those
values. Both return NULL for missing paths; invalid paths and type hints still
raise errors. Use quoted bracket paths for
literal field names, such as `variant_get(v, '["http.status"]')` for the
key `http.status`.
See [literal-key tests](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_get_literal_keys.slt) for
examples and skipped cases awaiting an upstream Arrow path-parser fix.

These expressions share the same extraction/conversion kernel:

```sql
variant_get(v, 'price', 'DECIMAL(12,2)')
CAST(variant_get(v, 'price') AS DECIMAL(12,2))
v:price::DECIMAL(12,2)
```

`TRY_CAST` uses the try-getter's conversion policy. Casts of already typed getters
retain the intermediate conversion. With the current parser, group colon access
on the right of arithmetic: `v:a::INT + (v:b::INT)`.

Type strings use DataFusion's SQL cast planner during planning. The registration
helper follows session type settings, including `SET` / `RESET`.

`VARIANT` resolves to a struct with `metadata` and `value` binary-view fields and
the `arrow.parquet.variant` extension metadata. Getter hints such as
`variant_get(v, 'price', 'VARIANT')` and getter casts such as `v:price::VARIANT`
use Arrow's native Variant target conversion.
Use SQL casts or `to_variant_object` to create Variant columns. Cast stored or
composed Variant values to SQL types with `CAST(v AS type)` or `TRY_CAST(v AS type)`;
these use the same root conversion as `variant_get(v, '', 'type')` and
`try_variant_get(v, '', 'type')` and inherit the getter kernel's conversion limitations.
Nested Variant getter targets remain unsupported.

See the [Spark field-access tests](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/spark_variant_field_extractions.slt)
for remaining conversion and syntax gaps.
