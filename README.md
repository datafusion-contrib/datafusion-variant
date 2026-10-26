# datafusion-variant

Query semi-structured data in Apache DataFusion using the Parquet Variant type.
Parse JSON once, extract typed fields with SQL, and store the results in Parquet.

This crate is under active development and targets Apache Spark Variant behavior.
Compatibility is incomplete, and APIs may change. See the
[roadmap](https://github.com/datafusion-contrib/datafusion-variant/issues/69).

## Quick start

The [quickstart example](https://github.com/datafusion-contrib/datafusion-variant/blob/main/examples/quickstart.rs)
creates a Variant column from JSON, then filters events by a field inside each
document.

### 1. Register Variant

Inside your application's async entry point, create a session and register the
functions and SQL field-access syntax:

```rust
use datafusion::prelude::*;
use datafusion_variant::register_all;

let mut ctx = SessionContext::new();
register_all(&mut ctx)?;
```

> [!WARNING]
> `register_all` replaces existing functions with the same names. Individual UDFs
> remain available for selective registration.

### 2. Create the events table

Each input row is a JSON document. `parse_json` converts it to a Variant value
in the `event` column:

```sql
CREATE TABLE events AS
SELECT parse_json(json_data) AS event
FROM (VALUES
    ('{"service":"api","status":200,"latency_ms":12,"tags":["cached"]}'),
    ('{"service":"api","status":500,"latency_ms":85,"error":null}')
) AS input(json_data);
```

Execute the statement with `ctx.sql(sql).await?.collect().await?`.

> [!TIP]
> For malformed input, use `try_parse_json(json_data)` to return SQL NULL instead
> of failing the query. Valid JSON `null` remains a Variant null.

### 3. Query fields

Select the service and latency for events with an error status using the
`variant_get` UDF:

```sql
SELECT variant_get(event, 'service', 'VARCHAR') AS service,
       variant_get(event, 'latency_ms', 'BIGINT') AS latency_ms
FROM events
WHERE variant_get(event, 'status', 'BIGINT') >= 400;
```

Or use SQL's `:` field-access syntax with `::` casts:

```sql
SELECT event:service::VARCHAR AS service,
       event:latency_ms::BIGINT AS latency_ms
FROM events
WHERE event:status::BIGINT >= 400;
```

Both forms return the same result. Display it with
`ctx.sql(sql).await?.show().await?`:

```text
+---------+------------+
| service | latency_ms |
+---------+------------+
| api     | 85         |
+---------+------------+
```

### Run the complete example

Run the example from a checkout; it also writes and reads Parquet and
runs the recipes below, without downloading data:

```sh
cargo run --example quickstart
```

For an embedding application, use the DataFusion revision and Arrow versions in
[Cargo.toml](https://github.com/datafusion-contrib/datafusion-variant/blob/main/Cargo.toml).
Tokio needs the `macros` and `rt-multi-thread` features.

For Parquet Variant files, also enable Arrow's experimental Variant integration
in the embedding application's dependencies (already enabled for these examples):

```toml
parquet = { version = "60", features = ["variant_experimental"] }
```

## What is Variant?

A Variant column can hold objects, arrays, and primitive values, with different
fields and types in each row. Think of it as a
[tagged union](https://arrow.apache.org/docs/format/Columnar.html#union-layout):
each value has a type, and different rows can hold different types.

For example, the first event above has a `tags`
array while the second has an `error` field. The contents do not need a fixed
schema, but each value retains its type. `parse_json` converts JSON text to the
[Parquet Variant encoding](https://parquet.apache.org/docs/file-format/types/variantencoding/);
`cast_to_variant` converts existing typed SQL values, including decimals.

Extract a field as a SQL type before filtering, aggregating, or joining on it.
A missing field returns SQL NULL. A JSON `null` is a distinct Variant value:
use `is_variant_null` to detect it, and `IS NULL` to detect SQL NULL.

For background on the encoding and shredding, see
[Variant Type in Apache Parquet for Semi-Structured Data](https://parquet.apache.org/blog/2026/02/27/variant-type-in-apache-parquet-for-semi-structured-data/).

## Cookbook

These recipes use the `events` table from the quickstart in a session with
`register_all` installed. The Parquet recipes create a local `events.parquet`
output; choose a fresh path when repeating them.

### Store JSON as a Variant column in Parquet

The quickstart's `SELECT parse_json(json_data) AS event` converts each JSON
string to Variant. Save that column with DataFusion's `COPY` statement:

```sql
COPY events TO 'events.parquet' STORED AS PARQUET;
```

```text
+-------+
| count |
+-------+
| 2     |
+-------+
```

> [!NOTE]
> Planned: choose unshredded storage, an explicit shredding schema, or automatic
> shredding when writing Parquet. Shredding stores selected fields in typed
> Parquet columns alongside the remaining Variant data.
> These write options are not implemented; the API and SQL syntax
> are TBD. A [DuckDB-style option](https://duckdb.org/docs/2.0/sql/data_types/variant#writing-variant-to-parquet)
> could look like:

```sql
-- Proposed syntax only; not yet supported.
COPY events TO 'events.parquet' (
    FORMAT parquet,
    SHREDDING {'event': 'STRUCT(service VARCHAR, status BIGINT)'}
);
```

### Read Parquet and select fields

Keep Parquet extension metadata during schema inference so DataFusion recognizes
the Variant column:

```sql
SET datafusion.execution.parquet.skip_metadata = false;
CREATE EXTERNAL TABLE saved_events
STORED AS PARQUET LOCATION 'events.parquet';

SELECT variant_get(event, 'service', 'VARCHAR') AS service,
       event:tags[0]::VARCHAR AS first_tag
FROM saved_events
ORDER BY event:status::BIGINT;
```

```text
+---------+-----------+
| service | first_tag |
+---------+-----------+
| api     | cached    |
| api     |           |
+---------+-----------+
```

An empty cell represents SQL NULL. The first column uses the `variant_get` UDF;
the second uses SQL field-access syntax. Both support nested fields and
zero-based array indices. `variant_get` raises an error if a value cannot be
converted to the requested type; use `try_variant_get` or `TRY_CAST` to return
NULL instead.

Reading a mixture of shredded and unshredded files as one table is not yet a
supported recipe. See the [upstream Variant integration epic](https://github.com/apache/datafusion/issues/26517)
for schema adaptation and extension metadata preservation work.

### Filter on shredded fields

Shredding stores fields such as `service` and `status` in typed Parquet columns,
so extraction can use those values without decoding them from the binary
Variant payload. The runnable quickstart uses Arrow's `shred_variant` to write
the same events to `shredded_events.parquet` with these two fields shredded:

```sql
CREATE EXTERNAL TABLE shredded_events
STORED AS PARQUET LOCATION 'shredded_events.parquet';

SELECT event:service::VARCHAR AS service,
       event:status::BIGINT AS status
FROM shredded_events
WHERE event:status::BIGINT >= 400
ORDER BY service, status;
```

```text
+---------+--------+
| service | status |
+---------+--------+
| api     | 500    |
+---------+--------+
```

> [!NOTE]
> Further scan optimizations are TBD in DataFusion: pruning unused fields of the
> Variant struct, skipping row groups/pages using shredded-field statistics,
> and exposing those statistics to the optimizer. The goal is to read only the
> required fields and skip data that cannot match the filter. See sections 1
> and 4 of the [Variant integration epic](https://github.com/apache/datafusion/issues/26517).

### Convert Variant back to JSON

```sql
SELECT to_json(event) AS json_data
FROM saved_events
ORDER BY event:status::BIGINT;
```

```text
+------------------------------------------------------------------+
| json_data                                                        |
+------------------------------------------------------------------+
| {"latency_ms":12,"service":"api","status":200,"tags":["cached"]} |
| {"error":null,"latency_ms":85,"service":"api","status":500}      |
+------------------------------------------------------------------+
```

`to_json` returns JSON text, so string values include JSON quotes. Use a typed
getter such as `variant_get(event, 'service', 'VARCHAR')` for an unquoted SQL
string. JSON formatting and object key order need not match the input text.

### Discover fields and distinguish nulls

```sql
SELECT schema_of_variant_agg(event) AS schema FROM saved_events;
```

```text
+-----------------------------------------------------------------------------------------------+
| schema                                                                                        |
+-----------------------------------------------------------------------------------------------+
| OBJECT<error: VOID, latency_ms: BIGINT, service: STRING, status: BIGINT, tags: ARRAY<STRING>> |
+-----------------------------------------------------------------------------------------------+
```

Use `schema_of_variant(event)` for a schema per row.

```sql
SELECT event:status::BIGINT AS status,
       variant_contains(event, 'error') AS has_error,
       is_variant_null(event:error) AS error_is_json_null,
       (event:error) IS NULL AS error_is_sql_null
FROM saved_events
ORDER BY status;
```

```text
+--------+-----------+--------------------+-------------------+
| status | has_error | error_is_json_null | error_is_sql_null |
+--------+-----------+--------------------+-------------------+
| 200    | false     | false              | true              |
| 500    | true      | true               | false             |
+--------+-----------+--------------------+-------------------+
```

## Functions

Links lead to executable SQL examples, or Rust tests for `variant_normalize`.
See [function behavior and SQL compatibility](https://github.com/datafusion-contrib/datafusion-variant/blob/main/docs/functions.md)
for conversion rules, SQL `VARIANT` type registration, and known limitations.

| Area | Functions and examples |
| --- | --- |
| JSON conversion | [`parse_json`, `try_parse_json`, `to_json`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/parse_json.slt), [`variant_get_json`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_get_json.slt) |
| Type conversion | [`cast_to_variant`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/cast_to_variant.slt) |
| Field and path access | [`variant_get`, `try_variant_get`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_get.slt), [`variant_contains`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_contains.slt) |
| Null inspection | [`is_variant_null`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/is_variant_null.slt) |
| Schema inference | [`schema_of_variant`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/schema_of_variant.slt), [`schema_of_variant_agg`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/schema_of_variant_agg.slt) |
| Objects | [`variant_object_construct`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_construct.slt), [`variant_object_keys`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_keys.slt), [`variant_object_insert`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_insert.slt), [`variant_object_delete`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_delete.slt) |
| Arrays | [`variant_list_construct`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_list_construct.slt), [`variant_list_insert`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_list_insert.slt), [`variant_list_delete`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_list_delete.slt) |
| Utilities | [`variant_normalize`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/src/variant_normalize.rs), [`variant_pretty`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_pretty.slt) |

## More examples and benchmarks

The [interactive CLI](https://github.com/datafusion-contrib/datafusion-variant/blob/main/examples/cli.rs)
loads Bluesky JSON into a `bsky(json_data)` table. Run `bash download_data.sh`
(choose option `1`), then `cargo run --example cli`. Queries end with a semicolon;
enter `q` or `quit` to exit. Use `parse_json(json_data)` to query these strings
with the Variant functions.

See the [benchmark guide](https://github.com/datafusion-contrib/datafusion-variant/blob/main/benches/README.md)
for extraction, parsing, serialization, schema inference, and JSONBench workloads,
including baseline comparison commands.

## Contributing

Contributions are welcome, including compatibility fixes, regression tests,
benchmarks, documentation, and integration examples.

Use a current stable Rust toolchain with `rustfmt` and `clippy`, and install
the Protocol Buffers compiler (`protoc`).

Run the development checks from the repository root:

```sh
cargo test
cargo fmt --all -- --check
cargo clippy -- -D warnings
```

For Spark compatibility changes, reference the upstream behavior and add
regression coverage in the [SQL tests](tests/test_files).

The [issue tracker](https://github.com/datafusion-contrib/datafusion-variant/issues)
contains individual tasks and discussions. Overall direction is tracked
in [#69](https://github.com/datafusion-contrib/datafusion-variant/issues/69).

## References

- [Apache Spark 4.2.0 documentation](https://spark.apache.org/docs/4.2.0/)
- [Parquet Variant encoding](https://parquet.apache.org/docs/file-format/types/variantencoding/)
- [Arrow canonical extension types](https://arrow.apache.org/docs/format/CanonicalExtensions.html)
- [Custom types using metadata in DataFusion](https://datafusion.apache.org/blog/2025/09/21/custom-types-using-metadata)
- [Iceberg Variant proposal — design background](https://docs.google.com/document/d/1sq70XDiWJ2DemWyA5dVB80gKzwi0CWoM0LOWM7VJVd8/edit)

## License

Licensed under the [Apache License, Version 2.0](LICENSE.txt).
