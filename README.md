# datafusion-variant

This crate provides a [Variant] SQL type and functions for querying
semi-structured data using [Apache DataFusion], as well as reading and writing
such data to [Apache Parquet] files. 

This crate is under active development and targets Apache Spark Variant behavior.
Compatibility is incomplete, and APIs may change. See the
[roadmap](https://github.com/datafusion-contrib/datafusion-variant/issues/69).

## Quick start

First, enable Variant in your project by adding the following to `Cargo.toml`:

```toml
[dependencies]
datafusion-variant = { git = "https://github.com/datafusion-contrib/datafusion-variant.git" }
datafusion = { git = "https://github.com/apache/datafusion.git", rev = "484183dc34ea565190d9cd4a66db6d24789f95d0" } # match this crate
parquet = { version = "60", features = ["variant_experimental"] } # enable Parquet Variant support
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Then, configure your DataFusion [`SessionContext`] to use the Variant functions
and SQL type with `register_all`. See the [Apache DataFusion documentation] for
more details on creating a session and executing queries.

```rust
use datafusion::prelude::*;
use datafusion_variant::register_all;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let mut ctx = SessionContext::new();
    register_all(&mut ctx)?;
    Ok(())
}
```

> [!WARNING]
> `register_all` replaces existing functions with the same names. 

Congratulations -- you can now query semi-structured data with the Variant type!

## Common tasks

This section shows how to perform common tasks with the Variant type. You can
run the SQL examples using the `SessionContext` e.g. using 
[`ctx.sql(sql).await?.collect().await?`](Docs for session context.sql()).

This repository also includes a `dfv` command-line tool with the Variant
functions and SQL type preinstalled for you to try the examples interactively. Run it like
```shell
# Note you can drop --release` if you want faster compile times and aren't measuring performance.
cargo run --release
```

### Convert JSON to Variant

`parse_json` converts JSON strings to Variant. You can use it to load semi
structured documents. For example, to create an `events` table with a Variant
column named `event`, run:

```sql
CREATE TABLE events AS
SELECT parse_json(json_data) AS event
FROM (VALUES
    ('{"service":"api","status":200,"latency_ms":12,"tags":["cached"]}'),
    ('{"service":"api","status":500,"latency_ms":85,"error":null}'),
    ('{"service":"api","status":504,"error":"Somethings not working"}')
) AS input(json_data);
```

> [!TIP]
> For malformed input, use `try_parse_json(json_data)` to return SQL NULL instead
> of failing the query. Valid JSON `null` remains a Variant null.

### Query 

You can extract fields from a Variant column using the `variant_get` UDF or
the SQL `:` field-access syntax. For example, to select the `service` and
`latency_ms` fields from the `event` Variant column, run:

```sql
SELECT event:service::VARCHAR AS service,
       event:latency_ms::BIGINT AS latency_ms
FROM events
WHERE event:status::BIGINT >= 400;
```

```text
+---------+------------+
| service | latency_ms |
+---------+------------+
| api     | 85         |
| api     | NULL       |
+---------+------------+
```

The `:` syntax is automatically rewritten to the `variant_get` function to the
following equivalent query, which you can also run directly:


```sql
SELECT variant_get(event, 'service', 'VARCHAR') AS service,
       variant_get(event, 'latency_ms', 'BIGINT') AS latency_ms
FROM events
WHERE variant_get(event, 'status', 'BIGINT') >= 400;
```

```text
+---------+------------+
| service | latency_ms |
+---------+------------+
| api     | 85         |
+---------+------------+
```

### Convert Variant to JSON

Convert the Variant values back to JSON text using `to_json`. For example, to
select the entire `event` column as JSON text, run:

```sql
SELECT to_json(event) AS json_data
FROM events
ORDER BY event:status::BIGINT;
```

```text
+------------------------------------------------------------------+
| json_data                                                        |
+------------------------------------------------------------------+
| {"latency_ms":12,"service":"api","status":200,"tags":["cached"]} |
| {"error":null,"latency_ms":85,"service":"api","status":500}      |
| {"error":"Somethings not working","service":"api","status":504}  |
+------------------------------------------------------------------+
```

### Store Variant in Parquet

You can write the Variant column with DataFusion's `COPY` statement:

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

You can see the Variant column is stored in Parquet as a struct:

```sql
> select path_in_schema, type from  parquet_metadata('events.parquet');
+------------------+------------+
| path_in_schema   | type       |
+------------------+------------+
| "event.metadata" | BYTE_ARRAY |
| "event.value"    | BYTE_ARRAY |
+------------------+------------+
2 row(s) fetched.
Elapsed 0.001 seconds.
```

### Read Variant from Parquet

You can `events.parquet` with the Variant column preserved. 

> [!WARNING]
> DataFusion's Parquet reader ignores this metadata by default today,
> so you must set `datafusion.execution.parquet.skip_metadata`

```sql
-- Preserve the schema information that identifies the Variant column.
SET datafusion.execution.parquet.skip_metadata = false;
SELECT variant_get(event, 'service', 'VARCHAR') AS service,
       event:tags[0]::VARCHAR AS first_tag -- Array indexing via `[]` operator
FROM 'events.parquet'
ORDER BY event:status::BIGINT;
```

```text
+---------+-----------+
| service | first_tag |
+---------+-----------+
| api     | cached    |
| api     | NULL      |
| api     | NULL      |
+---------+-----------+
```


## What is Variant?

A Variant column can hold objects, arrays, and primitive values, with different
fields and types in each row. Think of it as a
[tagged union](https://arrow.apache.org/docs/format/Columnar.html#union-layout):
each value has a type, and different rows can hold different types.

For example, a single Variant column can contain all of these values, shown as JSON:

| Value | Type |
| --- | --- |
| `{"service":"api","status":200}` | Object |
| `["cached","retry"]` | Array |
| `42` | Integer |
| `"api"` | String |

`parse_json` converts JSON text to the
[Parquet Variant encoding](https://parquet.apache.org/docs/file-format/types/variantencoding/);
`CAST(value AS VARIANT)` converts typed scalars and arrays, including decimals;
`to_variant_object(value)` also accepts structs and string-keyed maps.
`CAST(variant_value AS type)` converts back to a SQL type, with `TRY_CAST` returning
SQL NULL for failed conversions.
SQL `VARIANT` type syntax requires the [type planner](docs/functions.md#registration).

Extract a field as a SQL type before filtering, aggregating, or joining on it.
A missing field returns SQL NULL. A JSON `null` is a distinct Variant value:
use `is_variant_null` to detect it, and `IS NULL` to detect SQL NULL.

For background on the encoding and shredding, see
[Variant Type in Apache Parquet for Semi-Structured Data](https://parquet.apache.org/blog/2026/02/27/variant-type-in-apache-parquet-for-semi-structured-data/).

## 🚧 Work in Progress 🚧

These features are not yet implemented:


> [!WARNING]
> **NOT YET IMPLEMENTED:** choosing between unshredded storage, an explicit
> shredding schema, or automatic shredding when writing Parquet.
> Shredding stores selected fields in typed
> Parquet columns alongside the remaining Variant data.
> The API and SQL syntax are TBD. A [DuckDB-style option](https://duckdb.org/docs/2.0/sql/data_types/variant#writing-variant-to-parquet)
> could look like:

<!-- readme-test: skip: proposed shredding syntax is not implemented -->
```sql
-- Proposed syntax only; not yet supported.
COPY events TO 'events.parquet' (
    FORMAT parquet,
    SHREDDING {'event': 'STRUCT(service VARCHAR, status BIGINT)'}
);
```



> [!WARNING]
> **NOT YET IMPLEMENTED:** reading a mixture of shredded and unshredded files
> as one table. See the [upstream Variant integration epic](https://github.com/apache/datafusion/issues/26517)
> for schema adaptation and extension metadata preservation work.

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
| Type conversion | [`CAST` / `TRY_CAST` to VARIANT, `to_variant_object`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/spark_variant_conversions.slt) |
| Field and path access | [`variant_get`, `try_variant_get`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_get.slt), [`variant_contains`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_contains.slt) |
| Validation and null inspection | [`is_valid_variant`](tests/test_files/is_valid_variant.slt), [`is_variant_null`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/is_variant_null.slt) |
| Schema inference | [`schema_of_variant`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/schema_of_variant.slt), [`schema_of_variant_agg`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/schema_of_variant_agg.slt) |
| Objects | [`variant_object_construct`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_construct.slt), [`variant_object_keys`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_keys.slt), [`variant_object_insert`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_insert.slt), [`variant_object_delete`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_delete.slt) |
| Arrays | [`variant_list_construct`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_list_construct.slt), [`variant_list_insert`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_list_insert.slt), [`variant_list_delete`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_list_delete.slt) |
| Utilities | [`variant_normalize`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/src/variant_normalize.rs), [`variant_pretty`](https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_pretty.slt) |

## More examples and benchmarks

The [interactive CLI](https://github.com/datafusion-contrib/datafusion-variant/blob/main/examples/cli.rs)
loads Bluesky JSON into a `bsky(json_data)` table. Run `bash download_data.sh`,
then `cargo run --example cli`. Queries end with a semicolon;
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

`cargo test` also checks this README's Rust example and SQL results.

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
