# datafusion-variant

This crate provides a [Variant] SQL type and functions for querying
semi-structured data using [Apache DataFusion], and reading and writing
such data to [Apache Parquet] files.

This crate is under active development and targets Apache Spark Variant
behavior. Compatibility is incomplete, and APIs may change. See the
[roadmap][roadmap].

## What is Variant?

Variant is a type for storing semi-structured data, similar to JSON. Each value
can hold objects, arrays, and primitive values, with different fields and types
in each row.

For example, a single Variant column can contain all of these values, shown as JSON:

| Value | Type |
| --- | --- |
| `{"service":"api","status":200}` | Object |
| `["cached","retry"]` | Array |
| `42` | Integer |
| `"api"` | String |

Variant also supports "shredding" as an optimization for the common case where
the values share a common structure. For more background, see [Variant Type in
Apache Parquet for Semi-Structured Data][parquet-variant-blog].

## Quick start

First, enable Variant in your project by adding the following to `Cargo.toml`:

```toml
[dependencies]
datafusion-variant = { git = "https://github.com/datafusion-contrib/datafusion-variant.git" }
datafusion = { git = "https://github.com/apache/datafusion.git", rev = "484183dc34ea565190d9cd4a66db6d24789f95d0" } # match this crate
parquet = { version = "60", features = ["variant_experimental"] } # enable Parquet Variant support
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Then, configure your DataFusion [`SessionContext`] to use the Variant functions,
SQL field access, and `VARIANT` type with `register_all`. See the
[Apache DataFusion documentation] for more details on creating a session and
executing queries.

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
> Call `register_all` during session setup, before preparing statements. It replaces
> existing functions with the same names and the type planner, and clears prepared
> statements.

Congratulations -- you can now query semi-structured data with the Variant type!

## Common tasks

This section shows common queries with the Variant type. You can run the SQL
examples like any other DataFusion SQL statement using the `SessionContext`,
e.g. with [`ctx.sql(sql).await?.collect().await?`][session-context-sql].

This repository also includes a `dfv` command-line tool with the Variant
functionality preinstalled to try the examples interactively. Run it from the
repository root with:

```shell
# Drop `--release` for faster compile times if you aren't measuring performance.
cargo run --release
```

### Convert JSON to Variant

`parse_json` converts JSON strings to Variant. You can use it to load
semi-structured documents. For example, create an `events` table with a Variant
column named `event`:

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
> For potentially malformed input, use `try_parse_json(json_data)` to return
> SQL NULL instead of failing the query.

### Query

You can extract fields from a Variant column using the Spark SQL-style `:`
field-access syntax or the `variant_get` UDF. For example, to select the
`service` and `latency_ms` fields of events whose `status` is at least 400, run:

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

The `:` syntax is automatically rewritten to `variant_get`, so this is an
equivalent query:

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
| api     | NULL       |
+---------+------------+
```

### Store Variant in Parquet

You can write Variant columns to Parquet with the DataFusion `COPY` statement:

```sql
COPY events TO 'events.parquet' STORED AS PARQUET;
```

```text
+-------+
| count |
+-------+
| 3     |
+-------+
```

The Variant column is stored in Parquet as a struct with `metadata` and `value`
fields, as you can see with the `parquet_metadata` table function:

<!-- readme-test: skip: parquet_metadata is provided by dfv, not this crate -->
```sql
-- Note: parquet_metadata is part of dfv and datafusion-cli, not this crate.
SELECT path_in_schema, type FROM parquet_metadata('events.parquet');
```

```console
+------------------+------------+
| path_in_schema   | type       |
+------------------+------------+
| "event.metadata" | BYTE_ARRAY |
| "event.value"    | BYTE_ARRAY |
+------------------+------------+
```

### Read Variant from Parquet

You can read `events.parquet` back with the Variant column preserved.

> [!WARNING]
> DataFusion's Parquet reader ignores the schema metadata that identifies the
> Variant column by default today, so you must set
> `datafusion.execution.parquet.skip_metadata` to `false` first.

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

### Convert between Variant and other SQL types

`CAST(value AS VARIANT)` converts SQL values to Variant. For example, to
convert an integer and a decimal to Variant and show them as JSON, run:

```sql
SELECT CAST(42 AS VARIANT) AS int_value,
       CAST(1.23 AS VARIANT) AS decimal_value;
```

```text
+-----------+---------------+
| int_value | decimal_value |
+-----------+---------------+
| 42        | 1.23          |
+-----------+---------------+
```

To convert a struct or list to Variant, use `to_variant_object`:

<!-- CAST support tracked by https://github.com/datafusion-contrib/datafusion-variant/issues/112 -->
```sql
SELECT to_variant_object([1, 2, 3]) AS list_value,
       to_variant_object([{a: 1, b: 2}, {b: 3, a: 4}]) AS struct_list_value;
```

```text
+------------+-------------------------------+
| list_value | struct_list_value             |
+------------+-------------------------------+
| [1,2,3]    | [{"a":1,"b":2},{"a":4,"b":3}] |
+------------+-------------------------------+
```

### Inspect the schema of Variant values

Due to their dynamic nature, Variant values can have different fields and types in
each row. You can inspect the schema of a Variant column using the following
functions:

```sql
-- Get the schema of each Variant value
SELECT schema_of_variant(event) FROM events;
```

```text
+----------------------------------------------------------------------------------+
| schema_of_variant(events.event)                                                  |
+----------------------------------------------------------------------------------+
| OBJECT<latency_ms: BIGINT, service: STRING, status: BIGINT, tags: ARRAY<STRING>> |
| OBJECT<error: VOID, latency_ms: BIGINT, service: STRING, status: BIGINT>         |
| OBJECT<error: STRING, service: STRING, status: BIGINT>                           |
+----------------------------------------------------------------------------------+
```

```sql
-- Print the merged schema (union) of all Variant values in the column
SELECT schema_of_variant_agg(event) AS schema FROM events;
```

```text
+-------------------------------------------------------------------------------------------------+
| schema                                                                                          |
+-------------------------------------------------------------------------------------------------+
| OBJECT<error: STRING, latency_ms: BIGINT, service: STRING, status: BIGINT, tags: ARRAY<STRING>> |
+-------------------------------------------------------------------------------------------------+
```

## Functions

Links lead to executable SQL examples, or Rust tests for `variant_normalize`.
See [function behavior and SQL compatibility][functions-doc]
for conversion rules, SQL `VARIANT` type registration, and known limitations.

| Area | Functions and examples |
| --- | --- |
| JSON conversion | [`parse_json`, `try_parse_json`, `to_json`][parse-json-slt], [`variant_get_json`][variant-get-json-slt] |
| Type conversion | [`CAST` / `TRY_CAST` to VARIANT, `to_variant_object`][spark-variant-conversions-slt] |
| Field and path access | [`variant_get`, `try_variant_get`][variant-get-slt], [`variant_contains`][variant-contains-slt] |
| Validation and null inspection | [`is_valid_variant`][is-valid-variant-slt], [`is_variant_null`][is-variant-null-slt] |
| Schema inference | [`schema_of_variant`][schema-of-variant-slt], [`schema_of_variant_agg`][schema-of-variant-agg-slt] |
| Objects | [`variant_object_construct`][variant-object-construct-slt], [`variant_object_keys`][variant-object-keys-slt], [`variant_object_insert`][variant-object-insert-slt], [`variant_object_delete`][variant-object-delete-slt] |
| Arrays | [`variant_list_construct`][variant-list-construct-slt], [`variant_list_insert`][variant-list-insert-slt], [`variant_list_delete`][variant-list-delete-slt] |
| Utilities | [`variant_normalize`][variant-normalize-src] |

## Common pitfalls

Extract a field as a SQL type before filtering, aggregating, or joining on it.
A missing field returns SQL NULL. A JSON `null` is a distinct Variant null
value: use `is_variant_null` to detect it, and `IS NULL` to detect SQL NULL.

## More examples and benchmarks

For an interactive SQL shell with the Variant functions preinstalled, see
`dfv` ([#102](https://github.com/datafusion-contrib/datafusion-variant/pull/102)).

See the [benchmark guide][benchmark-guide]
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
regression coverage in the [SQL tests][sql-tests].

The [issue tracker][issue-tracker]
contains individual tasks and discussions. Overall direction is tracked
in [#69][roadmap].

## References

- [Apache Spark 4.2.0 documentation][spark-docs]
- [Parquet Variant encoding][parquet-variant-encoding]
- [Arrow canonical extension types][arrow-canonical-extensions]
- [Custom types using metadata in DataFusion][datafusion-custom-types-blog]
- [Iceberg Variant proposal — design background][iceberg-variant-proposal]

## License

Licensed under the [Apache License, Version 2.0][license].

[Variant]: https://parquet.apache.org/docs/file-format/types/variantencoding/
[Apache DataFusion]: https://datafusion.apache.org/
[Apache Parquet]: https://parquet.apache.org/
[`SessionContext`]: https://docs.rs/datafusion/latest/datafusion/execution/context/struct.SessionContext.html
[Apache DataFusion documentation]: https://datafusion.apache.org/library-user-guide/using-the-sql-api.html
[roadmap]: https://github.com/datafusion-contrib/datafusion-variant/issues/69
[session-context-sql]: https://docs.rs/datafusion/latest/datafusion/execution/context/struct.SessionContext.html#method.sql
[parquet-variant-encoding]: https://parquet.apache.org/docs/file-format/types/variantencoding/
[functions-registration]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/docs/functions.md#registration
[parquet-variant-blog]: https://parquet.apache.org/blog/2026/02/27/variant-type-in-apache-parquet-for-semi-structured-data/
[functions-doc]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/docs/functions.md
[parse-json-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/parse_json.slt
[variant-get-json-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_get_json.slt
[spark-variant-conversions-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/spark_variant_conversions.slt
[variant-get-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_get.slt
[variant-contains-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_contains.slt
[is-valid-variant-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/is_valid_variant.slt
[is-variant-null-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/is_variant_null.slt
[schema-of-variant-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/schema_of_variant.slt
[schema-of-variant-agg-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/schema_of_variant_agg.slt
[variant-object-construct-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_construct.slt
[variant-object-keys-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_keys.slt
[variant-object-insert-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_insert.slt
[variant-object-delete-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_object_delete.slt
[variant-list-construct-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_list_construct.slt
[variant-list-insert-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_list_insert.slt
[variant-list-delete-slt]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_list_delete.slt
[variant-normalize-src]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/src/variant_normalize.rs
[benchmark-guide]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/benches/README.md
[sql-tests]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files
[issue-tracker]: https://github.com/datafusion-contrib/datafusion-variant/issues
[spark-docs]: https://spark.apache.org/docs/4.2.0/
[arrow-canonical-extensions]: https://arrow.apache.org/docs/format/CanonicalExtensions.html
[datafusion-custom-types-blog]: https://datafusion.apache.org/blog/2025/09/21/custom-types-using-metadata
[iceberg-variant-proposal]: https://docs.google.com/document/d/1sq70XDiWJ2DemWyA5dVB80gKzwi0CWoM0LOWM7VJVd8/edit
[license]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/LICENSE.txt
