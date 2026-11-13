# datafusion-variant

<!-- The docs home page (docs/index.md) includes the text between these markers. -->
<!-- docs-intro-start -->
[![CI](https://github.com/datafusion-contrib/datafusion-variant/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/datafusion-contrib/datafusion-variant/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/datafusion-variant.svg)](https://crates.io/crates/datafusion-variant)
[![Docs](https://img.shields.io/badge/docs-datafusion--contrib.github.io-blue)](https://datafusion-contrib.github.io/datafusion-variant/)
[![API docs](https://docs.rs/datafusion-variant/badge.svg)](https://docs.rs/datafusion-variant)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/datafusion-contrib/datafusion-variant/blob/main/LICENSE.txt)

This crate provides a
[Variant](https://parquet.apache.org/docs/file-format/types/variantencoding/)
SQL type and functions for querying semi-structured data using
[Apache DataFusion](https://datafusion.apache.org/), and reading and writing
such data to [Apache Parquet](https://parquet.apache.org/) files.

> [!NOTE]
> This crate is under active development and aims to match
> [Apache Spark](https://spark.apache.org/) Variant behavior. Compatibility is
> incomplete, and APIs may change. See the
> [roadmap](https://github.com/datafusion-contrib/datafusion-variant/issues/69)
> for more details.
<!-- docs-intro-end -->

Please see the [main site][docs] for documentation, and [docs.rs][api-docs] for
the Rust API reference.

For example, to read fields from JSON documents:

```sql
SELECT event:service::VARCHAR, event:status::BIGINT, event:tags[0]::VARCHAR
FROM (VALUES
    (parse_json('{"service":"api","status":200,"tags":["cached"]}')),
    (parse_json('{"service":"web","status":504,"error":"timeout"}'))
) AS events(event);
```

```text
+----------------------+---------------------+----------------------+
| events.event:service | events.event:status | events.event:tags[0] |
+----------------------+---------------------+----------------------+
| api                  | 200                 | cached               |
| web                  | 504                 | NULL                 |
+----------------------+---------------------+----------------------+
```

## Features

- A `VARIANT` SQL type with Spark-style `:` field access, such as `event:tags[0]::VARCHAR`
- JSON conversion with `parse_json`, `try_parse_json`, and `to_json`
- Typed extraction with `variant_get` and `try_variant_get`, and casts between Variant and SQL types
- Reading and writing Variant columns in Parquet files, including shredded columns
- `dfv`, a SQL shell with the Variant functions preinstalled

## What is Variant?

Variant is a type for storing semi-structured data, similar to JSON. Each value
can hold objects, arrays, and primitive values, with different fields and types
in each row. Variants are stored in an efficient binary encoding, and fields that
many rows share can be "shredded" into typed Parquet columns for faster queries.
See the [introduction][docs] for more background.

## Getting started

Add the crate to your project and call `register_all` on your DataFusion
`SessionContext` to use the `VARIANT` type, the Variant functions, and `:` field
access in SQL. The [getting started guide][getting-started] walks through the
setup and the bundled `dfv` SQL shell, and the [cookbook][cookbook] shows common
tasks: converting JSON, querying fields, and reading and writing Parquet.

## Documentation

- [User guide][docs]
- [SQL function reference][sql-reference]
- [Rust API reference (docs.rs)][api-docs]
- [Contributing][contributing]

The documentation sources are in [`docs/`](docs/).

## License

Licensed under the [Apache License, Version 2.0][license].

[docs]: https://datafusion-contrib.github.io/datafusion-variant/
[getting-started]: https://datafusion-contrib.github.io/datafusion-variant/getting-started.html
[cookbook]: https://datafusion-contrib.github.io/datafusion-variant/cookbook/load-and-query-json.html
[sql-reference]: https://datafusion-contrib.github.io/datafusion-variant/sql-reference.html
[api-docs]: https://docs.rs/datafusion-variant
[contributing]: https://datafusion-contrib.github.io/datafusion-variant/contributing.html
[license]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/LICENSE.txt
