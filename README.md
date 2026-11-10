# datafusion-variant

<!-- The docs home page (docs/index.md) includes the text between these markers. -->
<!-- docs-intro-start -->
This crate provides a
[Variant](https://parquet.apache.org/docs/file-format/types/variantencoding/)
SQL type and functions for querying semi-structured data using
[Apache DataFusion](https://datafusion.apache.org/), and reading and writing
such data to [Apache Parquet](https://parquet.apache.org/) files.

This crate is under active development and aims to match
[Apache Spark](https://spark.apache.org/) Variant behavior. Compatibility is
incomplete, and APIs may change. See the
[roadmap](https://github.com/datafusion-contrib/datafusion-variant/issues/69)
for more details.
<!-- docs-intro-end -->

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
- [Contributing][contributing]

The documentation sources are in [`docs/`](docs/).

## License

Licensed under the [Apache License, Version 2.0][license].

[docs]: https://datafusion-contrib.github.io/datafusion-variant/
[getting-started]: https://datafusion-contrib.github.io/datafusion-variant/getting-started.html
[cookbook]: https://datafusion-contrib.github.io/datafusion-variant/cookbook/load-and-query-json.html
[sql-reference]: https://datafusion-contrib.github.io/datafusion-variant/sql-reference.html
[contributing]: https://datafusion-contrib.github.io/datafusion-variant/contributing.html
[license]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/LICENSE.txt
