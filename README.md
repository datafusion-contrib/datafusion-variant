# datafusion-variant

This crate provides user-defined functions for efficient Variant type handling in Datafusion. Variant types enable semi-structured data storage and querying, supporting JSON-like nested structs with dynamic schemas.

This crate aims to achieve complete feature parity with Spark and Databricks Variant functions.

Contributers are welcomed! [Now check this out](https://www.youtube.com/watch?v=1dj1kCrUFCY)

# Status

`datafusion-variant` is still under development. Progress is tracked in https://github.com/datafusion-contrib/datafusion-variant/issues/2; once it's closed, the crate's output should be considered stable.

# Usage

```sh
# run the example
cargo run --example cli

# run all tests
cargo test

# run sqllogictests
cargo test --test sqllogictests
```

# Reading

## Specifications

- Iceberg Variant proposal: https://docs.google.com/document/d/1sq70XDiWJ2DemWyA5dVB80gKzwi0CWoM0LOWM7VJVd8/edit?tab=t.0#heading=h.rt0cvesdzsj7<br>
- Databricks Variant functions: https://docs.databricks.com/gcp/en/sql/language-manual/sql-ref-functions-builtin#variant-functions<br>
- Spark Variant functions: https://spark.apache.org/docs/latest/api/sql/index.html<br>

## Miscellaneous

- https://datafusion.apache.org/blog/2025/09/21/custom-types-using-metadata
