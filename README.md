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

## Benchmarks

The `variant_get` benchmark extracts the literal field `a` from 8,192 unshredded
Variant objects containing deterministic Int64, Float64, Boolean, or string values.
Each input type compares Variant output, an explicit type hint (strings use
`Utf8View`), and its typed helper (`variant_get_int`, `variant_get_float`,
`variant_get_bool`, or `variant_get_str`), for 12 cases total.
Timings include argument cloning (shared input buffers),
UDF execution, and output disposal. Input construction, return-field resolution,
and output correctness checks run outside timing; SQL planning and I/O are excluded.
Results include time per batch and rows/second.

```sh
cargo bench --bench variant_get
# Filter to the explicit Int64 type-hint case
cargo bench --bench variant_get -- variant_get_int64
# Save a baseline, then compare after changing the implementation
cargo bench --bench variant_get -- --save-baseline before
cargo bench --bench variant_get -- --baseline before
```

Criterion stores baselines under `target/criterion`; preserve that directory and
use the same machine, build settings, and workload when comparing prototypes.

# Reading

## Specifications

- Iceberg Variant proposal: https://docs.google.com/document/d/1sq70XDiWJ2DemWyA5dVB80gKzwi0CWoM0LOWM7VJVd8/edit?tab=t.0#heading=h.rt0cvesdzsj7<br>
- Databricks Variant functions: https://docs.databricks.com/gcp/en/sql/language-manual/sql-ref-functions-builtin#variant-functions<br>
- Spark Variant functions: https://spark.apache.org/docs/latest/api/sql/search.html?q=variant<br>

## Miscellaneous

- https://datafusion.apache.org/blog/2025/09/21/custom-types-using-metadata
