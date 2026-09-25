# datafusion-variant

Variant support for Apache DataFusion, with functions and SQL syntax for
querying and manipulating semi-structured data.

Variant values can contain primitives, objects, and arrays without requiring
a fixed schema for their contents.

## Project status

This crate is under active development. The compatibility target is
**Apache Spark 4.2.0**. Compatibility is incomplete, and function names,
signatures, and behavior may change.

Basic `:` field access is available. SQL `VARIANT` type integration and
Variant casts through `::` are planned.

See [#69](https://github.com/datafusion-contrib/datafusion-variant/issues/69)
for the compatibility roadmap, integration work, and benchmarking plans.

## Getting started

Register the functions you need with a DataFusion `SessionContext`.
Register `VariantExprPlanner` to enable `:` field access:

```rust
use std::sync::Arc;

use datafusion::error::Result;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::SessionContext;
use datafusion_variant::{JsonToVariantUdf, VariantExprPlanner, VariantToJsonUdf};

#[tokio::main]
async fn main() -> Result<()> {
    let mut ctx = SessionContext::new();

    ctx.register_udf(ScalarUDF::new_from_impl(JsonToVariantUdf::default()));
    ctx.register_udf(ScalarUDF::new_from_impl(VariantToJsonUdf::default()));
    ctx.register_expr_planner(Arc::new(VariantExprPlanner))?;

    ctx.sql(
        r#"
        SELECT variant_to_json(
            json_to_variant('{"name":"Alice","age":30}'):name
        ) AS name
        "#,
    )
    .await?
    .show()
    .await?;

    Ok(())
}
```

This returns the JSON string `"Alice"`.

Use compatible DataFusion and Arrow dependencies when embedding the crate.
See [Cargo.toml](Cargo.toml) for the current versions and dependency patches.
The example also requires Tokio with its `macros` and `rt-multi-thread`
features.

## Functionality

The current API includes:

| Area | Functions |
| --- | --- |
| JSON conversion | `json_to_variant`, `variant_to_json` |
| Type conversion | `cast_to_variant` |
| Field and path access | `variant_get`, `variant_get_field`, `variant_contains` |
| Typed extraction | `variant_get_int`, `variant_get_float`, `variant_get_bool`, `variant_get_str`, `variant_get_json` |
| Null inspection | `is_variant_null` |
| Objects | `variant_object_construct`, `variant_object_keys`, `variant_object_insert`, `variant_object_delete` |
| Arrays | `variant_list_construct`, `variant_list_insert`, `variant_list_delete` |
| Utilities | `variant_normalize`, `variant_pretty` |

`variant_get` accepts an optional result type, currently expressed as an
Arrow type name such as `Int64` or `Utf8View`. `variant_get_field` treats
its key as a literal field name, allowing access to keys containing dots.

The [Spark field-access tests](tests/test_files/spark_variant_field_extractions.slt)
record known compatibility gaps.

## Examples

The [CLI example](examples/cli.rs) loads a sample Bluesky JSON dataset
into an in-memory table named `bsky` and registers Variant functions
and `:` field access.

From the repository root, download the sample data and choose option `1`:

```sh
bash download_data.sh
```

Start the CLI:

```sh
cargo run --example cli
```

Queries must end with a semicolon:

```sql
SELECT variant_to_json(json_to_variant(json_data))
FROM bsky
LIMIT 5;
```

Enter `quit` or `q` to exit. The example loads
`data/bluesky/file_0001.json.gz` into memory before accepting queries.

See [examples](examples) for the source code.

## Benchmarks

Run the extraction benchmarks:

```sh
cargo bench --bench variant_get
```

The suite has four groups, each using 8,192 rows. The first three use literal paths:

- `output_types`: compares Variant output, explicit type hints, and typed helpers
  for integer, float, boolean, and string values at path `a` in unshredded objects.
- `storage_layout`: extracts Variant output from identical two-field objects in
  unshredded, fully shredded, and partially shredded layouts. The partial layout
  shreds only sibling field `b`, leaving `a` in binary storage.
- `path_traversal`: extracts the same integer as Variant from the same unshredded
  documents via `top`, `obj.b.c`, `arr[1][1]`, and `mix[1].b`. The nested paths each
  have three steps; `top` is a one-step control. Timing includes parsing the path
  once per batch, per-row traversal, and output construction.
- `path_columns`: compares a literal `a`, a UTF-8 column repeating `a`, and a
  UTF-8 column alternating `a`/`b` over the same unshredded objects containing
  positive integers at `a` and their negatives at `b`, with Variant output.
  These call the UDF directly, exercising scalar versus array path dispatch
  without SQL optimization. Path columns are constructed outside timing.

Run one group by filtering its name:

```sh
cargo bench --bench variant_get -- output_types
cargo bench --bench variant_get -- storage_layout
cargo bench --bench variant_get -- path_traversal
cargo bench --bench variant_get -- path_columns
```

See [the benchmark source](benches/variant_get.rs) for the workloads.

Run serialization benchmarks with `cargo bench --bench variant_to_json`.
They cover scalar SQL NULL, plus scalar inputs and 8,192-row arrays of integers,
Variant nulls, escaped strings, and nested values, plus arrays with 25% and 100% SQL NULLs.
The `storage_layout` groups compare identical integers and escaped strings in
unshredded and typed-column storage, for both scalar and column inputs.
Inputs are constructed before timing; measurements include UDF invocation,
argument cloning, and output allocation/drop. Shredded objects
are excluded until their reconstruction issues are fixed
([#71](https://github.com/datafusion-contrib/datafusion-variant/issues/71)).
See [the source](benches/variant_to_json.rs).

To compare a change, save a baseline before making it, then rerun afterward:

```sh
# Before the change
cargo bench --bench variant_get -- --save-baseline before

# After the change
cargo bench --bench variant_get -- --baseline before
```

Use the same machine and build settings, and preserve `target/criterion`
between runs. Include relevant results when proposing performance changes.

Broader benchmark coverage is tracked in
[#19](https://github.com/datafusion-contrib/datafusion-variant/issues/19).

For realistic Parquet workloads, [JSONBench](benchmarks/jsonbench/README.md)
prepares the original Bluesky data as unshredded, partially shredded, and
query-field-shredded Variant. Its default run executes all five adapted queries
against all three layouts, reusing prepared fixtures:

```sh
cargo run --release --example jsonbench -- prepare
cargo run --release --example jsonbench -- run
```

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
