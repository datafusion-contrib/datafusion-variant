# Benchmarks

Run these commands from the repository root.

Run the extraction benchmarks:

```sh
cargo bench --bench variant_get
```

The suite has four groups, each using 8,192 rows. The first three use literal paths:

- `output_types`: compares Variant output and explicit type hints
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

See [the benchmark source](variant_get.rs) for the workloads.

Run JSON parsing benchmarks with `cargo bench --bench parse_json`.
See [the source](parse_json.rs) for scalar and array workloads and timing boundaries.

Run serialization benchmarks with `cargo bench --bench to_json`.
They cover scalar SQL NULL, plus scalar inputs and 8,192-row arrays of integers,
Variant nulls, escaped strings, and nested values, plus arrays with 25% and 100% SQL NULLs.
The `storage_layout` groups compare identical integers and escaped strings in
unshredded and typed-column storage, for both scalar and column inputs.
Inputs are constructed before timing; measurements include UDF invocation,
argument cloning, and output allocation/drop. Shredded objects
are excluded until their reconstruction issues are fixed
([#71](https://github.com/datafusion-contrib/datafusion-variant/issues/71)).
See [the source](to_json.rs).

Run schema inference benchmarks with `cargo bench --bench schema_of_variant`.
They cover scalar and array inputs, aggregate updates, state serialization, and
partial-state merging, including nulls, nested values, widening, and shredded integers.
Input construction and validation are outside timing.
See [the source](schema_of_variant.rs) for workloads and timing boundaries.

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

For realistic Parquet workloads, [JSONBench](../benchmarks/jsonbench/README.md)
prepares the original Bluesky data as unshredded, partially shredded, and
query-field-shredded Variant. Its default run executes all five adapted queries
against all three layouts, reusing prepared fixtures:

```sh
cargo run --release --example jsonbench -- prepare
cargo run --release --example jsonbench -- run
```
