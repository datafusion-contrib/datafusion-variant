# Benchmarks

datafusion-variant has two kinds of benchmarks:

- **Function benchmarks** time one function at a time on in-memory arrays with
  [Criterion]. Run them with `cargo bench`, for example
  `cargo bench --bench variant_get`. The [function benchmark guide][benches]
  describes each benchmark and how to compare a change against a baseline.
- **SQL benchmarks** time whole queries over Parquet files with `dfv`.
  JSONBench, below, is the first.

```{include} ../benchmarks/jsonbench/README.md
:heading-offset: 1
```

[Criterion]: https://github.com/bheisler/criterion.rs
[benches]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/benches/README.md
