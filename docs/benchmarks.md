# Benchmarks

datafusion-variant has two kinds of benchmarks:

- **Function benchmarks** time one Variant function at a time on in-memory
  arrays with [Criterion]. They live in [`benches`][benches].
- **SQL benchmarks** time whole queries, including Parquet reads. They are
  DataFusion [`.benchmark` files][df-sql-benchmarks] run with `dfv bench`, and
  live in [`benchmarks`][benchmarks].

Run the commands below from the repository root. Only compare results from the
same machine and build settings, and include them when proposing a performance
change.

## Function benchmarks

```sh
cargo bench --bench variant_get
cargo bench --bench variant_get -- path_traversal
```

The benchmark targets are `variant_get`, `parse_json`, `to_json`, and
`schema_of_variant`. The [function benchmark README][benches] describes the
groups in each target and what their timing covers.

To compare a change, save a baseline before making it, then compare against it
afterwards. Keep `target/criterion` between the runs.

```sh
cargo bench --bench variant_get -- --save-baseline before
# Make the change
cargo bench --bench variant_get -- --baseline before
```

## JSONBench

[JSONBench] runs five queries over Bluesky events. The SQL suite stores the
events as Parquet Variant in four layouts so their timings can be compared:

| Layout | Fields shredded |
| --- | --- |
| `unshredded` | None |
| `partial` | `kind`, `did`, `time_us` |
| `shredded` | The above plus `commit.collection` and `commit.operation` |
| `mixed` | None in file 1, as in `shredded` in files 2 to 5 |

### Preparing the data

Build `dfv`, then write each layout. Each layout's `prepare.sql` reads Bluesky
files 1 to 5 (5M rows) with `load_bsky`, which downloads missing files to
`data/bluesky` (about 135 MB each), and writes Parquet to
`data/jsonbench/<layout>`.

```sh
cargo build --release --bin dfv
for layout in unshredded partial shredded mixed; do
  target/release/dfv -f benchmarks/jsonbench/$layout/prepare.sql
done
```

`dfv -f` reports a failed statement but still exits successfully, so check the
output for errors.

### Running

```sh
target/release/dfv bench jsonbench
target/release/dfv bench jsonbench --subgroup shredded --query 1 -i 10
target/release/dfv bench --list jsonbench
```

Each case runs one query against one layout and is named like
`jsonbench/Q01/shredded`. The layout's `table.sql` registers the `bluesky`
table outside timing, and the timing covers planning through the last result
batch. `dfv bench` takes the same options as DataFusion's `benchmark_runner`,
such as `-i/--iterations`, `-n/--partitions`, and `--memory-limit`, and reads
`DATAFUSION_*` environment variables for session settings. See
`dfv bench --help`.

### Checking results

`--result-mode validate` checks every case against the expected results in
`benchmarks/jsonbench/results`, which were recorded from the five default
files. `--result-mode persist` rewrites them. Persisting writes SQL NULL as an
empty field but validation expects `NULL`, so replace empty fields with `NULL`
afterwards.

### Comparing revisions

Write each run as JSON and compare the two with DataFusion's
[`compare.py`][compare-py]:

```sh
target/release/dfv bench jsonbench -o base.json
# Rebuild dfv on the other revision
target/release/dfv bench jsonbench -o change.json
python ../datafusion/benchmarks/compare.py base.json change.json
```

### Quick manual runs

To try a change to a query or layout by hand, `dfv -i N` runs SQL files N
times and prints the time of each run. The first run also creates the table.

```sh
target/release/dfv -i 10 -f benchmarks/jsonbench/shredded/table.sql \
  -f benchmarks/jsonbench/queries/q01.sql
```

[Criterion]: https://github.com/bheisler/criterion.rs
[benches]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/benches/README.md
[benchmarks]: https://github.com/datafusion-contrib/datafusion-variant/tree/main/benchmarks
[df-sql-benchmarks]: https://github.com/apache/datafusion/blob/main/benchmarks/sql_benchmarks/README.md
[JSONBench]: https://jsonbench.com/
[compare-py]: https://github.com/apache/datafusion/blob/main/benchmarks/compare.py
