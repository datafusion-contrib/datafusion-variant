# SQL benchmarks

SQL benchmark suites written as DataFusion [`.benchmark` files](https://github.com/apache/datafusion/blob/main/benchmarks/sql_benchmarks/README.md) and run with `dfv`. Rust microbenchmarks for individual functions are in [`benches`](../benches/README.md).

| Suite | Description |
| --- | --- |
| [`jsonbench`](jsonbench/README.md) | The five JSONBench queries over Bluesky events stored as unshredded, partially shredded, shredded, and mixed Parquet Variant |

## Running

Run the commands from the repository root. Build `dfv` with `cargo build --release --bin dfv`, then put `target/release` on your `PATH` or replace `dfv` below with `target/release/dfv`.

`dfv bench` runs a suite's `.benchmark` files, each in a fresh session with the Variant functions installed:

```sh
dfv bench --list jsonbench
dfv bench jsonbench
dfv bench jsonbench --subgroup shredded --query 2 -i 10
```

The options match DataFusion's `benchmark_runner`: `-i/--iterations`, `-n/--partitions`, `-s/--batch-size`, `--memory-limit`, and `DATAFUSION_*` environment variables for session settings. See `dfv bench --help`.

`--result-mode validate` checks each query's results against the suite's `results/` files, and `--result-mode persist` rewrites them. Persisting writes SQL NULL as an empty field but validation expects `NULL`, so replace empty fields with `NULL` after persisting.

To compare two revisions, write each run as JSON and use DataFusion's [`compare.py`](https://github.com/apache/datafusion/blob/main/benchmarks/compare.py):

```sh
dfv bench jsonbench -o base.json
# Rebuild dfv on the other revision
dfv bench jsonbench -o change.json
python ../datafusion/benchmarks/compare.py base.json change.json
```

For quick manual checks, `dfv -i N` runs SQL files N times and prints the time of each run:

```sh
dfv -i 10 -f benchmarks/jsonbench/shredded/table.sql -f benchmarks/jsonbench/queries/q01.sql
```

Every run includes `table.sql`, whose `CREATE EXTERNAL TABLE IF NOT EXISTS` only does work the first time.

DataFusion's `benchmark_runner` builds its own session without the Variant functions, so it cannot run these suites. `dfv bench` runs the same files through DataFusion's unpublished `datafusion-benchmarks` crate, which it pulls from git at the same revision as the other DataFusion crates.
