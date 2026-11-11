# SQL benchmarks

SQL benchmark suites written as DataFusion [`.benchmark` files](https://github.com/apache/datafusion/blob/main/benchmarks/sql_benchmarks/README.md) and run with `dfv bench`. See [Benchmarks](../docs/benchmarks.md) for how to prepare the data, run the suites, check results, and compare revisions.

| Suite | Description |
| --- | --- |
| [`jsonbench`](jsonbench/README.md) | The five JSONBench queries over Bluesky events stored as unshredded, partially shredded, shredded, and mixed Parquet Variant |

DataFusion's `benchmark_runner` builds its own session without the Variant functions, so it cannot run these suites. `dfv bench` runs the same files through DataFusion's unpublished `datafusion-benchmarks` crate, which it pulls from git at the same revision as the other DataFusion crates.
