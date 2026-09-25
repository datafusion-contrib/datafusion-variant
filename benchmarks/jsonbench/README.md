# JSONBench on Parquet Variant

The five [JSONBench queries](https://github.com/ClickHouse/JSONBench/blob/e6c7c98dc766394d51f7d506a3dd2b5d51165d70/clickhouse/queries.sql)
use `variant_get(data, path, type)` and scan Parquet directly.

```sh
cargo run --release --example jsonbench -- prepare
cargo run --release --example jsonbench -- run
```

Preparation downloads the original million-row Bluesky gzip file to
`data/bluesky` (also used by `download_data.sh`) and writes these layouts under
`data/jsonbench`:

| Layout | Fields shredded |
| --- | --- |
| `unshredded` | None |
| `partial` | `kind`, `did`, `time_us` |
| `query-fields` | The above plus `commit.collection`, `commit.operation` |

Other fields remain in the residual Variant. Strings use Utf8 and `time_us`
uses Int64. Conversion reads gzip in 8,192-row batches and writes Snappy Parquet
with 65,536-row groups (`prepare --batch-size` and `--row-group-size`).
Each layout is converted separately. Temporary output is renamed after closing
the Parquet writer.

Like DataFusion's ClickBench downloader, source reuse checks file size and
incomplete downloads resume. Sizes come from HTTP Content-Length. Prepared
Parquet is reused when its footer has the expected row count; when all layouts
exist, preparation needs no network access. There are no manifests or checksums.
Delete prepared files to regenerate after changing conversion settings or code.

Both commands accept `--size 1m|10m|100m|1000m` and `--rows N` (a smaller prefix).
Use the same selection for preparation and execution; filenames include the row
count so smaller runs can coexist. Override paths with `--data-dir` and
`prepare --source-dir`.

```sh
cargo run --release --example jsonbench -- prepare --rows 10000
cargo run --release --example jsonbench -- run --rows 10000 --iterations 1
```

By default the runner requires all three layouts and executes all five queries,
five iterations each. Use `--layout partial --query 3` to select a case.
Other run options are `--partitions` (4), `--batch-size` (8192),
`--memory-limit-mib` (1024), and `--debug` (show plans outside timing).
UTC and Parquet extension metadata are enabled for the Variant queries.

Timing follows DataFusion ClickBench: SQL planning through result collection,
including Parquet reads. Preparation and table registration are outside timing.
There is no explicit warmup; the OS may cache data between iterations.
Results default to `data/jsonbench/results.json` and use DataFusion's comparison
format (milliseconds and result row counts):

```sh
cargo run --release --example jsonbench -- run -o data/jsonbench/base.json
# Repeat on the other revision with -o data/jsonbench/change.json.
python ../datafusion/benchmarks/compare.py \
  data/jsonbench/base.json data/jsonbench/change.json
```

Use identical fixtures and options when comparing. Python's `rich` is required.
Query errors stop the run; output is written after all selected cases succeed.

`cargo test --example jsonbench` checks document preservation, all five queries
across the three layouts, and expected results.
SQL uses `Utf8View`/`Int64` extraction, UTC hours, and deterministic tie ordering.
Q5 divides each positive epoch-microsecond endpoint by 1,000 before subtraction
to match ClickHouse's millisecond-boundary `dateDiff` semantics.

Sources (Apache-2.0): [JSONBench data and queries](https://github.com/ClickHouse/JSONBench),
[Xiangpeng Hao's conversion example](https://github.com/XiangpengHao/liquid-cache-bench/tree/aa9559b451b85e6dec430321bf29a2eb174d0e46/json_bench),
and [DataFusion's ClickBench runner](https://github.com/apache/datafusion/blob/main/benchmarks/src/clickbench.rs)
and [result format](https://github.com/apache/datafusion/blob/main/benchmarks/src/util/run.rs).
