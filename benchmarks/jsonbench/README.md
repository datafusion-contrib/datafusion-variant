# JSONBench on Parquet Variant

The five [JSONBench queries](https://github.com/ClickHouse/JSONBench/blob/e6c7c98dc766394d51f7d506a3dd2b5d51165d70/clickhouse/queries.sql) over the Bluesky event data, stored as Parquet Variant in four layouts ([#175](https://github.com/datafusion-contrib/datafusion-variant/issues/175)). See [Benchmarks](../../docs/benchmarks.md#jsonbench) for the layouts and how to prepare and run the suite.

| Path | Contents |
| --- | --- |
| `<layout>/prepare.sql` | Writes the layout's Parquet files to `data/jsonbench/<layout>` |
| `<layout>/table.sql` | Registers the `bluesky` table over those files |
| `<layout>/q01.benchmark` to `q05.benchmark` | One case per query, filled in from `jsonbench.benchmark.template` |
| `queries/` | The five queries, shared by every layout |
| `results/` | Expected results for the five default files |

The prepare scripts write Snappy Parquet with 65,536-row row groups. Fields a layout does not shred stay in the residual Variant. Shredded strings are `VARCHAR` and `time_us` is `BIGINT`. Some source documents are truncated at 65,535 bytes (line 91,840 of `file_0005.json.gz`), so the scripts parse with `try_parse_json`, which stores them as NULL. Edit the `COPY` statements to change a layout or add files.

## Queries

The queries extract `VARCHAR` and `BIGINT` values with `variant_get(data, path, type)`, report UTC hours, and order ties deterministically. Q5 divides each positive epoch-microsecond endpoint by 1,000 before subtracting, to count millisecond boundaries as ClickHouse's `dateDiff` does.

Sources (Apache-2.0): [JSONBench data and queries](https://github.com/ClickHouse/JSONBench) and [DataFusion's SQL benchmark runner](https://github.com/apache/datafusion/tree/main/benchmarks/sql_benchmarks).
