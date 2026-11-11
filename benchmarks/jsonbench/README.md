# JSONBench on Parquet Variant

The five [JSONBench queries](https://github.com/ClickHouse/JSONBench/blob/e6c7c98dc766394d51f7d506a3dd2b5d51165d70/clickhouse/queries.sql) over the Bluesky event data, stored as Parquet Variant in four layouts ([#175](https://github.com/datafusion-contrib/datafusion-variant/issues/175)). See the [SQL benchmarks README](../README.md) for how to run suites.

| Layout | Fields shredded |
| --- | --- |
| `unshredded` | None |
| `partial` | `kind`, `did`, `time_us` |
| `shredded` | The above plus `commit.collection` and `commit.operation` |
| `mixed` | None in file 1, as in `shredded` in files 2 to 5 |

Other fields stay in the residual Variant. Strings are `VARCHAR` and `time_us` is `BIGINT`.

## Preparing the data

Each layout directory has a `prepare.sql` that reads Bluesky files 1 to 5 (5M rows) with `load_bsky` and writes Snappy Parquet with 65,536-row row groups to `data/jsonbench/<layout>`. `load_bsky` downloads missing files to `data/bluesky` (about 135 MB each).

```sh
for layout in unshredded partial shredded mixed; do
  dfv -f benchmarks/jsonbench/$layout/prepare.sql
done
```

Some source documents are truncated at 65,535 bytes (line 91,840 of `file_0005.json.gz`), so the scripts parse with `try_parse_json`, which stores them as NULL. `dfv -f` reports a failed statement but still exits successfully, so check the output for errors. Edit the `COPY` statements to change a layout or add files.

## Running

```sh
dfv bench jsonbench
dfv bench jsonbench --subgroup mixed --query 3 -i 10
dfv bench jsonbench --result-mode validate
```

Each case is one query over one layout, named like `jsonbench/Q01/shredded`. Its `table.sql` registers the `bluesky` table outside timing, and the timing covers planning through the last result batch. The `results/` files were recorded from the five default files, so `--result-mode validate` only applies to that data.

To try a change to a query or layout by hand:

```sh
dfv -i 10 -f benchmarks/jsonbench/shredded/table.sql -f benchmarks/jsonbench/queries/q1.sql
```

## Queries

The queries extract `VARCHAR` and `BIGINT` values with `variant_get(data, path, type)`, report UTC hours, and order ties deterministically. Q5 divides each positive epoch-microsecond endpoint by 1,000 before subtracting, to count millisecond boundaries as ClickHouse's `dateDiff` does.

Sources (Apache-2.0): [JSONBench data and queries](https://github.com/ClickHouse/JSONBench) and [DataFusion's SQL benchmark runner](https://github.com/apache/datafusion/tree/main/benchmarks/sql_benchmarks).
