# JSONBench on Parquet Variant

The five [JSONBench queries](https://github.com/ClickHouse/JSONBench/blob/e6c7c98dc766394d51f7d506a3dd2b5d51165d70/clickhouse/queries.sql)
over the Bluesky events, stored as Parquet Variant in three layouts
([#175](https://github.com/datafusion-contrib/datafusion-variant/issues/175))
and run with `dfv`:

| Layout | Fields shredded |
| --- | --- |
| `unshredded` | None |
| `shredded` | `kind`, `did`, `time_us`, `commit.collection`, `commit.operation` |
| `mixed` | None in files 1 to 20, as in `shredded` in files 21 to 100 |

Run the commands below from the repository root.

## Preparing the data

Each layout's `load.sql` reads Bluesky files 1 to 100 (100M rows) with
`load_bsky`, which downloads missing files to `data/bluesky` (about 135 MB
each), and writes Parquet to `data/jsonbench/<layout>`:

```sh
cargo build --release --bin dfv
target/release/dfv -f benchmarks/jsonbench/unshredded/load.sql
target/release/dfv -f benchmarks/jsonbench/shredded/load.sql
target/release/dfv -f benchmarks/jsonbench/mixed/load.sql
```

To load fewer files, change the `load_bsky` ranges: for example
`load_bsky(1, 10)` for 10M rows, and `load_bsky(1, 2)` and `load_bsky(3, 10)` in
`mixed/load.sql`. `COPY` adds files to the layout directory, so delete it before
loading again. `dfv -f` reports a failed statement but still exits successfully,
so check the output for errors.

The scripts write Snappy Parquet with 65,536-row row groups and about 1M rows
per file. Fields a layout does not shred stay in the residual Variant. Shredded
strings are `VARCHAR` and `time_us` is `BIGINT`. Some source documents are
truncated, so the scripts parse with `try_parse_json`, which stores them as
NULL.

## Running the queries

Each layout's `create.sql` creates the `bluesky` table over its files, so pass
it before the query files. `-i N` runs the files N times in one session and
prints the time of each run, then the minimum, median, and maximum:

```sh
target/release/dfv -i 10 -f benchmarks/jsonbench/shredded/create.sql \
  -f benchmarks/jsonbench/queries/q1.sql
```

The first run also creates the table. `dfv` prints the time of each statement
as well, so passing several query files times each of them.

## Queries

The queries extract `VARCHAR` and `BIGINT` values with
`variant_get(data, path, type)`, report UTC hours, and order ties
deterministically. Q5 divides each positive epoch-microsecond endpoint by 1,000
before subtracting, to count millisecond boundaries as ClickHouse's `dateDiff`
does.

Sources (Apache-2.0): [JSONBench data and queries](https://github.com/ClickHouse/JSONBench).
