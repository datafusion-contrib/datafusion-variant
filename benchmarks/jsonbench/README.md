# JSONBench on Parquet Variant

The five [JSONBench queries](https://github.com/ClickHouse/JSONBench/blob/e6c7c98dc766394d51f7d506a3dd2b5d51165d70/clickhouse/queries.sql)
over the Bluesky events, stored as Parquet Variant in three layouts
([#175](https://github.com/datafusion-contrib/datafusion-variant/issues/175))
and run with `dfv`:

| Layout | Fields shredded |
| --- | --- |
| `unshredded` | None |
| `shredded` | `kind`, `did`, `time_us`, `commit.collection`, `commit.operation` |
| `mixed` | As in `shredded` in every fifth file (5, 10, 15, ...), none in the rest |

Run the commands below from the repository root.

## Preparing the data

By default, `load.sh <layout>` converts Bluesky files 1 to 10 (10M rows) to
Parquet files in `data/jsonbench/<layout>`, such as
`data/jsonbench/mixed/file_0001.parquet`. It downloads any missing Bluesky files
to `data/bluesky` (about 135 MB each):

```sh
cargo build --release --bin dfv
./benchmarks/jsonbench/load.sh unshredded
./benchmarks/jsonbench/load.sh shredded
./benchmarks/jsonbench/load.sh mixed
```

The script can be configured in various way, for example to convert all 100M rows use

```sh
NUM_FILES=100 ./benchmarks/jsonbench/load.sh mixed
```

See the script source for more options.

The resulting Parquet files are Snappy encoded with 65,536-rows in each row
group and 1M rows per file.  Shredded strings are stored as `VARCHAR` and
`time_us` is `BIGINT`. Note that some source documents are truncated.

JSONBench also reports how much storage each layout takes:

```sh
du -sh data/jsonbench/*
```

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
