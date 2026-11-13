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

Each layout's `load.sh` converts Bluesky files 1 to 100 (100M rows) to one
Parquet file each in `data/jsonbench/<layout>`, such as
`data/jsonbench/mixed/file_0001.parquet`. It downloads missing Bluesky files to
`data/bluesky` (about 135 MB each):

```sh
cargo build --release --bin dfv
./benchmarks/jsonbench/unshredded/load.sh
./benchmarks/jsonbench/shredded/load.sh
./benchmarks/jsonbench/mixed/load.sh
```

The scripts download and convert as many files at once as the machine has
cores. They skip Parquet files that are already written, so run a script again
to pick up an interrupted load or to retry files that failed. To convert a
layout again from scratch, delete its directory.

Environment variables change the defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DFV` | `target/release/dfv` | The `dfv` binary to run |
| `FILES` | `100` | Convert Bluesky files 1 to `FILES`. `mixed` leaves the first fifth unshredded |
| `JOBS` | Number of cores | Files to download and convert at once |
| `DFV_BSKY_DIR` | `data/bluesky` | Where the Bluesky files are downloaded |

For example, to load 10M rows with a `dfv` on the `PATH`:

```sh
DFV=dfv FILES=10 ./benchmarks/jsonbench/mixed/load.sh
```

The scripts write Snappy Parquet with 65,536-row row groups and 1M rows per
file. Fields a layout does not shred stay in the residual Variant. Shredded
strings are `VARCHAR` and `time_us` is `BIGINT`. Some source documents are
truncated, so the scripts parse with `try_parse_json`, which stores them as
NULL.

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
