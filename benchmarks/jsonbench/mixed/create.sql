-- The JSONBench table over the files written by `load.sh mixed`.
CREATE EXTERNAL TABLE IF NOT EXISTS bluesky
STORED AS PARQUET
LOCATION 'data/jsonbench/mixed/';
