-- The JSONBench table over the files written by prepare.sql in this directory.
CREATE EXTERNAL TABLE IF NOT EXISTS bluesky
STORED AS PARQUET
LOCATION 'data/jsonbench/partial/';
