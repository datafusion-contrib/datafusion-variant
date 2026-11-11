-- Writes JSONBench files 1-5 (5M rows) as unshredded Variant.
-- Run from the repository root: dfv -f benchmarks/jsonbench/unshredded/prepare.sql
-- Some source documents are truncated at 65,535 bytes, so try_parse_json stores them as NULL.

COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky(1))
TO 'data/jsonbench/unshredded/file_0001.parquet' STORED AS PARQUET
OPTIONS ('format.compression' 'snappy', 'format.max_row_group_size' '65536');

COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky(2))
TO 'data/jsonbench/unshredded/file_0002.parquet' STORED AS PARQUET
OPTIONS ('format.compression' 'snappy', 'format.max_row_group_size' '65536');

COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky(3))
TO 'data/jsonbench/unshredded/file_0003.parquet' STORED AS PARQUET
OPTIONS ('format.compression' 'snappy', 'format.max_row_group_size' '65536');

COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky(4))
TO 'data/jsonbench/unshredded/file_0004.parquet' STORED AS PARQUET
OPTIONS ('format.compression' 'snappy', 'format.max_row_group_size' '65536');

COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky(5))
TO 'data/jsonbench/unshredded/file_0005.parquet' STORED AS PARQUET
OPTIONS ('format.compression' 'snappy', 'format.max_row_group_size' '65536');
