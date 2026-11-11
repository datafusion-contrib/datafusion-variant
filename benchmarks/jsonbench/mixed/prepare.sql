-- Writes JSONBench files 1-5 (5M rows) with file 1 unshredded and files 2-5 shredded as in shredded/.
-- Run from the repository root: dfv -f benchmarks/jsonbench/mixed/prepare.sql
-- Some source documents are truncated at 65,535 bytes, so try_parse_json stores them as NULL.

SET datafusion.execution.parquet.compression = 'snappy';
SET datafusion.execution.parquet.max_row_group_size = 65536;

COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky(1))
TO 'data/jsonbench/mixed/file_0001.parquet' STORED AS PARQUET;

COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky(2))
TO 'data/jsonbench/mixed/file_0002.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"data": "STRUCT<kind VARCHAR, did VARCHAR, time_us BIGINT, commit STRUCT<collection VARCHAR, operation VARCHAR>>"}');

COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky(3))
TO 'data/jsonbench/mixed/file_0003.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"data": "STRUCT<kind VARCHAR, did VARCHAR, time_us BIGINT, commit STRUCT<collection VARCHAR, operation VARCHAR>>"}');

COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky(4))
TO 'data/jsonbench/mixed/file_0004.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"data": "STRUCT<kind VARCHAR, did VARCHAR, time_us BIGINT, commit STRUCT<collection VARCHAR, operation VARCHAR>>"}');

COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky(5))
TO 'data/jsonbench/mixed/file_0005.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"data": "STRUCT<kind VARCHAR, did VARCHAR, time_us BIGINT, commit STRUCT<collection VARCHAR, operation VARCHAR>>"}');
