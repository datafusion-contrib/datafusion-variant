-- Writes Bluesky files 1 to 100 (100M rows) to data/jsonbench/shredded with
-- every field the queries read shredded. Run from the repository root:
--   dfv -f benchmarks/jsonbench/shredded/load.sql
-- Change the load_bsky range to load fewer files. COPY adds files to the
-- directory, so delete it before loading again.

SET datafusion.execution.parquet.compression = 'snappy';
SET datafusion.execution.parquet.max_row_group_size = 65536;
SET datafusion.execution.soft_max_rows_per_output_file = 1000000;

COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky(1, 100))
TO 'data/jsonbench/shredded/' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"data": "STRUCT<kind VARCHAR, did VARCHAR, time_us BIGINT, commit STRUCT<collection VARCHAR, operation VARCHAR>>"}');
