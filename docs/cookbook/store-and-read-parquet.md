# Store and read Parquet

These examples use the `events` table from [Load and query JSON](load-and-query-json.md).

<!--
```sql
CREATE TABLE events AS
SELECT parse_json(json_data) AS event
FROM (VALUES
    ('{"service":"api","status":200,"latency_ms":12,"tags":["cached"]}'),
    ('{"service":"api","status":500,"latency_ms":85,"error":null}'),
    ('{"service":"api","status":504,"error":"Somethings not working"}')
) AS input(json_data);
```
-->

## Store Variant in Parquet

You can write Variant columns to Parquet like any other column with the normal
DataFusion `COPY` statement:

```sql
COPY events TO 'events.parquet' STORED AS PARQUET;
```

The Variant column is stored in Parquet following the [Parquet Variant
encoding specification][parquet-variant-encoding] as a struct with `metadata`
and `value` fields. You can see the structure using the
`parquet_metadata` table function:

```sql
-- Note: parquet_metadata is part of dfv and datafusion-cli, not this crate.
SELECT path_in_schema, type FROM parquet_metadata('events.parquet');
```

```text
+------------------+------------+
| path_in_schema   | type       |
+------------------+------------+
| "event.metadata" | BYTE_ARRAY |
| "event.value"    | BYTE_ARRAY |
+------------------+------------+
```

### Write shredded Parquet columns

You can choose which fields to store as typed Parquet columns with the
`variant.shredding` option. For example, to store the `service` and `status`
fields of the `event` column as typed Parquet columns, run:

```sql
COPY events TO 'shredded_events.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"event": "STRUCT<service VARCHAR, status BIGINT>"}');
```

See the [shredding guide](../shredding.md) for the full format of the
`variant.shredding` option.

## Read Variant from Parquet

You can read `events.parquet` back with the Variant column preserved. The same
query works with `shredded_events.parquet`.

```sql
SELECT variant_get(event, '$.service', 'VARCHAR') AS service,
       event:tags[0]::VARCHAR AS first_tag -- Array indexing via `[]` operator
FROM 'events.parquet'
ORDER BY event:status::BIGINT;
```

```text
+---------+-----------+
| service | first_tag |
+---------+-----------+
| api     | cached    |
| api     | NULL      |
| api     | NULL      |
+---------+-----------+
```

[parquet-variant-encoding]: https://parquet.apache.org/docs/file-format/types/variantencoding/
