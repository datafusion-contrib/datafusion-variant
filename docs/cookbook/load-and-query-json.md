# Load and query JSON

These pages contain examples of common queries using the Variant type. See
[Getting started](../getting-started.md) to run them from Rust or in `dfv`.

## Convert JSON to Variant

`parse_json` converts JSON strings to Variant. You can use it to load
semi-structured documents. For example, to create an `events` table with a Variant
column named `event`, run:

```sql
CREATE TABLE events AS
SELECT parse_json(json_data) AS event
FROM (VALUES
    ('{"service":"api","status":200,"latency_ms":12,"tags":["cached"]}'),
    ('{"service":"api","status":500,"latency_ms":85,"error":null}'),
    ('{"service":"api","status":504,"error":"Somethings not working"}')
) AS input(json_data);
```

> [!TIP]
> For potentially malformed input, use `try_parse_json(json_data)` to return
> SQL NULL instead of failing the query.

## Query

You can extract fields from a Variant column using the Spark SQL-style `:`
field-access operator or the `variant_get` UDF. For example, to select the
`service` and `latency_ms` fields of events whose `status` is at least 400, run:

<!-- TODO: remove the `::BIGINT` explicit typecasts when https://github.com/datafusion-contrib/datafusion-variant/issues/159 is ready -->
```sql
SELECT event:service::VARCHAR,
       event:latency_ms::BIGINT
FROM events
WHERE event:status::BIGINT >= 400;
```

```text
+----------------------+-------------------------+
| events.event:service | events.event:latency_ms |
+----------------------+-------------------------+
| api                  | 85                      |
| api                  | NULL                    |
+----------------------+-------------------------+
```

The `:` syntax is automatically rewritten to `variant_get`, so this is an
equivalent query:

```sql
SELECT variant_get(event, '$.service', 'VARCHAR') AS service,
       variant_get(event, '$.latency_ms', 'BIGINT') AS latency_ms
FROM events
WHERE variant_get(event, '$.status', 'BIGINT') >= 400;
```

```text
+---------+------------+
| service | latency_ms |
+---------+------------+
| api     | 85         |
| api     | NULL       |
+---------+------------+
```

## Convert Variant to JSON

`to_json` converts Variant values to JSON strings. To return the entire `event`
column as JSON text, run:

```sql
SELECT to_json(event) AS json_data
FROM events
ORDER BY event:status::BIGINT;
```

```text
+------------------------------------------------------------------+
| json_data                                                        |
+------------------------------------------------------------------+
| {"latency_ms":12,"service":"api","status":200,"tags":["cached"]} |
| {"error":null,"latency_ms":85,"service":"api","status":500}      |
| {"error":"Somethings not working","service":"api","status":504}  |
+------------------------------------------------------------------+
```

## Inspect the schema of Variant values

Due to their dynamic nature, Variant values can have different fields and types in
each row. You can inspect the schema of a Variant column using the following
functions:

```sql
-- Get the schema of each Variant value
SELECT schema_of_variant(event) FROM events;
```

```text
+----------------------------------------------------------------------------------+
| schema_of_variant(events.event)                                                  |
+----------------------------------------------------------------------------------+
| OBJECT<latency_ms: BIGINT, service: STRING, status: BIGINT, tags: ARRAY<STRING>> |
| OBJECT<error: VOID, latency_ms: BIGINT, service: STRING, status: BIGINT>         |
| OBJECT<error: STRING, service: STRING, status: BIGINT>                           |
+----------------------------------------------------------------------------------+
```

```sql
-- Print the merged schema (union) of all Variant values in the column
SELECT schema_of_variant_agg(event) AS schema FROM events;
```

```text
+-------------------------------------------------------------------------------------------------+
| schema                                                                                          |
+-------------------------------------------------------------------------------------------------+
| OBJECT<error: STRING, latency_ms: BIGINT, service: STRING, status: BIGINT, tags: ARRAY<STRING>> |
+-------------------------------------------------------------------------------------------------+
```
