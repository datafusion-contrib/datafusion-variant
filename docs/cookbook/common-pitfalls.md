# Common pitfalls

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

## Cast fields before comparing them

<!-- readme-test: error: Cannot infer common argument type -->
<!--
```sql
SELECT count(*) FROM events WHERE event:status >= 400;
```
-->

Comparison operators do not accept Variant values, so
`WHERE event:status >= 400` fails to plan. Cast the field to a SQL type first,
as in [Query](load-and-query-json.md#query).

## Cast fields before grouping or joining on them

Grouping and join keys compare Variant values by their binary encoding, which
includes the field names of the row each value came from. Equal values from rows
with different fields do not match, so grouping by `event:service` splits one
service into three groups:

```sql
SELECT event:service AS service, count(*) AS events
FROM events
GROUP BY event:service;
```

```text
+---------+--------+
| service | events |
+---------+--------+
| "api"   | 1      |
| "api"   | 1      |
| "api"   | 1      |
+---------+--------+
```

<!--
```sql
SELECT count(*) AS joined
FROM events JOIN (SELECT parse_json('{"service":"api"}') AS o) AS owners
ON event:service = o:service;
```

```text
+--------+
| joined |
+--------+
| 0      |
+--------+
```
-->

A join on `event:service` matches no rows for the same reason. Cast the field to
a SQL type first:

```sql
SELECT event:service::VARCHAR AS service, count(*) AS events
FROM events
GROUP BY event:service::VARCHAR;
```

```text
+---------+--------+
| service | events |
+---------+--------+
| api     | 3      |
+---------+--------+
```

## Tell SQL NULL from Variant null

A missing field returns SQL NULL. A JSON `null` is a Variant null value, which
`IS NULL` does not match. Use `is_variant_null` to detect it:

```sql
SELECT event:status::BIGINT AS status,
       event:error IS NULL AS error_is_null,
       is_variant_null(event:error) AS error_is_variant_null
FROM events
ORDER BY status;
```

```text
+--------+---------------+-----------------------+
| status | error_is_null | error_is_variant_null |
+--------+---------------+-----------------------+
| 200    | true          | false                 |
| 500    | false         | true                  |
| 504    | false         | false                 |
+--------+---------------+-----------------------+
```
