# Variant shredding

The standard Variant binary encoding is efficient and compact, but it must still
read and decode each value for every query. The [Parquet Variant shredding
specification][parquet-variant-shredding] describes an optional on-disk layout
for Variant columns that stores some fields in typed Parquet columns. Queries on
the shredded fields read the typed columns directly, which is significantly
faster.

## Writing shredded Parquet columns

### Top-level fields

Given the following table:

```sql
CREATE TABLE events AS
SELECT parse_json(json_data) AS event
FROM (VALUES
    ('{"service":"api","status":200,"latency_ms":12,"tags":["cached"]}'),
    ('{"service":"api","status":500,"latency_ms":85,"error":null}'),
    ('{"service":"api","status":504,"error":"Somethings not working"}')
) AS input(json_data);
```

You can store the `service` and `status` fields in typed columns by providing a
`variant.shredding` option to the `COPY` statement:

```sql
COPY events TO 'shredded_events.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"event": "STRUCT<service VARCHAR, status BIGINT>"}');
```

The `variant.shredding` option is a JSON object that maps the name of the
Variant column to a SQL type. The full format is described
[below](#format-of-the-variantshredding-option).

Storing the shredded column is logically equivalent to storing the typed
`service` and `status` values plus a residual Variant column with the remaining
fields:

<!--
```sql
SELECT residual_json(event) AS "event (residual)",
       event['typed_value']['service']['typed_value'] AS "event.service",
       event['typed_value']['status']['typed_value'] AS "event.status"
FROM 'shredded_events.parquet';
```
-->

```text
+-------------------------------------+---------------+--------------+
| event (residual)                    | event.service | event.status |
+-------------------------------------+---------------+--------------+
| {"latency_ms":12,"tags":["cached"]} | api           | 200          |
| {"error":null,"latency_ms":85}      | api           | 500          |
| {"error":"Somethings not working"}  | api           | 504          |
+-------------------------------------+---------------+--------------+
```

> [!NOTE]
> The actual encoding is more complicated: the typed columns and the binary
> residual are stored together in a single struct, as described in the
> [Parquet Variant shredding specification][parquet-variant-shredding].

### Nested fields

You can also shred nested objects and arrays. For example, if you have a
`requests` table with a nested `detail` object and `tags` array:

```sql
CREATE TABLE requests AS
SELECT parse_json(json_data) AS request
FROM (VALUES
    ('{"service":"api","detail":{"latency_ms":12,"region":"us-east"},"tags":["cached"]}'),
    ('{"service":"api","detail":{"latency_ms":85},"tags":[]}'),
    ('{"service":"web","detail":{"region":"eu-west"},"error":"timeout"}')
) AS input(json_data);
```

To shred the `latency_ms` field of the nested `detail` object and the `tags`
array, use the following `COPY` statement:

```sql
COPY requests TO 'shredded_requests.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{
  "request": "STRUCT<detail STRUCT<latency_ms BIGINT>, tags VARCHAR[]>"
}');
```

Fields that are not in the layout stay in the residual of the object that
contains them: `service` in the residual of `request`, and `region` in the
residual of `request.detail`. Rows without a shredded field have `NULL` in its
typed column:

<!-- TODO: remove `to_json` below when https://github.com/apache/datafusion/issues/24860 is fixed -->
<!--
```sql
SELECT residual_json(request) AS "request (residual)",
       residual_json(request, 'detail') AS "request.detail (residual)",
       request['typed_value']['detail']['typed_value']['latency_ms']['typed_value'] AS "request.detail.latency_ms",
       CASE WHEN request['typed_value']['tags']['typed_value'] IS NOT NULL
            THEN to_json(variant_get(request, 'tags')) END AS "request.tags"
FROM 'shredded_requests.parquet';
```
-->

```text
+-------------------------------------+---------------------------+---------------------------+--------------+
| request (residual)                  | request.detail (residual) | request.detail.latency_ms | request.tags |
+-------------------------------------+---------------------------+---------------------------+--------------+
| {"service":"api"}                   | {"region":"us-east"}      | 12                        | ["cached"]   |
| {"service":"api"}                   | NULL                      | 85                        | []           |
| {"error":"timeout","service":"web"} | {"region":"eu-west"}      | NULL                      | NULL         |
+-------------------------------------+---------------------------+---------------------------+--------------+
```

## Format of the `variant.shredding` option

`variant.shredding` is parsed as JSON, and its value must be a JSON object.

Each key is the case-sensitive name of a top-level Variant column in the `COPY`
input that will be shredded. The value of each field is the SQL type used to
store the shredded column.

The option is rejected for formats other than Parquet, and partition columns
cannot be shredded.

One statement can shred several Variant columns. For example, given a table
with two Variant columns, `event` and `metric`:

```sql
CREATE TABLE metrics AS
SELECT parse_json(event) AS event, parse_json(metric) AS metric
FROM (VALUES
    ('{"service":"api","status":200}', '42'),
    ('{"service":"api","status":500}', '7'),
    ('{"service":"web","status":504}', '"n/a"')
) AS input(event, metric);
```

The following `COPY` statement shreds the `status` field of `event` and the
whole `metric` column:

```sql
COPY metrics TO 'shredded_metrics.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{
  "event":  "STRUCT<status BIGINT>",
  "metric": "BIGINT"
}');
```

The shredding is described as SQL types, which are parsed by DataFusion's SQL
parser. The `COPY` statement attempts to "shred" the matching parts of the
Variant into values of that Parquet type. If the Variant value can be converted
to the specified type, the shredded column is populated with that value.
Otherwise, a `NULL` is stored in the shredded column and the value remains in
the residual Variant column.

For example, if you try to shred this Variant object as a `BIGINT`:

```json
{
  "service": "api",
  "status": "200"
}
```

the shredded column will be `NULL`:

<!--
```sql
CREATE TABLE status_objects AS SELECT parse_json('{"service":"api","status":"200"}') AS v;
COPY status_objects TO 'shredded_status.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"v": "BIGINT"}');
SELECT residual_json(v) AS "residual Variant",
       v['typed_value'] AS "shredded BIGINT"
FROM 'shredded_status.parquet';
```
-->

```text
+----------------------------------+-----------------+
| residual Variant                 | shredded BIGINT |
+----------------------------------+-----------------+
| {"service":"api","status":"200"} | NULL            |
+----------------------------------+-----------------+
```

If a Variant object is shredded into a `STRUCT`, fields in the Variant that are
not in the `STRUCT` remain in the residual Variant column. For example, if
you shred a Variant object like

```json
{
  "a": 1,
  "b": 2,
  "c": 3
}
```

into a `STRUCT` with only fields `a` and `b` (`STRUCT<a BIGINT, b BIGINT>`), the
result is:

<!--
```sql
CREATE TABLE objects AS SELECT parse_json('{"a":1,"b":2,"c":3}') AS v;
COPY objects TO 'shredded_objects.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"v": "STRUCT<a BIGINT, b BIGINT>"}');
SELECT residual_json(v) AS "residual Variant",
       named_struct('a', v['typed_value']['a']['typed_value'],
                    'b', v['typed_value']['b']['typed_value']) AS "shredded STRUCT"
FROM 'shredded_objects.parquet';
```
-->

```text
+------------------+-----------------+
| residual Variant | shredded STRUCT |
+------------------+-----------------+
| {"c":3}          | {a: 1, b: 2}    |
+------------------+-----------------+
```

#### Files with different shredding layouts

It is also possible to read files with different shredding layouts as a single
table, which is often referred to as schema evolution. This can happen when the
optimal shredding evolves over time, or different producers shred the same data
differently. DataFusion transparently unifies the different shredding layouts
into a single table.

For example, if you write `file1.parquet` with the `id` and `status` fields
shredded, and `file2.parquet` with just the `id` field shredded:

```sql
-- file1.parquet: shredded on id and status
COPY (
  SELECT parse_json(event) AS event FROM (VALUES
      ('{"id": 5, "service":"api", "status":200}'),
      ('{"id": 6, "service":"api", "status":500}'),
      ('{"id": 7, "service":"web", "status":504}')
  ) AS input(event)
) TO 'mixed_events/file1.parquet' STORED AS PARQUET
OPTIONS (
  'variant.shredding'
  '{"event": "STRUCT<id BIGINT, status BIGINT>"}' -- shred id and status
);

-- file2.parquet: shredded on id
COPY (
  SELECT parse_json(event) AS event FROM (VALUES
      ('{"id": 8, "service":"api", "status":200}'),
      ('{"id": 9, "service":"db", "status":503}')
  ) AS input(event)
) TO 'mixed_events/file2.parquet' STORED AS PARQUET
OPTIONS (
  'variant.shredding'
  '{"event": "STRUCT<id BIGINT>"}' -- shred id
);
```

Create a table over the directory, and DataFusion reads `event` from both files
as a single column:

```sql
CREATE EXTERNAL TABLE mixed_events STORED AS PARQUET LOCATION 'mixed_events/';

SELECT event:id, event FROM mixed_events ORDER BY event:id::BIGINT;
```

```text
+-----------------------+---------------------------------------+
| mixed_events.event:id | event                                 |
+-----------------------+---------------------------------------+
| 5                     | {"id":5,"service":"api","status":200} |
| 6                     | {"id":6,"service":"api","status":500} |
| 7                     | {"id":7,"service":"web","status":504} |
| 8                     | {"id":8,"service":"api","status":200} |
| 9                     | {"id":9,"service":"db","status":503}  |
+-----------------------+---------------------------------------+
```

Use `CREATE EXTERNAL TABLE` for such directories. Reading the directory directly
with `SELECT ... FROM 'mixed_events/'` does not unify the layouts yet and
returns an error or NULL for the fields that are shredded differently
([apache/datafusion#27089](https://github.com/apache/datafusion/issues/27089)).

See the documentation on [`VariantExprAdapterFactory`](../src/expr_adapter.rs)
for more details.

## Supported types

You can use any SQL type supported by DataFusion and listed in the [shredding
specification][parquet-variant-shredding]. For example:

* `BOOLEAN`
* `TINYINT`, `SMALLINT`, `INT`, `BIGINT`
* `FLOAT`, `DOUBLE`, `DECIMAL(p, s)`
* `DATE`, `TIMESTAMP`, `TIMESTAMP WITH TIME ZONE` (see below)
* `VARCHAR`, `BYTEA`
* `STRUCT<name type, ...>` and `type[]`, nested to any depth

<!-- These checks fail once a type listed below as rejected becomes supported, so this list can't silently go stale. -->
<!-- readme-test: error: not a valid variant shredding type -->
<!--
```sql
COPY (SELECT parse_json('1') AS v) TO 'rejected.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"v": "INT UNSIGNED"}');
COPY (SELECT parse_json('1') AS v) TO 'rejected.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"v": "INTERVAL"}');
COPY (SELECT parse_json('1') AS v) TO 'rejected.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"v": "TIME"}');
```
-->

<!-- readme-test: error: SHREDDING layout cannot contain VARIANT -->
<!--
```sql
COPY (SELECT parse_json('1') AS v) TO 'rejected.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"v": "STRUCT<a VARIANT>"}');
```
-->

The following are rejected:

* Unsigned integers, intervals, and other types without a shredded Parquet
  representation
* `TIME`, because DataFusion's `TIME` has nanosecond precision and Variant
  times are microseconds
* `VARIANT` fields inside a layout, which are not supported yet

To keep a field as Variant, leave it out of the layout.

<!--
```sql
CREATE TABLE times AS SELECT * FROM (VALUES
    (CAST(arrow_cast(TIMESTAMP '2024-01-01T00:00:00', 'Timestamp(Microsecond, Some("UTC"))') AS VARIANT))
) AS input(v);
COPY times TO 'unset_zone_times.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"v": "TIMESTAMP WITH TIME ZONE"}');
SELECT v['typed_value'] AS shredded FROM 'unset_zone_times.parquet';
```

```text
+----------+
| shredded |
+----------+
| NULL     |
+----------+
```

```sql
SET datafusion.execution.time_zone = 'UTC';
COPY times TO 'utc_zone_times.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"v": "TIMESTAMP WITH TIME ZONE"}');
SELECT v['typed_value'] AS shredded FROM 'utc_zone_times.parquet';
```

```text
+----------------------+
| shredded             |
+----------------------+
| 2024-01-01T00:00:00Z |
+----------------------+
```
-->

`TIMESTAMP WITH TIME ZONE` uses the session time zone,
`datafusion.execution.time_zone`, which is unset by default. Without it, the
layout is timezone-naive like `TIMESTAMP`, and timezone-aware Variant timestamps
stay in the residual. Variant stores these timestamps in UTC, so any zone works:
run `SET datafusion.execution.time_zone = 'UTC'` before the `COPY`.

[parquet-variant-shredding]: https://parquet.apache.org/docs/file-format/types/variantshredding/
