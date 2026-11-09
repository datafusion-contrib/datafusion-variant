# Variant shredding

The standard Variant binary encoding is efficient and compact, but it must still
be read and decoded for every query. The [Parquet Variant shredding
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
Variant into values of that Parquet type. A value is shredded when its Variant
type is in the same [equivalence class][parquet-variant-encoding-types] as the
layout type and it converts without loss. Otherwise, a `NULL` is stored in the
shredded column and the value remains in the residual Variant column. Integers
and decimals are all in the Exact Numeric class, while `DOUBLE`, `FLOAT`,
strings, dates, and timestamps with and without a time zone each have their own
class.

For example, shredding these Variant values as `BIGINT`:

<!--
```sql
CREATE TABLE samples AS SELECT * FROM (VALUES
    (1, CAST(arrow_cast(300, 'Int16') AS VARIANT)),
    (2, CAST(arrow_cast(2.00, 'Decimal128(3, 2)') AS VARIANT)),
    (3, CAST(arrow_cast(1.50, 'Decimal128(3, 2)') AS VARIANT)),
    (4, CAST(2.0 AS VARIANT)),
    (5, CAST('200' AS VARIANT))
) AS input(id, v);
COPY samples TO 'shredded_samples.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"v": "BIGINT"}');
SELECT s.v AS "Variant value",
       schema_of_variant(s.v) AS "Variant type",
       residual_json(f.v) AS "residual Variant",
       f.v['typed_value'] AS "shredded BIGINT"
FROM samples s JOIN 'shredded_samples.parquet' f ON s.id = f.id
ORDER BY s.id;
```
-->

```text
+---------------+--------------+------------------+-----------------+
| Variant value | Variant type | residual Variant | shredded BIGINT |
+---------------+--------------+------------------+-----------------+
| 300           | BIGINT       | NULL             | 300             |
| 2             | DECIMAL(3,2) | NULL             | 2               |
| 1.5           | DECIMAL(3,2) | 1.5              | NULL            |
| 2.0           | DOUBLE       | 2.0              | NULL            |
| "200"         | STRING       | "200"            | NULL            |
+---------------+--------------+------------------+-----------------+
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

## Supported types

You can use any SQL type supported by DataFusion and listed in the [shredding
specification][parquet-variant-shredding]. For example:

* `BOOLEAN`
* `TINYINT`, `SMALLINT`, `INT`, `BIGINT`
* `FLOAT`, `DOUBLE`, `DECIMAL(p, s)`
* `DATE`, `TIMESTAMP`, `TIMESTAMP WITH TIME ZONE` (see below)
* `VARCHAR`, `BYTEA`
* `STRUCT<name type, ...>` and `type[]`, nested to any depth

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

[parquet-variant-encoding-types]: https://parquet.apache.org/docs/file-format/types/variantencoding/#encoding-types
[parquet-variant-shredding]: https://parquet.apache.org/docs/file-format/types/variantshredding/
