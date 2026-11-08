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

Storing the shredded column is logically equivalent to storing the Variant
values as three columns:

| `event` (Variant)                                                  | `event.service` (VARCHAR) | `event.status` (BIGINT) |
|--------------------------------------------------------------------|---------------------------|-------------------------|
| `{"service":"api","status":200,"latency_ms":12,"tags":["cached"]}` | `api`                     | `200`                   |
| `{"service":"api","status":500,"latency_ms":85,"error":null}`      | `api`                     | `500`                   |
| `{"service":"api","status":504,"error":"Somethings not working"}`  | `api`                     | `504`                   |

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

This is logically equivalent to storing the Variant column as three columns.
Fields that are not in the layout, such as `service`, stay in the Variant
column, and rows without a shredded field have `NULL` in its typed column:

| `request` (Variant)                                                                 | `request.detail.latency_ms` (BIGINT) | `request.tags` (VARCHAR[]) |
|-------------------------------------------------------------------------------------|--------------------------------------|----------------------------|
| `{"service":"api","detail":{"latency_ms":12,"region":"us-east"},"tags":["cached"]}` | `12`                                 | `["cached"]`               |
| `{"service":"api","detail":{"latency_ms":85},"tags":[]}`                            | `85`                                 | `[]`                       |
| `{"service":"web","detail":{"region":"eu-west"},"error":"timeout"}`                 | `NULL`                               | `NULL`                     |

## Format of the `variant.shredding` option

`variant.shredding` is parsed as JSON, and its value must be a JSON object.

Each key is the case-sensitive name of a top-level Variant column in the `COPY`
input that will be shredded. The value of each field is the SQL type used to
store the shredded column.

Note that this option only applies when copying to Parquet, and partition
columns cannot be shredded.

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

| residual Variant                   | shredded BIGINT |
|------------------------------------|-----------------|
| `{"service":"api","status":"200"}` | `NULL`          |

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
result is logically:

| residual Variant | shredded STRUCT |
|------------------|-----------------|
| `{"c":3}`        | `{"a":1,"b":2}` |

## Supported types

You can use any SQL type supported by DataFusion and listed in the [shredding
specification][parquet-variant-shredding]. For example:

* `BOOLEAN`
* `TINYINT`, `SMALLINT`, `INT`, `BIGINT`
* `FLOAT`, `DOUBLE`, `DECIMAL(p, s)`
* `DATE`, `TIME`, `TIMESTAMP`, `TIMESTAMP WITH TIME ZONE`
* `VARCHAR`, `BYTEA`
* `STRUCT<name type, ...>` and `type[]`, nested to any depth

Unsigned integers, intervals, and other types without a shredded Parquet
representation are rejected.

[parquet-variant-shredding]: https://parquet.apache.org/docs/file-format/types/variantshredding/
