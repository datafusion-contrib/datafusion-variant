# Variant shredding

The standard Variant binary encoding is efficient and compact, but it must still
be read and decoded for every query. the [Parquet Variant shredding
specification][parquet-variant-shredding] describes an optional on-disk layout for
Variant columns that stores some fields in typed Parquet columns. Queries on the
shredded fields read the typed columns directly which is significantly faster.

## Writing Shredded Parquet columns

## Top-level fields

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
`variant.shredding` option to the `COPY` statement.

```sql
COPY events TO 'shredded_events.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{"event": "STRUCT<service VARCHAR, status BIGINT>"}');
```

The `variant.shredding` option is a JSON object that maps the name of the
Variant column to a SQL type and the full format is described [below]
(#format-of-the-variantshredding-option).

Storing the shredded column is logically equivalent to storing the Variant
values as three columns:

| `event` (Variant)                                                  | `event.service` (VARCHAR) | `event.status` (BIGINT) |
|--------------------------------------------------------------------|---------------------------|-------------------------|
| `{"service":"api","status":200,"latency_ms":12,"tags":["cached"]}` | `api`                     | `200`                   |
| `{"service":"api","status":500,"latency_ms":85,"error":null}`      | `api`                     | `500`                   |
| `{"service":"api","status":504,"error":"Somethings not working"}`  | `api`                     | `504`                   |


> [!INFO]
> Note that the actual encoding is more complicated: the typed columns and the
> binary residual are stored together in a single struct, as described in the
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

To shred the `detail:latency_ms` field and the `tags` array, use the following `COPY` statement:

```sql
COPY requests TO 'shredded_requests.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{
  "request": "STRUCT<detail STRUCT<latency_ms BIGINT>, tags VARCHAR[]>"
}');
```

This is logically equivalent to storing the Variant column as three columns.
Fields that are not in the layout, such as `service` stay in the Variant column,
and rows without a shredded field have `NULL` in its typed column:

| `request` (Variant)                                                                 | `request.detail.latency_ms` (BIGINT) | `request.tags` (VARCHAR[]) |
|-------------------------------------------------------------------------------------|--------------------------------------|----------------------------|
| `{"service":"api","detail":{"latency_ms":12,"region":"us-east"},"tags":["cached"]}` | `12`                                 | `["cached"]`               |
| `{"service":"api","detail":{"latency_ms":85},"tags":[]}`                            | `85`                                 | `[]`                       |
| `{"service":"web","detail":{"region":"eu-west"},"error":"timeout"}`                 | `NULL`                               | `NULL`                     |


## Format of the `variant.shredding` option

`variant.shredding` is a JSON object. Each key is the case-sensitive name of a
top-level Variant column in the `COPY` input, and each value is the SQL type to
shred that column into. Columns that are not listed keep their existing
storage. The option is only accepted when copying to Parquet, and partition
columns cannot be shredded.

Layouts can nest structs and arrays, and one statement can shred several
columns. For example, create an `events` table with two Variant columns,
`event` and `metric`:

```sql
CREATE TABLE events AS
SELECT parse_json(event) AS event, parse_json(metric) AS metric
FROM (VALUES
    ('{"service":"api","detail":{"latency":12},"tags":["cached"]}', '42'),
    ('{"service":"api","detail":{"latency":85},"error":null}', '7'),
    ('{"service":"web","tags":["retry","slow"],"error":"timeout"}', '"n/a"')
) AS input(event, metric);
```

To shred the `service` field, the `latency` field of the nested `detail`
object, and the `tags` array of `event`, nest the SQL types the same way as the
JSON. The whole `metric` column is shredded as a `BIGINT`:

```sql
COPY events TO 'shredded_events.parquet' STORED AS PARQUET
OPTIONS ('variant.shredding' '{
  "event":  "STRUCT<service VARCHAR, detail STRUCT<latency BIGINT>, tags VARCHAR[]>",
  "metric": "BIGINT"
}');
```

The `parquet_metadata` table function (part of `dfv` and `datafusion-cli`)
shows the resulting Parquet schema. Each shredded field has a `typed_value`
column for values that match the layout and a `value` column for values that
do not:

```sql
SELECT path_in_schema, type FROM parquet_metadata('shredded_events.parquet');
```

```text
+------------------------------------------------------------+------------+
| path_in_schema                                             | type       |
+------------------------------------------------------------+------------+
| "event.metadata"                                           | BYTE_ARRAY |
| "event.value"                                              | BYTE_ARRAY |
| "event.typed_value.service.value"                          | BYTE_ARRAY |
| "event.typed_value.service.typed_value"                    | BYTE_ARRAY |
| "event.typed_value.detail.value"                           | BYTE_ARRAY |
| "event.typed_value.detail.typed_value.latency.value"       | BYTE_ARRAY |
| "event.typed_value.detail.typed_value.latency.typed_value" | INT64      |
| "event.typed_value.tags.value"                             | BYTE_ARRAY |
| "event.typed_value.tags.typed_value.list.item.value"       | BYTE_ARRAY |
| "event.typed_value.tags.typed_value.list.item.typed_value" | BYTE_ARRAY |
| "metric.metadata"                                          | BYTE_ARRAY |
| "metric.value"                                             | BYTE_ARRAY |
| "metric.typed_value"                                       | INT64      |
+------------------------------------------------------------+------------+
```

Fields that are not in the layout, such as `error`, are stored in the
top-level `event.value` column. Values that do not match their type, such as
the string `"n/a"` in the third `metric`, are stored in the `value` column next
to their `typed_value` column. Nothing is lost either way.

## Supported types

Any SQL type that DataFusion can `CAST` to is accepted as long as the
[shredding specification][parquet-variant-shredding] has a shredded encoding
for the resulting Arrow type:

* `BOOLEAN`
* `TINYINT`, `SMALLINT`, `INT`, `BIGINT`
* `FLOAT`, `DOUBLE`, `DECIMAL(p, s)`
* `DATE`, `TIME`, `TIMESTAMP`, `TIMESTAMP WITH TIME ZONE`
* `VARCHAR`, `BYTEA`
* `STRUCT<name type, ...>` and `type[]`, nested to any depth

`VARIANT` cannot appear inside a layout; leave a field out to keep it in the
residual instead. Unsigned integers, intervals, and other types without a
shredded Parquet representation are rejected.

[parquet-variant-shredding]: https://parquet.apache.org/docs/file-format/types/variantshredding/
