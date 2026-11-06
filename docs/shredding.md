# Variant shredding

Shredding stores some fields of a Variant column as typed Parquet columns while
keeping the remaining fields in the standard Variant binary encoding. Queries on
the shredded fields read the typed columns directly, and all other fields are
preserved. The on-disk layout is defined by the [Parquet Variant shredding
specification][parquet-variant-shredding].

This document describes the `variant.shredding` option of the DataFusion `COPY`
statement, which chooses the layout. See [Write shredded Parquet
columns](../README.md#write-shredded-parquet-columns) in the README for an
introduction. You can run the examples interactively with the `dfv`
command-line tool in this repository.

## The `variant.shredding` option

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
