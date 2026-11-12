# Variant paths

`variant_get`, `try_variant_get`, `variant_delete`, and the `:` operator use a
path to select a value inside a Variant. Paths follow the Spark 4.2 syntax, a
small subset of JSONPath.

The examples on this page use this table:

```sql
CREATE TABLE requests AS
SELECT parse_json('{"service":"api","user":{"id":7,"roles":["admin","dev"]},"http.status":200,"user-agent":"curl"}') AS request;
```

## Syntax

A path starts with `$`, which selects the whole value, followed by any number of
steps:

| Step                     | Selects                                                                        |
|--------------------------|--------------------------------------------------------------------------------|
| `.name`                  | The field `name`. The name ends at the next `.` or `[`.                        |
| `['name']` or `["name"]` | The field `name`, which can contain any character except the quote around it. |
| `[n]`                    | The array element at index `n`, starting from 0.                               |

<!--
```sql
SELECT to_json(variant_get(request, '$')) = to_json(request) AS whole
FROM requests;
```

```text
+-------+
| whole |
+-------+
| true  |
+-------+
```
-->

```sql
SELECT variant_get(request, '$.user.id', 'INT') AS id,
       variant_get(request, '$.user.roles[1]', 'VARCHAR') AS second_role,
       variant_get(request, '$["http.status"]', 'INT') AS status
FROM requests;
```

```text
+----+-------------+--------+
| id | second_role | status |
+----+-------------+--------+
| 7  | dev         | 200    |
+----+-------------+--------+
```

<!--
```sql
SELECT variant_get(request, '$.user.name') IS NULL AS missing_field,
       variant_get(request, '$.user.roles[2]') IS NULL AS past_end,
       variant_get(request, '$.user.roles.id') IS NULL AS field_of_array,
       variant_get(request, '$.User') IS NULL AS other_case
FROM requests;
```

```text
+---------------+----------+----------------+------------+
| missing_field | past_end | field_of_array | other_case |
+---------------+----------+----------------+------------+
| true          | true     | true           | true       |
+---------------+----------+----------------+------------+
```
-->

Field names are case-sensitive. A path that does not match the value, such as a
missing field, an index past the end of an array, or a field step on an array,
returns SQL NULL.

<!-- readme-test: error: invalid Variant path -->
<!--
```sql
SELECT try_variant_get(request, '$.user.roles[*]') FROM requests;
SELECT variant_get(request, '$.user.roles[-1]') FROM requests;
SELECT variant_get(request, '$..id') FROM requests;
```
-->

<!--
```sql
SELECT variant_get(parse_json('{"*":1}'), '$.*', 'INT') AS star;
```

```text
+------+
| star |
+------+
| 1    |
+------+
```
-->

Paths have no wildcards, slices, filters, negative indexes, or recursive
descent (`..`). `$.*` selects a field named `*`. An invalid path raises an
error, even in `try_variant_get`.

<!--
```sql
SELECT p, variant_get(request, p, 'VARCHAR') AS value
FROM requests, (VALUES ('$.service'), ('$.user.roles[0]')) AS paths(p)
ORDER BY p;
```

```text
+-----------------+-------+
| p               | value |
+-----------------+-------+
| $.service       | api   |
| $.user.roles[0] | admin |
+-----------------+-------+
```
-->

The path is a string expression, so it can also come from a column.

## Field names with special characters

Use a quoted step for a field name that contains `.` or `[`. `$.http.status`
selects the field `status` inside an object `http`, while `$["http.status"]`
selects the field named `http.status`:

```sql
SELECT variant_get(request, '$.http.status') IS NULL AS nested_missing,
       variant_get(request, '$["http.status"]', 'INT') AS status
FROM requests;
```

```text
+----------------+--------+
| nested_missing | status |
+----------------+--------+
| true           | 200    |
+----------------+--------+
```

<!--
```sql
SELECT variant_get(v, '$["a]b"]', 'INT') AS bracket,
       variant_get(v, '$["$"]', 'INT') AS dollar,
       variant_get(v, '$["a\b"]', 'INT') AS backslash,
       variant_get(v, '$["it''s"]', 'INT') AS quote
FROM (SELECT parse_json('{"a]b":1,"$":2,"a\\b":3,"it''s":4}') AS v);
```

```text
+---------+--------+-----------+-------+
| bracket | dollar | backslash | quote |
+---------+--------+-----------+-------+
| 1       | 2      | 3         | 4     |
+---------+--------+-----------+-------+
```
-->

Every character between the quotes is part of the name, including `]`, `$`, and
`\`. To select a name that contains `'`, quote it with `"`, and the reverse.

## The `:` operator

`value:path` is shorthand for `variant_get(value, 'path')`, where `path` is a
[rootless path](#rootless-paths), written without the leading `$`:

```sql
SELECT request:user.roles[0]::VARCHAR AS first_role,
       request:['http.status']::INT AS status,
       request:['user-agent']::VARCHAR AS agent
FROM requests;
```

```text
+------------+--------+-------+
| first_role | status | agent |
+------------+--------+-------+
| admin      | 200    | curl  |
+------------+--------+-------+
```

<!-- readme-test: error: No field named agent -->
<!--
```sql
SELECT request:user-agent FROM requests;
```
-->

Quote a field name that is not a plain SQL identifier, such as `user-agent`.
Without quotes, `request:user-agent` subtracts a column named `agent` from
`request:user`.

<!--
```sql
SELECT request:`user-agent` IS NULL AS backtick
FROM requests;
```

```text
+----------+
| backtick |
+----------+
| true     |
+----------+
```
-->

Backtick-quoted names, such as `` request:`user-agent` ``, return SQL NULL
instead of the field. Use `request:['user-agent']`. Support is tracked in
[#141](https://github.com/datafusion-contrib/datafusion-variant/issues/141).

## Deleting with `variant_delete`

<!-- readme-test: error: variant_delete: invalid path -->
<!--
```sql
SELECT variant_delete(request, 'user') FROM requests;
SELECT variant_delete(request, '$') FROM requests;
```
-->

`variant_delete` only accepts paths that start with `$`, and the path must
select a field or array element, not `$` itself.

## Rootless paths

`variant_get` and `try_variant_get` also accept paths without the leading `$`,
and the `:` operator always uses them:

| With `$`           | Without `$`         |
|--------------------|---------------------|
| `$`                | `''` (empty string) |
| `$.user.roles[0]`  | `user.roles[0]`     |
| `$["http.status"]` | `["http.status"]`   |

<!--
```sql
SELECT to_json(variant_get(request, '')) = to_json(request) AS whole,
       variant_get(request, 'user.roles[0]', 'VARCHAR') AS first_role,
       variant_get(request, '["http.status"]', 'INT') AS status
FROM requests;
```

```text
+-------+------------+--------+
| whole | first_role | status |
+-------+------------+--------+
| true  | admin      | 200    |
+-------+------------+--------+
```
-->

<!-- readme-test: error: invalid Variant path -->
<!--
```sql
SELECT variant_get(parse_json('{"$type":1}'), '$type');
```
-->

<!--
```sql
SELECT variant_get(parse_json('{"$type":1}'), '$.$type', 'INT') AS rooted,
       variant_get(parse_json('{"$type":1}'), '["$type"]', 'INT') AS quoted,
       variant_get(parse_json('{"a\\b":3,"ab":4}'), '["a\\b"]', 'INT') AS escaped,
       variant_get(parse_json('{"a\\b":3,"ab":4}'), '["a\b"]', 'INT') AS unescaped;
```

```text
+--------+--------+---------+-----------+
| rooted | quoted | escaped | unescaped |
+--------+--------+---------+-----------+
| 1      | 1      | 3       | 4         |
+--------+--------+---------+-----------+
```
-->

They differ from `$` paths in two ways:

- A path that starts with `$` is always a `$` path, so `'$type'` is an invalid
  path, not the field `$type`. Use `'$.$type'` or `'["$type"]'`.
- Inside brackets, `\` escapes the next character, so `'["a\\b"]'` selects the
  field `a\b` and `'["a\b"]'` selects the field `ab`.

Unifying the two forms is tracked in
[#166](https://github.com/datafusion-contrib/datafusion-variant/issues/166).

## Differences from Spark

<!--
```sql
SELECT to_json(variant_get(request, p)) = to_json(request) AS whole
FROM requests, (VALUES (CAST(NULL AS VARCHAR))) AS paths(p);
```

```text
+-------+
| whole |
+-------+
| true  |
+-------+
```
-->

- Spark only accepts `$` paths in function arguments.
- A SQL NULL path, such as a NULL in a path column, selects the whole value.
  Spark returns SQL NULL.
- After `:`, Spark decodes backtick-quoted names and keeps `\` as part of quoted
  names.
