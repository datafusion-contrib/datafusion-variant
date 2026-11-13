# Variant paths

`variant_get`, `try_variant_get`, `variant_delete`, and the `:` operator use a
path to select a value inside a Variant. Paths follow the Spark 4.2 syntax, a
small subset of [JSONPath](https://www.rfc-editor.org/rfc/rfc9535.html).

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
SELECT to_json(variant_get(request, '$')) = to_json(request) AS whole,
       variant_get(request, '$.user.name') IS NULL AS missing_field,
       variant_get(request, '$.user.roles[2]') IS NULL AS past_end,
       variant_get(request, '$.user.roles.id') IS NULL AS field_of_array,
       variant_get(request, '$.User') IS NULL AS other_case,
       variant_get(parse_json('{"*":1}'), '$.*', 'INT') AS star
FROM requests;
```

```text
+-------+---------------+----------+----------------+------------+------+
| whole | missing_field | past_end | field_of_array | other_case | star |
+-------+---------------+----------+----------------+------------+------+
| true  | true          | true     | true           | true       | 1    |
+-------+---------------+----------+----------------+------------+------+
```
-->

Field names are case-sensitive, and `$.*` selects a field named `*`, not every
field. A path that does not match the value, such as a missing field, an index
past the end of an array, or a field step on an array, returns SQL NULL.

<!--
```sql
SELECT p, variant_get(request, p, 'VARCHAR') AS value
FROM requests, (VALUES ('$.service'), ('$.user.roles[0]'), (NULL)) AS paths(p)
ORDER BY p;
```

```text
+-----------------+-------+
| p               | value |
+-----------------+-------+
| $.service       | api   |
| $.user.roles[0] | admin |
| NULL            | NULL  |
+-----------------+-------+
```
-->

The path is a string expression, so it can also come from a column. A SQL NULL
path returns SQL NULL.

<!-- readme-test: error: invalid Variant path -->
<!--
```sql
SELECT variant_get(request, 'user.id') FROM requests;
SELECT try_variant_get(request, '') FROM requests;
SELECT variant_get(request, '$.user.roles[*]') FROM requests;
SELECT variant_get(request, '$.user.roles[-1]') FROM requests;
SELECT variant_get(request, '$..id') FROM requests;
```
-->

A path that does not start with `$`, such as `user.id` or an empty string, is
invalid. Paths have no wildcards, slices, filters, negative indexes, or
recursive descent (`..`). An invalid path raises an error, even in
`try_variant_get`.

<!-- readme-test: error: variant_delete: invalid path -->
<!--
```sql
SELECT variant_delete(request, '$') FROM requests;
```
-->

`variant_delete` also rejects `$` alone, because it deletes a field or an array
element.

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

Every character between the quotes is part of the name, including `]` and `$`.
Unlike JSONPath, `\` is not an escape character, so `$["a\b"]` selects the field
`a\b`. To select a name that contains `'`, quote it with `"`, and the reverse.

## The `:` operator

`value:path` is shorthand for `variant_get` with `$.` in front of the path, or
`$` when the path starts with `[`. So `request:user.roles[0]` reads
`$.user.roles[0]`, and `request:['http.status']` reads `$['http.status']`:

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

Write `:` once, at the start of the path, and continue it with `.` and `[...]`:

| Expression              | Reads                                      |
|-------------------------|--------------------------------------------|
| `request:user.roles[0]` | `$.user.roles[0]`                          |
| `request:user:roles[0]` | `$.user`, then `$.roles[0]` of that result |
| `request.user`          | The struct field `user` (an error)         |

<!--
```sql
SELECT request:user.roles[0]::VARCHAR AS one_path,
       request:user:roles[0]::VARCHAR AS chained
FROM requests;
```

```text
+----------+---------+
| one_path | chained |
+----------+---------+
| admin    | admin   |
+----------+---------+
```
-->

<!-- readme-test: error: Field user not found in struct -->
<!--
```sql
SELECT request.user FROM requests;
```
-->

The chained form returns the same value, but it extracts the whole `user` object
before reading `roles[0]`, which is slower, especially for shredded columns
([#176](https://github.com/datafusion-contrib/datafusion-variant/issues/176)).
Without `:`, `.` is DataFusion's struct field access. Variant columns are stored
as Arrow structs with no `user` field, so `request.user` fails with
`Field user not found in struct`.

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

Backtick-quoted names, such as `` request:`user-agent` ``, return SQL NULL,
while Spark selects the field. Use `request:['user-agent']` instead. Support is
tracked in [#141](https://github.com/datafusion-contrib/datafusion-variant/issues/141).
