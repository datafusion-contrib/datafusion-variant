# Variant paths

`variant_get`, `try_variant_get`, `variant_delete`, and the `:` operator use a
path to select a value inside a Variant. Paths follow the Spark 4.2 syntax, a
small subset of [JSONPath](https://www.rfc-editor.org/rfc/rfc9535.html).

The examples on this page use this table:

```sql
CREATE TABLE requests AS
SELECT parse_json('{
  "service": "api",
  "user": {
    "id": 7,
    "roles": ["admin", "dev"]
  },
  "http.status": 200,
  "user-agent": "curl"
}') AS request;
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
       variant_get(request, '$["http.status"]', 'INT') AS status,
       variant_get(request, '$.http.status', 'INT') AS nested_status
FROM requests;
```

```text
+----+-------------+--------+---------------+
| id | second_role | status | nested_status |
+----+-------------+--------+---------------+
| 7  | dev         | 200    | NULL          |
+----+-------------+--------+---------------+
```

<!--
```sql
SELECT to_json(variant_get(request, '$')) = to_json(request) AS whole,
       variant_get(request, '$.User') IS NULL AS other_case,
       variant_get(v, '$.*', 'INT') AS star,
       variant_get(v, '$["a]b"]', 'INT') AS bracket,
       variant_get(v, '$["$"]', 'INT') AS dollar,
       variant_get(v, '$["a\b"]', 'INT') AS backslash,
       variant_get(v, '$["it''s"]', 'INT') AS quote
FROM requests, (SELECT parse_json('{"*":0,"a]b":1,"$":2,"a\\b":3,"it''s":4}') AS v);
```

```text
+-------+------------+------+---------+--------+-----------+-------+
| whole | other_case | star | bracket | dollar | backslash | quote |
+-------+------------+------+---------+--------+-----------+-------+
| true  | true       | 0    | 1       | 2      | 3         | 4     |
+-------+------------+------+---------+--------+-----------+-------+
```
-->

`$.http.status` looks for a field `status` inside an object `http`, so quote a
name that contains `.` or `[`. Unlike JSONPath, `\` inside quotes is not an
escape character, so `$["a\b"]` selects the field `a\b`. Field names are
case-sensitive.

Paths have no wildcards, slices, filters, negative indexes, or recursive descent
(`..`), and `$.*` selects a field named `*`.

## NULLs and errors

<!--
```sql
SELECT p, variant_get(request, p, 'VARCHAR') AS value
FROM requests,
     (VALUES ('$.service'), ('$.user.name'), ('$.user.roles[2]'), ('$.user.roles.id'), (NULL)) AS paths(p)
ORDER BY p;
```

```text
+-----------------+-------+
| p               | value |
+-----------------+-------+
| $.service       | api   |
| $.user.name     | NULL  |
| $.user.roles.id | NULL  |
| $.user.roles[2] | NULL  |
| NULL            | NULL  |
+-----------------+-------+
```
-->

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

<!-- readme-test: error: variant_delete: invalid path -->
<!--
```sql
SELECT variant_delete(request, '$') FROM requests;
```
-->

- A path that does not match the value, such as a missing field, an index past
  the end of an array, or a field step on an array, returns SQL NULL.
- The path can be any string expression, such as a column. A SQL NULL path
  returns SQL NULL.
- A path that does not start with `$`, such as `user.id` or an empty string, is
  invalid. An invalid path raises an error, even in `try_variant_get`.
- `variant_delete` also rejects `$` alone, because it deletes a field or an
  array element.

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

<!--
```sql
SELECT request:user.roles[0]::VARCHAR AS one_path,
       request:user:roles[0]::VARCHAR AS chained,
       request:`user-agent` IS NULL AS backtick
FROM requests;
```

```text
+----------+---------+----------+
| one_path | chained | backtick |
+----------+---------+----------+
| admin    | admin   | true     |
+----------+---------+----------+
```
-->

<!-- readme-test: error: Field user not found in struct -->
<!--
```sql
SELECT request.user.roles[0] FROM requests;
```
-->

<!-- readme-test: error: No field named agent -->
<!--
```sql
SELECT request:user-agent FROM requests;
```
-->

Chaining `:` reads the same path, so `request:user:roles[0]` returns the same
value as `request:user.roles[0]`. These forms do not work as expected:

| Write                    | Not                        | Because                                                                                                                    |
|--------------------------|----------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `request:user.roles[0]`  | `request.user.roles[0]`    | Without `:`, `.` is struct field access on the Variant's Arrow storage, which fails with `Field user not found in struct`. |
| `request:['user-agent']` | `request:user-agent`       | `-` subtracts a column named `agent` from `request:user`.                                                                  |
| `request:['user-agent']` | `` request:`user-agent` `` | Backtick-quoted names return SQL NULL for now, while Spark selects the field ([#141]).                                     |

[#141]: https://github.com/datafusion-contrib/datafusion-variant/issues/141
