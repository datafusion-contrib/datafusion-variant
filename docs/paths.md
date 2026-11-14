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

`$.http.status` looks for a field `status` inside an object `http`, so quote a
name that contains `.` or `[`. Unlike JSONPath, `\` inside quotes is not an
escape character, and paths have no wildcards, slices, filters, negative
indexes, or recursive descent (`..`), so `$.*` selects a field named `*`:

```sql
SELECT variant_get(v, '$["a]b"]', 'INT') AS bracket,
       variant_get(v, '$["a\b"]', 'INT') AS backslash,
       variant_get(v, '$["it''s"]', 'INT') AS quote,
       variant_get(v, '$.*', 'INT') AS star
FROM (SELECT parse_json('{"a]b": 1, "a\\b": 2, "it''s": 3, "*": 4}') AS v);
```

```text
+---------+-----------+-------+------+
| bracket | backslash | quote | star |
+---------+-----------+-------+------+
| 1       | 2         | 3     | 4    |
+---------+-----------+-------+------+
```

## NULLs and errors

A path that does not match the value returns SQL NULL, and so does a SQL NULL
path. The path can be any string expression, such as a column:

```sql
SELECT path, variant_get(request, path, 'VARCHAR') AS value
FROM requests,
     (VALUES ('$.service'), ('$.User'), ('$.user.name'),
             ('$.user.roles[2]'), ('$.user.roles.id'), (NULL)) AS paths(path)
ORDER BY path;
```

```text
+-----------------+-------+
| path            | value |
+-----------------+-------+
| $.User          | NULL  |
| $.service       | api   |
| $.user.name     | NULL  |
| $.user.roles.id | NULL  |
| $.user.roles[2] | NULL  |
| NULL            | NULL  |
+-----------------+-------+
```

Field names are case-sensitive, so `$.User` does not match `user`, and a field
step on an array, such as `$.user.roles.id`, does not match either.

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

An invalid path raises an error, even in `try_variant_get`. This includes a path
that does not start with `$`, such as `user.id` or an empty string.
`variant_delete` also rejects `$` alone, because it deletes a field or an array
element.

## The `:` operator

`value:path` is shorthand for `variant_get` with `$.` in front of the path, or
`$` when the path starts with `[`. So `request:user.roles[0]` reads
`$.user.roles[0]`, and `request:['http.status']` reads `$['http.status']`:

```sql
SELECT request:user.roles[0]::VARCHAR AS first_role,
       request:['http.status']::INT AS status
FROM requests;
```

```text
+------------+--------+
| first_role | status |
+------------+--------+
| admin      | 200    |
+------------+--------+
```

Chaining `:` reads the same path. Backtick-quoted names return SQL NULL for now,
while Spark selects the field
([#141](https://github.com/datafusion-contrib/datafusion-variant/issues/141)),
so quote them with brackets instead:

```sql
SELECT request:user:roles[0]::VARCHAR AS chained,
       request:`user-agent` AS backtick,
       request:['user-agent']::VARCHAR AS bracket
FROM requests;
```

```text
+---------+----------+---------+
| chained | backtick | bracket |
+---------+----------+---------+
| admin   | NULL     | curl    |
+---------+----------+---------+
```

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

These forms fail:

| Write                    | Not                     | Because                                                                                                                    |
|--------------------------|-------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `request:user.roles[0]`  | `request.user.roles[0]` | Without `:`, `.` is struct field access on the Variant's Arrow storage, which fails with `Field user not found in struct`. |
| `request:['user-agent']` | `request:user-agent`    | `-` subtracts a column named `agent` from `request:user`, which fails with `No field named agent`.                         |
