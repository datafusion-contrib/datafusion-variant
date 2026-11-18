# Variant paths

`variant_get`, `try_variant_get`, `variant_delete`, and the `:` operator use a
path to select a value inside a Variant. Paths follow the Spark 4.2 syntax,
which is based on [JSONPath](https://www.rfc-editor.org/rfc/rfc9535.html).

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
  "user-agent": "curl",
  "trace\\id": "abc"
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
name that contains `.` or `[`.

Unlike JSONPath, paths have no wildcards, slices, filters, negative indexes, or
recursive descent (`..`), so `$.*` selects a field named `*`.

## Escaping

Paths have no escape sequences. Unlike JSONPath, `\` inside quotes is an ordinary
character, so `$["trace\id"]` selects the field named `trace\id`. To select a
name that contains a quote, put it inside the other kind of quote.

```sql
SELECT variant_get(request, '$["http.status"]', 'INT') AS status,
       variant_get(request, '$["trace\id"]', 'VARCHAR') AS trace_id
FROM requests;
```

```text
+--------+----------+
| status | trace_id |
+--------+----------+
| 200    | abc      |
+--------+----------+
```

## Missing paths and errors

A path that does not exist returns NULL. See
[Tell SQL NULL from Variant null](cookbook/common-pitfalls.md#tell-sql-null-from-variant-null)
for how this differs from a JSON `null`. The path can be any string expression,
such as a column:

```sql
SELECT path, variant_get(request, path, 'VARCHAR') AS value
FROM requests,
     (VALUES ('$.service'), ('$.User'), ('$.user.roles[2]'),
             ('$.user.roles.id'), ('$.*')) AS paths(path)
ORDER BY path;
```

```text
+-----------------+-------+
| path            | value |
+-----------------+-------+
| $.*             | NULL  |
| $.User          | NULL  |
| $.service       | api   |
| $.user.roles.id | NULL  |
| $.user.roles[2] | NULL  |
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

`value:path` is shorthand for `variant_get`. Thus, `request:user.roles[0]` reads the path
`$.user.roles[0]`, and `request:['http.status']` reads the path `$['http.status']`:

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

`request:user:roles[0]` reads the same path as `request:user.roles[0]`.
Backtick-quoted names return SQL NULL for now,
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

See [Common pitfalls](cookbook/common-pitfalls.md#read-variant-fields-with-the-colon-operator)
for `:` forms that fail.
