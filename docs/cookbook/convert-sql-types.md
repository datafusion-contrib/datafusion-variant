# Convert to and from SQL types

`CAST(value AS VARIANT)` converts SQL values to Variant. For example, to
convert an integer and a decimal to Variant and show them as JSON, run:

```sql
SELECT CAST(42 AS VARIANT) AS int_value,
       CAST(1.23 AS VARIANT) AS decimal_value;
```

```text
+-----------+---------------+
| int_value | decimal_value |
+-----------+---------------+
| 42        | 1.23          |
+-----------+---------------+
```

To convert a list, struct, or map to Variant, use `to_variant_object`:

```sql
SELECT to_variant_object([1, 2, 3]) AS list_value,
       to_variant_object([{a: 1, b: 2}, {b: 3, a: 4}]) AS struct_list_value;
```

```text
+------------+-------------------------------+
| list_value | struct_list_value             |
+------------+-------------------------------+
| [1,2,3]    | [{"a":1,"b":2},{"a":4,"b":3}] |
+------------+-------------------------------+
```

<!-- readme-test: error: Failed to extract primitive -->
<!--
```sql
SELECT CAST(parse_json('{"a":1}') AS INT);
```
-->

To convert a Variant value to a SQL type, use `CAST` or `::`. `TRY_CAST`
returns SQL NULL instead of an error when the value does not fit the type:

```sql
SELECT CAST(parse_json('42') AS BIGINT) AS int_value,
       parse_json('"api"')::VARCHAR AS string_value,
       TRY_CAST(parse_json('{"a":1}') AS INT) AS not_an_int;
```

```text
+-----------+--------------+------------+
| int_value | string_value | not_an_int |
+-----------+--------------+------------+
| 42        | api          | NULL       |
+-----------+--------------+------------+
```

To extract a field as a SQL type, see [Query](load-and-query-json.md#query).
