# Spark compatibility

This crate aims to match Apache Spark's Variant behavior. The first release
targets the [Variant functions of Spark 4.2.0][spark-functions] (`32f72996`),
and later releases follow Spark's development.

Spark's tests for these functions are ported to the [SQL tests][sql-tests],
and each file names the Spark revision its expected results come from. The
ported cases pass, except for the differences below. Each difference links the
skipped or adapted test cases that record Spark's result.

Paths follow Spark 4.2 syntax. See [Variant paths](paths.md).

If you find a difference that is not listed here, please
[open an issue][issues].

## JSON parsing and schema inference

| Difference from Spark | Cause and evidence |
| --- | --- |
| Fractional JSON numbers become `DOUBLE`, where Spark keeps an exact decimal. Integers outside the `i64` range, such as `9223372036854775809`, lose precision. This also changes `schema_of_variant` and `schema_of_variant_agg`, for example `DOUBLE` instead of `DECIMAL(1,0)` for `1.0`. | **Arrow JSON parser:** numbers are read as `i64`, then `f64`. [Parser][arrow-json], [arrow-rs#11169](https://github.com/apache/arrow-rs/issues/11169), [`parse_json.slt`][t-parse-json], [`schema_of_variant.slt`][t-schema]. Schema merging itself matches Spark: [merge cases][t-schema-merge]. |
| Duplicate object keys are accepted and the last value wins. Spark rejects them by default. | **Arrow JSON parser:** duplicates collapse while parsing. [`parse_json.slt`][t-parse-json]. |
| `1 2` is rejected, where Spark reads the first value. `1e309` is rejected, where Spark reads Double infinity. `try_parse_json` returns NULL for both. | **Arrow JSON parser:** trailing input and numeric overflow are errors. [`parse_json.slt`][t-parse-json]. |
| Spark's 16 MiB limit on a Variant value or its metadata is not enforced. | **Not implemented here.** Spark's builder enforces [`SIZE_LIMIT`][spark-util]. [`parse_json.slt`][t-parse-json]. |
| Spark's settings for allowing duplicate keys and for legacy Unicode validation are not available. | **Crate API:** `parse_json` and `try_parse_json` take one argument. [`parse_json.slt`][t-parse-json]. |

## Field access and casts

Typed getters, `:` casts, and casts of Variant values all use the same Arrow
conversion, so each limitation below applies to all of them.

| Difference from Spark | Cause and evidence |
| --- | --- |
| A backtick-quoted name after `:`, such as ``v:`a b` ``, returns SQL NULL instead of the field. Arithmetic needs parentheses around a right-hand `:` access, as in `v:a::int + (v:b::int)`. | **Crate `:` planner and the DataFusion SQL parser.** [Field extraction cases 23, 29 and 39][t-fields]. |
| Numeric strings, such as `"1"`, `"-0.0"` and `"inf"`, do not convert to integer or floating-point types. `variant_get` fails and `try_variant_get` returns NULL. | **Arrow conversion:** numeric targets do not parse strings. [Primitive cast cases][t-casts]. |
| Strings do not convert to dates, numbers do not convert to timestamps, and Booleans do not convert to decimals. | **Arrow conversion:** these conversions are missing. [arrow-rs#10102](https://github.com/apache/arrow-rs/issues/10102), [primitive cast cases][t-casts]. |
| Casting a number to a string fails, where Spark returns its text form. Use `to_json` for JSON text. | **Arrow conversion:** string targets accept only Variant strings. [arrow-rs#10024](https://github.com/apache/arrow-rs/issues/10024), [field extraction case 03][t-fields]. |
| Nested Variant targets, such as `ARRAY<VARIANT>` or `STRUCT<a VARIANT>`, are rejected. | **Arrow conversion:** nested Variant fields are not supported, so the crate rejects them while planning. [`variant_get_sql_types.slt`][t-sql-types]. |
| Type names follow DataFusion SQL. For example, Spark's `BINARY` is `BYTEA` here. | **DataFusion type planner:** type hints resolve like SQL casts. [Primitive cast cases][t-casts]. |
| `coalesce(v, 'default')` does not coerce the string to Variant. Cast it explicitly with `CAST('default' AS VARIANT)`. | **DataFusion coercion:** Variant has no implicit coercion rules. [Field extraction case 39][t-fields]. |
| `explode(CAST(v:scores AS ARRAY<INT>))` is not supported. `variant_explode` works with `CROSS JOIN LATERAL` and `INNER JOIN LATERAL`, but not with `LEFT JOIN LATERAL`. Use `variant_explode_outer` to keep rows with empty input. | **Not ported:** Spark's `explode`. **DataFusion:** correlated outer lateral joins, [apache/datafusion#21201](https://github.com/apache/datafusion/issues/21201). [Field extraction case 30][t-fields], [`variant_explode.slt`][t-explode]. |

## Conversion to Variant and JSON output

| Difference from Spark | Cause and evidence |
| --- | --- |
| `to_variant_object` drops fields whose value is NULL, where Spark keeps them as Variant null. Duplicate keys keep the last value, where Spark fails. | **Arrow `cast_to_variant`.** [arrow-rs#11947](https://github.com/apache/arrow-rs/issues/11947), [arrow-rs#11948](https://github.com/apache/arrow-rs/issues/11948), [conversion cases][t-conversions]. |
| Arrays and structs that contain Variant values cannot be converted to Variant. | **Arrow `cast_to_variant`** has no support for nested Variant fields, and DataFusion's `named_struct` drops their extension metadata. [arrow-rs#11942](https://github.com/apache/arrow-rs/issues/11942), [conversion cases][t-conversions]. |
| `to_json(v, options)` ignores its second argument, including `timeZone`. Timestamps are not formatted in the session time zone. | **Crate and Arrow:** the options are not read, and Arrow's serializer has no time zone setting. [arrow-rs#11945](https://github.com/apache/arrow-rs/issues/11945), [conversion cases][t-conversions]. |
| `to_json` accepts only Variant. Spark also serializes arrays, maps and structs. | **Crate:** registering the Variant `to_json` replaces any other function with that name. Convert other values with `to_variant_object` first. |

Like Spark 4.2, `CAST(struct_or_map AS VARIANT)` is an error and
`to_variant_object` is the way to convert them. SQL NULL and Variant null stay
distinct, as described in
[Tell SQL NULL from Variant null](cookbook/common-pitfalls.md#tell-sql-null-from-variant-null),
except in the shredded cases below.

## Shredded data and validation

| Difference from Spark | Cause and evidence |
| --- | --- |
| Reading a missing field from a shredded object as Variant returns Variant null instead of SQL NULL. Casting a shredded Variant null to a struct returns a struct of NULL fields instead of NULL. | **Arrow `variant_get`.** [arrow-rs#11191](https://github.com/apache/arrow-rs/issues/11191) and [arrow-rs#11301](https://github.com/apache/arrow-rs/issues/11301). [`variant_sql_type.slt`][t-variant-sql-type]. |
| Some `variant_get` calls on shredded data, or on Variant columns stored as `Binary` or `LargeBinary`, fail because the result does not match the declared output type. | **Crate:** the declared return type does not always match what Arrow returns. [`variant_sql_type.slt`][t-variant-sql-type], [`is_variant_null.slt`][t-is-variant-null]. |
| `is_valid_variant` rejects some bytes that Spark accepts. It checks metadata, UTF-8, key order and uniqueness, offsets, nesting, and date and time ranges. | **Arrow validation** is stricter than Spark, and this crate keeps Arrow's rules. [`is_valid_variant.slt`][t-is-valid]. |
| `variant_explode` and `variant_delete` validate the whole input, so they reject malformed values that Spark never reads. | **Crate:** inputs are fully validated before they are traversed. [`variant_delete.slt`][t-delete]. |
| Malformed shredded metadata can panic while a value is reconstructed. | **Arrow `unshred_variant`.** [arrow-rs#11069](https://github.com/apache/arrow-rs/issues/11069). |

## Beyond Spark 4.2

- Arrow types without a Spark equivalent, such as unsigned integers, `Float16`
  and nanosecond timestamps, convert to Variant.
  [`variant_conversion.slt`][t-variant-conversion].
- `is_valid_variant` accepts Variant `TIME` values, which the Spark revision in
  [`is_valid_variant.slt`][t-is-valid] rejects.
- UUID values in Variant can be read, but SQL UUID values cannot be converted to
  Variant yet. [arrow-rs#11937](https://github.com/apache/arrow-rs/issues/11937).
- `variant_object_construct`, `variant_object_insert`, `variant_object_keys`,
  `variant_list_construct`, `variant_list_insert` and `variant_delete` are not
  Spark 4.2 functions. Spark added `variant_delete` in 4.3.
- Errors use DataFusion and Arrow messages, not Spark error classes. The ported
  tests check the kind of error.
- `typeof` and other general Spark functions are not provided.
  [Field extraction cases 33 to 36][t-fields].

## Keeping this page current

When a difference is fixed, enable its test cases and remove its row. Upstream
fixes and their releases are tracked in
[#141](https://github.com/datafusion-contrib/datafusion-variant/issues/141).

[spark-functions]: https://spark.apache.org/docs/4.2.0/sql-ref-functions-builtin.html#variant-functions
[spark-util]: https://github.com/apache/spark/blob/32f7299601108917fb01920a54e084595b7b3bf8/common/variant/src/main/java/org/apache/spark/types/variant/VariantUtil.java
[arrow-json]: https://github.com/apache/arrow-rs/blob/60.1.0/parquet-variant-json/src/from_json.rs
[sql-tests]: https://github.com/datafusion-contrib/datafusion-variant/tree/main/tests/test_files
[issues]: https://github.com/datafusion-contrib/datafusion-variant/issues
[t-parse-json]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/parse_json.slt
[t-schema]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/schema_of_variant.slt
[t-schema-merge]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/spark_schema_of_variant_merge.slt
[t-fields]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/spark_variant_field_extractions.slt
[t-casts]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/spark_variant_get_casts_primitives.slt
[t-sql-types]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_get_sql_types.slt
[t-explode]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_explode.slt
[t-conversions]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/spark_variant_conversions.slt
[t-variant-sql-type]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_sql_type.slt
[t-is-variant-null]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/is_variant_null.slt
[t-is-valid]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/is_valid_variant.slt
[t-delete]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_delete.slt
[t-variant-conversion]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files/variant_conversion.slt
