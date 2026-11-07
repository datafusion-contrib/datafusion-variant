# Spark compatibility

This crate targets Spark Variant behavior, but compatibility is incomplete.
This document collects the known differences recorded in the implementation and
tests, explains where they originate, and links the supporting evidence. It is
not a claim of exhaustive compatibility with every Spark SQL feature.

## Baseline and evidence

Reviewed against datafusion-variant `642248a` with the Arrow Variant crates and
`arrow-cast` resolved to **60.1.0**, and DataFusion pinned to `484183dc34ea`.
The comparison uses Spark 4.2.0 and the exact Spark revisions recorded in the
linked tests. Newer ports, such as `variant_delete`, use their own pinned
revision; they do not imply coverage of all later Spark additions.

The test headers distinguish expectations previously executed on Spark from
expectations checked against Spark source/tests. This documentation review did
not rerun Spark. A `skipif DataFusion` record preserves a desired result; it is
not proof that the discrepancy still exists after a dependency upgrade. See
[stale skips](#skipped-expectations-needing-revalidation) below.

Primary Spark references:

- [Spark 4.2.0 Variant function reference][spark-functions].
- [Variant expression tests][spark-expressions] and [evaluation tests][spark-eval]
  at `32f7299601108917fb01920a54e084595b7b3bf8`: JSON parsing, primitive casts,
  and conversion to Variant.
- [SQL field-extraction cases][spark-fields] at
  `5c63836176a2a15a5d43a61f2e18890eb4ac7ece`.
- [Schema inference and merge tests][spark-schema] at
  `4ccf07b6d91e2b126b7001741778ee28f22c6609`.
- [Variant expressions and path parser][spark-paths] and
  [VariantBuilder][spark-builder] at
  `046f574fce3ffd93925f7bcdd459221fa11a730d`: deletion and path semantics.

## JSON parsing and schema inference

| Difference from Spark | Origin and evidence |
| --- | --- |
| Fractional JSON numbers become `DOUBLE` instead of Spark's exact decimals where representable. Integers outside `i64`, such as `9223372036854775809`, can lose precision. | **Arrow JSON parser:** `variant_from_number` uses `i64`, then `f64`, rather than decimal parsing. [Arrow source][arrow-json], [parse tests](../tests/test_files/parse_json.slt), [Arrow #11169](https://github.com/apache/arrow-rs/issues/11169). |
| `schema_of_variant(parse_json('1.0'))` reports `DOUBLE` rather than `DECIMAL(1,0)`. The difference propagates through nested schemas and `schema_of_variant_agg`. | **Parsing, not a separate schema-merge rule:** the inference functions see the types produced by the parser. Explicit decimal-to-Variant inputs exercise Spark's decimal inference and merging. [Scalar tests](../tests/test_files/schema_of_variant.slt), [aggregate tests](../tests/test_files/schema_of_variant_agg.slt), [225 merge pairs](../tests/test_files/spark_schema_of_variant_merge.slt). |
| Duplicate JSON object keys are accepted, keeping the last value; Spark rejects them by default. This includes nested objects and keys equal after Unicode decoding. | **Arrow/serde_json:** parsing through `serde_json::Value` collapses duplicate keys before building Variant. [Parser][arrow-json], [pending Spark expectations](../tests/test_files/parse_json.slt). |
| Input `1 2` is rejected here; the recorded Spark behavior consumes the first root. Input `1e309` is rejected here; Spark parses it as Double infinity. | **Parser policy:** serde_json rejects trailing input and numeric overflow. These require compatibility decisions, rather than being Variant encoding bugs. [Recorded comparisons and local error tests](../tests/test_files/parse_json.slt), [Spark parser][spark-builder]. `try_parse_json` returns SQL NULL for the locally rejected input. |
| Spark's 16 MiB value/metadata limit is not enforced here. | **Engine policy:** Spark's builder enforces `SIZE_LIMIT`; this crate delegates to Arrow without adding that limit. [Recorded comparison](../tests/test_files/parse_json.slt), [Spark builder][spark-builder]. |
| Spark's configurable duplicate-key allowance and legacy Unicode-validation behavior are not exposed. | **Crate API:** `parse_json` and `try_parse_json` accept one string argument and use the Arrow parser. [Implementation](../src/parse_json.rs), [scope notes](../tests/test_files/parse_json.slt). |

## Extraction, casts, and SQL syntax

Typed getters, casts of untyped getters, `:` casts, and casts from stored Variant
values share the Arrow extraction/conversion path. A kernel limitation therefore
affects more than just the three-argument `variant_get` call.

| Difference from Spark | Origin and evidence |
| --- | --- |
| Getter paths use `a.b[0]` and `''` for the root, rather than `$.a.b[0]` and `$`. `$` is a literal field name here. A SQL NULL path also selects the root. | **Crate integration with Arrow's path grammar:** paths are passed to `VariantPath`; Spark uses its own JSONPath parser and NULL propagation. [Getter](../src/variant_get.rs), [getter tests](../tests/test_files/variant_get.slt), [Spark implementation][spark-paths]. |
| Backtick-quoted names after `:` can return SQL NULL instead of selecting the field, including ordinary names and names containing spaces. Arithmetic may need parentheses around a right-hand `:` expression. | **DataFusion/sqlparser and the crate's colon planner:** the planner forwards the parsed right operand to the getter. [Planner](../src/expr_planner.rs), [cases 23 and 29 plus additional quoting cases](../tests/test_files/spark_variant_field_extractions.slt). |
| Numeric strings do not convert to integer or floating-point targets, including strings such as `"1"`, `"-0.0"`, and `"inf"`. Strict getters fail and try getters return NULL where Spark succeeds. | **Arrow conversion kernel:** numeric conversion does not parse string variants. String-to-decimal and string-to-Boolean conversions have passing coverage and are separate cases. [Kernel][arrow-conversions], [primitive cast tests](../tests/test_files/spark_variant_get_casts_primitives.slt). |
| String-to-date, numeric-to-timestamp, and Boolean-to-decimal conversions covered by Spark remain unsupported. | **Arrow conversion kernel:** the relevant conversion branches are missing. [Kernel][arrow-conversions], [pending primitive cast cases](../tests/test_files/spark_variant_get_casts_primitives.slt). |
| Casting a numeric Variant to a SQL string fails instead of producing Spark's string representation. | **Arrow string extraction:** `VariantToStringArrowBuilder` uses `as_string`, not general Variant serialization. [Kernel][arrow-to-arrow], [field-extraction case 03](../tests/test_files/spark_variant_field_extractions.slt). Use `to_json` when JSON serialization is intended; it retains quotes around Variant strings. |
| Nested Variant targets such as `ARRAY<VARIANT>` or `STRUCT<a VARIANT>` are rejected. | **Crate guard for Arrow metadata support:** the kernel interprets nested extension fields as ordinary storage structs. [Target validation](../src/variant_type.rs), [SQL type tests](../tests/test_files/variant_get_sql_types.slt). |
| Type-hint spellings and meanings follow DataFusion SQL and session settings. The Spark cast suite uses `BYTEA` for Spark `BINARY` and explicit timestamp precision/time-zone syntax. | **DataFusion type planner:** hints resolve through the same planner as SQL casts. [Resolver](../src/sql_type.rs), [type tests](../tests/test_files/variant_get_sql_types.slt), [Spark cast adaptations](../tests/test_files/spark_variant_get_casts_primitives.slt). |
| `coalesce(variant_value, 'default_value')` does not perform Spark's common-type coercion to Variant. | **DataFusion coercion:** the crate supplies explicit cast rewrites, not general implicit Variant coercion. [Pending field-extraction case 39](../tests/test_files/spark_variant_field_extractions.slt). Explicitly cast the fallback to `VARIANT`. |
| Spark's `typeof(v)` result `variant` is not provided. | **SQL introspection integration:** `arrow_typeof` exposes the physical Arrow type. The crate already adapts `DESCRIBE` and `information_schema.columns`; those are distinct from Spark's `typeof`. [Pending cases 33–36](../tests/test_files/spark_variant_field_extractions.slt), [schema adapter](../src/schema_display_shim.rs). |
| Spark's `SELECT explode(CAST(v:scores AS ARRAY<INT>))` form is not ported. Correlated `variant_explode` supports CROSS/INNER JOIN LATERAL, not arbitrary outer lateral joins. | **SQL planning:** general Spark generator syntax and DataFusion lateral decorrelation differ. `variant_explode` and `variant_explode_outer` themselves are implemented. Use the latter to preserve empty inputs. [Pending case 30](../tests/test_files/spark_variant_field_extractions.slt), [planner](../src/variant_explode.rs), [working table-function cases](../tests/test_files/variant_explode.slt). |

## Conversion to Variant and JSON output

| Difference from Spark | Origin and evidence |
| --- | --- |
| `to_variant_object` omits null-valued object fields instead of retaining Variant null, and keeps the last duplicate key instead of rejecting duplicates. | **Arrow object building/conversion:** `ObjectFieldBuilder::append_null` is a no-op; conversion does not enable duplicate-field validation. [Object builder][arrow-object-builder], [pending conversion tests](../tests/test_files/spark_variant_conversions.slt), [Arrow #11947](https://github.com/apache/arrow-rs/issues/11947), [#11948](https://github.com/apache/arrow-rs/issues/11948). |
| Arrays/objects containing Variant values cannot reliably be converted back to Variant. Preserved nested Variant metadata is explicitly rejected; constructors that lose it can expose physical storage fields instead. | **Arrow plus DataFusion:** Arrow conversion lacks nested extension-field handling, and DataFusion constructors such as `named_struct` can lose that metadata. [Validator](../src/variant_conversion.rs), [pending tests](../tests/test_files/spark_variant_conversions.slt), [Arrow #11942](https://github.com/apache/arrow-rs/issues/11942). |
| `to_json(v, options)` ignores the second argument, including `timeZone`; timestamp JSON formatting and session-time-zone parity are incomplete. | **Crate option handling plus Arrow serialization:** both scalar and batch serializers need time-zone support, then the UDF must pass options/session settings through. [Implementation](../src/to_json.rs), [conversion test notes](../tests/test_files/spark_variant_conversions.slt), [Arrow #11945](https://github.com/apache/arrow-rs/issues/11945). |
| This crate's `to_json` accepts only Variant, whereas Spark also serializes ordinary arrays, maps, and structs. | **Crate function registration:** the Variant-only UDF replaces any existing function with that name. Convert supported containers with `to_variant_object` first, subject to the limitations above. [UDF](../src/to_json.rs), [registration](../src/lib.rs). |

`CAST(struct_or_map AS VARIANT)` requiring `to_variant_object` is shared with
the pinned Spark implementation, so it is not a discrepancy. SQL NULL versus
Variant null is also intentionally preserved for supported unshredded paths;
the exceptions below concern shredded storage and result schemas.

## Shredded values and validation

| Difference or limitation | Origin and evidence |
| --- | --- |
| Extracting Variant from shredded objects/lists can lose the distinction between a missing path and a present Variant null. Converting a shredded Variant null to a struct can produce a valid struct of NULL fields instead of a SQL NULL struct. | **Arrow extraction:** pending parity cases cover both getters and casts. [Shredded tests](../tests/test_files/variant_sql_type.slt), upstream work [#11504](https://github.com/apache/arrow-rs/pull/11504) and [#11052](https://github.com/apache/arrow-rs/pull/11052). |
| Materializing getter results can fail when the actual output contains `typed_value` or uses Binary/LargeBinary storage while the declared return field promises the canonical layout. | **Crate return-field inference:** the declared physical schema does not always match Arrow's output. [Shredded predicate tests](../tests/test_files/is_variant_null.slt), [binary-layout tests](../tests/test_files/variant_sql_type.slt), [getter](../src/variant_get.rs). |
| `is_valid_variant` can reject bytes Spark accepts: metadata, UTF-8, key ordering/uniqueness, offsets, nesting, and date/time ranges are checked more strictly. | **Arrow validation policy:** the UDF uses `Variant::try_new`. This is a validation-contract difference, not a promise to reproduce Spark's permissiveness. [Implementation](../src/is_valid_variant.rs), [raw-byte fixtures](../tests/test_files/is_valid_variant.slt). |
| `variant_delete` and `variant_explode` can reject malformed nested payloads even when Spark's lazy traversal need not decode them, such as a subtree being deleted. | **Crate use of full Arrow validation:** inputs are validated before traversal/copying. [Deletion code](../src/variant_delete.rs), [deletion regression](../tests/test_files/variant_delete.slt), [explode code and rationale](../src/variant_explode.rs). |
| Malformed shredded metadata can still panic during reconstruction; unsupported reconstruction/layouts can raise errors rather than returning a validation Boolean. | **Arrow unshredding:** missing field names in metadata are a known failure path shared by validation, deletion, and explosion. [Validation code](../src/is_valid_variant.rs), [Arrow #11069](https://github.com/apache/arrow-rs/issues/11069), [fallible decoder request #11260](https://github.com/apache/arrow-rs/issues/11260). |
| `variant_delete` uses Spark's default escaped JSONPath grammar, including the leading `$`, but does not expose Spark's legacy literal-backslash setting. | **Crate API choice:** deletion has a separate parser because Arrow's getter grammar differs. Do not reuse rootless getter paths for deletion. [Parser rationale](../src/variant_delete.rs), [Spark-derived tests](../tests/test_files/variant_delete.slt). |

## Additional types and integration scope

- Arrow inputs include unsigned integers, Float16, narrow decimals, and additional
  storage layouts without direct Spark SQL equivalents. Variant `TIME` and
  nanosecond timestamp encodings are also accepted by Arrow validation where the
  Spark revision pinned by the validation tests rejects them.
  [Conversion cases](../tests/test_files/variant_conversion.slt),
  [schema cases](../tests/test_files/schema_of_variant.slt),
  [validation contract](../src/is_valid_variant.rs).
- UUID Variant values can be inspected from fixtures, but SQL UUID construction
  and preserving UUID extension metadata during conversion remain pending. This
  is an integration gap, not a claim that Spark has an equivalent SQL UUID type.
  [Pending conversion test](../tests/test_files/variant_conversion.slt),
  [Arrow #11937](https://github.com/apache/arrow-rs/issues/11937).
- Variant is an Arrow extension over struct storage here. Parquet reads currently
  require `datafusion.execution.parquet.skip_metadata = false` to preserve its
  identifying metadata. See the [README](../README.md#read-variant-from-parquet).
- `variant_get_json`, `variant_object_construct`, `variant_object_insert`,
  `variant_object_keys`, `variant_list_construct`, and `variant_list_insert` are
  additional crate APIs, not a promise of equivalent Spark functions. In
  particular, the object/list constructors currently require scalar arguments.
  [Registration](../src/lib.rs), [object constructor](../src/variant_object_construct.rs),
  [list constructor](../src/variant_list_construct.rs).
- Errors use DataFusion/Arrow messages rather than Spark error classes and exact
  wording. Ported tests check the relevant error category or NULL behavior.

## Skipped expectations needing revalidation

The following historical discrepancies are still marked as skipped in the tests,
but **60.1.0 source contains the corresponding fixes**. They should not be counted
as confirmed current failures. Re-enable and run the cases before declaring
end-to-end parity; this documentation change does not alter the tests.

| Historical discrepancy | Current dependency evidence | Tests to revisit |
| --- | --- | --- |
| A `]` inside a quoted getter key required an extra backslash. | Arrow's `parse_in_bracket` now tracks quotes and only terminates at an unquoted `]`. [Source][arrow-paths]. | [Literal keys](../tests/test_files/variant_get_literal_keys.slt), [additional field-extraction case](../tests/test_files/spark_variant_field_extractions.slt). The getter documentation also retains the old restriction. |
| Float-to-decimal conversion accepted values exceeding the target precision, such as `12345.678` to `DECIMAL(3,2)`. | The Variant conversion calls `float_to_decimal_opt`, whose checked helper now validates the result precision. [Variant conversion][arrow-conversions], [decimal helper][arrow-decimal]. | [Field-extraction case 04](../tests/test_files/spark_variant_field_extractions.slt). |
| Schema inference rejected an empty object key because metadata offsets repeated. | `VariantMetadata::with_full_validation` now allows equal offsets for an empty entry. [Source][arrow-metadata]. | [Final scalar-schema case](../tests/test_files/schema_of_variant.slt). |

## Keeping this inventory current

When fixing a discrepancy, update its linked tests and this document together.
Record the Spark revision, expected result, responsible implementation layer,
and tracking issue where available. Recheck skipped cases after Arrow or
DataFusion upgrades instead of carrying their comments forward as current facts.
The main compatibility tracker is [#71][compatibility-issue]; overall scope is
tracked in the [roadmap][roadmap].

[spark-functions]: https://spark.apache.org/docs/4.2.0/sql-ref-functions-builtin.html#variant-functions
[spark-expressions]: https://github.com/apache/spark/blob/32f7299601108917fb01920a54e084595b7b3bf8/sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/expressions/variant/VariantExpressionSuite.scala
[spark-eval]: https://github.com/apache/spark/blob/32f7299601108917fb01920a54e084595b7b3bf8/sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/expressions/variant/VariantExpressionEvalUtilsSuite.scala
[spark-fields]: https://github.com/apache/spark/blob/5c63836176a2a15a5d43a61f2e18890eb4ac7ece/sql/core/src/test/resources/sql-tests/inputs/variant-field-extractions.sql
[spark-schema]: https://github.com/apache/spark/blob/4ccf07b6d91e2b126b7001741778ee28f22c6609/sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/expressions/variant/VariantExpressionSuite.scala
[spark-paths]: https://github.com/apache/spark/blob/046f574fce3ffd93925f7bcdd459221fa11a730d/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/variant/variantExpressions.scala
[spark-builder]: https://github.com/apache/spark/blob/046f574fce3ffd93925f7bcdd459221fa11a730d/common/variant/src/main/java/org/apache/spark/types/variant/VariantBuilder.java
[arrow-json]: https://github.com/apache/arrow-rs/blob/60.1.0/parquet-variant-json/src/from_json.rs
[arrow-conversions]: https://github.com/apache/arrow-rs/blob/60.1.0/parquet-variant-compute/src/type_conversion.rs
[arrow-to-arrow]: https://github.com/apache/arrow-rs/blob/60.1.0/parquet-variant-compute/src/variant_to_arrow.rs
[arrow-object-builder]: https://github.com/apache/arrow-rs/blob/60.1.0/parquet-variant/src/builder/object.rs
[arrow-paths]: https://github.com/apache/arrow-rs/blob/60.1.0/parquet-variant/src/utils.rs
[arrow-decimal]: https://github.com/apache/arrow-rs/blob/60.1.0/arrow-cast/src/cast/decimal.rs
[arrow-metadata]: https://github.com/apache/arrow-rs/blob/60.1.0/parquet-variant/src/variant/metadata.rs
[compatibility-issue]: https://github.com/datafusion-contrib/datafusion-variant/issues/71
[roadmap]: https://github.com/datafusion-contrib/datafusion-variant/issues/69
