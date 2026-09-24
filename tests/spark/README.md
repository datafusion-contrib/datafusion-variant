These extraction tests and golden results come from Apache Spark commit
`5c63836176a2a15a5d43a61f2e18890eb4ac7ece`:

- https://github.com/apache/spark/blob/5c63836176a2a15a5d43a61f2e18890eb4ac7ece/sql/core/src/test/resources/sql-tests/inputs/variant-field-extractions.sql
- https://github.com/apache/spark/blob/5c63836176a2a15a5d43a61f2e18890eb4ac7ece/sql/core/src/test/resources/sql-tests/results/variant-field-extractions.sql.out

The only change to these reference files is `parse_json` → `json_to_variant`.
The executable adaptation is `../test_files/spark_variant_field_extractions.slt`.
Spark's `isnull(expr)` and `isnotnull(expr)` are expressed as `expr IS NULL` and `expr IS NOT NULL`.

Its case numbers follow the 39 SELECT statements in the input. Variant outputs
are serialized with `variant_to_json`. Cases 23–28 test special-key extraction
without the original `::string`, so cast support cannot mask path-handling bugs.
Case 03 separately covers conversion to SQL string. The complete original
queries remain in the reference files; extraction-only string expectations
include JSON quotes. Other SQL and Spark's expected values are preserved.
Unsupported cases use `skipif DataFusion`
with a TODO explaining the missing capability; they are parity targets, not
assertions that the current incompatible behavior is correct. Spark error
classes are recorded in comments; parser rejection uses DataFusion's diagnostic.

The four `typeof` cases remain skipped: `arrow_typeof` reports the Arrow storage
type rather than Spark's logical `variant` type, so it is not an equivalent
substitution.

Currently, 27 cases execute and 12 are skipped.
Case 23 is skipped for the backtick field-name bug, independently of cast support.

The 39 original queries and the six extraction-only adaptations were also
validated against Apache Spark 4.2.0 in local mode. The source pin above records
where the fixture was copied from, independently of that runtime validation.
