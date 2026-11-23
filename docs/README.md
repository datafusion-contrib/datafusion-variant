## What is Variant?

Variant is a type for storing semi-structured data, similar to JSON. Each value
can hold objects, arrays, and primitive values, with different fields and types
in each row.

For example, a single Variant column can contain all of these values, shown as JSON:

| Value                            | Type    |
|----------------------------------|---------|
| `{"service":"api","status":200}` | Object  |
| `["cached","retry"]`             | Array   |
| `42`                             | Integer |
| `"api"`                          | String  |

Variants are stored using an efficient binary encoding, similar to BSON, with
optimizations for repeated field names. For more background, see [Variant Type
in Apache Parquet for Semi-Structured Data][parquet-variant-blog].

This crate also supports the Variant "shredding" optimization for when many rows
share a common structure. You specify some subset of the fields to store as
typed Parquet columns, and the remaining fields are stored in the standard
Variant binary representation. Queries on the common fields use the typed
columns for better performance, while preserving all other fields in Variant
values. See [Write shredded Parquet columns](cookbook/store-and-read-parquet.md#write-shredded-parquet-columns)
and the [shredding guide](shredding.md) for more details.

## References

- [Apache Spark 4.2.0 documentation][spark-docs]
- [Parquet Variant encoding][parquet-variant-encoding]
- [Arrow canonical extension types][arrow-canonical-extensions]
- [Custom types using metadata in DataFusion][datafusion-custom-types-blog]
- [Iceberg Variant proposal (design background)][iceberg-variant-proposal]

[parquet-variant-blog]: https://parquet.apache.org/blog/2026/02/27/variant-type-in-apache-parquet-for-semi-structured-data/
[spark-docs]: https://spark.apache.org/docs/4.2.0/
[parquet-variant-encoding]: https://parquet.apache.org/docs/file-format/types/variantencoding/
[arrow-canonical-extensions]: https://arrow.apache.org/docs/format/CanonicalExtensions.html
[datafusion-custom-types-blog]: https://datafusion.apache.org/blog/2025/09/21/custom-types-using-metadata
[iceberg-variant-proposal]: https://docs.google.com/document/d/1sq70XDiWJ2DemWyA5dVB80gKzwi0CWoM0LOWM7VJVd8/edit
