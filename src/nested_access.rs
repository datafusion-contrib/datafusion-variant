// SPDX-License-Identifier: Apache-2.0

//! Tells the Parquet reader which parts of a shredded Variant column
//! `variant_get` and `try_variant_get` read, via DataFusion's
//! [`NestedInputAccess`].
//!
//! [`NestedInputAccess`]: datafusion::logical_expr::NestedInputAccess

use arrow_schema::{DataType, Field, Fields};
use datafusion::common::{ColumnStatistics, Statistics, stats::Precision};
use datafusion::logical_expr::{FieldPath, ReturnFieldArgs};
use parquet_variant::VariantPathElement;

use crate::variant_path::parse_variant_path;
use crate::variant_type::is_variant;

const METADATA: &str = "metadata";
const VALUE: &str = "value";
const TYPED_VALUE: &str = "typed_value";

/// The parts of a Variant column that a getter with a literal path reads.
///
/// # Example
///
/// Given a column `v` shredded as `STRUCT<a BIGINT, b STRUCT<c VARCHAR>>`, its
/// storage is:
///
/// ```text
/// v: Struct<
///   metadata,
///   value,                       <- fields that are not shredded
///   typed_value: Struct<
///     a: Struct<value, typed_value: Int64>,
///     b: Struct<value, typed_value: Struct<
///       c: Struct<value, typed_value: Utf8View>>>>>
/// ```
///
/// Then the leaves read are:
///
/// | path      | leaves (as well as `metadata`)                               |
/// |-----------|--------------------------------------------------------------|
/// | `$.a`     | `typed_value.a.value`, `typed_value.a.typed_value`           |
/// | `$.b.c`   | `typed_value.b.typed_value.c.{value, typed_value}`           |
/// | `$.b`     | `typed_value.b.value` and all of `typed_value.b.typed_value` |
/// | `$.other` | `value`                                                      |
/// | `$.a.x`   | `typed_value.a.value`                                        |
///
/// This follows the path the same way as the `variant_get` kernel does: the
/// kernel steps into a shredded field without looking at the `value` beside it,
/// and reads only `value` from the point where the path is no longer shredded.
#[derive(Debug, PartialEq)]
pub(crate) struct GetterAccess {
    /// Paths of the leaves the getter reads, from the root of the column.
    paths: Vec<FieldPath>,
    /// Set when the path ends at a shredded primitive.
    shredded: Option<ShreddedLeaf>,
}

/// A path that ends at a shredded primitive: `Struct<value, typed_value: T>`.
#[derive(Debug, PartialEq)]
struct ShreddedLeaf {
    /// Index in [`GetterAccess::paths`] of `value`, if the file has it.
    value: Option<usize>,
    /// Index in [`GetterAccess::paths`] of `typed_value`.
    typed_value: usize,
    /// Type of `typed_value`.
    data_type: DataType,
}

impl GetterAccess {
    /// Returns what a getter called with `args` reads from its Variant
    /// argument, or `None` if it must read the whole column.
    pub(crate) fn try_new(args: &ReturnFieldArgs) -> Option<Self> {
        let variant = args.arg_fields.first()?;
        if !is_variant(variant) {
            return None;
        }
        let path = args.scalar_arguments.get(1).copied().flatten()?;
        let path = parse_variant_path(path.try_as_str().flatten()?, true).ok()?;
        // `$` reads everything
        if path.is_empty() {
            return None;
        }
        let DataType::Struct(root) = variant.data_type() else {
            return None;
        };
        let mut group = root;
        child(group, METADATA)?;

        let mut paths = vec![vec![METADATA.to_string()]];
        // Path to `group`, the (value, typed_value) pair reached so far
        let mut prefix = FieldPath::new();
        for element in path.iter() {
            let typed_value = child(group, TYPED_VALUE).map(Field::data_type);
            let next = match (element, typed_value) {
                (VariantPathElement::Field { name }, Some(DataType::Struct(object))) => {
                    match child(object, name) {
                        Some(field) => {
                            // A shredded field must be a (value, typed_value)
                            // pair. Read the column whole and let the kernel
                            // report the error otherwise.
                            let DataType::Struct(next) = field.data_type() else {
                                return None;
                            };
                            Some((name, next))
                        }
                        None => None,
                    }
                }
                // Lists can't be narrowed by struct field paths.
                (
                    VariantPathElement::Index { .. },
                    Some(
                        DataType::List(_)
                        | DataType::LargeList(_)
                        | DataType::ListView(_)
                        | DataType::LargeListView(_)
                        | DataType::FixedSizeList(_, _),
                    ),
                ) => return None,
                (VariantPathElement::ListElement, _) => return None,
                _ => None,
            };
            let Some((name, next)) = next else {
                // Not shredded from here on: the rest is read from `value`.
                child(group, VALUE)?;
                paths.push(join(&prefix, VALUE));
                return Some(Self {
                    paths,
                    shredded: None,
                });
            };
            prefix.push(TYPED_VALUE.to_string());
            prefix.push(name.to_string());
            group = next;
        }

        // The path ends here: read all of `group`.
        let value = child(group, VALUE).map(|_| {
            paths.push(join(&prefix, VALUE));
            paths.len() - 1
        });
        let typed_value = child(group, TYPED_VALUE);
        if value.is_none() && typed_value.is_none() {
            return None;
        }
        let mut shredded = None;
        if let Some(typed_value) = typed_value {
            let typed_value_path = join(&prefix, TYPED_VALUE);
            if !typed_value.data_type().is_nested() {
                shredded = Some(ShreddedLeaf {
                    value,
                    typed_value: paths.len(),
                    data_type: typed_value.data_type().clone(),
                });
            }
            push_leaves(typed_value_path, typed_value.data_type(), &mut paths)?;
        }
        Some(Self { paths, shredded })
    }

    /// The paths of the leaves the getter reads.
    pub(crate) fn into_paths(self) -> Vec<FieldPath> {
        self.paths
    }

    /// Returns the statistics of a getter that returns `return_type`, given
    /// the statistics of [`Self::into_paths`] within some container of rows (a
    /// row group or a page).
    ///
    /// Statistics are only known when the output is exactly the `typed_value`
    /// column: its type is the requested one, and no row of the container
    /// stores its value in `value` instead (for example a string in a field
    /// shredded as `BIGINT`).
    pub(crate) fn output_statistics(
        &self,
        return_type: &DataType,
        inputs: &Statistics,
    ) -> Option<ColumnStatistics> {
        let shredded = self.shredded.as_ref()?;
        if shredded.data_type != *return_type {
            return None;
        }
        if let Some(value) = shredded.value {
            let Precision::Exact(num_rows) = inputs.num_rows else {
                return None;
            };
            let value = inputs.column_statistics.get(value)?;
            if value.null_count != Precision::Exact(num_rows) {
                return None;
            }
        }
        // Parquet counts a leaf as null where any of its ancestors is null,
        // which is where the getter returns NULL.
        let typed_value = inputs.column_statistics.get(shredded.typed_value)?;
        Some(ColumnStatistics {
            null_count: typed_value.null_count,
            min_value: typed_value.min_value.clone(),
            max_value: typed_value.max_value.clone(),
            ..ColumnStatistics::new_unknown()
        })
    }
}

fn child<'a>(fields: &'a Fields, name: &str) -> Option<&'a Field> {
    fields
        .iter()
        .find(|field| field.name() == name)
        .map(|field| field.as_ref())
}

fn join(prefix: &FieldPath, name: &str) -> FieldPath {
    let mut path = prefix.clone();
    path.push(name.to_string());
    path
}

/// Adds the path of every leaf below `path`, whose type is `data_type`.
///
/// Paths only traverse structs, so a path that reaches a list ends there.
/// Returns `None` if there is a struct without fields.
fn push_leaves(path: FieldPath, data_type: &DataType, paths: &mut Vec<FieldPath>) -> Option<()> {
    let DataType::Struct(fields) = data_type else {
        paths.push(path);
        return Some(());
    };
    if fields.is_empty() {
        return None;
    }
    for field in fields {
        push_leaves(join(&path, field.name()), field.data_type(), paths)?;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow_schema::FieldRef;
    use datafusion::scalar::ScalarValue;
    use parquet_variant_compute::VariantType;

    /// A (value, typed_value) pair
    fn pair(typed_value: DataType) -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(VALUE, DataType::BinaryView, true),
            Field::new(TYPED_VALUE, typed_value, true),
        ]))
    }

    fn object(fields: Vec<(&str, DataType)>) -> DataType {
        DataType::Struct(
            fields
                .into_iter()
                .map(|(name, data_type)| Field::new(name, data_type, false))
                .collect(),
        )
    }

    /// `v` shredded as `STRUCT<a BIGINT, b STRUCT<c VARCHAR>, l ARRAY<BIGINT>>`
    fn shredded() -> FieldRef {
        let typed_value = object(vec![
            ("a", pair(DataType::Int64)),
            ("b", pair(object(vec![("c", pair(DataType::Utf8View))]))),
            (
                "l",
                pair(DataType::List(Arc::new(Field::new_list_field(
                    pair(DataType::Int64),
                    false,
                )))),
            ),
        ]);
        variant(vec![
            Field::new(METADATA, DataType::BinaryView, false),
            Field::new(VALUE, DataType::BinaryView, true),
            Field::new(TYPED_VALUE, typed_value, true),
        ])
    }

    fn variant(fields: Vec<Field>) -> FieldRef {
        Arc::new(
            Field::new("v", DataType::Struct(fields.into()), true).with_extension_type(VariantType),
        )
    }

    fn access(variant: &FieldRef, path: &str) -> Option<GetterAccess> {
        let path = ScalarValue::from(path);
        GetterAccess::try_new(&ReturnFieldArgs {
            arg_fields: &[
                Arc::clone(variant),
                Arc::new(Field::new("path", DataType::Utf8, false)),
            ],
            scalar_arguments: &[None, Some(&path)],
        })
    }

    fn paths(variant: &FieldRef, path: &str) -> Option<Vec<String>> {
        let paths = access(variant, path)?.into_paths();
        Some(paths.iter().map(|path| path.join(".")).collect())
    }

    #[test]
    fn required_paths() {
        let v = shredded();
        #[rustfmt::skip]
        let cases: Vec<(&str, Option<Vec<&str>>)> = vec![
            ("$.a", Some(vec![
                "metadata",
                "typed_value.a.value",
                "typed_value.a.typed_value",
            ])),
            ("$.b.c", Some(vec![
                "metadata",
                "typed_value.b.typed_value.c.value",
                "typed_value.b.typed_value.c.typed_value",
            ])),
            // A whole shredded object
            ("$.b", Some(vec![
                "metadata",
                "typed_value.b.value",
                "typed_value.b.typed_value.c.value",
                "typed_value.b.typed_value.c.typed_value",
            ])),
            // A whole shredded list
            ("$.l", Some(vec![
                "metadata",
                "typed_value.l.value",
                "typed_value.l.typed_value",
            ])),
            // Not shredded: read from the residual value
            ("$.other", Some(vec!["metadata", "value"])),
            ("$.other.x", Some(vec!["metadata", "value"])),
            ("$.a.x", Some(vec!["metadata", "typed_value.a.value"])),
            ("$.b.other", Some(vec!["metadata", "typed_value.b.value"])),
            ("$[0]", Some(vec!["metadata", "value"])),
            // Reads the whole column
            ("$", None),
            ("$.l[0]", None),
            ("not a path", None),
        ];
        for (path, expected) in cases {
            let expected =
                expected.map(|paths| paths.into_iter().map(String::from).collect::<Vec<_>>());
            assert_eq!(paths(&v, path), expected, "{path}");
        }
    }

    #[test]
    fn required_paths_unshredded() {
        let v = variant(vec![
            Field::new(METADATA, DataType::BinaryView, false),
            Field::new(VALUE, DataType::BinaryView, true),
        ]);
        assert_eq!(
            paths(&v, "$.a"),
            Some(vec!["metadata".into(), "value".into()])
        );
        // Not a Variant
        let v = Arc::new(Field::new("v", v.data_type().clone(), true));
        assert_eq!(paths(&v, "$.a"), None);
    }

    /// Statistics of `metadata`, `typed_value.a.value` and
    /// `typed_value.a.typed_value` for a container of 10 rows
    fn statistics(value_nulls: usize) -> Statistics {
        let column = |nulls: usize| ColumnStatistics {
            null_count: Precision::Exact(nulls),
            ..ColumnStatistics::new_unknown()
        };
        Statistics {
            num_rows: Precision::Exact(10),
            total_byte_size: Precision::Absent,
            column_statistics: vec![
                column(0),
                column(value_nulls),
                ColumnStatistics {
                    min_value: Precision::Exact(ScalarValue::Int64(Some(1))),
                    max_value: Precision::Exact(ScalarValue::Int64(Some(7))),
                    ..column(2)
                },
            ],
        }
    }

    #[test]
    fn output_statistics() {
        let v = shredded();
        let a = access(&v, "$.a").unwrap();

        // Every value is in `typed_value`
        let stats = a
            .output_statistics(&DataType::Int64, &statistics(10))
            .unwrap();
        assert_eq!(stats.null_count, Precision::Exact(2));
        assert_eq!(
            stats.min_value,
            Precision::Exact(ScalarValue::Int64(Some(1)))
        );
        assert_eq!(
            stats.max_value,
            Precision::Exact(ScalarValue::Int64(Some(7)))
        );

        // One row stores its value in `value`
        assert!(
            a.output_statistics(&DataType::Int64, &statistics(9))
                .is_none()
        );
        // A different type than the shredded one needs a conversion
        assert!(
            a.output_statistics(&DataType::Int32, &statistics(10))
                .is_none()
        );
        // Not a shredded primitive
        for path in ["$.b", "$.other", "$.a.x"] {
            let access = access(&v, path).unwrap();
            assert!(
                access
                    .output_statistics(&DataType::Int64, &statistics(10))
                    .is_none(),
                "{path}"
            );
        }
    }
}
