// SPDX-License-Identifier: Apache-2.0

//! Read Parquet files whose Variant shredding layout differs from the table's.
//!
//! A `ListingTable` merges the schemas of its files. When one file stores a
//! Variant column as `Struct<metadata, value>` and another as
//! `Struct<metadata, value, typed_value: Struct<...>>`, the merged table type
//! has `typed_value`, and DataFusion's default [`PhysicalExprAdapter`] casts the
//! first file's column to it, filling `typed_value` with nulls. For an ordinary
//! struct that is right. For Variant it is not: the shredding specification reads
//! a null `typed_value` next to a non-null `value` as "this value is not an
//! object", so every shredded field of every row from that file comes back NULL.
//!
//! [`VariantExprAdapterFactory`] wraps another factory (the default by default)
//! and replaces that cast, for Variant columns only, with a per-batch rewrite:
//! the file's column is unshredded and then shredded again to the table's
//! `typed_value` schema, so each row is valid for the table's layout. Files whose
//! layout already matches the table are not touched.
//!
//! Install it per table:
//!
//! ```ignore
//! use datafusion::datasource::listing::{
//!     ListingTable, ListingTableConfig, ListingTableConfigExt, ListingTableUrl,
//! };
//! use datafusion_variant::VariantExprAdapterFactory;
//!
//! let config = ListingTableConfig::new(ListingTableUrl::parse("data/")?)
//!     .infer_options(&ctx.state())
//!     .await?
//!     .infer_schema(&ctx.state())
//!     .await?
//!     .with_expr_adapter_factory(Arc::new(VariantExprAdapterFactory::default()));
//! ctx.register_table("events", Arc::new(ListingTable::try_new(config)?))?;
//! ```
//!
//! Wiring this in automatically for every table with a Variant column needs a
//! DataFusion hook; see <https://github.com/apache/datafusion/issues/27089>.

use std::fmt::{self, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion};
use datafusion::common::{Result, exec_err, plan_err};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr_adapter::{
    DefaultPhysicalExprAdapterFactory, PhysicalExprAdapter, PhysicalExprAdapterFactory,
};
use datafusion::physical_plan::ColumnarValue;
use parquet_variant_compute::{VariantArray, shred_variant, unshred_variant};

use crate::variant_type::is_variant;

/// Creates [`VariantExprAdapter`]s. Wraps another factory and post-processes its
/// output; see the [module documentation](self).
#[derive(Debug, Clone)]
pub struct VariantExprAdapterFactory {
    inner: Arc<dyn PhysicalExprAdapterFactory>,
}

impl Default for VariantExprAdapterFactory {
    fn default() -> Self {
        Self::new(Arc::new(DefaultPhysicalExprAdapterFactory))
    }
}

impl VariantExprAdapterFactory {
    /// Wrap `inner`. Its rewrite runs first; Variant struct casts in the result
    /// are then replaced.
    pub fn new(inner: Arc<dyn PhysicalExprAdapterFactory>) -> Self {
        Self { inner }
    }
}

impl PhysicalExprAdapterFactory for VariantExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Result<Arc<dyn PhysicalExprAdapter>> {
        let inner = self.inner.create(
            Arc::clone(&logical_file_schema),
            Arc::clone(&physical_file_schema),
        )?;
        Ok(Arc::new(VariantExprAdapter {
            inner,
            logical_file_schema,
            physical_file_schema,
        }))
    }

    fn is_equivalent_to_default(&self) -> bool {
        false
    }
}

/// Replaces the struct cast the wrapped adapter inserts on a Variant column with
/// [`VariantReshredExpr`].
#[derive(Debug)]
pub struct VariantExprAdapter {
    inner: Arc<dyn PhysicalExprAdapter>,
    logical_file_schema: SchemaRef,
    physical_file_schema: SchemaRef,
}

impl PhysicalExprAdapter for VariantExprAdapter {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
        // 1. Cut out every Variant column whose file layout differs from the
        //    table layout, before the wrapped adapter sees it: the default
        //    adapter either rejects the struct cast at planning time or
        //    null-fills `typed_value`. Each one is replaced by a childless
        //    placeholder the wrapped adapter leaves alone.
        let mut cut: Vec<Arc<dyn PhysicalExpr>> = Vec::new();
        let expr = expr
            .transform_down(|expr| {
                let Some(column) = expr.downcast_ref::<Column>() else {
                    return Ok(Transformed::no(expr));
                };
                let Some(reshred) = self.reshred_expr(column) else {
                    return Ok(Transformed::no(expr));
                };
                let hole: Arc<dyn PhysicalExpr> = Arc::new(Hole {
                    id: cut.len(),
                    data_type: reshred.data_type(&self.physical_file_schema)?,
                });
                cut.push(reshred);
                Ok(Transformed::new(hole, true, TreeNodeRecursion::Jump))
            })
            .data()?;

        // 2. Everything else: column indices, missing columns, partition values.
        let expr = self.inner.rewrite(expr)?;

        // 3. Put the reshred expressions back. Their column child already refers
        //    to the file schema, so the scan projects it like any other column.
        expr.transform_down(|expr| match expr.downcast_ref::<Hole>() {
            Some(hole) => Ok(Transformed::new(
                Arc::clone(&cut[hole.id]),
                true,
                TreeNodeRecursion::Jump,
            )),
            None => Ok(Transformed::no(expr)),
        })
        .data()
    }
}

impl VariantExprAdapter {
    /// A [`VariantReshredExpr`] over the file column for `column`, when it is a
    /// Variant column whose file layout differs from the table layout; `None`
    /// otherwise, including when the file lacks the column entirely (the
    /// wrapped adapter null-fills it).
    fn reshred_expr(&self, column: &Column) -> Option<Arc<dyn PhysicalExpr>> {
        let logical = self
            .logical_file_schema
            .field_with_name(column.name())
            .ok()?;
        if !is_variant(logical) {
            return None;
        }
        let index = self.physical_file_schema.index_of(column.name()).ok()?;
        let physical = self.physical_file_schema.field(index);
        if physical.data_type() == logical.data_type()
            || !is_variant_storage(physical.data_type())
            || !is_variant_storage(logical.data_type())
        {
            return None;
        }
        Some(Arc::new(VariantReshredExpr::new(
            Arc::new(Column::new(column.name(), index)),
            Arc::new(logical.clone()),
        )))
    }
}

/// Stand-in for a cut-out subtree while the wrapped adapter runs. Never
/// evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Hole {
    id: usize,
    data_type: DataType,
}

impl Display for Hole {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "variant_hole({})", self.id)
    }
}

impl PhysicalExpr for Hole {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.data_type.clone())
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, _batch: &RecordBatch) -> Result<ColumnarValue> {
        exec_err!(
            "variant_hole({}) was not replaced after the schema rewrite",
            self.id
        )
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(self)
    }

    fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

/// A struct with `metadata` and `value` children: the storage shape of a
/// Variant, shredded or not.
fn is_variant_storage(data_type: &DataType) -> bool {
    match data_type {
        DataType::Struct(fields) => {
            fields.iter().any(|f| f.name() == "metadata")
                && fields.iter().any(|f| f.name() == "value")
        }
        _ => false,
    }
}

/// The `typed_value` child of a Variant storage type, if it has one.
fn typed_value_type(data_type: &DataType) -> Option<&DataType> {
    match data_type {
        DataType::Struct(fields) => fields
            .iter()
            .find(|f| f.name() == "typed_value")
            .map(|f| f.data_type()),
        _ => None,
    }
}

/// Turn the storage type of a `typed_value` column back into the shredding
/// schema [`shred_variant`] takes: each shredded field `Struct<value,
/// typed_value: T>` becomes `T`, recursively through objects and lists.
fn shredding_schema(storage: &DataType) -> Result<DataType> {
    Ok(match storage {
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|f| Ok(Field::new(f.name(), shredded_field_type(f)?, true)))
                .collect::<Result<Vec<_>>>()?
                .into(),
        ),
        DataType::List(child) => DataType::List(Arc::new(Field::new_list_field(
            shredded_field_type(child)?,
            true,
        ))),
        DataType::LargeList(child) => DataType::LargeList(Arc::new(Field::new_list_field(
            shredded_field_type(child)?,
            true,
        ))),
        primitive => primitive.clone(),
    })
}

/// `field` is a shredded object field or list element: `Struct<value,
/// typed_value: T>`. Returns the shredding schema for `T`.
fn shredded_field_type(field: &Field) -> Result<DataType> {
    match typed_value_type(field.data_type()) {
        Some(typed_value) => shredding_schema(typed_value),
        None => plan_err!(
            "Variant shredded field '{}' has no typed_value column; cannot derive a shredding schema from {}",
            field.name(),
            field.data_type()
        ),
    }
}

/// Reads a Variant column stored in one shredding layout and returns it in
/// another. Each batch is unshredded and shredded again to the target
/// `typed_value` schema, so every row is valid for the target layout.
#[derive(Debug, Clone, Eq)]
pub struct VariantReshredExpr {
    input: Arc<dyn PhysicalExpr>,
    target: FieldRef,
}

// Manual impls, as DataFusion does for its own expressions, to work around
// https://github.com/rust-lang/rust/issues/78808 with `Arc<dyn PhysicalExpr>`.
impl PartialEq for VariantReshredExpr {
    fn eq(&self, other: &Self) -> bool {
        self.input.eq(&other.input) && self.target == other.target
    }
}

impl Hash for VariantReshredExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.target.hash(state);
    }
}

impl VariantReshredExpr {
    /// `input` evaluates to the file's Variant column; `target` is the table's
    /// field for it, including the Variant extension metadata.
    pub fn new(input: Arc<dyn PhysicalExpr>, target: FieldRef) -> Self {
        Self { input, target }
    }

    /// The expression evaluated against the file.
    pub fn input(&self) -> &Arc<dyn PhysicalExpr> {
        &self.input
    }

    /// The table field the result matches.
    pub fn target(&self) -> &FieldRef {
        &self.target
    }

    fn reshred(&self, array: ArrayRef) -> Result<ArrayRef> {
        let variant = VariantArray::try_new(array.as_ref())?;
        let unshredded = unshred_variant(&variant)?;
        let reshredded = match typed_value_type(self.target.data_type()) {
            Some(typed_value) => shred_variant(&unshredded, &shredding_schema(typed_value)?)?,
            None => unshredded,
        };
        let out: ArrayRef = Arc::new(reshredded.into_inner());
        if out.data_type() == self.target.data_type() {
            return Ok(out);
        }
        // Same field set and order by construction; align nullability and
        // binary encodings with the table type.
        match cast(&out, self.target.data_type()) {
            Ok(aligned) => Ok(aligned),
            Err(e) => exec_err!(
                "Variant column '{}' reshredded to {} but the table expects {}: {e}",
                self.target.name(),
                out.data_type(),
                self.target.data_type()
            ),
        }
    }
}

impl Display for VariantReshredExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "variant_reshred({})", self.input)
    }
}

impl PhysicalExpr for VariantReshredExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.target.data_type().clone())
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.input.nullable(input_schema)
    }

    fn return_field(&self, input_schema: &Schema) -> Result<FieldRef> {
        let nullable = self.nullable(input_schema)?;
        Ok(Arc::new(
            Field::new(
                self.target.name(),
                self.target.data_type().clone(),
                nullable,
            )
            .with_metadata(self.target.metadata().clone()),
        ))
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let array = self.input.evaluate(batch)?.into_array(batch.num_rows())?;
        Ok(ColumnarValue::Array(self.reshred(array)?))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            Arc::clone(&self.target),
        )))
    }

    fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "variant_reshred(")?;
        self.input.fmt_sql(f)?;
        write!(f, ")")
    }
}
