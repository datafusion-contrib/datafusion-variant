// SPDX-License-Identifier: Apache-2.0

//! Adapters to read Parquet files with different Variant shredding layouts.
//!
//! * [`VariantListingTableFactory`]: wraps an existing table factory to attach
//!   [`VariantExprAdapterFactory`] to tables
//! * [`VariantExprAdapterFactory`]: creates a `VariantExprAdapter` for each file
//! * `VariantExprAdapter`: per-file adapter
//! * `VariantReshredExpr`: expression that converts a Variant column from one
//!   shredding layout to another

use std::fmt::{self, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider, TableProviderFactory};
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion};
use datafusion::common::{Result, exec_err, internal_err, plan_err};
use datafusion::datasource::listing::{ListingTable, ListingTableConfig};
use datafusion::execution::session_state::SessionState;
use datafusion::logical_expr::CreateExternalTable;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr_adapter::{
    DefaultPhysicalExprAdapterFactory, PhysicalExprAdapter, PhysicalExprAdapterFactory,
};
use datafusion::physical_plan::ColumnarValue;
use parquet_variant_compute::{VariantArray, shred_variant, unshred_variant};

use crate::variant_type::is_variant;

/// Adapts the output of Parquet files whose Variant shredding layout differs
/// from the table's.
///
/// Implemented as a [`PhysicalExprAdapterFactory`] that wraps another factory
/// and, when layouts differ, reconstructs Variant values and shreds them into
/// the table's layout. Matching layouts use the wrapped adapter unchanged.
///
/// # Example
///
/// Consider a directory `events/` with two files that have a Variant column
/// `event`. One file stores the column unshredded, and a newer one stores it
/// with `service` and `status` shredded into typed columns:
///
/// ```text
/// events/unshredded.parquet
///   Schema:
///     event: Struct<metadata, value>
///
///   value (variant bytes)
///   ----------------------------------------------
///   {"service":"api","status":200,"latency_ms":12}
///   {"service":"api","status":500,"latency_ms":85}
///   {"service":"db","status":504}
///
/// events/shredded.parquet
///   Schema:
///     event: Struct<metadata, value, typed_value: Struct<service, status>>
///
///   value (variant bytes)  typed_value.service  typed_value.status
///   ---------------------  -------------------  ------------------
///   {"latency_ms":12}      api                  200
///   {"latency_ms":85}      api                  500
///   NULL                   db                   504
/// ```
///
/// The table's schema is the result of merging the two file schemas. In this
/// case that is:
///
/// ```text
/// Unified Schema:
///   event: Struct<metadata, value, typed_value: Struct<service, status>>
/// ```
///
/// This adapter factory adds a per-file expression to convert the
/// unshredded file's column to the table's shredded layout.
///
/// With this adapter factory on the table, both files return every field. For
/// example, this query:
///
/// ```sql
/// CREATE EXTERNAL TABLE events STORED AS PARQUET LOCATION 'events/';
///
/// SELECT event:service::VARCHAR AS service,
///        event:status::BIGINT AS status,
///        event:latency_ms::BIGINT AS latency_ms
/// FROM events;
/// ```
///
/// returns:
///
/// ```text
/// +---------+--------+------------+
/// | service | status | latency_ms |
/// +---------+--------+------------+
/// | api     | 200    | 12         |  <- shredded.parquet
/// | api     | 500    | 85         |
/// | db      | 504    | NULL       |
/// | api     | 200    | 12         |  <- unshredded.parquet
/// | api     | 500    | 85         |
/// | db      | 504    | NULL       |
/// +---------+--------+------------+
/// ```
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
    /// Adapt Variant layouts and delegate other schema rewriting to `inner`.
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

/// A [`PhysicalExprAdapter`] that adapts expressions written against the table
/// schema to one Parquet file's schema.
///
/// These are created by [`VariantExprAdapterFactory`].
///
/// It wraps each reference to a Variant column in a `VariantReshredExpr` when
/// the file's shredding layout differs from the table's. Everything else is
/// rewritten by the wrapped adapter.
#[derive(Debug)]
struct VariantExprAdapter {
    inner: Arc<dyn PhysicalExprAdapter>,
    logical_file_schema: SchemaRef,
    physical_file_schema: SchemaRef,
}

impl PhysicalExprAdapter for VariantExprAdapter {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
        // 1. Leave a placeholder for every Variant column reference whose file
        // layout differs from the table layout so the inner adapter doesn't
        // modify it.
        let mut placeholders = Vec::new();
        let expr = expr
            .transform_down(|expr| {
                let Some(column) = expr.downcast_ref::<Column>() else {
                    return Ok(Transformed::no(expr));
                };
                let Some(reshred) = self.reshred_expr(column) else {
                    return Ok(Transformed::no(expr));
                };
                let placeholder: Arc<dyn PhysicalExpr> = Arc::new(Placeholder {
                    id: placeholders.len(),
                    data_type: reshred.data_type(&self.physical_file_schema)?,
                });
                placeholders.push(reshred);
                Ok(Transformed::new(placeholder, true, TreeNodeRecursion::Jump))
            })
            .data()?;

        // 2. Call the wrapped adapter to do schema rewriting as normal. It
        // ignores the placeholders.
        let expr = self.inner.rewrite(expr)?;

        // 3. Put the reshredding expressions back in place of the placeholders.
        expr.transform_down(|expr| match expr.downcast_ref::<Placeholder>() {
            Some(placeholder) => Ok(Transformed::new(
                Arc::clone(&placeholders[placeholder.id]),
                true,
                TreeNodeRecursion::Jump,
            )),
            None => Ok(Transformed::no(expr)),
        })
        .data()
    }
}

impl VariantExprAdapter {
    /// Returns a `VariantReshredExpr` that reshreds a Variant column whose
    /// file layout differs from the table layout.
    ///
    /// Returns `None` if the column doesn't need reshredding.
    ///
    /// `column` refers to the table (logical) schema, so `column.index()` is
    /// its position in the table, not in the file.
    fn reshred_expr(&self, column: &Column) -> Option<Arc<dyn PhysicalExpr>> {
        // Look the column up by name in both schemas, as DataFusion's default
        // adapter does. `column.index()` is only valid for the table schema.
        let logical = self
            .logical_file_schema
            .field_with_name(column.name())
            .ok()?;
        if !is_variant(logical) {
            return None;
        }
        // The file may store its columns in a different order than the table,
        // or lack some of them, so the column's index in the file can differ
        // from `column.index()`. Thus, look up by name;
        let index = self.physical_file_schema.index_of(column.name()).ok()?;
        let physical = self.physical_file_schema.field(index);
        // Does it need reshredding?
        if !is_variant(physical) || physical.data_type() == logical.data_type() {
            return None;
        }
        Some(Arc::new(VariantReshredExpr::new(
            Arc::new(Column::new(column.name(), index)),
            Arc::new(logical.clone()),
        )))
    }
}

/// Stands in for a reshredded Variant column while the wrapped adapter rewrites
/// the rest of the expression tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Placeholder {
    id: usize,
    data_type: DataType,
}

impl Display for Placeholder {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "variant_placeholder({})", self.id)
    }
}

impl PhysicalExpr for Placeholder {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.data_type.clone())
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, _batch: &RecordBatch) -> Result<ColumnarValue> {
        exec_err!(
            "variant_placeholder({}) was not replaced after the schema rewrite",
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

/// Returns the `typed_value` child of a Variant storage type, if it has one.
fn typed_value_type(data_type: &DataType) -> Option<&DataType> {
    match data_type {
        DataType::Struct(fields) => fields
            .iter()
            .find(|f| f.name() == "typed_value")
            .map(|f| f.data_type()),
        _ => None,
    }
}

/// Returns the shredding schema that [`shred_variant`] needs to produce the
/// given `typed_value` storage type.
///
/// The storage type wraps every object field and list element in
/// `Struct<value, typed_value: T>`; the shredding schema has just `T`. For
/// example:
/// * Input: `Struct<service: Struct<value, typed_value: Utf8>>`
/// * Output: `Struct<service: Utf8>`
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
        DataType::ListView(child) => DataType::ListView(Arc::new(Field::new_list_field(
            shredded_field_type(child)?,
            true,
        ))),
        DataType::LargeListView(child) => DataType::LargeListView(Arc::new(Field::new_list_field(
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

/// [`PhysicalExpr`] that reads a Variant column stored in one shredding layout
/// and returns it in another.
///
/// Each batch is unshredded and shredded again to the target `typed_value`
/// schema, so every row is valid for the target layout.
#[derive(Debug, Clone, Eq)]
struct VariantReshredExpr {
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
    /// Create a new reshredding expression.
    ///
    /// * `input`: evaluates to a Variant column.
    /// * `target`: the output field: the Variant storage type to produce.
    fn new(input: Arc<dyn PhysicalExpr>, target: FieldRef) -> Self {
        Self { input, target }
    }

    /// Perform the actual reshredding.
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
            Err(e) => internal_err!(
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

    fn return_field(&self, _input_schema: &Schema) -> Result<FieldRef> {
        Ok(Arc::clone(&self.target))
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

/// A [`TableProviderFactory`] that wraps another factory and attaches
/// [`VariantExprAdapterFactory`] to the [`ListingTable`]s it creates.
///
/// Tables of any other kind are returned unchanged.
#[derive(Debug)]
pub struct VariantListingTableFactory {
    inner: Arc<dyn TableProviderFactory>,
}

impl VariantListingTableFactory {
    /// Wrap `inner`, attaching the Variant adapter to the [`ListingTable`]s it creates.
    pub fn new(inner: Arc<dyn TableProviderFactory>) -> Self {
        Self { inner }
    }

    /// Replace `state`'s `PARQUET` table factory with one that wraps it and
    /// knows how to adapt Variant shredding layouts.
    ///
    /// Does nothing if `state` has no `PARQUET` table factory.
    pub fn register(state: &mut SessionState) {
        let factories = state.table_factories_mut();
        let Some(existing_factory) = factories.remove(PARQUET) else {
            return;
        };
        let wrapped_factory = Self::new(existing_factory);
        factories.insert(PARQUET.into(), Arc::new(wrapped_factory));
    }
}

/// Key of the Parquet factory in `SessionState::table_factories`.
const PARQUET: &str = "PARQUET";

#[async_trait]
impl TableProviderFactory for VariantListingTableFactory {
    /// Create a new table provider for `cmd`.
    ///
    /// If the provider returned by `self.inner` is a [`ListingTable`], rebuilds
    /// it with [`VariantExprAdapterFactory`] attached.
    async fn create(
        &self,
        state: &dyn Session,
        cmd: &CreateExternalTable,
    ) -> Result<Arc<dyn TableProvider>> {
        let provider = self.inner.create(state, cmd).await?;
        let Some(table) = provider.downcast_ref::<ListingTable>() else {
            return Ok(provider);
        };
        if table.table_paths().is_empty() {
            return Ok(provider);
        }

        // TODO: avoid recreating the ListingTable when we have an API
        // to get a ListingTableConfig from the listing table
        // <https://github.com/apache/datafusion/issues/27314>

        // `ListingTable::try_new` removes partition columns from this schema
        // before appending their definitions from the listing options.
        let config = ListingTableConfig::new_with_multi_paths(table.table_paths().clone())
            .with_listing_options(table.options().clone())
            .with_schema(table.schema())
            .with_expr_adapter_factory(Arc::new(VariantExprAdapterFactory::default()));
        let table = ListingTable::try_new(config)?
            .with_cache(state.runtime_env().cache_manager.get_file_statistic_cache())
            .with_definition(cmd.definition.clone())
            .with_constraints(cmd.constraints.clone())
            .with_column_defaults(cmd.column_defaults.clone());
        Ok(Arc::new(table))
    }
}
