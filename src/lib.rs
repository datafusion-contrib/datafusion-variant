// SPDX-License-Identifier: Apache-2.0

#![warn(clippy::all)]

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod shared;
mod shredding;
mod sql_type;

mod placeholders;
mod expr_planner;
mod impl_variant_get;
mod is_valid_variant;
mod is_variant_null;
mod parse_json;
mod schema_of_variant;
mod to_json;
mod variant_conversion;
mod variant_delete;
mod variant_display;
mod variant_explode;
mod variant_get;
mod variant_get_rewrite;
mod variant_list_construct;
mod variant_list_insert;
mod variant_object_construct;
mod variant_object_insert;
mod variant_object_keys;
mod variant_path;
mod variant_type;

pub use placeholders::*;
pub use expr_planner::*;
pub use is_valid_variant::*;
pub use is_variant_null::*;
pub use parse_json::*;
pub use schema_of_variant::*;
pub use to_json::*;
pub use variant_conversion::*;
pub use variant_delete::*;
pub use variant_explode::*;
pub use variant_get::*;
pub use variant_get_rewrite::*;
pub use variant_list_construct::*;
pub use variant_list_insert::*;
pub use variant_object_construct::*;
pub use variant_object_insert::*;
pub use variant_object_keys::*;
pub use variant_type::*;

use datafusion::error::Result;
use datafusion::execution::{FunctionRegistry, SessionStateBuilder};
use datafusion::logical_expr::{AggregateUDF, ScalarUDF};
use datafusion::prelude::SessionContext;
use std::sync::Arc;

/// Register all Variant functionality with a DataFusion [`SessionContext`]
///
/// JSON display applies to DataFusion's extension-aware formatters, including
/// `DataFrame::show` and `DataFrame::to_string`.
/// `DESCRIBE`, `SHOW COLUMNS`, and `information_schema.columns` report top-level
/// Variant fields as `Variant`.
/// Parquet `COPY` supports `variant.shredding`, mapping column names to SQL types.
/// Tables made with `CREATE EXTERNAL TABLE ... STORED AS PARQUET` read files
/// whose Variant shredding layouts differ; see [`VariantListingTableFactory`].
///
/// Existing functions with the same names, the Variant extension registration,
/// and the session's type planner are replaced. The session's `PARQUET` table
/// factory is wrapped, not replaced. Call during session setup, before
/// preparing statements: installing the type planner rebuilds the session state
/// and clears prepared statements.
pub fn register_all(ctx: &mut SessionContext) -> Result<()> {
    {
        let state = ctx.state_ref();
        let mut state = state.write();
        *state = SessionStateBuilder::new_from_existing(state.clone())
            .with_session_id(state.session_id().to_owned())
            .with_type_planner(Arc::new(VariantTypePlanner))
            .build();
        VariantListingTableFactory::register(&mut state);
    }

    variant_display::register(ctx)?;
    ctx.add_analyzer_rule(Arc::new(shredding::VariantShredding));
    ctx.register_table_options_extension(shredding::VariantWriteOptions::default());

    for udf in [
        ScalarUDF::new_from_impl(ParseJsonUdf::default()),
        ScalarUDF::new_from_impl(TryParseJsonUdf::default()),
        ScalarUDF::new_from_impl(VariantToJsonUdf::default()),
        ScalarUDF::new_from_impl(IsVariantNullUdf::default()),
        ScalarUDF::new_from_impl(IsValidVariantUdf::default()),
        ScalarUDF::new_from_impl(SchemaOfVariantUdf::default()),
        ScalarUDF::new_from_impl(VariantGetJsonUdf::default()),
        ScalarUDF::new_from_impl(VariantObjectConstruct::default()),
        ScalarUDF::new_from_impl(VariantObjectKeys::default()),
        ScalarUDF::new_from_impl(VariantObjectInsert::default()),
        ScalarUDF::new_from_impl(VariantDeleteUdf::default()),
        ScalarUDF::new_from_impl(VariantListConstruct::default()),
        ScalarUDF::new_from_impl(VariantListInsert::default()),
    ] {
        ctx.register_udf(Arc::new(udf))?;
    }
    ctx.register_udaf(Arc::new(AggregateUDF::new_from_impl(
        SchemaOfVariantAggUdf::default(),
    )))?;
    register_variant_get_functions(ctx)?;
    register_variant_conversion_functions(ctx)?;
    register_variant_explode_functions(ctx)?;
    ctx.register_expr_planner(Arc::new(VariantExprPlanner))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::AsArray;
    use arrow::datatypes::Int64Type;
    use datafusion::catalog::{Session, TableProvider, TableProviderFactory};
    use datafusion::datasource::provider::DefaultTableFactory;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::logical_expr::CreateExternalTable;

    /// Counts the tables it is asked to create, then defers to the stock factory.
    #[derive(Debug, Default)]
    struct CountingFactory {
        calls: AtomicUsize,
        inner: DefaultTableFactory,
    }

    #[async_trait]
    impl TableProviderFactory for CountingFactory {
        async fn create(
            &self,
            state: &dyn Session,
            cmd: &CreateExternalTable,
        ) -> Result<Arc<dyn TableProvider>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.create(state, cmd).await
        }
    }

    /// A `PARQUET` table factory the session already has is wrapped, not replaced.
    #[tokio::test]
    async fn register_all_wraps_existing_parquet_factory() -> Result<()> {
        let counting = Arc::new(CountingFactory::default());
        let state = SessionStateBuilder::new()
            .with_default_features()
            .with_table_factory(String::from("PARQUET"), counting.clone())
            .build();
        let mut ctx = SessionContext::new_with_state(state);
        crate::register_all(&mut ctx)?;

        // One unshredded and one shredded file with the same three rows.
        let dir = tempfile::tempdir()?;
        let dir = dir.path().display();
        for (file, options) in [
            ("unshredded", ""),
            (
                "shredded",
                r#"OPTIONS ('variant.shredding' '{"event":"STRUCT<service VARCHAR>"}')"#,
            ),
        ] {
            ctx.sql(&format!(
                "COPY (SELECT parse_json(column1) AS event FROM VALUES \
                 ('{{\"service\":\"api\"}}'), ('{{\"service\":\"api\"}}'), ('{{\"service\":\"db\"}}')) \
                 TO '{dir}/{file}.parquet' STORED AS PARQUET {options}"
            ))
                .await?
                .collect()
                .await?;
        }

        ctx.sql(&format!(
            "CREATE EXTERNAL TABLE t STORED AS PARQUET LOCATION '{dir}/'"
        ))
        .await?;
        assert_eq!(counting.calls.load(Ordering::SeqCst), 1);

        // The Variant adapter is attached: no row of the unshredded file is lost.
        let batches = ctx
            .sql("SELECT count(variant_get(event, 'service', 'VARCHAR')) FROM t")
            .await?
            .collect()
            .await?;
        assert_eq!(batches[0].column(0).as_primitive::<Int64Type>().value(0), 6);
        Ok(())
    }
}
