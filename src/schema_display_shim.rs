// SPDX-License-Identifier: Apache-2.0

//! Temporary Variant type names for schema introspection.

use std::{
    collections::{HashMap, hash_map::Entry},
    sync::Arc,
};

use arrow::{
    array::{AsArray, StringBuilder},
    datatypes::{SchemaRef, UInt64Type},
    record_batch::RecordBatch,
};
use async_trait::async_trait;
use futures::{TryStreamExt, stream};

use datafusion::{
    catalog::{
        CatalogProviderList, Session, TableProvider, information_schema::INFORMATION_SCHEMA,
    },
    common::{
        Result,
        config::ConfigOptions,
        tree_node::{Transformed, TransformedResult},
    },
    datasource::{provider_as_source, source_as_provider},
    execution::TaskContext,
    logical_expr::{Expr, LogicalPlan, TableType, Values, lit},
    optimizer::AnalyzerRule,
    physical_plan::{
        ExecutionPlan, SendableRecordBatchStream, execute_stream,
        stream::RecordBatchStreamAdapter,
        streaming::{PartitionStream, StreamingTableExec},
    },
};

use crate::variant_type::is_variant;

// TODO: remove when DataFusion uses extension type names in schema introspection:
// https://github.com/apache/datafusion/issues/26866
#[derive(Debug)]
pub(crate) struct VariantSchemaDisplay;

impl AnalyzerRule for VariantSchemaDisplay {
    fn analyze(&self, plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_up_with_subqueries(|mut plan| {
            if let LogicalPlan::TableScan(scan) = &mut plan {
                let name = scan.table_name.clone().resolve(
                    &config.catalog.default_catalog,
                    &config.catalog.default_schema,
                );
                if config.catalog.information_schema
                    && name.schema.as_ref() == INFORMATION_SCHEMA
                    && name.table.eq_ignore_ascii_case("columns")
                {
                    let inner = source_as_provider(&scan.source)?;
                    // Plans can be analyzed repeatedly, including through views.
                    if !inner.is::<VariantColumns>() {
                        scan.source = provider_as_source(Arc::new(VariantColumns { inner }));
                        return Ok(Transformed::yes(plan));
                    }
                }
                return Ok(Transformed::no(plan));
            }
            let LogicalPlan::DescribeTable(describe) = &plan else {
                return Ok(Transformed::no(plan));
            };
            if !describe
                .schema
                .fields()
                .iter()
                .any(|field| is_variant(field))
            {
                return Ok(Transformed::no(plan));
            }

            let values = describe
                .schema
                .fields()
                .iter()
                .map(|field| {
                    let data_type = if is_variant(field) {
                        "Variant".to_owned()
                    } else {
                        field.data_type().to_string()
                    };
                    vec![
                        lit(field.name().clone()),
                        lit(data_type),
                        lit(if field.is_nullable() { "YES" } else { "NO" }),
                    ]
                })
                .collect();
            Ok(Transformed::yes(LogicalPlan::Values(Values {
                schema: describe.output_schema.clone(),
                values,
            })))
        })
        .data()
    }

    fn name(&self) -> &str {
        "variant_schema_display"
    }
}

#[derive(Debug)]
struct VariantColumns {
    inner: Arc<dyn TableProvider>,
}

#[async_trait]
impl TableProvider for VariantColumns {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }

    // Keep filters above the adapter so they see logical type names. Fetch all
    // source columns: projected queries may omit the table/column identity needed
    // to recover field metadata. StreamingTableExec applies projection and limit.
    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&[usize]>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let type_index = self.schema().index_of("data_type")?;
        if projection.is_some_and(|indices| !indices.contains(&type_index)) {
            // Type predicates remain above this scan, so their input column is
            // included in the projection. Other queries need no metadata lookup.
            return self.inner.scan(state, projection, &[], limit).await;
        }
        let input = self.inner.scan(state, None, &[], None).await?;
        let schema = input.schema();
        let partition = Arc::new(VariantColumnsStream {
            schema: schema.clone(),
            input,
            catalogs: state.catalog_list(),
        });
        Ok(Arc::new(StreamingTableExec::try_new(
            schema,
            vec![partition],
            projection,
            [],
            false,
            limit,
        )?))
    }
}

#[derive(Debug)]
struct VariantColumnsStream {
    schema: SchemaRef,
    input: Arc<dyn ExecutionPlan>,
    catalogs: Arc<dyn CatalogProviderList>,
}

impl PartitionStream for VariantColumnsStream {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let input = self.input.clone();
        let catalogs = self.catalogs.clone();
        let stream = stream::once(async move { execute_stream(input, ctx) })
            .try_flatten()
            .and_then(move |batch| format_column_types(batch, catalogs.clone()));
        Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), stream))
    }
}

async fn format_column_types(
    batch: RecordBatch,
    catalogs: Arc<dyn CatalogProviderList>,
) -> Result<RecordBatch> {
    let schema = batch.schema_ref();
    let catalog_names = batch
        .column(schema.index_of("table_catalog")?)
        .as_string::<i32>();
    let schema_names = batch
        .column(schema.index_of("table_schema")?)
        .as_string::<i32>();
    let table_names = batch
        .column(schema.index_of("table_name")?)
        .as_string::<i32>();
    let column_names = batch
        .column(schema.index_of("column_name")?)
        .as_string::<i32>();
    let positions = batch
        .column(schema.index_of("ordinal_position")?)
        .as_primitive::<UInt64Type>();
    let type_index = schema.index_of("data_type")?;
    let types = batch.column(type_index).as_string::<i32>();
    let mut formatted = StringBuilder::new();
    let mut schemas = HashMap::new();

    for row in 0..batch.num_rows() {
        let key = (
            catalog_names.value(row),
            schema_names.value(row),
            table_names.value(row),
        );
        let table_schema = match schemas.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let provider = catalogs
                    .catalog(key.0)
                    .and_then(|catalog| catalog.schema(key.1));
                let table_schema = if let Some(provider) = provider {
                    provider.table(key.2).await?.map(|table| table.schema())
                } else {
                    None
                };
                entry.insert(table_schema)
            }
        };
        // DataFusion's information schema uses zero-based ordinal positions.
        // A table may have disappeared since the source batch was produced.
        let field = table_schema
            .as_ref()
            .and_then(|schema| schema.fields().get(positions.value(row) as usize));
        let is_variant =
            field.is_some_and(|field| field.name() == column_names.value(row) && is_variant(field));
        formatted.append_value(if is_variant {
            "Variant"
        } else {
            types.value(row)
        });
    }

    let (schema, mut columns, _) = batch.into_parts();
    columns[type_index] = Arc::new(formatted.finish());
    Ok(RecordBatch::try_new(schema, columns)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::{
        physical_plan::{collect, execution_plan::reset_plan_states},
        prelude::{SessionConfig, SessionContext},
    };

    #[tokio::test]
    async fn information_schema_resolves_types_at_execution() -> Result<()> {
        let mut ctx =
            SessionContext::new_with_config(SessionConfig::new().with_information_schema(true));
        crate::register_all(&mut ctx)?;
        // Plan before the table exists, then execute the same physical plan
        // after replacing it. Neither the source rows nor type names may freeze.
        let plan = ctx
            .sql("SELECT data_type FROM information_schema.columns WHERE table_name = 'late'")
            .await?
            .create_physical_plan()
            .await?;
        for (sql_type, expected) in [("VARIANT", "Variant"), ("BIGINT", "Int64")] {
            ctx.sql(&format!("CREATE TABLE late (v {sql_type})"))
                .await?
                .collect()
                .await?;
            let batches = collect(reset_plan_states(plan.clone())?, ctx.task_ctx()).await?;
            assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
            let batch = batches.iter().find(|batch| batch.num_rows() != 0).unwrap();
            assert_eq!(batch.column(0).as_string::<i32>().value(0), expected);
            ctx.sql("DROP TABLE late").await?.collect().await?;
        }
        Ok(())
    }
}
