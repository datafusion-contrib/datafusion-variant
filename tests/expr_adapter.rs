// SPDX-License-Identifier: Apache-2.0

//! A `ListingTable` over Parquet files with different Variant shredding layouts
//! must return the same rows as reading each file alone.

use std::fs::File;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Schema};
use arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::listing::{
    ListingTable, ListingTableConfig, ListingTableConfigExt, ListingTableUrl,
};
use datafusion::physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion::prelude::*;
use datafusion_variant::{VariantExprAdapterFactory, register_all};
use parquet::arrow::ArrowWriter;
use parquet_variant_compute::{
    ShreddedSchemaBuilder, VariantArray, json_to_variant, shred_variant,
};

const EVENTS: [&str; 3] = [
    r#"{"service":"api","status":200,"latency_ms":12}"#,
    r#"{"service":"api","status":500,"latency_ms":85}"#,
    r#"{"service":"db","status":504}"#,
];

fn variant() -> VariantArray {
    let json: ArrayRef = Arc::new(StringArray::from(EVENTS.to_vec()));
    json_to_variant(&json).expect("json_to_variant")
}

fn shredded(paths: &[(&str, DataType)]) -> VariantArray {
    let mut b = ShreddedSchemaBuilder::new();
    for (p, t) in paths {
        b = b.with_path(*p, t).expect("path");
    }
    shred_variant(&variant(), &b.build()).expect("shred_variant")
}

fn write(path: &std::path::Path, v: VariantArray) {
    let schema = Arc::new(Schema::new(vec![v.field("event")]));
    let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema.clone(), None).unwrap();
    w.write(&RecordBatch::try_new(schema, vec![Arc::new(v.into_inner())]).unwrap())
        .unwrap();
    w.close().unwrap();
}

async fn ctx() -> SessionContext {
    let config = SessionConfig::new()
        .set_bool("datafusion.execution.parquet.skip_metadata", false)
        .with_target_partitions(1);
    let mut ctx = SessionContext::new_with_config(config);
    register_all(&mut ctx).unwrap();
    ctx
}

async fn register_dir(
    ctx: &SessionContext,
    name: &str,
    dir: &std::path::Path,
    factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
) {
    let url = ListingTableUrl::parse(format!("{}/", dir.display())).unwrap();
    let state = ctx.state();
    let mut config = ListingTableConfig::new(url)
        .infer_options(&state)
        .await
        .unwrap()
        .infer_schema(&state)
        .await
        .unwrap();
    if let Some(factory) = factory {
        config = config.with_expr_adapter_factory(factory);
    }
    ctx.register_table(name, Arc::new(ListingTable::try_new(config).unwrap()))
        .unwrap();
}

async fn run(ctx: &SessionContext, sql: &str) -> String {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

const FIELDS: &str = "SELECT variant_get(event, 'service', 'VARCHAR') AS service, \
                             event:status::BIGINT AS status, \
                             event:latency_ms::BIGINT AS latency_ms \
                      FROM t ORDER BY status, latency_ms NULLS FIRST";

const EXPECTED_TWICE: &str = "\
+---------+--------+------------+
| service | status | latency_ms |
+---------+--------+------------+
| api     | 200    | 12         |
| api     | 200    | 12         |
| api     | 500    | 85         |
| api     | 500    | 85         |
| db      | 504    |            |
| db      | 504    |            |
+---------+--------+------------+";

fn factory() -> Arc<dyn PhysicalExprAdapterFactory> {
    Arc::new(VariantExprAdapterFactory::default())
}

#[tokio::test]
async fn shredded_and_unshredded_files() {
    let dir = tempfile::tempdir().unwrap();
    write(
        &dir.path().join("a_shredded.parquet"),
        shredded(&[("service", DataType::Utf8), ("status", DataType::Int64)]),
    );
    write(&dir.path().join("b_unshredded.parquet"), variant());

    // Without the adapter the unshredded file's shredded fields are NULL.
    let ctx_default = ctx().await;
    register_dir(&ctx_default, "t", dir.path(), None).await;
    let before = run(
        &ctx_default,
        "SELECT count(variant_get(event, 'service', 'VARCHAR')) FROM t",
    )
    .await;
    assert!(
        before.contains("| 3 "),
        "default adapter should lose 3 rows, got:\n{before}"
    );

    let ctx = ctx().await;
    register_dir(&ctx, "t", dir.path(), Some(factory())).await;
    assert_eq!(run(&ctx, FIELDS).await, EXPECTED_TWICE);
    assert_eq!(
        run(
            &ctx,
            "SELECT count(variant_get(event, 'service', 'VARCHAR')) FROM t"
        )
        .await,
        "+-------------------------------------------------------------+\n\
         | count(variant_get(t.event,Utf8(\"service\"),Utf8(\"VARCHAR\"))) |\n\
         +-------------------------------------------------------------+\n\
         | 6                                                           |\n\
         +-------------------------------------------------------------+"
    );
}

#[tokio::test]
async fn files_shredded_on_different_fields() {
    let dir = tempfile::tempdir().unwrap();
    write(
        &dir.path().join("a.parquet"),
        shredded(&[("service", DataType::Utf8), ("status", DataType::Int64)]),
    );
    write(
        &dir.path().join("b.parquet"),
        shredded(&[("latency_ms", DataType::Int64)]),
    );

    let ctx = ctx().await;
    register_dir(&ctx, "t", dir.path(), Some(factory())).await;
    assert_eq!(run(&ctx, FIELDS).await, EXPECTED_TWICE);
}

#[tokio::test]
async fn same_layout_is_not_rewritten() {
    let dir = tempfile::tempdir().unwrap();
    let shred = &[("service", DataType::Utf8), ("status", DataType::Int64)];
    write(&dir.path().join("a.parquet"), shredded(shred));
    write(&dir.path().join("b.parquet"), shredded(shred));

    let ctx = ctx().await;
    register_dir(&ctx, "t", dir.path(), Some(factory())).await;
    assert_eq!(run(&ctx, FIELDS).await, EXPECTED_TWICE);
    let plan = run(&ctx, "EXPLAIN SELECT event:service::VARCHAR FROM t").await;
    assert!(
        !plan.contains("variant_reshred"),
        "same-layout files must not be rewritten:\n{plan}"
    );
}

#[tokio::test]
async fn whole_column_read_and_json() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir.path().join("a_unshredded.parquet"), variant());
    write(
        &dir.path().join("b_shredded.parquet"),
        shredded(&[("status", DataType::Int64)]),
    );

    let ctx = ctx().await;
    register_dir(&ctx, "t", dir.path(), Some(factory())).await;
    // Every row is a valid Variant for the table layout: status comes back for all six.
    let out = run(&ctx, "SELECT count(event:status::BIGINT) FROM t").await;
    assert!(
        out.contains("| 6 "),
        "expected 6 non-null status values, got:\n{out}"
    );
}
