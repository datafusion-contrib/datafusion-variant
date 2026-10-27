//! Run with `cargo run --example quickstart`. No external dataset is required.
use std::fs::File;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion_variant::register_all;
use parquet::arrow::ArrowWriter;
use parquet_variant_compute::{VariantArray, shred_variant};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut ctx = SessionContext::new();
    register_all(&mut ctx)?;

    ctx.sql(
        r#"
        CREATE TABLE events AS
        SELECT parse_json(json_data) AS event
        FROM (VALUES
            ('{"service":"api","status":200,"latency_ms":12,"tags":["cached"]}'),
            ('{"service":"api","status":500,"latency_ms":85,"error":null}')
        ) AS input(json_data)
        "#,
    )
    .await?
    .collect()
    .await?;

    // Equivalent field extraction through the UDF and SQL field-access syntax.
    for sql in [
        "SELECT variant_get(event, 'service', 'VARCHAR') AS service,
                variant_get(event, 'latency_ms', 'BIGINT') AS latency_ms
         FROM events
         WHERE variant_get(event, 'status', 'BIGINT') >= 400",
        "SELECT event:service::VARCHAR AS service,
                event:latency_ms::BIGINT AS latency_ms
         FROM events
         WHERE event:status::BIGINT >= 400",
    ] {
        ctx.sql(sql).await?.show().await?;
    }

    // Keep the example repeatable and remove its Parquet output on exit.
    let output = tempfile::tempdir()?;
    let path = output.path().join("events.parquet");
    ctx.sql(&format!(
        "COPY events TO '{}' STORED AS PARQUET",
        path.display()
    ))
    .await?
    .show()
    .await?;

    // Preserve the Variant extension annotation when inferring Parquet schemas.
    ctx.sql("SET datafusion.execution.parquet.skip_metadata = false")
        .await?
        .collect()
        .await?;
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE saved_events STORED AS PARQUET LOCATION '{}'",
        path.display()
    ))
    .await?
    .collect()
    .await?;

    // Shred the same events with Arrow until SQL COPY supports a shredding option.
    let events = ctx.table("events").await?;
    let schema = Arc::clone(events.schema().inner());
    let batch = concat_batches(&schema, &events.collect().await?)?;
    let shredded = shred_variant(
        &VariantArray::try_new(batch.column(0).as_ref())?,
        &DataType::Struct(
            vec![
                Field::new("service", DataType::Utf8, true),
                Field::new("status", DataType::Int64, true),
            ]
            .into(),
        ),
    )?;
    let schema = Arc::new(Schema::new(vec![shredded.field("event")]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(shredded.into_inner())])?;
    let shredded_path = output.path().join("shredded_events.parquet");
    let mut writer = ArrowWriter::try_new(File::create(&shredded_path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE shredded_events STORED AS PARQUET LOCATION '{}'",
        shredded_path.display()
    ))
    .await?
    .collect()
    .await?;

    for sql in [
        "SELECT variant_get(event, 'service', 'VARCHAR') AS service,
                event:tags[0]::VARCHAR AS first_tag
         FROM saved_events ORDER BY event:status::BIGINT",
        "SELECT event:service::VARCHAR AS service,
                event:status::BIGINT AS status
         FROM shredded_events
         WHERE event:status::BIGINT >= 400
         ORDER BY service, status",
        "SELECT to_json(event) AS json_data FROM events
         ORDER BY event:status::BIGINT",
        "SELECT schema_of_variant_agg(event) AS schema FROM saved_events",
        "SELECT event:status::BIGINT AS status,
                variant_contains(event, 'error') AS has_error,
                is_variant_null(event:error) AS error_is_json_null,
                (event:error) IS NULL AS error_is_sql_null
         FROM saved_events ORDER BY status",
    ] {
        ctx.sql(sql).await?.show().await?;
    }
    Ok(())
}
