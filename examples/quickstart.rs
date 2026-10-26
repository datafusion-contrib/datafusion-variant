//! Run with `cargo run --example quickstart`. No external dataset is required.
use datafusion::prelude::*;
use datafusion_variant::register_all;

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

    for sql in [
        "SELECT variant_get(event, 'service', 'VARCHAR') AS service,
                event:tags[0]::VARCHAR AS first_tag
         FROM saved_events ORDER BY event:status::BIGINT",
        "SELECT to_json(event) AS json_data FROM saved_events
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
