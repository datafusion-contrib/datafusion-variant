use datafusion::assert_batches_eq;
use datafusion::error::Result;
use datafusion::prelude::*;
use datafusion_variant::register_all;

#[tokio::test]
async fn register_all_preserves_session_and_enables_variant_types() -> Result<()> {
    let mut ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(2));
    ctx.sql("CREATE TABLE existing AS SELECT 7 AS n")
        .await?
        .collect()
        .await?;
    let shared_ctx = ctx.clone();
    let session_id = ctx.session_id();

    register_all(&mut ctx)?;

    assert_eq!(shared_ctx.state().session_id(), session_id);
    assert_eq!(shared_ctx.copied_config().target_partitions(), 2);
    let batches = shared_ctx
        .sql("SELECT CAST(CAST(n AS VARIANT) AS INT) AS n FROM existing")
        .await?
        .collect()
        .await?;
    assert_batches_eq!(["+---+", "| n |", "+---+", "| 7 |", "+---+"], &batches);
    Ok(())
}
