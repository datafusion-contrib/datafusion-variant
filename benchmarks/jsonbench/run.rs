//! Adapted from DataFusion's ClickBench runner and BenchmarkRun JSON format:
//! https://github.com/apache/datafusion/blob/main/benchmarks/src/clickbench.rs
//! Copyright The Apache Software Foundation, licensed under Apache-2.0.
use std::fs::{self, File};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use arrow::array::RecordBatch;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_variant::VariantGetUdf;
use serde::Serialize;

use super::prepare::has_rows;
use super::{Layout, Run};

pub(super) const QUERIES: [&str; 5] = [
    include_str!("queries/q1.sql"),
    include_str!("queries/q2.sql"),
    include_str!("queries/q3.sql"),
    include_str!("queries/q4.sql"),
    include_str!("queries/q5.sql"),
];

#[derive(Serialize)]
struct Report {
    context: serde_json::Value,
    queries: Vec<Case>,
}

#[derive(Serialize)]
struct Case {
    query: String,
    start_time: u64,
    success: bool,
    iterations: Vec<Iteration>,
}

#[derive(Serialize)]
struct Iteration {
    elapsed: f64,
    row_count: usize,
}

fn now() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

pub(super) async fn context(options: &Run, paths: &[PathBuf]) -> Result<SessionContext> {
    let memory_bytes = options
        .memory_limit_mib
        .checked_mul(1024 * 1024)
        .context("memory limit overflow")?;
    let config = SessionConfig::new()
        .with_target_partitions(options.partitions)
        .with_batch_size(options.batch_size)
        .set_str("datafusion.execution.time_zone", "UTC")
        // DataFusion normally discards Parquet field metadata during schema
        // inference. Variant UDFs require the canonical extension annotation.
        .set_bool("datafusion.execution.parquet.skip_metadata", false);
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_limit(memory_bytes, 1.0)
        .build_arc()?;
    let ctx = SessionContext::new_with_config_rt(config, runtime);
    ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::default()));
    let paths: Vec<String> = paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let table = ctx.read_parquet(paths, Default::default()).await?;
    ctx.register_table("bluesky", table.into_view())?;
    Ok(ctx)
}

pub(super) async fn run(options: &Run) -> Result<()> {
    let layouts = options
        .layout
        .map(|l| vec![l])
        .unwrap_or_else(|| Layout::ALL.to_vec());
    let query_ids: Vec<u8> = options
        .query
        .map(|q| vec![q])
        .unwrap_or_else(|| (1..=5).collect());
    let parts = options.dataset.parts()?;
    let mut fixtures = Vec::new();
    // Check all selected layouts before running; never silently omit a case.
    for layout in layouts {
        let mut paths = Vec::new();
        for &(number, rows) in &parts {
            let path = options.dataset.path(layout, number, rows);
            ensure!(
                has_rows(&path, rows),
                "missing or incomplete {}; run prepare",
                path.display()
            );
            paths.push(path);
        }
        fixtures.push((layout, paths));
    }
    let mut report = Report {
        context: serde_json::json!({
            "benchmark_version": env!("CARGO_PKG_VERSION"),
            "datafusion_version": datafusion::DATAFUSION_VERSION,
            "num_cpus": std::thread::available_parallelism()?.get(),
            "start_time": now()?,
            "arguments": std::env::args().skip(1).collect::<Vec<_>>(),
            "full_suite": options.layout.is_none() && options.query.is_none(),
        }),
        queries: vec![],
    };
    for (layout, paths) in fixtures {
        let ctx = context(options, &paths).await?;
        for &query_id in &query_ids {
            let sql = QUERIES[usize::from(query_id - 1)];
            let mut case = Case {
                query: format!("{}/Q{query_id}", layout.name()),
                start_time: now()?,
                success: true,
                iterations: vec![],
            };
            if options.debug {
                ctx.sql(sql).await?.explain(false, false)?.show().await?;
            }
            for iteration in 0..options.iterations {
                // Same timing boundary as DataFusion's ClickBench runner.
                let start = Instant::now();
                let batches = ctx.sql(sql).await?.collect().await?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                let row_count = batches.iter().map(RecordBatch::num_rows).sum();
                println!(
                    "{} iteration {iteration}: {elapsed:.3} ms ({row_count} rows)",
                    case.query
                );
                case.iterations.push(Iteration { elapsed, row_count });
            }
            report.queries.push(case);
        }
    }
    let output = options.output_path();
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    serde_json::to_writer_pretty(File::create(&output)?, &report)?;
    println!("Results: {}", output.display());
    Ok(())
}
