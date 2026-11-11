// SPDX-License-Identifier: Apache-2.0

//! `dfv bench`: runs DataFusion SQL benchmark (`.benchmark`) files in a
//! session with the Variant functions installed.
//!
//! DataFusion's `benchmark_runner` builds its own `SessionContext`, so it
//! cannot run suites that call Variant functions. This mirrors its non
//! Criterion mode on top of the `datafusion-benchmarks` library, installing
//! [`crate::session_context`] on each case's fresh `make_ctx` session.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::ValueEnum;
use datafusion::error::{DataFusionError, Result};
use datafusion::prelude::SessionContext;
use datafusion_benchmarks::sql_benchmark::SqlBenchmark;
use datafusion_benchmarks::sql_benchmark_runner::{
    BenchmarkFilter, SqlRunConfig, ensure_selection, filter_benchmarks, finish_benchmark,
    format_benchmark_list, load_benchmark_definitions, make_ctx, prepare_benchmark,
};
use datafusion_benchmarks::util::{BenchmarkRun, CommonOpt, print_memory_stats};

#[derive(Debug, clap::Args)]
pub struct BenchArgs {
    /// Suite to run, a directory under --benchmark-dir such as `jsonbench`. Omit to run every suite
    suite: Option<String>,

    /// Run only the matching query, such as `1` or `Q01`
    #[arg(short, long)]
    query: Option<String>,

    /// Run only this subgroup, such as `shredded`
    #[arg(long)]
    subgroup: Option<String>,

    /// Directory holding the suites
    #[arg(long, default_value = "benchmarks")]
    benchmark_dir: PathBuf,

    /// Write the timings as JSON, for DataFusion's benchmarks/compare.py
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Save each query's results to its `result` file, or check the results against it
    #[arg(long, value_enum, default_value_t = ResultMode::None)]
    result_mode: ResultMode,

    /// List the selected benchmarks and exit
    #[arg(long)]
    list: bool,

    #[command(flatten)]
    common: CommonOpt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ResultMode {
    None,
    Persist,
    Validate,
}

pub async fn run(args: BenchArgs) -> Result<()> {
    if args.common.iterations == 0 {
        return Err(DataFusionError::Configuration(
            "--iterations must be greater than zero".to_string(),
        ));
    }
    let benchmark_dir = args.benchmark_dir.canonicalize()?;
    let config = SqlRunConfig {
        common: args.common,
        filter: BenchmarkFilter {
            name: args.suite,
            subgroup: args.subgroup,
            query: args.query,
        },
        replacements: HashMap::new(),
        query_filename: None,
        persist_results: args.result_mode == ResultMode::Persist,
        validate_results: args.result_mode == ResultMode::Validate,
        output: args.output,
    };

    let listing_ctx = context(&config.common).await?;
    let all_benchmarks = load_benchmark_definitions(
        &config.filter,
        &listing_ctx,
        &benchmark_dir,
        &config.replacements,
    )
    .await?;
    let selected = filter_benchmarks(&config.filter, all_benchmarks.clone());
    ensure_selection(&config.filter, &all_benchmarks, &selected)?;
    if args.list {
        println!("{}", format_benchmark_list(&selected));
        return Ok(());
    }

    let mut run = BenchmarkRun::new();
    for benchmarks in selected.into_values() {
        for mut benchmark in benchmarks {
            let ctx = context(&config.common).await?;
            let result = run_case(&ctx, &mut benchmark, &config, &mut run).await;
            let cleanup_result = benchmark.cleanup(&ctx).await;
            finish_benchmark(result, cleanup_result)?;
        }
    }
    run.maybe_write_json(config.output.as_ref())
}

async fn context(common: &CommonOpt) -> Result<SessionContext> {
    crate::session_context(make_ctx(common)?).await
}

async fn run_case(
    ctx: &SessionContext,
    benchmark: &mut SqlBenchmark,
    config: &SqlRunConfig,
    run: &mut BenchmarkRun,
) -> Result<()> {
    prepare_benchmark(ctx, benchmark, config).await?;

    let mut case_name = format!("{}/{}", benchmark.group(), benchmark.name());
    if !benchmark.subgroup().is_empty() {
        case_name = format!("{case_name}/{}", benchmark.subgroup());
    }
    run.set_memory_pool(&ctx.runtime_env().memory_pool);
    run.start_new_case(&case_name);

    for iteration in 0..config.common.iterations {
        let start = Instant::now();
        let row_count = benchmark.run(ctx, false).await?;
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        println!("{case_name} iteration {iteration}: {ms:.1} ms, {row_count} rows");
        run.write_iter(elapsed, row_count);
    }
    print_memory_stats(&*ctx.runtime_env().memory_pool);
    Ok(())
}
