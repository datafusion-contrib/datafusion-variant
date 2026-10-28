//! `dfv`: DataFusion CLI with the Variant functions preinstalled.
mod cli_context;
#[path = "../../src/display_shim.rs"]
mod display;

use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use datafusion::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionConfig;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::ExplainFormat;
use datafusion::prelude::SessionContext;
use datafusion_cli::catalog::DynamicObjectStoreCatalog;
use datafusion_cli::functions::{
    ListFilesCacheFunc, MetadataCacheFunc, ParquetFileMetadataFunc, ParquetMetadataFunc,
    ParquetPageIndexFunc, StatisticsCacheFunc,
};
use datafusion_cli::object_storage::instrumented::InstrumentedObjectStoreRegistry;
use datafusion_cli::object_storage::{StdinCarriesCommands, is_stdin_location};
use datafusion_cli::print_format::PrintFormat;
use datafusion_cli::print_options::{MaxRows, PrintOptions};
use datafusion_cli::{DATAFUSION_CLI_VERSION, exec};
use datafusion_variant::register_all;
use mimalloc::MiMalloc;

mod load_bsky;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Parser, PartialEq)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(
        short = 'c',
        long,
        num_args = 0..,
        help = "Execute the given command string(s), then exit. Commands are expected to be non empty.",
        value_parser(parse_command)
    )]
    command: Vec<String>,

    #[clap(
        short,
        long,
        num_args = 0..,
        help = "Execute commands from file(s), then exit",
        value_parser(parse_valid_file)
    )]
    file: Vec<String>,
}

impl Args {
    /// Without -c/-f the CLI enters the REPL, which reads its SQL from
    /// stdin.
    fn repl_mode(&self) -> bool {
        self.command.is_empty() && self.file.is_empty()
    }

    /// Return true if the CLI reads from stdin
    fn reads_sql_from_stdin(&self) -> bool {
        self.repl_mode() || self.file.iter().any(|f| is_stdin_location(f))
    }
}

#[tokio::main]
pub async fn main() -> ExitCode {
    if let Err(e) = main_inner().await {
        println!("Error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

async fn main_inner() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!(
        "dfv (DataFusion Variant CLI) v{} based on DataFusion CLI v{DATAFUSION_CLI_VERSION}",
        env!("CARGO_PKG_VERSION")
    );

    // This is mostly copied setup code from datafusion-cli -- it might be nice to encapsulate this more
    let session_config = get_session_config(&args)?;

    let instrumented_registry = Arc::new(InstrumentedObjectStoreRegistry::new());
    let runtime_env = RuntimeEnvBuilder::new()
        .with_object_store_registry(instrumented_registry.clone())
        .build_arc()?;

    let state = SessionStateBuilder::new()
        .with_config(session_config)
        .with_runtime_env(runtime_env)
        .with_default_features()
        .build();

    // enable dynamic file query (`SELECT .. FROM 'file.parquet'`)
    let mut ctx = SessionContext::new_with_state(state).enable_url_table();
    // register the Variant functions, `:` field access, and SQL `VARIANT` type
    register_all(&mut ctx)?;
    ctx.refresh_catalogs().await?;
    // install dynamic catalog provider that can register required object stores
    ctx.register_catalog_list(Arc::new(DynamicObjectStoreCatalog::new(
        ctx.state().catalog_list().clone(),
        ctx.state_weak_ref(),
    )));

    // register the datafusion-cli table functions
    let cache_manager = ctx.task_ctx().runtime_env().cache_manager.clone();
    ctx.register_udtf("parquet_metadata", Arc::new(ParquetMetadataFunc {}));
    ctx.register_udtf(
        "parquet_file_metadata",
        Arc::new(ParquetFileMetadataFunc {}),
    );
    ctx.register_udtf("parquet_page_index", Arc::new(ParquetPageIndexFunc {}));
    ctx.register_udtf(
        "metadata_cache",
        Arc::new(MetadataCacheFunc::new(cache_manager.clone())),
    );
    ctx.register_udtf(
        "statistics_cache",
        Arc::new(StatisticsCacheFunc::new(cache_manager.clone())),
    );
    ctx.register_udtf(
        "list_files_cache",
        Arc::new(ListFilesCacheFunc::new(cache_manager)),
    );
    // register `load_bsky(n)` to download and load sample Bluesky JSON data
    ctx.register_udtf("load_bsky", Arc::new(load_bsky::LoadBskyFunc));

    let mut print_options = PrintOptions {
        format: PrintFormat::Automatic,
        quiet: false,
        maxrows: MaxRows::Limited(40),
        color: false,
        instrumented_registry,
    };

    let cli_ctx = cli_context::DisplayContext(&ctx);
    if args.repl_mode() {
        print_examples();
        return exec::exec_from_repl(&cli_ctx, &mut print_options)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)));
    }

    if !args.file.is_empty() {
        exec::exec_from_files(&cli_ctx, args.file, &print_options).await?;
    }

    if !args.command.is_empty() {
        exec::exec_from_commands(&cli_ctx, args.command, &print_options).await?;
    }

    Ok(())
}

/// Print example commands when starting the interactive shell.
fn print_examples() {
    let examples = r###"
Try:
  -- Build a table of Variant values and query it
  CREATE TABLE events AS
  SELECT parse_json(json_data) AS event
  FROM (VALUES
      ('{"service":"api","status":200,"latency_ms":12,"tags":["cached"]}'),
      ('{"service":"api","status":500,"latency_ms":85,"error":null}'),
      ('{"service":"api","status":504,"error":"Somethings not working"}')
  ) AS input(json_data);

  -- Select fields from the Variant values using the `:` operator and filter by a field value
  SELECT
     event:service::VARCHAR AS service
  FROM events
  WHERE event:status::BIGINT >= 400;

  -- Write Variant to Parquet and read it back
  COPY events TO 'events.parquet' STORED AS PARQUET;
  SET datafusion.execution.parquet.skip_metadata = false;
  SELECT event FROM 'events.parquet';

  -- Inspect the schema of Variant values
  SELECT schema_of_variant(event) FROM events;

  -- Load 1M real Bluesky events (downloads ~135MB to data/bluesky on first use)
  CREATE TABLE bsky AS SELECT parse_json(json_data) AS event FROM load_bsky(1);
  SELECT event:kind::VARCHAR AS kind, count(*) FROM bsky GROUP BY kind ORDER BY 2 DESC;

See https://github.com/datafusion-contrib/datafusion-variant#functions for all functions.

Enter \q to exit.
"###;
    println!("{examples}");
}

fn get_session_config(args: &Args) -> Result<SessionConfig> {
    // Read options from environment variables
    let mut config_options = ConfigOptions::from_env()?;

    // use easier to understand "tree" mode by default
    // if the user hasn't specified an explain format in the environment
    if env::var_os("DATAFUSION_EXPLAIN_FORMAT").is_none() {
        config_options.explain.format = ExplainFormat::Tree;
    }

    // in the CLI, we want to show NULL values rather the empty strings
    if env::var_os("DATAFUSION_FORMAT_NULL").is_none() {
        config_options.format.null = String::from("NULL");
    }

    let mut session_config = SessionConfig::from(config_options).with_information_schema(true);

    if args.reads_sql_from_stdin() {
        session_config = session_config.with_extension(Arc::new(StdinCarriesCommands));
    }

    Ok(session_config)
}

fn parse_valid_file(file: &str) -> Result<String, String> {
    if Path::new(file).is_file() {
        Ok(file.to_string())
    } else {
        Err(format!("Invalid file '{file}'"))
    }
}

fn parse_command(command: &str) -> Result<String, String> {
    if !command.is_empty() {
        Ok(command.to_string())
    } else {
        Err("-c flag expects only non empty commands".to_string())
    }
}
