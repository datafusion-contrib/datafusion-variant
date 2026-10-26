// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! `dfv`: DataFusion CLI with the Variant functions preinstalled.
//!
//! This file is a copy of `datafusion-cli/src/main.rs` from the pinned
//! DataFusion revision (see `Cargo.toml`). The only intentional differences
//! are marked with `// dfv:` comments: the startup banner, installing
//! [`VariantTypePlanner`] for the SQL `VARIANT` type, and calling
//! [`register_all`] to register the Variant functions. Keep the rest in sync
//! with upstream when bumping the DataFusion revision.

use std::collections::HashMap;
use std::env;
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::ExitCode;
use std::sync::{Arc, LazyLock};

use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionConfig;
use datafusion::execution::memory_pool::{
    FairSpillPool, GreedyMemoryPool, MemoryPool, TrackConsumersPool,
};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::ExplainFormat;
use datafusion::prelude::SessionContext;
use datafusion_cli::catalog::DynamicObjectStoreCatalog;
use datafusion_cli::functions::{
    ListFilesCacheFunc, MetadataCacheFunc, ParquetFileMetadataFunc, ParquetMetadataFunc,
    ParquetPageIndexFunc, StatisticsCacheFunc,
};
use datafusion_cli::object_storage::instrumented::{
    InstrumentedObjectStoreMode, InstrumentedObjectStoreRegistry,
};
use datafusion_cli::object_storage::{StdinCarriesCommands, is_stdin_location};
use datafusion_cli::{
    DATAFUSION_CLI_VERSION, exec,
    pool_type::PoolType,
    print_format::PrintFormat,
    print_options::{MaxRows, PrintOptions},
};

use clap::Parser;
use datafusion::common::config::Dialect;
use datafusion::common::config_err;
use datafusion::config::ConfigOptions;
use datafusion::execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion_spark::SessionStateBuilderSpark;
use datafusion_variant::{VariantTypePlanner, register_all};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Parser, PartialEq)]
#[clap(author, version, about, long_about= None)]
struct Args {
    #[clap(
        short = 'p',
        long,
        help = "Path to your data, default to current directory",
        value_parser(parse_valid_data_dir)
    )]
    data_path: Option<String>,

    #[clap(
        short = 'b',
        long,
        help = "The batch size of each query, or use DataFusion default",
        value_parser(parse_batch_size)
    )]
    batch_size: Option<usize>,

    #[clap(
        short = 'c',
        long,
        num_args = 0..,
        help = "Execute the given command string(s), then exit. Commands are expected to be non empty.",
        value_parser(parse_command)
    )]
    command: Vec<String>,

    #[clap(
        short = 'm',
        long,
        help = "The memory pool limitation (e.g. '10g'), default to None (no limit)",
        value_parser(extract_memory_pool_size)
    )]
    memory_limit: Option<usize>,

    #[clap(
        short,
        long,
        num_args = 0..,
        help = "Execute commands from file(s), then exit",
        value_parser(parse_valid_file)
    )]
    file: Vec<String>,

    #[clap(
        short = 'r',
        long,
        num_args = 0..,
        help = "Run the provided files on startup instead of ~/.datafusionrc",
        value_parser(parse_valid_file),
        conflicts_with = "file"
    )]
    rc: Option<Vec<String>>,

    #[clap(long, value_enum, default_value_t = PrintFormat::Automatic)]
    format: PrintFormat,

    #[clap(
        short,
        long,
        help = "Reduce printing other than the results and work quietly"
    )]
    quiet: bool,

    #[clap(
        long,
        help = "Specify the memory pool type 'greedy' or 'fair'",
        default_value_t = PoolType::Greedy
    )]
    mem_pool_type: PoolType,

    #[clap(
        long,
        help = "The number of top memory consumers to display when query fails due to memory exhaustion. To disable memory consumer tracking, set this value to 0",
        default_value = "3"
    )]
    top_memory_consumers: usize,

    #[clap(
        long,
        help = "The max number of rows to display for 'Table' format\n[possible values: numbers(0/10/...), inf(no limit)]",
        default_value = "40"
    )]
    maxrows: MaxRows,

    #[clap(long, help = "Enables console syntax highlighting")]
    color: bool,

    #[clap(
        long,
        help = "Enable Apache Spark-compatible SQL dialect, functions, and expression planning"
    )]
    spark: bool,

    #[clap(
        short = 'd',
        long,
        help = "Available disk space for spilling queries (e.g. '10g'), default to None (uses DataFusion's default value of '100g')",
        value_parser(extract_disk_limit)
    )]
    disk_limit: Option<usize>,

    #[clap(
        long,
        help = "Specify the default object_store_profiling mode, defaults to 'disabled'.\n[possible values: disabled, summary, trace]",
        default_value_t = InstrumentedObjectStoreMode::Disabled
    )]
    object_store_profiling: InstrumentedObjectStoreMode,
}

impl Args {
    /// Without -c/-f the CLI enters the REPL, which reads its SQL from
    /// stdin — interactively or piped.
    fn repl_mode(&self) -> bool {
        self.command.is_empty() && self.file.is_empty()
    }

    /// Whether the CLI consumes stdin for its own SQL input. This covers the
    /// REPL (no -c/-f, reading SQL interactively or piped) as well as an
    /// explicit `-f /dev/stdin` (or the other stdin pseudo-paths), where the
    /// SQL file *is* stdin. In either case stdin is already spoken for and
    /// cannot also back a `LOCATION '/dev/stdin'` table.
    fn reads_sql_from_stdin(&self) -> bool {
        self.repl_mode() || self.file.iter().any(|f| is_stdin_location(f))
    }
}

#[tokio::main]
/// Calls [`main_inner`], then handles printing errors and returning the correct exit code
pub async fn main() -> ExitCode {
    if let Err(e) = main_inner().await {
        println!("Error: {e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

/// Main CLI entrypoint
async fn main_inner() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    if !args.quiet {
        // dfv: identify this build and the DataFusion CLI it wraps
        println!(
            "dfv (DataFusion Variant CLI) v{} based on DataFusion CLI v{DATAFUSION_CLI_VERSION}",
            env!("CARGO_PKG_VERSION")
        );
    }

    if let Some(ref path) = args.data_path {
        let p = Path::new(path);
        env::set_current_dir(p).unwrap();
    }

    let session_config = get_session_config(&args)?;

    let mut rt_builder = RuntimeEnvBuilder::new();
    // set memory pool size
    if let Some(memory_limit) = args.memory_limit {
        // set memory pool type
        let pool: Arc<dyn MemoryPool> = match args.mem_pool_type {
            PoolType::Fair if args.top_memory_consumers == 0 => {
                Arc::new(FairSpillPool::new(memory_limit))
            }
            PoolType::Fair => Arc::new(TrackConsumersPool::new(
                FairSpillPool::new(memory_limit),
                NonZeroUsize::new(args.top_memory_consumers).unwrap(),
            )),
            PoolType::Greedy if args.top_memory_consumers == 0 => {
                Arc::new(GreedyMemoryPool::new(memory_limit))
            }
            PoolType::Greedy => Arc::new(TrackConsumersPool::new(
                GreedyMemoryPool::new(memory_limit),
                NonZeroUsize::new(args.top_memory_consumers).unwrap(),
            )),
        };

        rt_builder = rt_builder.with_memory_pool(pool)
    }

    // set disk limit
    if let Some(disk_limit) = args.disk_limit {
        let builder = DiskManagerBuilder::default()
            .with_mode(DiskManagerMode::OsTmpDirectory)
            .with_max_temp_directory_size(disk_limit.try_into().unwrap());
        rt_builder = rt_builder.with_disk_manager_builder(builder);
    }

    let instrumented_registry = Arc::new(
        InstrumentedObjectStoreRegistry::new().with_profile_mode(args.object_store_profiling),
    );
    rt_builder = rt_builder.with_object_store_registry(instrumented_registry.clone());

    let runtime_env = rt_builder.build_arc()?;

    let mut state_builder = SessionStateBuilder::new()
        .with_config(session_config)
        .with_runtime_env(runtime_env)
        .with_default_features();
    if args.spark {
        state_builder = state_builder.with_spark_features();
    }
    // dfv: support the SQL `VARIANT` type name (e.g. `CAST(x AS VARIANT)`)
    state_builder = state_builder.with_type_planner(Arc::new(VariantTypePlanner));

    // enable dynamic file query
    let mut ctx = SessionContext::new_with_state(state_builder.build()).enable_url_table();
    // dfv: register the Variant functions and `:` field access
    register_all(&mut ctx)?;
    ctx.refresh_catalogs().await?;
    // install dynamic catalog provider that can register required object stores
    ctx.register_catalog_list(Arc::new(DynamicObjectStoreCatalog::new(
        ctx.state().catalog_list().clone(),
        ctx.state_weak_ref(),
    )));
    // register `parquet_metadata` table function to get metadata from parquet files
    ctx.register_udtf("parquet_metadata", Arc::new(ParquetMetadataFunc {}));

    // register `parquet_file_metadata` table function to get file level metadata from parquet files
    ctx.register_udtf(
        "parquet_file_metadata",
        Arc::new(ParquetFileMetadataFunc {}),
    );

    // register `parquet_page_index` table function to get the page index of parquet files
    ctx.register_udtf("parquet_page_index", Arc::new(ParquetPageIndexFunc {}));

    // register `metadata_cache` table function to get the contents of the file metadata cache
    ctx.register_udtf(
        "metadata_cache",
        Arc::new(MetadataCacheFunc::new(
            ctx.task_ctx().runtime_env().cache_manager.clone(),
        )),
    );

    // register `statistics_cache` table function to get the contents of the file statistics cache
    ctx.register_udtf(
        "statistics_cache",
        Arc::new(StatisticsCacheFunc::new(
            ctx.task_ctx().runtime_env().cache_manager.clone(),
        )),
    );

    ctx.register_udtf(
        "list_files_cache",
        Arc::new(ListFilesCacheFunc::new(
            ctx.task_ctx().runtime_env().cache_manager.clone(),
        )),
    );

    let mut print_options = PrintOptions {
        format: args.format,
        quiet: args.quiet,
        maxrows: args.maxrows,
        color: args.color,
        instrumented_registry: Arc::clone(&instrumented_registry),
    };

    let repl_mode = args.repl_mode();
    let commands = args.command;
    let files = args.file;
    let rc = match args.rc {
        Some(file) => file,
        None => {
            let mut files = Vec::new();
            let home = dirs::home_dir();
            if let Some(p) = home {
                let home_rc = p.join(".datafusionrc");
                if home_rc.exists() {
                    files.push(home_rc.into_os_string().into_string().unwrap());
                }
            }
            files
        }
    };

    if repl_mode {
        if !rc.is_empty() {
            exec::exec_from_files(&ctx, rc, &print_options).await?;
        }
        // TODO maybe we can have thiserror for cli but for now let's keep it simple
        return exec::exec_from_repl(&ctx, &mut print_options)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)));
    }

    if !files.is_empty() {
        exec::exec_from_files(&ctx, files, &print_options).await?;
    }

    if !commands.is_empty() {
        exec::exec_from_commands(&ctx, commands, &print_options).await?;
    }

    Ok(())
}

/// Get the session configuration based on the provided arguments
/// and environment settings.
fn get_session_config(args: &Args) -> Result<SessionConfig> {
    // Read options from environment variables and merge with command line options
    let mut config_options = ConfigOptions::from_env()?;

    if let Some(batch_size) = args.batch_size {
        if batch_size == 0 {
            return config_err!("batch_size must be greater than 0");
        }
        config_options.execution.batch_size =
            datafusion_common::config::ConfigNonZeroUsize::try_new(batch_size)?;
    }

    // use easier to understand "tree" mode by default
    // if the user hasn't specified an explain format in the environment
    if env::var_os("DATAFUSION_EXPLAIN_FORMAT").is_none() {
        config_options.explain.format = ExplainFormat::Tree;
    }

    // in the CLI, we want to show NULL values rather the empty strings
    if env::var_os("DATAFUSION_FORMAT_NULL").is_none() {
        config_options.format.null = String::from("NULL");
    }

    if args.spark {
        config_options.sql_parser.dialect = Dialect::Spark;
    }

    let mut session_config = SessionConfig::from(config_options).with_information_schema(true);

    if args.reads_sql_from_stdin() {
        // When stdin carries the session's SQL — the REPL (including any rc
        // file run before it) or an explicit `-f /dev/stdin` — it cannot also
        // serve as a data source for `LOCATION '/dev/stdin'`.
        session_config = session_config.with_extension(Arc::new(StdinCarriesCommands));
    }

    Ok(session_config)
}

fn parse_valid_file(dir: &str) -> Result<String, String> {
    if Path::new(dir).is_file() {
        Ok(dir.to_string())
    } else {
        Err(format!("Invalid file '{dir}'"))
    }
}

fn parse_valid_data_dir(dir: &str) -> Result<String, String> {
    if Path::new(dir).is_dir() {
        Ok(dir.to_string())
    } else {
        Err(format!("Invalid data directory '{dir}'"))
    }
}

fn parse_batch_size(size: &str) -> Result<usize, String> {
    match size.parse::<usize>() {
        Ok(size) if size > 0 => Ok(size),
        _ => Err(format!("Invalid batch size '{size}'")),
    }
}

fn parse_command(command: &str) -> Result<String, String> {
    if !command.is_empty() {
        Ok(command.to_string())
    } else {
        Err("-c flag expects only non empty commands".to_string())
    }
}

#[derive(Debug, Clone, Copy)]
enum ByteUnit {
    Byte,
    KiB,
    MiB,
    GiB,
    TiB,
}

impl ByteUnit {
    fn multiplier(&self) -> u64 {
        match self {
            ByteUnit::Byte => 1,
            ByteUnit::KiB => 1 << 10,
            ByteUnit::MiB => 1 << 20,
            ByteUnit::GiB => 1 << 30,
            ByteUnit::TiB => 1 << 40,
        }
    }
}

fn parse_size_string(size: &str, label: &str) -> Result<usize, String> {
    static BYTE_SUFFIXES: LazyLock<HashMap<&'static str, ByteUnit>> = LazyLock::new(|| {
        let mut m = HashMap::new();
        m.insert("b", ByteUnit::Byte);
        m.insert("k", ByteUnit::KiB);
        m.insert("kb", ByteUnit::KiB);
        m.insert("m", ByteUnit::MiB);
        m.insert("mb", ByteUnit::MiB);
        m.insert("g", ByteUnit::GiB);
        m.insert("gb", ByteUnit::GiB);
        m.insert("t", ByteUnit::TiB);
        m.insert("tb", ByteUnit::TiB);
        m
    });

    static SUFFIX_REGEX: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"^(-?[0-9]+)([a-z]+)?$").unwrap());

    let lower = size.to_lowercase();
    if let Some(caps) = SUFFIX_REGEX.captures(&lower) {
        let num_str = caps.get(1).unwrap().as_str();
        let num = num_str
            .parse::<usize>()
            .map_err(|_| format!("Invalid numeric value in {label} '{size}'"))?;

        let suffix = caps.get(2).map(|m| m.as_str()).unwrap_or("b");
        let unit = BYTE_SUFFIXES
            .get(suffix)
            .ok_or_else(|| format!("Invalid {label} '{size}'"))?;
        let total_bytes = usize::try_from(unit.multiplier())
            .ok()
            .and_then(|multiplier| num.checked_mul(multiplier))
            .ok_or_else(|| format!("{label} '{size}' is too large"))?;

        Ok(total_bytes)
    } else {
        Err(format!("Invalid {label} '{size}'"))
    }
}

pub fn extract_memory_pool_size(size: &str) -> Result<usize, String> {
    parse_size_string(size, "memory pool size")
}

pub fn extract_disk_limit(size: &str) -> Result<usize, String> {
    parse_size_string(size, "disk limit")
}
