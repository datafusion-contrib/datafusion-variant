//! JSONBench uses the original Bluesky NDJSON data and translated Variant SQL.
//! See README.md in this directory for provenance and timing boundaries.
mod prepare;
mod run;

use std::path::PathBuf;

use anyhow::{Result, ensure};
use arrow::datatypes::{DataType, Field};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(about = "Prepare and run JSONBench over Parquet Variant (no MemTables)")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Download/reuse JSON and prepare/reuse all three Parquet layouts.
    Prepare(Prepare),
    /// Validate prepared fixtures, then run all 15 cases by default.
    Run(Run),
}

#[derive(Clone, Debug, Args)]
struct Dataset {
    /// Directory for prepared Parquet files.
    #[arg(long, default_value = "data/jsonbench")]
    data_dir: PathBuf,
    #[arg(long, value_enum, default_value = "1m")]
    size: Size,
    /// Use only the first N records, for smaller development runs.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    rows: Option<u64>,
}

impl Dataset {
    fn parts(&self) -> Result<Vec<(usize, u64)>> {
        let available = self.size.files() as u64 * 1_000_000;
        let rows = self.rows.unwrap_or(available);
        ensure!(rows <= available, "--rows exceeds --size");
        Ok((0..rows.div_ceil(1_000_000))
            .map(|i| (i as usize + 1, (rows - i * 1_000_000).min(1_000_000)))
            .collect())
    }

    fn path(&self, layout: Layout, number: usize, rows: u64) -> PathBuf {
        self.data_dir
            .join(layout.name())
            .join(format!("file_{number:04}_{rows}.parquet"))
    }
}

#[derive(Clone, Debug, Args)]
struct Prepare {
    #[command(flatten)]
    dataset: Dataset,
    /// Original compressed JSON files; compatible with download_data.sh.
    #[arg(long, default_value = "data/bluesky")]
    source_dir: PathBuf,
    /// Maximum records per conversion batch.
    #[arg(long, default_value_t = 8192, value_parser = positive_usize)]
    batch_size: usize,
    #[arg(long, default_value_t = 65_536, value_parser = positive_usize)]
    row_group_size: usize,
}

#[derive(Clone, Debug, Args)]
struct Run {
    #[command(flatten)]
    dataset: Dataset,
    /// Omit to require and run all three layouts.
    #[arg(long, value_enum)]
    layout: Option<Layout>,
    /// Query IDs are 1 through 5. Omit to run all queries.
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=5))]
    query: Option<u8>,
    #[arg(long, default_value_t = 5, value_parser = positive_usize)]
    iterations: usize,
    #[arg(long, default_value_t = 4, value_parser = positive_usize)]
    partitions: usize,
    #[arg(long, default_value_t = 8192, value_parser = positive_usize)]
    batch_size: usize,
    /// DataFusion managed-memory limit in MiB; operators may spill to disk.
    #[arg(long, default_value_t = 1024, value_parser = positive_usize)]
    memory_limit_mib: usize,
    /// JSON output compatible with DataFusion benchmarks/compare.py (default: DATA_DIR/results.json).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Print physical plans outside timing.
    #[arg(long)]
    debug: bool,
}

impl Run {
    fn output_path(&self) -> PathBuf {
        self.output
            .clone()
            .unwrap_or_else(|| self.dataset.data_dir.join("results.json"))
    }
}

fn positive_usize(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| "expected a positive integer".to_string())
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Size {
    #[value(name = "1m")]
    One,
    #[value(name = "10m")]
    Ten,
    #[value(name = "100m")]
    Hundred,
    #[value(name = "1000m")]
    Thousand,
}

impl Size {
    fn files(self) -> usize {
        match self {
            Self::One => 1,
            Self::Ten => 10,
            Self::Hundred => 100,
            Self::Thousand => 1000,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum Layout {
    Unshredded,
    Partial,
    QueryFields,
}

impl Layout {
    const ALL: [Self; 3] = [Self::Unshredded, Self::Partial, Self::QueryFields];

    fn name(self) -> &'static str {
        match self {
            Self::Unshredded => "unshredded",
            Self::Partial => "partial",
            Self::QueryFields => "query-fields",
        }
    }

    fn schema(self) -> Option<DataType> {
        if self == Self::Unshredded {
            return None;
        }
        let mut fields = vec![
            Field::new("kind", DataType::Utf8, true),
            Field::new("did", DataType::Utf8, true),
            Field::new("time_us", DataType::Int64, true),
        ];
        if self == Self::QueryFields {
            fields.push(Field::new(
                "commit",
                DataType::Struct(
                    vec![
                        Field::new("collection", DataType::Utf8, true),
                        Field::new("operation", DataType::Utf8, true),
                    ]
                    .into(),
                ),
                true,
            ));
        }
        Some(DataType::Struct(fields.into()))
    }
}

pub async fn execute(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Prepare(options) => prepare::prepare(&options),
        Command::Run(options) => run::run(&options).await,
    }
}

#[cfg(test)]
mod tests;
