//! Parquet-only JSONBench preparation and query runner.
#[path = "../benchmarks/jsonbench/mod.rs"]
mod jsonbench;

use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    jsonbench::execute(jsonbench::Cli::parse()).await
}
