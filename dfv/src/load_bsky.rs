//! The `load_bsky(n)` table function: sample Bluesky JSON data.
//!
//! `load_bsky(n)` returns the `n`th file of the [ClickHouse Bluesky dataset]
//! as a table with a single `json_data` column holding one JSON document per
//! row. The file is read from `DFV_BSKY_DIR/data/bluesky/file_000n.json.gz`
//!
//! If the file does not exist it is downloaded first (about 135 MB per file).
//!
//! [ClickHouse Bluesky dataset]: https://clickhouse.com/docs/getting-started/example-datasets/bluesky

use std::fs::{File, TryLockError};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider};
use datafusion::common::{DataFusionError, Result, plan_err};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;
use flate2::read::GzDecoder;
use futures::StreamExt;
use object_store::ObjectStoreExt;
use object_store::http::HttpBuilder;
use object_store::path::Path as ObjectPath;
use tokio::io::AsyncWriteExt;

/// Where the dataset files are downloaded from.
const BASE_URL: &str = "https://clickhouse-public-datasets.s3.amazonaws.com";
/// Default local directory, relative to the current directory.
const DEFAULT_DIR: &str = "data/bluesky";
/// Environment variable overriding [`DEFAULT_DIR`].
const DIR_ENV: &str = "DFV_BSKY_DIR";
/// The dataset has files `file_0001.json.gz` through `file_1000.json.gz`.
const MAX_FILE_NUMBER: i64 = 1000;
/// Rows per output `RecordBatch`.
const BATCH_SIZE: usize = 8192;

/// `load_bsky(n)` table function. See the module documentation.
#[derive(Debug, Default)]
pub struct LoadBskyFunc;

impl TableFunctionImpl for LoadBskyFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let file_number = match args.exprs() {
            [] => 1,
            [Expr::Literal(ScalarValue::Int64(Some(n)), _)]
                if (1..=MAX_FILE_NUMBER).contains(n) =>
            {
                *n
            }
            _ => {
                return plan_err!(
                    "load_bsky expects a single file number between 1 and {MAX_FILE_NUMBER}, e.g. load_bsky(1)"
                );
            }
        };
        let dir = std::env::var_os(DIR_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_DIR));
        Ok(Arc::new(BskyTable {
            file_name: format!("file_{file_number:04}.json.gz"),
            path: dir.join(format!("file_{file_number:04}.json.gz")),
        }))
    }
}

fn bsky_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "json_data",
        DataType::Utf8,
        false,
    )]))
}

/// One Bluesky file as a table. The file is downloaded if needed and read
/// from disk on every scan.
#[derive(Debug)]
struct BskyTable {
    file_name: String,
    path: PathBuf,
}

impl BskyTable {
    async fn batches(&self) -> Result<Vec<RecordBatch>> {
        if !self.path.is_file() {
            download(&self.file_name, &self.path).await?;
        }
        eprintln!("Loading {}", self.path.display());
        let path = self.path.clone();
        let batches = tokio::task::spawn_blocking(move || read_json_lines(&path))
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))??;
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        eprintln!("Loaded {rows} rows");
        Ok(batches)
    }
}

#[async_trait]
impl TableProvider for BskyTable {
    fn schema(&self) -> SchemaRef {
        bsky_schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&[usize]>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let batches = self.batches().await?;
        // A single partition; DataFusion repartitions as needed.
        Ok(MemorySourceConfig::try_new_exec(
            &[batches],
            self.schema(),
            projection.map(<[usize]>::to_vec),
        )?)
    }
}

/// Download `bluesky/<file_name>` from [`BASE_URL`] to `dest`, writing to a
/// temporary `.part` file first so an interrupted download is not mistaken
/// for a complete file.
async fn download(file_name: &str, dest: &Path) -> Result<()> {
    let path = dest.to_path_buf();
    // Keep the lock file in place: removing it could let callers lock different
    // files for the same destination. Closing the handle releases the lock.
    let _lock = tokio::task::spawn_blocking(move || -> std::io::Result<File> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let lock = File::options()
            .write(true)
            .create(true)
            .truncate(false)
            .open(path.with_extension("lock"))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                eprintln!("Waiting for download of {}", path.display());
                lock.lock()?;
            }
            Err(TryLockError::Error(e)) => return Err(e),
        }
        Ok(lock)
    })
    .await
    .map_err(|e| DataFusionError::External(Box::new(e)))??;
    // Another scan or dfv process may have completed the download while we waited.
    if dest.is_file() {
        return Ok(());
    }
    let location = ObjectPath::from(format!("bluesky/{file_name}"));
    eprintln!("Downloading {BASE_URL}/{location} to {}", dest.display());
    let store = HttpBuilder::new().with_url(BASE_URL).build()?;
    let result = store.get(&location).await?;
    let total_bytes = result.meta.size;
    let mut stream = result.into_stream();

    let part = dest.with_extension("part");
    let mut file = tokio::fs::File::create(&part).await?;
    let mut downloaded: u64 = 0;
    let mut last_report: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if downloaded - last_report >= 10 << 20 {
            last_report = downloaded;
            eprint!("\r  {} / {} MB", downloaded >> 20, total_bytes >> 20);
        }
    }
    file.flush().await?;
    drop(file);
    eprintln!("\r  {} / {} MB", downloaded >> 20, total_bytes >> 20);
    std::fs::rename(&part, dest)?;
    Ok(())
}

/// Read a gzipped newline-delimited JSON file into `json_data` batches.
fn read_json_lines(path: &Path) -> Result<Vec<RecordBatch>> {
    let schema = bsky_schema();
    let reader = BufReader::new(GzDecoder::new(File::open(path)?));
    let mut batches = Vec::new();
    let mut rows = Vec::with_capacity(BATCH_SIZE);
    let mut flush = |rows: &mut Vec<String>| -> Result<()> {
        let array = StringArray::from(std::mem::take(rows));
        batches.push(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(array)],
        )?);
        Ok(())
    };
    for line in reader.lines() {
        rows.push(line?);
        if rows.len() == BATCH_SIZE {
            flush(&mut rows)?;
        }
    }
    if !rows.is_empty() {
        flush(&mut rows)?;
    }
    Ok(batches)
}
