// SPDX-License-Identifier: Apache-2.0

//! The `load_bsky` table function: sample Bluesky JSON data.
//!
//! `load_bsky(n)` returns the `n`th file of the [ClickHouse Bluesky dataset]
//! as a table with a single `json_data` column holding one JSON document per
//! row, and `load_bsky(first, last)` returns files `first` to `last` in order.
//! Files are read from `DFV_BSKY_DIR` (default `data/bluesky`), for example
//! `data/bluesky/file_0001.json.gz`.
//!
//! Missing files are downloaded first (about 135 MB per file). Files are read
//! one at a time as the query consumes them, so a range does not have to fit
//! in memory.
//!
//! [ClickHouse Bluesky dataset]: https://clickhouse.com/docs/getting-started/example-datasets/bluesky

use std::fs::{File, TryLockError};
use std::io::{BufRead, BufReader};
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::streaming::StreamingTable;
use datafusion::catalog::{TableFunctionArgs, TableFunctionImpl, TableProvider};
use datafusion::common::{DataFusionError, Result, plan_err};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::stream::RecordBatchReceiverStream;
use datafusion::physical_plan::streaming::PartitionStream;
use datafusion::scalar::ScalarValue;
use flate2::read::GzDecoder;
use futures::StreamExt;
use object_store::ObjectStoreExt;
use object_store::http::HttpBuilder;
use object_store::path::Path as ObjectPath;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::Sender;

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

/// `load_bsky(n)` and `load_bsky(first, last)` table function. See the module
/// documentation.
#[derive(Debug, Default)]
pub struct LoadBskyFunc;

impl TableFunctionImpl for LoadBskyFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let file_number = |expr: &Expr| match expr {
            Expr::Literal(ScalarValue::Int64(Some(n)), _) if (1..=MAX_FILE_NUMBER).contains(n) => {
                Some(*n)
            }
            _ => None,
        };
        let files = match args.exprs() {
            [] => Some(1..=1),
            [n] => file_number(n).map(|n| n..=n),
            [first, last] => file_number(first)
                .zip(file_number(last))
                .map(|(first, last)| first..=last)
                .filter(|files| !files.is_empty()),
            _ => None,
        };
        let Some(files) = files else {
            return plan_err!(
                "load_bsky expects a file number or a range of file numbers between 1 and {MAX_FILE_NUMBER}, e.g. load_bsky(1) or load_bsky(1, 10)"
            );
        };
        let dir = std::env::var_os(DIR_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_DIR));
        let schema = bsky_schema();
        let files = BskyFiles {
            schema: Arc::clone(&schema),
            dir,
            files,
        };
        Ok(Arc::new(StreamingTable::try_new(
            schema,
            vec![Arc::new(files)],
        )?))
    }
}

fn bsky_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "json_data",
        DataType::Utf8,
        false,
    )]))
}

/// Bluesky files read in order as a single partition, which DataFusion
/// repartitions as needed. Each file is downloaded if needed when the scan
/// reaches it.
#[derive(Debug)]
struct BskyFiles {
    schema: SchemaRef,
    dir: PathBuf,
    files: RangeInclusive<i64>,
}

impl PartitionStream for BskyFiles {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let mut builder = RecordBatchReceiverStream::builder(Arc::clone(&self.schema), 2);
        let tx = builder.tx();
        let schema = Arc::clone(&self.schema);
        let dir = self.dir.clone();
        let files = self.files.clone();
        builder.spawn(async move {
            for n in files {
                let file_name = format!("file_{n:04}.json.gz");
                let path = dir.join(&file_name);
                if !path.is_file() {
                    download(&file_name, &path).await?;
                }
                eprintln!("Loading {}", path.display());
                let (schema, tx) = (Arc::clone(&schema), tx.clone());
                let rows = tokio::task::spawn_blocking(move || read_json_lines(&path, schema, &tx))
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))??;
                eprintln!("Loaded {rows} rows");
            }
            Ok(())
        });
        builder.build()
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

/// Send the lines of a gzipped newline-delimited JSON file to `tx` as
/// `json_data` batches and return the number of rows sent. Stops early when
/// the query no longer needs rows.
fn read_json_lines(
    path: &Path,
    schema: SchemaRef,
    tx: &Sender<Result<RecordBatch>>,
) -> Result<usize> {
    let mut lines = BufReader::new(GzDecoder::new(File::open(path)?)).lines();
    let mut rows = 0;
    loop {
        let batch = lines
            .by_ref()
            .take(BATCH_SIZE)
            .collect::<std::io::Result<Vec<_>>>()?;
        if batch.is_empty() {
            return Ok(rows);
        }
        rows += batch.len();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(batch))],
        )?;
        if tx.blocking_send(Ok(batch)).is_err() {
            return Ok(rows);
        }
    }
}
