//! JSONBench preparation: download gzip NDJSON and convert to Parquet Variant.
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use arrow::array::{ArrayRef, RecordBatch, StructArray};
use arrow::datatypes::{Field, Schema};
use flate2::read::MultiGzDecoder;
use parquet::arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use parquet_variant_compute::{VariantArrayBuilder, VariantType, shred_variant};
use parquet_variant_json::JsonToVariant;
use tempfile::NamedTempFile;

use super::{Layout, Prepare};

pub(super) fn has_rows(path: &Path, rows: u64) -> bool {
    File::open(path)
        .ok()
        .and_then(|file| ParquetRecordBatchReaderBuilder::try_new(file).ok())
        .is_some_and(|reader| reader.metadata().file_metadata().num_rows() as u64 == rows)
}

fn download(path: &Path, number: usize) -> Result<()> {
    let url = format!(
        "https://clickhouse-public-datasets.s3.amazonaws.com/bluesky/file_{number:04}.json.gz"
    );
    // Like DataFusion's ClickBench script, check size and resume downloads.
    // Each JSONBench part has a different compressed size; get it from the server.
    let response = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--head",
            &url,
        ])
        .output()?;
    ensure!(response.status.success(), "cannot get source size: {url}");
    let size: u64 = String::from_utf8(response.stdout)?
        .lines()
        .filter_map(|line| line.split_once(':'))
        .rfind(|(key, _)| key.eq_ignore_ascii_case("content-length"))
        .context("missing source Content-Length")?
        .1
        .trim()
        .parse()?;
    if fs::metadata(path).is_ok_and(|m| m.len() == size) {
        println!("Reusing {}", path.display());
        return Ok(());
    }
    let status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--retry",
            "3",
            "--continue-at",
            "-",
            "--output",
        ])
        .arg(path)
        .arg(&url)
        .status()?;
    ensure!(status.success(), "download failed: {url}");
    ensure!(
        fs::metadata(path)?.len() == size,
        "unexpected download size"
    );
    Ok(())
}

pub(super) fn prepare(options: &Prepare) -> Result<()> {
    for (number, rows) in options.dataset.parts()? {
        let missing: Vec<_> = Layout::ALL
            .into_iter()
            .filter(|&layout| !has_rows(&options.dataset.path(layout, number, rows), rows))
            .collect();
        if missing.is_empty() {
            println!("Reusing all layouts for file_{number:04} ({rows} rows)");
            continue;
        }
        fs::create_dir_all(&options.source_dir)?;
        let source = options.source_dir.join(format!("file_{number:04}.json.gz"));
        download(&source, number)?;
        for layout in missing {
            let output = options.dataset.path(layout, number, rows);
            println!("Preparing {}", output.display());
            convert(&source, &output, layout, rows, options)?;
        }
    }
    Ok(())
}

pub(super) fn convert(
    source: &Path,
    output: &Path,
    layout: Layout,
    rows: u64,
    options: &Prepare,
) -> Result<()> {
    let parent = output.parent().context("output directory")?;
    fs::create_dir_all(parent)?;
    let temporary = NamedTempFile::new_in(parent)?;
    let mut writer: Option<ArrowWriter<File>> = None;
    let mut reader = BufReader::new(MultiGzDecoder::new(File::open(source)?));
    let mut line = String::new();
    let mut remaining = rows;
    while remaining > 0 {
        let count = remaining.min(options.batch_size as u64);
        let mut builder = VariantArrayBuilder::new(count as usize);
        for _ in 0..count {
            line.clear();
            ensure!(
                reader.read_line(&mut line)? != 0,
                "truncated source: {}",
                source.display()
            );
            builder.append_json(line.trim())?;
        }
        let mut array = builder.build();
        if let Some(schema) = layout.schema() {
            array = shred_variant(&array, &schema)?;
        }
        let array: ArrayRef = Arc::new(StructArray::from(array));
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", array.data_type().clone(), true).with_extension_type(VariantType),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array])?;
        if writer.is_none() {
            let properties = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .set_max_row_group_row_count(Some(options.row_group_size))
                .build();
            writer = Some(ArrowWriter::try_new(
                temporary.reopen()?,
                schema,
                Some(properties),
            )?);
        }
        writer.as_mut().context("Parquet writer")?.write(&batch)?;
        remaining -= count;
    }
    writer.context("empty source")?.close()?;
    temporary.persist(output)?;
    Ok(())
}
