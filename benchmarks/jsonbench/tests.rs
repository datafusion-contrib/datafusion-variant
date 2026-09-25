use std::fs::{self, File};
use std::io::Write;

use anyhow::Result;
use flate2::{Compression, write::GzEncoder};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet_variant_compute::{VariantArray, unshred_variant};
use parquet_variant_json::VariantToJson;
use tempfile::TempDir;

use super::prepare::convert;
use super::run::{QUERIES, context};
use super::*;
use arrow::array::RecordBatch;
use arrow::util::display::{ArrayFormatter, FormatOptions};

const JSON: &str = concat!(
    "{\"kind\":\"commit\",\"did\":\"alice\",\"time_us\":1999999,\"commit\":{\"operation\":\"create\",\"collection\":\"app.bsky.feed.post\",\"record\":{\"text\":\"a\",\"langs\":[\"en\"]}}}\n",
    "{\"kind\":\"commit\",\"did\":\"alice\",\"time_us\":3000001,\"commit\":{\"operation\":\"create\",\"collection\":\"app.bsky.feed.post\"}}\n",
    "{\"kind\":\"commit\",\"did\":\"bob\",\"time_us\":3600001000,\"commit\":{\"operation\":\"create\",\"collection\":\"app.bsky.feed.post\"}}\n",
    "{\"kind\":\"commit\",\"did\":\"bob\",\"time_us\":3600002999,\"commit\":{\"operation\":\"create\",\"collection\":\"app.bsky.feed.post\"}}\n",
    "{\"kind\":\"commit\",\"did\":\"carol\",\"time_us\":2,\"commit\":{\"operation\":\"create\",\"collection\":\"app.bsky.feed.repost\"}}\n",
    "{\"kind\":\"identity\",\"did\":\"other\",\"time_us\":3,\"identity\":{\"handle\":\"x\"}}\n",
    "{\"kind\":\"commit\",\"did\":\"alice\",\"time_us\":2,\"commit\":{\"operation\":\"create\",\"collection\":\"app.bsky.feed.like\"}}\n",
    "{\"kind\":\"commit\",\"did\":\"dave\",\"time_us\":1,\"commit\":{\"operation\":\"delete\",\"collection\":\"app.bsky.feed.post\"}}\n",
    "{\"kind\":\"commit\",\"did\":\"eve\",\"time_us\":2000000,\"commit\":{\"operation\":\"create\",\"collection\":\"app.bsky.feed.post\"}}\n",
    "{\"kind\":\"commit\",\"did\":\"frank\",\"time_us\":2000000,\"commit\":{\"operation\":\"create\",\"collection\":\"app.bsky.feed.post\"}}\n",
    "{\"kind\":\"commit\",\"did\":null,\"time_us\":4,\"commit\":{\"operation\":\"create\",\"collection\":null}}\n",
);

fn fixture() -> Result<(TempDir, Prepare)> {
    let temp = tempfile::tempdir()?;
    let source_dir = temp.path().join("source");
    fs::create_dir(&source_dir)?;
    let name = "file_0001.json.gz".to_string();
    let path = source_dir.join(&name);
    let mut gzip = GzEncoder::new(File::create(&path)?, Compression::fast());
    gzip.write_all(JSON.as_bytes())?;
    gzip.finish()?;
    let options = Prepare {
        dataset: Dataset {
            data_dir: temp.path().join("prepared"),
            size: Size::One,
            rows: Some(11),
        },
        source_dir,
        batch_size: 2,
        row_group_size: 3,
    };
    for layout in Layout::ALL {
        convert(
            &path,
            &options.dataset.path(layout, 1, 11),
            layout,
            11,
            &options,
        )?;
    }
    Ok((temp, options))
}

fn rows(batches: &[RecordBatch]) -> Result<Vec<Vec<Option<String>>>> {
    let mut rows = Vec::new();
    let options = FormatOptions::default();
    for batch in batches {
        let formatters = batch
            .columns()
            .iter()
            .map(|c| ArrayFormatter::try_new(c.as_ref(), &options))
            .collect::<Result<Vec<_>, _>>()?;
        for row in 0..batch.num_rows() {
            rows.push(
                batch
                    .columns()
                    .iter()
                    .zip(&formatters)
                    .map(|(c, f)| (!c.is_null(row)).then(|| f.value(row).to_string()))
                    .collect(),
            );
        }
    }
    Ok(rows)
}

fn run_options(options: &Prepare) -> Run {
    Run {
        dataset: options.dataset.clone(),
        layout: None,
        query: None,
        iterations: 1,
        partitions: 2,
        batch_size: 3,
        memory_limit_mib: 64,
        output: None,
        debug: false,
    }
}

#[tokio::test]
async fn parquet_layouts_preserve_documents_and_all_five_query_results() -> Result<()> {
    let (_temp, options) = fixture()?;
    let expected_documents: Vec<serde_json::Value> = JSON
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let mut references = Vec::new();
    for layout in Layout::ALL {
        let paths = vec![options.dataset.path(layout, 1, 11)];
        // Check preservation of the entire document, including residual fields
        // and arrays the benchmark's five paths do not visit.
        let mut documents = Vec::new();
        for path in &paths {
            let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
            let descriptor = builder.metadata().file_metadata().schema_descr();
            assert!(format!("{:?}", descriptor.root_schema()).contains("Variant"));
            assert_eq!(builder.metadata().num_row_groups(), 4);
            for batch in builder.build()? {
                let batch = batch?;
                let array = VariantArray::try_new(batch.column(0).as_ref())?;
                assert_eq!(
                    array.typed_value_field().is_some(),
                    layout != Layout::Unshredded
                );
                let array = unshred_variant(&array)?;
                for row in 0..batch.num_rows() {
                    documents.push(serde_json::from_str::<serde_json::Value>(
                        &array.value(row).to_json_string()?,
                    )?);
                }
            }
        }
        assert_eq!(documents, expected_documents);
        let ctx = context(&run_options(&options), &paths).await?;
        let mut results = Vec::new();
        for sql in QUERIES {
            results.push(rows(&ctx.sql(sql).await?.collect().await?)?);
        }
        if references.is_empty() {
            references = results;
        } else {
            assert_eq!(references, results);
        }
    }
    let text = |value: &str| Some(value.to_string());
    assert_eq!(
        references[0],
        vec![
            vec![text("app.bsky.feed.post"), text("7")],
            vec![None, text("2")],
            vec![text("app.bsky.feed.like"), text("1")],
            vec![text("app.bsky.feed.repost"), text("1")],
        ]
    );
    assert_eq!(
        references[1][0],
        vec![text("app.bsky.feed.post"), text("6"), text("4")]
    );
    assert_eq!(references[1][1], vec![None, text("1"), text("0")]);
    assert_eq!(references[2].len(), 4);
    assert_eq!(
        references[2][1],
        vec![text("app.bsky.feed.post"), text("0"), text("4")]
    );
    assert_eq!(
        references[2][3],
        vec![text("app.bsky.feed.post"), text("1"), text("2")]
    );
    assert_eq!(
        references[3]
            .iter()
            .map(|row| row[0].clone())
            .collect::<Vec<_>>(),
        vec![text("alice"), text("eve"), text("frank")]
    );
    assert_eq!(
        references[4],
        vec![
            vec![text("alice"), text("1001")],
            vec![text("bob"), text("1")],
            vec![text("eve"), text("0")]
        ]
    );
    Ok(())
}
