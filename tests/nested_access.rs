// SPDX-License-Identifier: Apache-2.0

//! `variant_get` and `try_variant_get` in a Parquet scan read only the
//! shredded columns that hold their path, and skip row groups using the
//! statistics of those columns.

use std::path::Path;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::common::Result;
use datafusion::physical_plan::metrics::MetricValue;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::*;
use datafusion_variant::register_all;

const ROWS: usize = 10_000;
const ROW_GROUPS: usize = 10;
/// A few hundred bytes that do not compress well, as the fields of a real
/// document that are not shredded.
const PAD: &str = "md5(value::varchar) || md5((value + 1)::varchar) || md5((value + 2)::varchar) \
                   || md5((value + 3)::varchar) || md5((value + 4)::varchar)";
const SHREDDING: &str = r#"{"v":"STRUCT<a BIGINT, b STRUCT<c VARCHAR>>"}"#;

/// Writes `ROWS` rows in `ROW_GROUPS` row groups to `path`, optionally
/// shredding `a` and `b.c`:
///
/// ```text
/// id    v
/// 1     {"a": 1, "b": {"c": "s1"}, "pad": "c4ca4238a0b9..."}
/// 2     {"a": 2, "b": {"c": "s2"}, "pad": "c81e728d9d4c..."}
/// ...
/// ```
///
/// If `fallback`, `a` is the string `"oops"` in the row with `id = 2500`. When
/// shredded as `BIGINT` that row stores `a` in `value`, not in `typed_value`.
async fn write(ctx: &SessionContext, path: &Path, shredding: bool, fallback: bool) -> Result<()> {
    let a = if fallback {
        r#"CASE WHEN value = 2500 THEN '"oops"' ELSE value::varchar END"#
    } else {
        "value::varchar"
    };
    let shredding = if shredding {
        format!(", 'variant.shredding' '{SHREDDING}'")
    } else {
        String::new()
    };
    let sql = format!(
        r#"COPY (
             SELECT value AS id,
                    parse_json('{{"a": ' || {a}
                      || ', "b": {{"c": "s' || value
                      || '"}}, "pad": "' || {PAD} || '"}}') AS v
             FROM generate_series(1, {ROWS})
           ) TO '{}' STORED AS PARQUET
           OPTIONS ('max_row_group_size' '{}' {shredding})"#,
        path.display(),
        ROWS / ROW_GROUPS,
    );
    ctx.sql(&sql).await?.collect().await?;
    Ok(())
}

/// What the Parquet scan of a query did
#[derive(Debug)]
struct Scan {
    /// The query's output: the values of each row
    output: Vec<Vec<String>>,
    bytes_scanned: usize,
    /// Row groups left after pruning with statistics
    row_groups_matched: usize,
}

fn rows(batches: &[RecordBatch]) -> Result<Vec<Vec<String>>> {
    let options = FormatOptions::default().with_null("NULL");
    let mut rows = vec![];
    for batch in batches {
        let columns = batch
            .columns()
            .iter()
            .map(|column| ArrayFormatter::try_new(column, &options))
            .collect::<Result<Vec<_>, _>>()?;
        for row in 0..batch.num_rows() {
            rows.push(columns.iter().map(|c| c.value(row).to_string()).collect());
        }
    }
    Ok(rows)
}

async fn run(ctx: &SessionContext, sql: &str) -> Result<Scan> {
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
    let mut scan: &Arc<dyn ExecutionPlan> = &plan;
    while let Some(child) = scan.children().first() {
        scan = child;
    }
    let metrics = scan.metrics().expect("scan metrics");
    let row_groups_matched = metrics
        .iter()
        .filter_map(|metric| match metric.value() {
            MetricValue::PruningMetrics {
                name,
                pruning_metrics,
            } if name == "row_groups_pruned_statistics" => Some(pruning_metrics.matched()),
            _ => None,
        })
        .sum();
    Ok(Scan {
        output: rows(&batches)?,
        bytes_scanned: metrics
            .sum_by_name("bytes_scanned")
            .expect("bytes_scanned")
            .as_usize(),
        row_groups_matched,
    })
}

/// The tables:
///
/// * `shredded`: one shredded file, with the `"oops"` row
/// * `clean`: one shredded file, without the `"oops"` row
/// * `unshredded`: one unshredded file, with the `"oops"` row
/// * `mixed`: the `shredded` and `unshredded` files in one table
/// * `doubled`: the `unshredded` file twice
async fn setup(dir: &Path, pushdown_filters: bool) -> Result<SessionContext> {
    let config = SessionConfig::new().with_target_partitions(1).set_bool(
        "datafusion.execution.parquet.pushdown_filters",
        pushdown_filters,
    );
    let mut ctx = SessionContext::new_with_config(config);
    register_all(&mut ctx)?;
    for (table, files) in [
        ("shredded", vec![(true, true)]),
        ("clean", vec![(true, false)]),
        ("unshredded", vec![(false, true)]),
        ("mixed", vec![(true, true), (false, true)]),
        ("doubled", vec![(false, true), (false, true)]),
    ] {
        let table_dir = dir.join(table);
        if !table_dir.exists() {
            std::fs::create_dir(&table_dir)?;
            for (i, (shredding, fallback)) in files.into_iter().enumerate() {
                let file = table_dir.join(format!("{i}.parquet"));
                write(&ctx, &file, shredding, fallback).await?;
            }
        }
        ctx.sql(&format!(
            "CREATE EXTERNAL TABLE {table} STORED AS PARQUET LOCATION '{}/'",
            table_dir.display()
        ))
        .await?;
    }
    Ok(ctx)
}

fn file_size(dir: &Path, table: &str) -> usize {
    std::fs::metadata(dir.join(table).join("0.parquet"))
        .unwrap()
        .len() as usize
}

/// Narrowing the scan does not change results: the shredded tables return
/// what the unshredded table returns.
#[tokio::test]
async fn same_results_as_unshredded() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let queries = [
        // Shredded primitive
        "SELECT sum(try_variant_get(v, '$.a', 'BIGINT')) FROM t",
        "SELECT count(*) FROM t WHERE try_variant_get(v, '$.a', 'BIGINT') > 9500",
        "SELECT count(*) FROM t WHERE try_variant_get(v, '$.a', 'BIGINT') BETWEEN 2400 AND 2600",
        "SELECT id FROM t WHERE try_variant_get(v, '$.a', 'BIGINT') = 7 ORDER BY id",
        "SELECT min(v:a::double), max(v:a::double) FROM t WHERE id != 2500",
        // The row whose `a` is in `value`
        "SELECT id FROM t WHERE try_variant_get(v, '$.a', 'BIGINT') IS NULL ORDER BY id",
        "SELECT id FROM t WHERE try_variant_get(v, '$.a', 'VARCHAR') = 'oops' ORDER BY id",
        "SELECT id, to_json(v:a) FROM t WHERE id BETWEEN 2499 AND 2501 ORDER BY id",
        // Nested shredded primitive, and a whole shredded object
        "SELECT count(v:b:c::varchar), min(v:b:c::varchar), max(v:b:c::varchar) FROM t",
        "SELECT count(*) FROM t WHERE v:b:c::varchar = 's4242'",
        "SELECT min(to_json(v:b)), max(to_json(v:b)) FROM t",
        // Not shredded
        "SELECT max(v:pad::varchar) FROM t",
        "SELECT count(v:missing), count(v:a:x), count(v:b:other), count(variant_get(v, '$[0]')) FROM t",
        // Several paths of one column, and the whole column
        "SELECT sum(try_variant_get(v, '$.a', 'BIGINT')), max(v:b:c::varchar), max(v:pad::varchar) FROM t",
        "SELECT sum(try_variant_get(v, '$.a', 'BIGINT')), count(v) FROM t",
    ];
    for pushdown_filters in [false, true] {
        let ctx = setup(dir.path(), pushdown_filters).await?;
        for query in queries {
            let message = format!("{query} (pushdown_filters={pushdown_filters})");
            let expected = run(&ctx, &query.replace("FROM t", "FROM unshredded")).await?;
            let shredded = run(&ctx, &query.replace("FROM t", "FROM shredded")).await?;
            assert_eq!(shredded.output, expected.output, "{message}");

            // `mixed` has each row twice, as `doubled` does
            let expected = run(&ctx, &query.replace("FROM t", "FROM doubled")).await?;
            let mixed = run(&ctx, &query.replace("FROM t", "FROM mixed")).await?;
            assert_eq!(mixed.output, expected.output, "{message}");
        }
    }
    Ok(())
}

/// A getter of a shredded path reads the columns of that path, not the whole
/// Variant column.
#[tokio::test]
async fn reads_only_shredded_columns() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let ctx = setup(dir.path(), false).await?;
    let file_size = file_size(dir.path(), "shredded");

    // Reads everything
    let whole = run(
        &ctx,
        "SELECT count(try_variant_get(v, '$', 'VARCHAR')) FROM shredded",
    )
    .await?;
    assert!(whole.bytes_scanned > file_size / 2, "{whole:?}");

    for query in [
        "SELECT sum(try_variant_get(v, '$.a', 'BIGINT')) FROM shredded",
        "SELECT max(try_variant_get(v, '$.b.c', 'VARCHAR')) FROM shredded",
        "SELECT max(TRY_CAST(v:b:c AS VARCHAR)) FROM shredded",
        // Untyped
        "SELECT count(v:a) FROM shredded",
        "SELECT count(v:b) FROM shredded",
        // In a filter
        "SELECT count(*) FROM shredded WHERE try_variant_get(v, '$.a', 'BIGINT') % 2 = 0",
        "SELECT count(*) FROM shredded WHERE TRY_CAST(v:b:c AS VARCHAR) LIKE '%7'",
    ] {
        let scan = run(&ctx, query).await?;
        assert!(
            scan.bytes_scanned * 10 < whole.bytes_scanned,
            "{query}: {scan:?} vs {}",
            whole.bytes_scanned
        );
    }

    // `pad` is not shredded, so it is read from `value`: most of the file
    let scan = run(&ctx, "SELECT max(v:pad::varchar) FROM shredded").await?;
    assert!(scan.bytes_scanned > file_size / 2, "{scan:?}");
    Ok(())
}

/// The shredded file of a table whose files have different layouts is also
/// read narrowly: the getter reads the file's own layout.
#[tokio::test]
async fn reads_only_shredded_columns_of_mixed_layouts() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let ctx = setup(dir.path(), false).await?;
    let query = "SELECT sum(try_variant_get(v, '$.a', 'BIGINT')) FROM";
    let shredded = run(&ctx, &format!("{query} shredded")).await?;
    let unshredded = run(&ctx, &format!("{query} unshredded")).await?;
    let mixed = run(&ctx, &format!("{query} mixed")).await?;
    assert_eq!(
        mixed.bytes_scanned,
        shredded.bytes_scanned + unshredded.bytes_scanned
    );
    assert!(shredded.bytes_scanned * 10 < unshredded.bytes_scanned);
    Ok(())
}

/// A filter on a shredded path skips row groups using the statistics of the
/// `typed_value` column, unless some row stores its value in `value`.
#[tokio::test]
async fn prunes_row_groups() -> Result<()> {
    let dir = tempfile::tempdir()?;
    for pushdown_filters in [false, true] {
        let ctx = setup(dir.path(), pushdown_filters).await?;
        #[rustfmt::skip]
        let cases = [
            // (query, row groups left to read, result)
            //
            // Only the last row group has values > 9500
            ("SELECT count(*) FROM clean WHERE v:a::BIGINT > 9500", 1, 500),
            ("SELECT count(*) FROM clean WHERE try_variant_get(v, '$.a', 'BIGINT') > 9500", 1, 500),
            ("SELECT count(*) FROM clean WHERE TRY_CAST(v:a AS BIGINT) BETWEEN 4500 AND 5500", 2, 1001),
            ("SELECT count(*) FROM clean WHERE v:a::BIGINT = 0", 0, 0),
            // Nothing is NULL
            ("SELECT count(*) FROM clean WHERE v:a::BIGINT IS NULL", 0, 0),
            // Strings too
            ("SELECT count(*) FROM clean WHERE v:b:c::varchar = 's1'", 1, 1),
            // The row group with "oops" has to be read: its `a` may be anything
            ("SELECT count(*) FROM shredded WHERE try_variant_get(v, '$.a', 'BIGINT') > 9500", 2, 500),
            ("SELECT count(*) FROM shredded WHERE try_variant_get(v, '$.a', 'BIGINT') IS NULL", 1, 1),
            // Converting to another type than the shredded one: no statistics
            ("SELECT count(*) FROM clean WHERE v:a::INT > 9500", ROW_GROUPS, 500),
            ("SELECT count(*) FROM clean WHERE v:a::DOUBLE > 9500", ROW_GROUPS, 500),
            // Not shredded: no statistics
            ("SELECT count(*) FROM clean WHERE v:pad::varchar = 'x'", ROW_GROUPS, 0),
        ];
        for (query, row_groups, count) in cases {
            let message = format!("{query} (pushdown_filters={pushdown_filters})");
            let scan = run(&ctx, query).await?;
            assert_eq!(scan.row_groups_matched, row_groups, "{message}");
            assert_eq!(scan.output, vec![vec![count.to_string()]], "{message}");
        }
    }
    Ok(())
}
