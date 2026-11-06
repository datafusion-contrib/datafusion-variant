// SPDX-License-Identifier: Apache-2.0

//! Execute the SQL examples in the README and in `docs/*.md` and compare their
//! results with the copy/pasted output.
//!
//! Each `sql` fence is run, and its output must match the next `text` fence
//! exactly. For example, this Markdown passes:
//!
//! ```sql
//! SELECT parse_json('{"a": 1}') AS value;
//! ```
//!
//! ```text
//! +---------+
//! | value   |
//! +---------+
//! | {"a":1} |
//! +---------+
//! ```
//!
//! Statements that return no rows, such as `CREATE TABLE` and `SET`, need no
//! `text` output. Neither do `COPY` statements, whose row count is not useful
//! in documentation.
//!
//! Several queries may share one expected `text` output.
//!
//! Skip SQL by adding `<!-- readme-test: skip: reason -->` before its fence.

use anyhow::{Context, Result, ensure};
use datafusion::prelude::*;
use datafusion::sql::parser::{DFParser, Statement};
use datafusion_variant::register_all;
use std::{io::Write, sync::Arc};

#[tokio::test]
async fn readme_sql() -> Result<()> {
    check_markdown("README", include_str!("../README.md")).await
}

#[tokio::test]
async fn shredding_doc_sql() -> Result<()> {
    check_markdown("docs/shredding.md", include_str!("../docs/shredding.md")).await
}

async fn check_markdown(name: &str, markdown: &str) -> Result<()> {
    let output = tempfile::tempdir()?;
    let mut ctx = SessionContext::new();
    let mut pending = Vec::<(usize, String)>::new();
    let mut skip = None;
    let mut skipped = Vec::new();
    let mut executed = 0;
    let mut checked = 0;
    let mut lines = markdown.lines().enumerate();

    while let Some((index, line)) = lines.next() {
        let line_number = index + 1;
        if line.starts_with("## ") || line.starts_with("### ") {
            ensure!(
                pending.is_empty(),
                "{name}:{line_number}: query result has no expected table"
            );
        }
        if line.starts_with("## ") {
            // Like `dfv`, allow `SELECT ... FROM 'file.parquet'`.
            let config = SessionConfig::new().set_str("datafusion.format.null", "NULL");
            ctx = SessionContext::new_with_config(config).enable_url_table();
            register_all(&mut ctx)?;
            ctx.register_udtf(
                "parquet_metadata",
                Arc::new(datafusion_cli::functions::ParquetMetadataFunc {}),
            );
        }
        if let Some(reason) = line
            .strip_prefix("<!-- readme-test: skip:")
            .and_then(|text| text.strip_suffix("-->"))
        {
            ensure!(skip.is_none(), "{name}:{line_number}: repeated skip marker");
            ensure!(
                !reason.trim().is_empty(),
                "{name}:{line_number}: missing skip reason"
            );
            skip = Some(reason.trim());
            continue;
        }
        let Some(language) = line.strip_prefix("```") else {
            continue;
        };
        let mut block = String::new();
        let mut closed = false;
        for (_, line) in lines.by_ref() {
            if line == "```" {
                closed = true;
                break;
            }
            block.push_str(line);
            block.push('\n');
        }
        ensure!(closed, "{name}:{line_number}: unclosed code fence");
        if let Some(reason) = skip.take() {
            ensure!(
                language == "sql",
                "{name}:{line_number}: skip marker must precede SQL"
            );
            skipped.push((line_number, reason));
            continue;
        }
        match language {
            "sql" => {
                executed += 1;
                // Keep documented relative filenames without changing the process cwd:
                // every `'name.parquet'` literal is redirected into the temporary directory.
                let mut sql = String::new();
                for (index, part) in block.split('\'').enumerate() {
                    if index > 0 {
                        sql.push('\'');
                    }
                    if index % 2 == 1 && part.ends_with(".parquet") && !part.contains('/') {
                        let path = output.path().join(part);
                        sql.push_str(&path.display().to_string().replace('\'', "''"));
                    } else {
                        sql.push_str(part);
                    }
                }
                let statements = DFParser::parse_sql(&sql)
                    .with_context(|| format!("{name}:{line_number}: parsing SQL"))?;
                for statement in statements {
                    let is_copy = matches!(statement, Statement::CopyTo(_));
                    let df = async {
                        let plan = ctx.state().statement_to_plan(statement).await?;
                        ctx.execute_logical_plan(plan).await
                    }
                    .await
                    .with_context(|| format!("{name}:{line_number}: executing SQL"))?;
                    if !is_copy && !df.schema().fields().is_empty() {
                        // Exercise the session's registered extension formatters.
                        let table = df.to_string().await.with_context(|| {
                            format!("{name}:{line_number}: displaying SQL result")
                        })?;
                        pending.push((line_number, table));
                    } else {
                        df.collect().await?;
                    }
                }
            }
            "text" => {
                ensure!(
                    !pending.is_empty(),
                    "{name}:{line_number}: expected table has no query"
                );
                for (query_line, actual) in pending.drain(..) {
                    // Arrow's formatter checks values, headers, padding and borders together.
                    ensure!(
                        actual.trim() == block.trim(),
                        "{name}:{query_line}: result differs from table at line {line_number}\nExpected:\n{block}\nActual:\n{actual}"
                    );
                    checked += 1;
                }
            }
            _ => {}
        }
    }
    ensure!(skip.is_none(), "{name}: unused skip marker");
    ensure!(
        pending.is_empty(),
        "{name}: query result has no expected table"
    );
    ensure!(executed > 0, "{name}: no SQL blocks run");
    // Write directly so the summary is visible even with libtest's default capture.
    let mut report = std::io::stderr().lock();
    writeln!(
        report,
        "{name} SQL: {executed} blocks run, {checked} results checked, {} skipped",
        skipped.len()
    )?;
    for (line, reason) in skipped {
        writeln!(report, "  {name}:{line}: skipped ({reason})")?;
    }
    Ok(())
}
