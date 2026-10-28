//! Execute README SQL fences and compare their results with the next `text` fence.
//! Each level-two section starts a fresh session. Multiple queries may share one
//! expected table; setup statements return no table. Proposed SQL is explicitly
//! excluded with `<!-- readme-test: skip: reason -->` before its fence.

use anyhow::{Context, Result, ensure};
use arrow::util::display::FormatOptions;
use arrow::util::pretty::pretty_format_batches_with_options;
use datafusion::prelude::*;
use datafusion::sql::parser::DFParser;
use datafusion_variant::register_all;
use std::io::Write;

#[tokio::test]
async fn readme_sql() -> Result<()> {
    let output = tempfile::tempdir()?;
    let parquet_path = output.path().join("events.parquet");
    let parquet_literal = format!(
        "'{}'",
        parquet_path.display().to_string().replace('\'', "''")
    );
    let mut ctx = SessionContext::new();
    let mut pending = Vec::<(usize, String)>::new();
    let mut skip = None;
    let mut skipped = Vec::new();
    let mut executed = 0;
    let mut checked = 0;
    let mut lines = include_str!("../README.md").lines().enumerate();

    while let Some((index, line)) = lines.next() {
        let line_number = index + 1;
        if line.starts_with("## ") || line.starts_with("### ") {
            ensure!(
                pending.is_empty(),
                "README:{line_number}: query result has no expected table"
            );
        }
        if line.starts_with("## ") {
            // Like `dfv`, allow `SELECT ... FROM 'file.parquet'`.
            ctx = SessionContext::new().enable_url_table();
            register_all(&mut ctx)?;
        }
        if let Some(reason) = line
            .strip_prefix("<!-- readme-test: skip:")
            .and_then(|text| text.strip_suffix("-->"))
        {
            ensure!(skip.is_none(), "README:{line_number}: repeated skip marker");
            ensure!(
                !reason.trim().is_empty(),
                "README:{line_number}: missing skip reason"
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
        ensure!(closed, "README:{line_number}: unclosed code fence");
        if let Some(reason) = skip.take() {
            ensure!(
                language == "sql",
                "README:{line_number}: skip marker must precede SQL"
            );
            skipped.push((line_number, reason));
            continue;
        }
        match language {
            "sql" => {
                executed += 1;
                // Keep documented relative filenames without changing the process cwd.
                let sql = block.replace("'events.parquet'", &parquet_literal);
                let statements = DFParser::parse_sql(&sql)
                    .with_context(|| format!("README:{line_number}: parsing SQL"))?;
                for statement in statements {
                    let batches = async {
                        let plan = ctx.state().statement_to_plan(statement).await?;
                        ctx.execute_logical_plan(plan).await?.collect().await
                    }
                    .await
                    .with_context(|| format!("README:{line_number}: executing SQL"))?;
                    if batches.iter().any(|batch| batch.num_columns() > 0) {
                        // Show SQL NULL as `NULL`, as `dfv` does.
                        let options = FormatOptions::default().with_null("NULL");
                        let table = pretty_format_batches_with_options(&batches, &options)?;
                        pending.push((line_number, table.to_string()));
                    }
                }
            }
            "text" => {
                ensure!(
                    !pending.is_empty(),
                    "README:{line_number}: expected table has no query"
                );
                for (query_line, actual) in pending.drain(..) {
                    // Arrow's formatter checks values, headers, padding and borders together.
                    ensure!(
                        actual.trim() == block.trim(),
                        "README:{query_line}: result differs from table at line {line_number}\nExpected:\n{block}\nActual:\n{actual}"
                    );
                    checked += 1;
                }
            }
            _ => {}
        }
    }
    ensure!(skip.is_none(), "README: unused skip marker");
    ensure!(
        pending.is_empty(),
        "README: query result has no expected table"
    );
    ensure!(checked > 0, "README: no query results checked");
    // Write directly so the summary is visible even with libtest's default capture.
    let mut report = std::io::stderr().lock();
    writeln!(
        report,
        "README SQL: {executed} blocks run, {checked} results checked, {} skipped",
        skipped.len()
    )?;
    for (line, reason) in skipped {
        writeln!(report, "  README:{line}: skipped ({reason})")?;
    }
    Ok(())
}
