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
//!
//! Checks that readers do not need to see can be wrapped in an HTML comment
//! (`<!--` ... `-->`). The fences inside still run and are compared.
//!
//! Require every statement in a fence to fail by adding
//! `<!-- readme-test: error: message -->` before it. Put the marker on its own
//! line before a hidden check's `<!--`, since its `-->` would end the comment.
//!
//! The test-only `residual_json(variant[, field, ...])` function shows the
//! residual stored at a shredded field path as JSON.

use anyhow::{Context, Result, bail, ensure};
use arrow::array::{Array, ArrayRef, AsArray, StructArray};
use arrow_schema::{DataType, Field, Fields};
use datafusion::common::{exec_datafusion_err, exec_err};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::prelude::*;
use datafusion::sql::parser::{DFParser, Statement};
use datafusion_variant::register_all;
use parquet_variant_compute::variant_to_json;
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
    let mut expect_error = None;
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
            ctx.register_udf(ScalarUDF::from(ResidualJson {
                signature: Signature::variadic_any(Volatility::Immutable),
            }));
        }
        if let Some(message) = line
            .strip_prefix("<!-- readme-test: error:")
            .and_then(|text| text.strip_suffix("-->"))
        {
            ensure!(
                expect_error.is_none(),
                "{name}:{line_number}: repeated error marker"
            );
            ensure!(
                !message.trim().is_empty(),
                "{name}:{line_number}: missing expected error"
            );
            expect_error = Some(message.trim());
            continue;
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
        let expected_error = expect_error.take();
        ensure!(
            expected_error.is_none() || language == "sql",
            "{name}:{line_number}: error marker must precede SQL"
        );
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
                    let result = async {
                        let plan = ctx.state().statement_to_plan(statement).await?;
                        let df = ctx.execute_logical_plan(plan).await?;
                        if !is_copy && !df.schema().fields().is_empty() {
                            // Exercise the session's registered extension formatters.
                            df.to_string().await.map(Some)
                        } else {
                            df.collect().await.map(|_| None)
                        }
                    }
                    .await;
                    match (expected_error, result) {
                        (Some(message), Err(error)) => {
                            ensure!(
                                error.to_string().contains(message),
                                "{name}:{line_number}: expected error containing {message:?}, got: {error}"
                            );
                            checked += 1;
                        }
                        (Some(message), Ok(_)) => {
                            bail!("{name}:{line_number}: expected error containing {message:?}")
                        }
                        (None, result) => {
                            let table = result
                                .with_context(|| format!("{name}:{line_number}: executing SQL"))?;
                            pending.extend(table.map(|table| (line_number, table)));
                        }
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
    ensure!(expect_error.is_none(), "{name}: unused error marker");
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

/// SQL cannot show a shredded residual otherwise: it is a binary column inside
/// the shredded struct.
#[derive(Debug, PartialEq, Eq, Hash)]
struct ResidualJson {
    signature: Signature,
}

impl ScalarUDFImpl for ResidualJson {
    fn name(&self) -> &str {
        "residual_json"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::common::Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(
        &self,
        args: ScalarFunctionArgs,
    ) -> datafusion::common::Result<ColumnarValue> {
        let Some((variant, path)) = args.args.split_first() else {
            return exec_err!("residual_json expects a Variant column");
        };
        let variant = variant.to_array(args.number_rows)?;
        let variant = variant.as_struct();
        let mut node = variant;
        for field in path {
            let ColumnarValue::Scalar(field) = field else {
                return exec_err!("residual_json path must be string literals");
            };
            let Some(Some(field)) = field.try_as_str() else {
                return exec_err!("residual_json path must be string literals");
            };
            node = node
                .column_by_name("typed_value")
                .and_then(|typed| typed.as_struct_opt())
                .and_then(|typed| typed.column_by_name(field))
                .and_then(|field| field.as_struct_opt())
                .ok_or_else(|| exec_datafusion_err!("no shredded field '{field}'"))?;
        }
        let missing = |column| exec_datafusion_err!("residual_json input has no {column}");
        let metadata = variant
            .column_by_name("metadata")
            .ok_or_else(|| missing("metadata"))?;
        let value = node
            .column_by_name("value")
            .ok_or_else(|| missing("value"))?;
        let fields = Fields::from(vec![
            Field::new("metadata", metadata.data_type().clone(), false),
            Field::new("value", value.data_type().clone(), true),
        ]);
        // Rows without a residual are SQL NULL, not Variant null.
        let residual: ArrayRef = Arc::new(StructArray::try_new(
            fields,
            vec![metadata.clone(), value.clone()],
            value.nulls().cloned(),
        )?);
        Ok(ColumnarValue::Array(Arc::new(variant_to_json(&residual)?)))
    }
}
