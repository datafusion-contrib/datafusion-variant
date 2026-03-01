use anyhow::{Context, Result};
use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::*;
use datafusion_variant::{
    CastToVariantUdf, IsVariantNullUdf, JsonToVariantUdf, VariantExprPlanner, VariantGetUdf,
    VariantListConstruct, VariantListInsert, VariantObjectConstruct, VariantObjectInsert,
    VariantPretty, VariantToJsonUdf,
};
use flate2::read::GzDecoder;
use rustyline::error::ReadlineError;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Completer, Config, Editor, Helper, Highlighter, Hinter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::sync::Arc;
use std::time::Instant;

/// Helper for rustyline that provides multi-line SQL query support
#[derive(Helper, Completer, Highlighter, Hinter, Default)]
struct SqlHelper;

impl Validator for SqlHelper {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        let input = ctx.input();

        // Check if query is complete (ends with semicolon)
        if is_complete_query(input) {
            Ok(ValidationResult::Valid(None))
        } else {
            Ok(ValidationResult::Incomplete)
        }
    }
}

fn is_complete_query(input: &str) -> bool {
    let trimmed = input.trim();

    if trimmed.is_empty() {
        return true;
    }

    if trimmed == "quit" || trimmed == "q" {
        return true;
    }

    trimmed.ends_with(';')
}

#[tokio::main]
async fn main() -> Result<()> {
    // todo: we can let users pass in arbitrary files to load
    // for now, we'll directly use bluesky's data
    // let args = env::args();

    // let file_path = args
    //     .next()
    //     .unwrap_or("data/bluesky/file_0001.json.gz".into());

    let file_path = "data/bluesky/file_0001.json.gz";

    println!("Loading data from: {}\n", file_path);

    // load data
    let json_strings = {
        let load_start = Instant::now();

        let file = File::open(file_path).context("make sure to run ./bin/download_data.sh")?;
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);

        let mut json_strings = Vec::new();
        let mut line_count = 0;

        for line in reader.lines() {
            let line = line?;
            json_strings.push(line);
            line_count += 1;

            if line_count % 100000 == 0 {
                print!("\rLoading: {} rows...", line_count);
                io::stdout().flush()?;
            }
        }

        let load_duration = load_start.elapsed();
        println!(
            "\r✓ Loaded {} rows in {:.2}s",
            line_count,
            load_duration.as_secs_f64()
        );

        json_strings
    };

    let ctx = {
        let setup_start = Instant::now();

        let mut ctx = SessionContext::new();
        let schema = Schema::new(vec![Field::new("json_data", DataType::Utf8, false)]);
        let string_array: ArrayRef = Arc::new(StringArray::from(json_strings));
        let batch = RecordBatch::try_new(Arc::new(schema), vec![string_array])?;

        let provider =
            datafusion::datasource::MemTable::try_new(batch.schema(), vec![vec![batch]])?;

        ctx.register_table("bsky", Arc::new(provider))?;

        // register variant udfs
        ctx.register_udf(ScalarUDF::new_from_impl(VariantToJsonUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(JsonToVariantUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(CastToVariantUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(IsVariantNullUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantPretty::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantObjectConstruct::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantListConstruct::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantListInsert::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantObjectInsert::default()));
        ctx.register_expr_planner(Arc::new(VariantExprPlanner))?;

        let setup_duration = setup_start.elapsed();
        println!(
            "✓ DataFusion context ready in {:.3}s\n",
            setup_duration.as_secs_f64()
        );

        ctx
    };

    println!("interactive query mode. type 'q' or 'quit' or ctrl+c to exit");
    println!("Available table: bsky (json_data: Utf8)");
    println!("Tip: Press Enter without ';' to continue query on next line\n");

    let config = Config::builder().auto_add_history(true).build();
    let helper = SqlHelper::default();

    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(helper));

    loop {
        let readline = rl.readline("> ");

        match readline {
            Ok(line) => {
                let query = line.trim();

                if query.is_empty() {
                    continue;
                }

                if query == "quit" || query == "q" {
                    println!("bye");
                    break;
                }

                if let Err(e) = run_query(&ctx, query).await {
                    eprintln!("\x1b[31;1merror:\x1b[0m {}", e);
                }
                println!();
            }
            Err(ReadlineError::Interrupted) => {
                println!("bye");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("bye");
                break;
            }
            Err(err) => {
                eprintln!("\x1b[31;1merror:\x1b[0m {}", err);
                break;
            }
        }
    }

    Ok(())
}

async fn run_query(ctx: &SessionContext, query: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}\n", query);

    let plan_start = Instant::now();
    let df = ctx.sql(query).await?;
    let plan_duration = plan_start.elapsed();

    let exec_start = Instant::now();
    df.show().await?;
    let exec_duration = exec_start.elapsed();

    println!("\n--- results ---");
    println!(
        "planning time:   {:.3}ms",
        plan_duration.as_secs_f64() * 1000.0
    );
    println!("exec time:  {:.3}ms", exec_duration.as_secs_f64() * 1000.0);
    println!(
        "total time:      {:.3}ms",
        (plan_duration + exec_duration).as_secs_f64() * 1000.0
    );

    Ok(())
}
