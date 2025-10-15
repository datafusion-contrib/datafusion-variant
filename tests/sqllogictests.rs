use datafusion::{logical_expr::ScalarUDF, prelude::*};
use datafusion_sqllogictest::{DataFusion, TestContext};
use datafusion_variant::{
    CastToVariantUdf, IsVariantNullUdf, JsonToVariantUdf, VariantGetUdf, VariantPretty,
    VariantToJsonUdf,
};
use indicatif::ProgressBar;
use sqllogictest::strict_column_validator;
use std::path::PathBuf;

#[tokio::test]
async fn run_sqllogictests() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let test_files_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_files");

    let mut test_files: Vec<_> = std::fs::read_dir(&test_files_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "slt" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    test_files.sort();

    for test_file in test_files {
        println!("Running test file: {:?}", test_file);

        let relative_path = test_file
            .strip_prefix(&test_files_dir)
            .unwrap_or(&test_file)
            .to_path_buf();

        let ctx = if let Some(test_ctx) = TestContext::try_new_for_test_file(&relative_path).await {
            test_ctx.session_ctx().clone()
        } else {
            SessionContext::new()
        };

        // register variant udfs
        ctx.register_udf(ScalarUDF::new_from_impl(VariantToJsonUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(JsonToVariantUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(CastToVariantUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(IsVariantNullUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantPretty::default()));

        let pb = ProgressBar::hidden();

        let mut runner = sqllogictest::Runner::new(|| async {
            Ok(DataFusion::new(
                ctx.clone(),
                relative_path.clone(),
                pb.clone(),
            ))
        });

        runner.with_column_validator(strict_column_validator);
        runner.run_file_async(test_file).await?;
    }

    Ok(())
}
