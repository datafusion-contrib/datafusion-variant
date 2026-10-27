use arrow::array::{Array, ArrayRef, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::*;
use datafusion_sqllogictest::{DataFusion, TestContext};
use datafusion_variant::{VariantTypePlanner, register_all};
use indicatif::ProgressBar;
use parquet_variant::{Uuid, Variant, VariantBuilderExt};
use parquet_variant_compute::{VariantArrayBuilder, VariantType, json_to_variant, shred_variant};
use sqllogictest::strict_column_validator;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn register_shredded_null_fixture(
    ctx: &SessionContext,
    name: &str,
    rows: Vec<Option<&str>>,
    shredding_type: DataType,
) -> datafusion::error::Result<()> {
    let json: ArrayRef = Arc::new(StringArray::from(rows));
    let original = json_to_variant(&json)?;
    let shredded = shred_variant(&original, &shredding_type)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        original.field("original"),
        shredded.field("v"),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(1..=json.len() as i32)),
            Arc::new(original.into_inner()),
            Arc::new(shredded.into_inner()),
        ],
    )?;
    ctx.register_batch(name, batch)?;
    Ok(())
}

fn register_binary_variant_fixture(ctx: &SessionContext) -> datafusion::error::Result<()> {
    let json: ArrayRef = Arc::new(StringArray::from(vec![Some(r#"{"x":7}"#), None]));
    let original = json_to_variant(&json)?;
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for (name, binary_type) in [("b", DataType::Binary), ("l", DataType::LargeBinary)] {
        let storage = DataType::Struct(
            vec![
                Field::new("metadata", binary_type.clone(), false),
                Field::new("value", binary_type, true),
            ]
            .into(),
        );
        columns.push(arrow::compute::cast(original.inner(), &storage)?);
        fields.push(Field::new(name, storage, true).with_extension_type(VariantType));
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
    ctx.register_batch("binary_variant_inputs", batch)?;
    Ok(())
}

#[tokio::test]
async fn run_sqllogictests() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let test_files_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("./tests/test_files");

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
        println!("Running test file: {test_file:?}");

        let relative_path = test_file
            .strip_prefix(&test_files_dir)
            .unwrap_or(&test_file)
            .to_path_buf();

        let ctx = if let Some(test_ctx) = TestContext::try_new_for_test_file(&relative_path).await {
            test_ctx.session_ctx().clone()
        } else {
            SessionContext::new()
        };

        let state = SessionStateBuilder::new_from_existing(ctx.state())
            .with_type_planner(Arc::new(VariantTypePlanner))
            .build();
        let mut ctx = SessionContext::new_with_state(state);

        register_all(&mut ctx)?;

        if relative_path == Path::new("variant_sql_type.slt") {
            register_binary_variant_fixture(&ctx)?;
        }

        if relative_path == Path::new("schema_of_variant.slt") {
            // UUID casting is pending upstream; variant_list_construct accepts
            // only scalar arguments, so also build the UUID array cases here.
            let uuid = Variant::Uuid(Uuid::from_bytes([1; 16]));
            let mut builder = VariantArrayBuilder::new(3);
            builder.append_variant(uuid.clone());
            for other in [Variant::Null, Variant::from("x")] {
                let mut list = builder.try_new_list()?;
                list.append_value(uuid.clone());
                list.append_value(other);
                list.finish();
            }
            let variants = builder.build();
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                variants.field("v"),
            ]));
            ctx.register_batch(
                "variant_uuid_inputs",
                RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(Int32Array::from(vec![1, 2, 3])),
                        Arc::new(variants.into_inner()),
                    ],
                )?,
            )?;
        }

        if relative_path == Path::new("is_variant_null.slt")
            || relative_path == Path::new("variant_sql_type.slt")
        {
            register_shredded_null_fixture(
                &ctx,
                "shredded_null_objects",
                vec![
                    None,
                    Some("null"),
                    Some("{}"),
                    Some(r#"{"a":null}"#),
                    Some(r#"{"a":1}"#),
                    Some(r#"{"b":2}"#),
                    Some(r#"{"a":null,"b":2}"#),
                ],
                DataType::Struct(vec![Field::new("a", DataType::Int64, true)].into()),
            )?;
            register_shredded_null_fixture(
                &ctx,
                "shredded_null_lists",
                vec![
                    None,
                    Some("null"),
                    Some("[]"),
                    Some("[null]"),
                    Some("[1]"),
                    Some("[1,null]"),
                    Some("[null,2]"),
                ],
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            )?;
        }

        let pb = ProgressBar::new(24);

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
