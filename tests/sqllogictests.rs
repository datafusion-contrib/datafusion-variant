use arrow::array::{
    Array, ArrayRef, BinaryViewArray, Int32Array, Int64Array, StringArray, StructArray,
    new_null_array,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use datafusion_sqllogictest::{
    DFColumnType, DFOutput, DFSqlLogicTestError, TestContext, convert_batches,
    convert_schema_to_types,
};
use datafusion_variant::register_all;
use parquet_variant::{EMPTY_VARIANT_METADATA_BYTES, Uuid, Variant, VariantBuilderExt};
use parquet_variant_compute::{
    VariantArray, VariantArrayBuilder, VariantType, json_to_variant, shred_variant,
};
use sqllogictest::{AsyncDB, DBOutput, strict_column_validator};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// Share the temporary presentation adapter with dfv without adding a library API.
#[path = "../src/display_shim.rs"]
mod display;

// TODO: return to the upstream DataFusion runner once
// DataFusion's SLT runner uses registered extension formatters. Tobias's follow-up:
// https://github.com/apache/datafusion/pull/21714#discussion_r3160465335
struct TestDatabase {
    ctx: SessionContext,
}

#[async_trait::async_trait]
impl AsyncDB for TestDatabase {
    type Error = DFSqlLogicTestError;
    type ColumnType = DFColumnType;

    async fn run(&mut self, sql: &str) -> Result<DFOutput, Self::Error> {
        let df = self.ctx.sql(sql).await?;
        let types = convert_schema_to_types(df.schema().fields());
        let mut options = self.ctx.state().config_options().format.clone();
        // SLTs always represent SQL NULL as NULL, independently of CLI settings.
        options.null = "NULL".into();
        let df = display::format_for_display(df, options)?;
        let schema = Arc::clone(df.schema().inner());
        let rows = convert_batches(&schema, df.collect().await?, false)?;
        if rows.is_empty() && types.is_empty() {
            Ok(DBOutput::StatementComplete(0))
        } else {
            Ok(DBOutput::Rows { types, rows })
        }
    }

    fn engine_name(&self) -> &str {
        "DataFusion"
    }

    async fn sleep(duration: std::time::Duration) {
        tokio::time::sleep(duration).await;
    }

    async fn shutdown(&mut self) {}
}

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

// SQL constructors only produce valid Variants. Supply raw storage here and assert
// UDF behavior in is_valid_variant.slt; Arrow owns exhaustive decoder coverage.
fn register_variant_validation_fixture(ctx: &SessionContext) -> datafusion::error::Result<()> {
    let empty = EMPTY_VARIANT_METADATA_BYTES;
    // metadata, encoded value, typed value, outer validity
    let rows = [
        (empty, Some(&[][..]), None, true),              // Empty payload.
        (empty, Some(&[3, 1, 0, 2, 24, 0]), None, true), // Nested truncation.
        (&[1], Some(&[0]), None, true),                  // Malformed metadata.
        (empty, Some(&[5, 255]), None, true),            // Invalid UTF-8.
        (empty, None, Some(3), true),                    // Missing raw / present typed value.
        (empty, Some(&[12, 1]), Some(4), true),          // Conflicting representations.
        (&[], Some(&[]), None, false),                   // SQL NULL masks malformed storage.
    ];
    let metadata: ArrayRef = Arc::new(BinaryViewArray::from_iter_values(rows.iter().map(|r| r.0)));
    let value: ArrayRef = Arc::new(BinaryViewArray::from_iter(rows.iter().map(|r| r.1)));
    let typed: ArrayRef = Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.2)));
    let nulls: NullBuffer = rows.iter().map(|r| r.3).collect();
    let mut columns: Vec<(FieldRef, ArrayRef)> = vec![(
        Arc::new(Field::new("id", DataType::Int32, false)),
        Arc::new(Int32Array::from_iter_values(1..=rows.len() as i32)),
    )];
    for (name, typed) in [
        ("raw", None),
        (
            "null_typed",
            Some(new_null_array(&DataType::Null, rows.len())),
        ),
        ("shredded", Some(typed)),
    ] {
        let mut children = vec![("metadata", metadata.clone()), ("value", value.clone())];
        if let Some(typed) = typed {
            children.push(("typed_value", typed));
        }
        let (child_fields, children, _) = StructArray::try_from(children)?.into_parts();
        let array = StructArray::new(child_fields, children, Some(nulls.clone()));
        let variant = VariantArray::try_new(&array)?;
        columns.push((Arc::new(variant.field(name)), Arc::new(array)));
    }
    ctx.register_batch(
        "variant_validation_inputs",
        RecordBatch::from(StructArray::from(columns)),
    )?;
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

        let mut ctx =
            if let Some(test_ctx) = TestContext::try_new_for_test_file(&relative_path).await {
                test_ctx.session_ctx().clone()
            } else {
                SessionContext::new()
            };

        register_all(&mut ctx)?;

        if relative_path == Path::new("is_valid_variant.slt")
            || relative_path == Path::new("variant_explode.slt")
            || relative_path == Path::new("variant_display.slt")
        {
            register_variant_validation_fixture(&ctx)?;
        }

        if relative_path == Path::new("variant_sql_type.slt")
            || relative_path == Path::new("is_valid_variant.slt")
            || relative_path == Path::new("variant_explode.slt")
            || relative_path == Path::new("variant_display.slt")
        {
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

        if relative_path == Path::new("is_valid_variant.slt")
            || relative_path == Path::new("is_variant_null.slt")
            || relative_path == Path::new("variant_sql_type.slt")
            || relative_path == Path::new("variant_explode.slt")
            || relative_path == Path::new("variant_display.slt")
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

        let mut runner =
            sqllogictest::Runner::new(|| async { Ok(TestDatabase { ctx: ctx.clone() }) });

        runner.with_column_validator(strict_column_validator);
        runner.run_file_async(test_file).await?;
    }

    Ok(())
}
