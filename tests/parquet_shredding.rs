// SPDX-License-Identifier: Apache-2.0

use std::fs::File;

use datafusion::{common::Result, prelude::*};
use datafusion_variant::register_all;
use parquet::{
    arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    basic::{LogicalType, Type},
};
use parquet_variant_compute::VariantType;

// parquet_metadata exposes column chunks, so it cannot inspect the schema of
// an empty file or the VARIANT annotation on its parent group. All SQL-visible
// behavior, including nonempty physical layouts, is covered in the SLT.
#[tokio::test]
async fn empty_file_variant_schema() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut ctx = SessionContext::new();
    register_all(&mut ctx)?;
    for parallel in [false, true] {
        let path = dir.path().join(format!("empty_{parallel}.parquet"));
        ctx.sql(&format!(
            r#"COPY (SELECT parse_json('{{"a":1}}') AS v WHERE false)
               TO '{}' STORED AS PARQUET
               OPTIONS ('variant.shredding' '{{"v":"STRUCT<a BIGINT>"}}',
                        'allow_single_file_parallelism' '{parallel}')"#,
            path.display()
        ))
        .await?
        .collect()
        .await?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?;
        assert!(
            reader
                .schema()
                .field(0)
                .has_valid_extension_type::<VariantType>()
        );
        assert!(matches!(
            reader.parquet_schema().root_schema().get_fields()[0]
                .get_basic_info()
                .logical_type_ref(),
            Some(LogicalType::Variant(_))
        ));
        assert!(
            reader
                .parquet_schema()
                .columns()
                .iter()
                .any(
                    |column| column.path().string() == "v.typed_value.a.typed_value"
                        && column.physical_type() == Type::INT64
                )
        );
    }
    Ok(())
}
