use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use datafusion::common::exec_err;
use datafusion::error::Result;
use parquet_variant_compute::VariantType;

pub fn is_variant_array(field: &Field) -> Result<()> {
    if !matches!(field.extension_type(), VariantType) {
        return exec_err!("field does not have extension type VariantType");
    }

    let variant_type = VariantType;
    variant_type.supports_data_type(field.data_type())?;

    Ok(())
}
