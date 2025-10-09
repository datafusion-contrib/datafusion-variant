use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use datafusion::common::exec_err;
use datafusion::error::Result;
use parquet_variant_compute::VariantType;

#[cfg(test)]
use parquet_variant_compute::{VariantArray, VariantArrayBuilder};

#[cfg(test)]
use parquet_variant_json::JsonToVariant;

pub fn is_variant_array(field: &Field) -> Result<()> {
    if !matches!(field.extension_type(), VariantType) {
        return exec_err!("field does not have extension type VariantType");
    }

    let variant_type = VariantType;
    variant_type.supports_data_type(field.data_type())?;

    Ok(())
}

#[cfg(test)]
pub fn build_variant_array_from_json(value: &serde_json::Value) -> VariantArray {
    let json_str = value.to_string();
    let mut builder = VariantArrayBuilder::new(1);
    builder.append_json(json_str.as_str()).unwrap();

    builder.build()
}

#[cfg(test)]
#[allow(unused)]
pub fn build_variant_array_from_json_array(jsons: &[Option<serde_json::Value>]) -> VariantArray {
    let mut builder = VariantArrayBuilder::new(jsons.len());

    jsons.into_iter().for_each(|v| match v.as_ref() {
        Some(json) => builder.append_json(json.to_string().as_str()).unwrap(),
        None => builder.append_null(),
    });

    builder.build()
}
