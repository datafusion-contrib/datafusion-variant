use std::sync::Arc;

use arrow::array::{Array, cast::AsArray};
use arrow_schema::extension::ExtensionType;
use arrow_schema::{DataType, Field};
use datafusion::common::exec_datafusion_err;
use datafusion::error::Result;
use datafusion::{common::exec_err, scalar::ScalarValue};
use parquet_variant_compute::VariantType;

#[cfg(test)]
use parquet_variant_compute::{VariantArray, VariantArrayBuilder};

#[cfg(test)]
use parquet_variant_json::JsonToVariant;

pub fn try_field_as_variant_array(field: &Field) -> Result<()> {
    if !matches!(field.extension_type(), VariantType) {
        return exec_err!("field does not have extension type VariantType");
    }

    let variant_type = VariantType;
    variant_type.supports_data_type(field.data_type())?;

    Ok(())
}

pub fn try_field_as_binary(field: &Field) -> Result<()> {
    if !matches!(
        field.data_type(),
        DataType::Binary | DataType::BinaryView | DataType::LargeBinary
    ) {
        return exec_err!("expected binary field");
    }

    Ok(())
}

pub fn try_parse_binary_scalar(scalar: &ScalarValue) -> Result<Option<&Vec<u8>>> {
    let b = match scalar {
        ScalarValue::Binary(b) | ScalarValue::BinaryView(b) | ScalarValue::LargeBinary(b) => b,
        unsupported => {
            return exec_err!(
                "expected binary scalar value, got data type: {}",
                unsupported.data_type()
            );
        }
    };

    Ok(b.as_ref())
}

pub fn try_parse_binary_columnar(array: &Arc<dyn Array>) -> Result<Vec<Option<&[u8]>>> {
    if let Some(binary_array) = array.as_binary_opt::<i32>() {
        return Ok(binary_array.into_iter().collect::<Vec<_>>());
    }

    if let Some(binary_view_array) = array.as_binary_view_opt() {
        return Ok(binary_view_array.into_iter().collect::<Vec<_>>());
    }

    if let Some(large_binary_array) = array.as_binary_opt::<i64>() {
        return Ok(large_binary_array.into_iter().collect::<Vec<_>>());
    }

    Err(exec_datafusion_err!("expected binary array"))
}

// test related methods

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
