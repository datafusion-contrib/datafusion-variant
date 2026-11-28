// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow::array::{Array, cast::AsArray};
#[cfg(test)]
use arrow::array::{ArrayRef, StructArray};
use arrow_schema::{DataType, Field};
use datafusion::common::exec_datafusion_err;
use datafusion::error::Result;
use datafusion::{common::exec_err, scalar::ScalarValue};
use parquet_variant_compute::{VariantArray, VariantType};

#[cfg(test)]
use parquet_variant_compute::VariantArrayBuilder;

#[cfg(test)]
use parquet_variant_json::JsonToVariant;

#[cfg(test)]
use crate::variant_type::variant_field;

/// Checks that `field` is a Variant: it has the Variant extension type and a
/// valid storage schema (a struct with `metadata` and `value` or `typed_value`).
///
/// The shredded `typed_value` schema is checked by `VariantArray::try_new`.
pub fn try_field_as_variant_array(field: &Field) -> Result<()> {
    ensure(
        field.has_valid_extension_type::<VariantType>(),
        "field does not have extension type VariantType",
    )
}

pub fn _try_field_as_binary(field: &Field) -> Result<()> {
    match field.data_type() {
        DataType::Binary | DataType::BinaryView | DataType::LargeBinary => {}
        unsupported => return exec_err!("expected binary field, got {unsupported} field"),
    }

    Ok(())
}

pub fn try_parse_variant_scalar(scalar: &ScalarValue) -> Result<VariantArray> {
    let v = match scalar {
        ScalarValue::Struct(v) => v,
        unsupported => {
            return exec_err!(
                "expected variant scalar value, got data type: {}",
                unsupported.data_type()
            );
        }
    };

    VariantArray::try_new(v.as_ref()).map_err(Into::into)
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

pub fn try_parse_string_scalar(scalar: &ScalarValue) -> Result<Option<&String>> {
    let b = match scalar {
        ScalarValue::Utf8(s) | ScalarValue::Utf8View(s) | ScalarValue::LargeUtf8(s) => s,
        unsupported => {
            return exec_err!(
                "expected binary scalar value, got data type: {}",
                unsupported.data_type()
            );
        }
    };

    Ok(b.as_ref())
}

pub fn try_parse_string_columnar(array: &Arc<dyn Array>) -> Result<Vec<Option<&str>>> {
    if let Some(string_array) = array.as_string_opt::<i32>() {
        return Ok(string_array.into_iter().collect::<Vec<_>>());
    }

    if let Some(string_view_array) = array.as_string_view_opt() {
        return Ok(string_view_array.into_iter().collect::<Vec<_>>());
    }

    if let Some(large_string_array) = array.as_string_opt::<i64>() {
        return Ok(large_string_array.into_iter().collect::<Vec<_>>());
    }

    Err(exec_datafusion_err!("expected string array"))
}

/// This is similar to anyhow's ensure! macro
/// If the `pred` fails, it will return a DataFusionError
pub fn ensure(pred: bool, err_msg: &str) -> Result<()> {
    if !pred {
        return exec_err!("{}", err_msg);
    }

    Ok(())
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
pub fn variant_scalar_from_json(json: serde_json::Value) -> ScalarValue {
    let mut builder = VariantArrayBuilder::new(1);
    builder.append_json(json.to_string().as_str()).unwrap();
    ScalarValue::Struct(Arc::new(builder.build().into()))
}

#[cfg(test)]
pub fn variant_array_from_json_rows(json_rows: &[serde_json::Value]) -> ArrayRef {
    let mut builder = VariantArrayBuilder::new(json_rows.len());
    for value in json_rows {
        builder.append_json(value.to_string().as_str()).unwrap();
    }
    let variant_array: StructArray = builder.build().into();
    Arc::new(variant_array) as ArrayRef
}

#[cfg(test)]
pub fn standard_variant_get_arg_fields() -> Vec<Arc<Field>> {
    vec![
        Arc::new(variant_field("input")),
        Arc::new(Field::new("path", DataType::Utf8, true)),
    ]
}

#[cfg(test)]
#[allow(unused)]
pub fn build_variant_array_from_json_array(jsons: &[Option<serde_json::Value>]) -> VariantArray {
    let mut builder = VariantArrayBuilder::new(jsons.len());

    jsons.iter().for_each(|v| match v.as_ref() {
        Some(json) => builder.append_json(json.to_string().as_str()).unwrap(),
        None => builder.append_null(),
    });

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Fields;
    use arrow_schema::extension::{EXTENSION_TYPE_NAME_KEY, ExtensionType};
    use std::collections::HashMap;

    fn tagged_as_variant(data_type: DataType) -> Field {
        // `with_extension_type` panics on invalid storage, so set the name directly.
        let name = (
            EXTENSION_TYPE_NAME_KEY.to_string(),
            VariantType::NAME.to_string(),
        );
        Field::new("v", data_type, true).with_metadata(HashMap::from([name]))
    }

    #[test]
    fn variant_field_requires_valid_storage() {
        try_field_as_variant_array(&variant_field("v")).unwrap();

        let untagged = Field::new("v", variant_field("v").data_type().clone(), true);
        try_field_as_variant_array(&untagged).unwrap_err();

        let empty_struct = tagged_as_variant(DataType::Struct(Fields::empty()));
        try_field_as_variant_array(&empty_struct).unwrap_err();

        let metadata_only = Fields::from(vec![Field::new("metadata", DataType::BinaryView, false)]);
        try_field_as_variant_array(&tagged_as_variant(DataType::Struct(metadata_only)))
            .unwrap_err();
    }
}
