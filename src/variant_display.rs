// SPDX-License-Identifier: Apache-2.0

use std::{fmt::Write, sync::Arc};

use arrow::{
    array::{Array, AsArray, BinaryViewArray},
    buffer::NullBuffer,
    compute::cast,
    util::display::{ArrayFormatter, DisplayIndex, FormatOptions, FormatResult},
};
use arrow_schema::{DataType, extension::ExtensionType};
use datafusion::{
    catalog::Session,
    common::{Result, types::DFExtensionType},
    logical_expr::registry::ExtensionTypeRegistration,
    prelude::SessionContext,
};
use parquet_variant::Variant;
use parquet_variant_compute::{VariantArray, VariantType, unshred_variant};
use parquet_variant_json::VariantToJson;

pub(crate) fn register(ctx: &SessionContext) -> Result<()> {
    ctx.state()
        .extension_type_registry()
        .add_extension_type_registration(ExtensionTypeRegistration::new_arc(
            VariantType::NAME,
            |data_type, metadata| {
                VariantType::try_new(data_type, VariantType::deserialize_metadata(metadata)?)?;
                Ok(Arc::new(VariantDisplay(data_type.clone())))
            },
        ))?;
    Ok(())
}

// TODO: reuse DataFusion's canonical Variant extension implementation once it
// provides this formatter: https://github.com/apache/datafusion/issues/21144
#[derive(Debug)]
struct VariantDisplay(DataType);

impl DFExtensionType for VariantDisplay {
    // TODO: expose the Variant type name once schema introspection supports it:
    // https://github.com/apache/datafusion/issues/26866
    fn storage_type(&self) -> DataType {
        self.0.clone()
    }

    fn serialize_metadata(&self) -> Option<String> {
        VariantType.serialize_metadata()
    }

    fn create_array_formatter<'fmt>(
        &self,
        array: &'fmt dyn Array,
        options: &FormatOptions<'fmt>,
    ) -> Result<Option<ArrayFormatter<'fmt>>> {
        // Reconstruct partially shredded objects before serialization so fields
        // remaining in the binary value column are included as well.
        let array = VariantArray::try_new(array)?;
        let array = if array
            .typed_value_column()
            .is_some_and(|typed| typed.data_type() != &DataType::Null)
        {
            unshred_variant(&array)?
        } else {
            array
        };
        let metadata = cast(array.metadata_column(), &DataType::BinaryView)?;
        let values = cast(array.value_column(), &DataType::BinaryView)?;
        Ok(Some(ArrayFormatter::new(
            Box::new(VariantValues {
                metadata: metadata.as_binary_view().clone(),
                values: values.as_binary_view().clone(),
                nulls: array.nulls().cloned(),
                null: options.null(),
            }),
            options.safe(),
        )))
    }
}

struct VariantValues<'a> {
    metadata: BinaryViewArray,
    values: BinaryViewArray,
    nulls: Option<NullBuffer>,
    null: &'a str,
}

impl DisplayIndex for VariantValues<'_> {
    fn write(&self, idx: usize, f: &mut dyn Write) -> FormatResult {
        if self.nulls.as_ref().is_some_and(|nulls| nulls.is_null(idx)) {
            f.write_str(self.null)?;
        } else if self.values.is_null(idx) {
            // A missing required value is interpreted as Variant null.
            f.write_str("null")?;
        } else {
            // Validate before decoding: VariantArray::value can panic on malformed bytes.
            let value = Variant::try_new(self.metadata.value(idx), self.values.value(idx))?;
            f.write_str(&value.to_json_string()?)?;
        }
        Ok(())
    }
}
