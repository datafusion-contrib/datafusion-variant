use arrow_schema::{extension::ExtensionType, ArrowError, DataType};

pub struct VariantExtensionType;

impl ExtensionType for VariantExtensionType {
    const NAME: &'static str = "arrow-rs.variant";

    type Metadata = ();

    fn metadata(&self) -> &Self::Metadata {
        &()
    }

    fn serialize_metadata(&self) -> Option<String> {
        None
    }

    fn deserialize_metadata(_metadata: Option<&str>) -> Result<Self::Metadata, ArrowError> {
        Ok(())
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        match data_type {
            DataType::BinaryView => Ok(()),
            _ => Err(ArrowError::InvalidArgumentError(
                "Variant must be BinaryView".to_string(),
            )),
        }
    }

    fn try_new(data_type: &DataType, _metadata: Self::Metadata) -> Result<Self, ArrowError> {
        match data_type {
            DataType::BinaryView => Ok(VariantExtensionType),
            _ => Err(ArrowError::InvalidArgumentError(
                "Variant must be BinaryView".to_string(),
            )),
        }
    }
}
