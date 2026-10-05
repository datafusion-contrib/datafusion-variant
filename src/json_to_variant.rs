// https://docs.databricks.com/gcp/en/sql/language-manual/functions/parse_json

use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields};
use datafusion::{
    common::exec_err,
    error::Result,
    functions::utils::make_scalar_function,
    logical_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    },
};
use parquet_variant_compute::{VariantType, json_to_variant};

/// Returns a Variant from a JSON string
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct JsonToVariantUdf {
    signature: Signature,
}

impl Default for JsonToVariantUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(
                TypeSignature::Uniform(
                    1,
                    vec![DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View],
                ),
                datafusion::logical_expr::Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for JsonToVariantUdf {
    fn name(&self) -> &str {
        "json_to_variant"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Struct(Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, false),
        ])))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        let [input] = args.arg_fields else {
            return exec_err!("expected 1 argument");
        };
        // Arrow preserves input SQL nulls; invalid JSON errors instead of adding nulls.
        let data_type = self.return_type(std::slice::from_ref(input.data_type()))?;
        Ok(Arc::new(
            Field::new(self.name(), data_type, input.is_nullable())
                .with_extension_type(VariantType),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(
            |args| {
                let [input] = args else {
                    return exec_err!("expected 1 argument");
                };
                Ok(Arc::new(json_to_variant(input)?.into_inner()))
            },
            vec![],
        )(&args.args)
    }
}
