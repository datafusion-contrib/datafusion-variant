// https://docs.databricks.com/gcp/en/sql/language-manual/functions/to_json

use std::sync::Arc;

use arrow::array::StringArray;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::{
    common::{exec_datafusion_err, exec_err},
    error::Result,
    logical_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
        TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet_variant_compute::VariantArray;
use parquet_variant_json::VariantToJson;

use crate::shared::is_variant_array;

/// Returns a JSON string from a Variant
///
/// ## Arguments
/// - expr: a DataType::Struct expression that represents a Variant
/// - options: an optional MAP (note, it seems arrow-rs' parquet-variant is pretty restrictive about the options)
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantToJsonUdf {
    signature: Signature,
}

impl Default for VariantToJsonUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(
                TypeSignature::OneOf(vec![TypeSignature::Any(1), TypeSignature::Any(2)]),
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for VariantToJsonUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_to_json"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        unimplemented!()
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [argument] = args.arg_fields else {
            return exec_err!("expected an argument");
        };

        is_variant_array(argument.as_ref())?;

        let nullable = argument.is_nullable();

        Ok(FieldRef::new(Field::new(
            self.name(),
            DataType::Utf8View,
            nullable,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arg = args
            .args
            .first()
            .ok_or_else(|| exec_datafusion_err!("empty argument, expected 1 argument"))?;

        let out = match arg {
            ColumnarValue::Scalar(scalar) => {
                let ScalarValue::Struct(variant_array) = scalar else {
                    return exec_err!("Unsupported data type: {}", scalar.data_type());
                };

                let variant_array = VariantArray::try_new(variant_array.as_ref())?;

                // in a scalar case, is it safe to assume variant_array has only 1 element?
                let v = variant_array.value(0);

                ColumnarValue::Scalar(ScalarValue::Utf8View(Some(v.to_json_string()?)))
            }
            ColumnarValue::Array(arr) => match arr.data_type() {
                DataType::Struct(_) => {
                    let variant_array = VariantArray::try_new(arr.as_ref())?;

                    // is there a reason why variant array doesn't implement Iterator?
                    let mut out = Vec::with_capacity(variant_array.len());

                    for i in 0..variant_array.len() {
                        let v = variant_array.value(i);
                        out.push(Some(v.to_json_string()?));
                    }

                    let out: StringArray = out.into();

                    ColumnarValue::Array(Arc::new(out))
                }
                unsupported => return exec_err!("Invalid data type: {unsupported}"),
            },
        };

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::{Field, Fields};
    use parquet_variant_compute::VariantArrayBuilder;
    use parquet_variant_json::JsonToVariant;
    use serde_json::Value;

    use super::*;

    #[test]
    fn test_variant_to_json_udf_scalar_complex() {
        let expected_json = serde_json::json!({
            "name": "norm",
            "age": 50,
            "list": [false, true, ()]
        });

        let json_str = expected_json.to_string();
        let mut builder = VariantArrayBuilder::new(1);
        builder.append_json(json_str.as_str()).unwrap();

        let input = builder.build().into();

        let variant_input = ScalarValue::Struct(Arc::new(input));

        let udf = VariantToJsonUdf::default();

        let return_field = Arc::new(Field::new("result", DataType::Utf8View, true));
        let arg_field = Arc::new(Field::new("input", DataType::Struct(Fields::empty()), true));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(variant_input)],
            return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(j))) = result else {
            panic!("expected valid json string")
        };

        let json: Value = serde_json::from_str(j.as_str()).unwrap();
        assert_eq!(json, expected_json);
    }
}
