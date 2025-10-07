// https://docs.databricks.com/gcp/en/sql/language-manual/functions/parse_json

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, LargeStringArray, StringArray, StringViewArray, StructArray};
use arrow_schema::{DataType, Field, Fields};
use datafusion::{
    common::{exec_datafusion_err, exec_err},
    error::Result as DataFusionResult,
    logical_expr::{ColumnarValue, ScalarUDFImpl, Signature, TypeSignature},
    scalar::ScalarValue,
};
use parquet_variant_compute::{VariantArrayBuilder, VariantType};
use parquet_variant_json::JsonToVariant as JsonToVariantExt;

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct JsonToVariantUDF {
    signature: Signature,
}

impl Default for JsonToVariantUDF {
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

impl ScalarUDFImpl for JsonToVariantUDF {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "json_to_variant"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
        Ok(DataType::BinaryView)
    }

    fn return_field_from_args(
        &self,
        args: datafusion::logical_expr::ReturnFieldArgs,
    ) -> datafusion::error::Result<Arc<Field>> {
        let arg_fields = args.arg_fields;

        if arg_fields.len() != 1 {
            return exec_err!(
                "Incorrect number of arguments for string_to_uuid. Expected json_to_variant(...)"
            );
        }

        let arg_field = &args.arg_fields[0];

        let is_argument_string = matches!(
            arg_field.data_type(),
            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8
        );

        if !is_argument_string {
            return exec_err!("Expected string argument");
        }

        let is_nullable = arg_field.is_nullable();

        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, is_nullable),
        ]));

        let return_field =
            Field::new(self.name(), data_type, is_nullable).with_extension_type(VariantType);

        Ok(Arc::new(return_field))
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> DataFusionResult<datafusion::logical_expr::ColumnarValue> {
        let arg = args
            .args
            .first()
            .ok_or_else(|| exec_datafusion_err!("empty argument, expected 1 argument"))?;

        let out = match arg {
            ColumnarValue::Scalar(scalar_value) => {
                let json_term = match scalar_value {
                    ScalarValue::Utf8(json)
                    | ScalarValue::Utf8View(json)
                    | ScalarValue::LargeUtf8(json) => json,
                    unsupported => {
                        return exec_err!("Unsupported data type {}", unsupported.data_type());
                    }
                };

                let mut builder = VariantArrayBuilder::new(1);

                match json_term {
                    Some(json_str) => builder.append_json(json_str.as_str())?,
                    None => builder.append_null(),
                }

                let struct_array: StructArray = builder.build().into();
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(struct_array)))
            }
            ColumnarValue::Array(arr) => match arr.data_type() {
                DataType::Utf8 => ColumnarValue::Array(from_utf8_arr(arr)?),
                DataType::LargeUtf8 => ColumnarValue::Array(from_large_utf8_arr(arr)?),
                DataType::Utf8View => ColumnarValue::Array(from_utf8view_arr(arr)?),
                _ => return exec_err!("Invalid data type {}", arr.data_type()),
            },
        };

        Ok(out)
    }
}

macro_rules! define_from_string_array {
    ($fn_name:ident, $array_type:ty) => {
        pub(crate) fn $fn_name(arr: &ArrayRef) -> DataFusionResult<ArrayRef> {
            let arr = arr
                .as_any()
                .downcast_ref::<$array_type>()
                .ok_or(exec_datafusion_err!(
                    "Unable to downcast array as expected by type."
                ))?;

            let mut builder = VariantArrayBuilder::new(arr.len());

            for v in arr {
                match v {
                    Some(json_str) => builder.append_json(json_str)?,
                    None => builder.append_null(),
                }
            }

            let variant_array: StructArray = builder.build().into();

            Ok(Arc::new(variant_array) as ArrayRef)
        }
    };
}

define_from_string_array!(from_utf8_arr, StringArray);
define_from_string_array!(from_utf8view_arr, StringViewArray);
define_from_string_array!(from_large_utf8_arr, LargeStringArray);

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::ScalarFunctionArgs;
    use parquet_variant::{Variant, VariantBuilder};
    use parquet_variant_compute::VariantArray;

    #[test]
    fn test_json_to_variant_udf_scalar_none() {
        let udf = JsonToVariantUDF::default();

        let json_input = ScalarValue::Utf8(None);

        let return_field = Arc::new(Field::new("result", DataType::BinaryView, true));
        let arg_field = Arc::new(Field::new("input", DataType::Utf8, true));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(json_input)],
            return_field: return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        match result {
            ColumnarValue::Scalar(ScalarValue::Struct(sv)) => {
                assert!(sv.is_null(0), "expected null struct");
            }
            _ => panic!("Expected null struct array result"),
        }
    }

    #[test]
    fn test_json_to_variant_udf_scalar_null() {
        let udf = JsonToVariantUDF::default();

        let json_input = ScalarValue::Utf8(Some("null".into()));

        let return_field = Arc::new(Field::new("result", DataType::BinaryView, true));
        let arg_field = Arc::new(Field::new("input", DataType::Utf8, true));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(json_input)],
            return_field: return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();
        match result {
            ColumnarValue::Scalar(ScalarValue::Struct(v)) => {
                let variant_array = VariantArray::try_new(v.as_ref()).unwrap();
                let variant = variant_array.value(0);
                assert_eq!(variant, Variant::from(()));
            }
            _ => panic!("Expected scalar BinaryView result"),
        }
    }

    #[test]
    fn test_json_to_variant_udf_scalar() {
        let udf = JsonToVariantUDF::default();

        let (expected_m, expected_v) = {
            let mut variant_builder = VariantBuilder::new();
            let mut object_builder = variant_builder.new_object();

            object_builder.insert("key", 123_u8);

            let mut inner_array_builder = object_builder.new_list("data");

            inner_array_builder.append_value(4u8);
            inner_array_builder.append_value(5u8);
            inner_array_builder.append_value("str");

            inner_array_builder.finish();

            object_builder.finish();

            variant_builder.finish()
        };

        let expected_variant = Variant::try_new(&expected_m, &expected_v).unwrap();

        let json_input =
            ScalarValue::Utf8(Some(r#"{"key": 123, "data": [4, 5, "str"]}"#.to_string()));

        let return_field = Arc::new(Field::new("result", DataType::BinaryView, true));
        let arg_field = Arc::new(Field::new("input", DataType::Utf8, true));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(json_input)],
            return_field: return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        match result {
            ColumnarValue::Scalar(ScalarValue::Struct(v)) => {
                let variant_array = VariantArray::try_new(v.as_ref()).unwrap();
                let variant = variant_array.value(0);
                assert_eq!(variant, expected_variant);
            }
            _ => panic!("Expected scalar BinaryView result"),
        }
    }

    #[test]
    fn test_json_to_variant_udf_scalar_number() {
        let udf = JsonToVariantUDF::default();

        let json_input = ScalarValue::Utf8(Some("123".to_string()));

        let return_field = Arc::new(Field::new("result", DataType::BinaryView, true));
        let arg_field = Arc::new(Field::new("input", DataType::Utf8, true));

        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(json_input)],
            return_field: return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        match result {
            ColumnarValue::Scalar(ScalarValue::Struct(v)) => {
                let variant_array = VariantArray::try_new(v.as_ref()).unwrap();
                let variant = variant_array.value(0);
                assert_eq!(variant, Variant::from(123_u8));
            }
            _ => panic!("Expected scalar BinaryView result"),
        }
    }
}
