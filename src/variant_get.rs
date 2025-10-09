use std::sync::Arc;

use arrow::array::{ArrayRef, StructArray};
use arrow_schema::{DataType, Field, Fields};
use datafusion::{
    common::exec_err,
    error::Result,
    logical_expr::{ColumnarValue, ScalarUDFImpl, Signature, TypeSignature, Volatility},
    scalar::ScalarValue,
};
use parquet_variant::VariantPath;
use parquet_variant_compute::{GetOptions, variant_get};

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantGetUdf {
    signature: Signature,
}

impl Default for VariantGetUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(
                TypeSignature::OneOf(vec![TypeSignature::Any(1), TypeSignature::Any(2)]),
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for VariantGetUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_get"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[arrow_schema::DataType]) -> Result<arrow_schema::DataType> {
        let fields = vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, true),
        ];

        Ok(DataType::Struct(Fields::from(fields)))
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let [variant_arg, variant_path] = args.args.as_slice() else {
            return exec_err!("expected 2 arguments");
        };

        let out = match (variant_arg, variant_path) {
            (ColumnarValue::Array(variant_array), ColumnarValue::Scalar(variant_path)) => {
                let variant_path = match variant_path.clone() {
                    ScalarValue::Utf8(path)
                    | ScalarValue::Utf8View(path)
                    | ScalarValue::LargeUtf8(path) => path,
                    unsupported => return exec_err!("unsupported data type: {unsupported}"),
                }
                .unwrap_or_default();

                let res = variant_get(
                    variant_array,
                    GetOptions::new_with_path(VariantPath::from(variant_path.as_str())),
                )?;

                ColumnarValue::Array(res)
            }
            (ColumnarValue::Scalar(scalar_variant), ColumnarValue::Scalar(variant_path)) => {
                let ScalarValue::Struct(variant_array) = scalar_variant else {
                    return exec_err!("expected struct array");
                };

                let variant_array = Arc::clone(variant_array) as ArrayRef;

                let variant_path = match variant_path.clone() {
                    ScalarValue::Utf8(path)
                    | ScalarValue::Utf8View(path)
                    | ScalarValue::LargeUtf8(path) => path,
                    unsupported => return exec_err!("unsupported data type: {unsupported}"),
                }
                .unwrap_or_default();

                let res = variant_get(
                    &variant_array,
                    GetOptions::new_with_path(VariantPath::from(variant_path.as_str())),
                )?
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .clone();

                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(res)))
            }
            (ColumnarValue::Array(_variant_array), ColumnarValue::Array(_variant_paths)) => {
                // i assume this would be reasonable? we'd essentially have to zip through each pair
                // but we would have a list of arrayrefs...
                todo!()
            }
            (ColumnarValue::Scalar(_scalar_value), ColumnarValue::Array(_array)) => {
                todo!("do we even support this case?")
            }
        };

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, BinaryViewArray};
    use arrow_schema::{Field, Fields};
    use datafusion::logical_expr::{ReturnFieldArgs, ScalarFunctionArgs};
    use parquet_variant::Variant;
    use parquet_variant_compute::{VariantArrayBuilder, VariantType};
    use parquet_variant_json::JsonToVariant;

    use super::*;

    #[test]
    fn test_get_variant_scalar() {
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
        let path = "name";

        let udf = VariantGetUdf::default();

        let arg_field = Arc::new(
            Field::new("input", DataType::Struct(Fields::empty()), true)
                .with_extension_type(VariantType),
        );
        let arg_field2 = Arc::new(Field::new("path", DataType::Utf8, true));

        let return_field = udf
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &[arg_field.clone(), arg_field2.clone()],
                scalar_arguments: &[],
            })
            .unwrap();

        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Scalar(variant_input),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(path.to_string()))),
            ],
            return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Struct(struct_arr)) = result else {
            panic!("expected ScalarValue struct");
        };

        assert_eq!(struct_arr.len(), 1);

        let metadata_arr = struct_arr
            .column(0)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();
        let value_arr = struct_arr
            .column(1)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();

        let metadata = metadata_arr.value(0);
        let value = value_arr.value(0);

        let v = Variant::try_new(metadata, value).unwrap();

        assert_eq!(v, Variant::from("norm"))
    }
}
