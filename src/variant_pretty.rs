use std::sync::Arc;

use arrow::array::StringViewArray;
use arrow_schema::DataType;
use datafusion::{
    common::{exec_datafusion_err, exec_err},
    error::Result,
    logical_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet_variant_compute::VariantArray;

use crate::shared::try_field_as_variant_array;

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantPretty {
    signature: Signature,
}

impl Default for VariantPretty {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(1), Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for VariantPretty {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_pretty"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8View)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let field = args
            .arg_fields
            .first()
            .ok_or_else(|| exec_datafusion_err!("empty argument, expected 1 argument"))?;

        try_field_as_variant_array(field.as_ref())?;

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
                let v = variant_array.value(0);

                ColumnarValue::Scalar(ScalarValue::Utf8View(Some(format!("{:?}", v))))
            }
            ColumnarValue::Array(arr) => match arr.data_type() {
                DataType::Struct(_) => {
                    let variant_array = VariantArray::try_new(arr.as_ref())?;

                    // is there a reason why variant array doesn't implement Iterator?
                    let mut out = Vec::with_capacity(variant_array.len());

                    for i in 0..variant_array.len() {
                        let v = variant_array.value(i);
                        out.push(Some(format!("{:?}", v)));
                    }

                    let out: StringViewArray = out.into();

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
    use insta::assert_snapshot;
    use parquet_variant_compute::VariantType;

    use crate::shared::build_variant_array_from_json;

    use super::*;

    #[test]
    fn test_pretty_primitive_scalar() {
        let expected_json = serde_json::json!("norm");
        let input = build_variant_array_from_json(&expected_json);

        let variant_input = ScalarValue::Struct(Arc::new(input.into()));

        let udf = VariantPretty::default();
        let return_field = Arc::new(Field::new("result", DataType::Utf8View, true));
        let arg_field = Arc::new(
            Field::new("input", DataType::Struct(Fields::empty()), true)
                .with_extension_type(VariantType),
        );

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

        assert_snapshot!(j, @r#"ShortString(ShortString("norm"))"#);
    }

    #[test]
    fn test_pretty_variant_object() {
        let expected_json = serde_json::json!({
            "name": "norm",
            "age": 50,
            "list": [false, true, ()]
        });

        let input = build_variant_array_from_json(&expected_json);

        let variant_input = ScalarValue::Struct(Arc::new(input.into()));

        let udf = VariantPretty::default();

        let return_field = Arc::new(Field::new("result", DataType::Utf8View, true));
        let arg_field = Arc::new(
            Field::new("input", DataType::Struct(Fields::empty()), true)
                .with_extension_type(VariantType),
        );

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

        assert_snapshot!(j, @r#"{"age": Int8(50), "list": [BooleanFalse, BooleanTrue, Null], "name": ShortString(ShortString("norm"))}"#);
    }
}
