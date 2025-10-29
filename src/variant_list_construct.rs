use std::sync::Arc;

use arrow::array::StructArray;
use arrow_schema::{DataType, Field, Fields};
use datafusion::{
    common::exec_datafusion_err,
    error::{DataFusionError, Result},
    logical_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet_variant::{Variant, VariantBuilder};
use parquet_variant_compute::{VariantArray, VariantType};

use crate::shared::{ensure, try_parse_string_scalar, try_parse_variant_scalar};

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantListConstruct {
    signature: Signature,
}

impl Default for VariantListConstruct {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::VariadicAny, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for VariantListConstruct {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_list_construct"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[arrow_schema::DataType]) -> Result<arrow_schema::DataType> {
        Err(DataFusionError::Internal(
            "implemented return_field_from_args instead".into(),
        ))
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> Result<Arc<Field>> {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, false),
        ]));

        Ok(Arc::new(
            Field::new(self.name(), data_type, true).with_extension_type(VariantType),
        ))
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        // validate arguments
        let argument_fields = args.arg_fields;
        let argument_values = args.args;

        ensure(
            argument_fields.len() == argument_values.len(),
            "argument fields and values must be of same length",
        )?;

        // note, should we have the ability to configure behavior for duplicate keys?

        let mut v = VariantBuilder::new();
        let mut o = v.new_object();

        for (k, v) in object_keys.iter().zip(object_values) {
            let v = v.value(0);
            o.try_insert(k, v)?;
        }

        o.finish();

        let (m, v) = v.finish();

        let v = Variant::new(m.as_ref(), v.as_ref());

        let out: StructArray = VariantArray::from_iter([v]).into();

        Ok(ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(out))))
    }
}
