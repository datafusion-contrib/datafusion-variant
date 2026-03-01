use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, StructArray},
    compute::concat,
};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields};
use datafusion::{
    common::{arrow_datafusion_err, exec_datafusion_err, exec_err},
    error::{DataFusionError, Result},
    logical_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet_variant::VariantPath;
use parquet_variant_compute::{GetOptions, VariantArray, VariantType, variant_get};

use crate::shared::{
    try_field_as_variant_array, try_parse_string_columnar, try_parse_string_scalar,
};

fn type_hint_from_scalar(field_name: &str, scalar: &ScalarValue) -> Result<FieldRef> {
    let type_name = match scalar {
        ScalarValue::Utf8(Some(value))
        | ScalarValue::Utf8View(Some(value))
        | ScalarValue::LargeUtf8(Some(value)) => value.as_str(),
        other => {
            return exec_err!(
                "type hint must be a non-null UTF8 literal, got {}",
                other.data_type()
            );
        }
    };

    let casted_type = match type_name.parse::<DataType>() {
        Ok(data_type) => Ok(data_type),
        Err(ArrowError::ParseError(e)) => Err(exec_datafusion_err!("{e}")),
        Err(e) => Err(arrow_datafusion_err!(e)),
    }?;

    Ok(Arc::new(Field::new(field_name, casted_type, true)))
}

fn type_hint_from_value(field_name: &str, arg: &ColumnarValue) -> Result<FieldRef> {
    match arg {
        ColumnarValue::Scalar(value) => type_hint_from_scalar(field_name, value),
        ColumnarValue::Array(_) => {
            exec_err!("type hint argument must be a scalar UTF8 literal")
        }
    }
}

fn build_get_options<'a>(path: VariantPath<'a>, as_type: &Option<FieldRef>) -> GetOptions<'a> {
    match as_type {
        Some(field) => GetOptions::new_with_path(path).with_as_type(Some(field.clone())),
        None => GetOptions::new_with_path(path),
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantGetUdf {
    signature: Signature,
}

impl Default for VariantGetUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(
                TypeSignature::OneOf(vec![TypeSignature::Any(2), TypeSignature::Any(3)]),
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
        Err(DataFusionError::Internal(
            "implemented return_field_from_args instead".into(),
        ))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        if let Some(maybe_scalar) = args.scalar_arguments.get(2) {
            let scalar = maybe_scalar.ok_or_else(|| {
                exec_datafusion_err!("type hint argument to variant_get must be a literal")
            })?;
            return type_hint_from_scalar(self.name(), scalar);
        }

        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, true),
        ]));

        Ok(Arc::new(
            Field::new(self.name(), data_type, true).with_extension_type(VariantType),
        ))
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let (variant_arg, variant_path, type_arg) = match args.args.as_slice() {
            [variant_arg, variant_path] => (variant_arg, variant_path, None),
            [variant_arg, variant_path, type_arg] => (variant_arg, variant_path, Some(type_arg)),
            _ => return exec_err!("expected 2 or 3 arguments"),
        };

        let variant_field = args
            .arg_fields
            .first()
            .ok_or_else(|| exec_datafusion_err!("expected argument field"))?;

        try_field_as_variant_array(variant_field.as_ref())?;

        let type_field = type_arg
            .map(|arg| type_hint_from_value(self.name(), arg))
            .transpose()?;

        let out = match (variant_arg, variant_path) {
            (ColumnarValue::Array(variant_array), ColumnarValue::Scalar(variant_path)) => {
                let variant_path = try_parse_string_scalar(variant_path)?
                    .map(|s| s.as_str())
                    .unwrap_or_default();

                let res = variant_get(
                    variant_array,
                    build_get_options(VariantPath::try_from(variant_path)?, &type_field),
                )?;

                ColumnarValue::Array(res)
            }
            (ColumnarValue::Scalar(scalar_variant), ColumnarValue::Scalar(variant_path)) => {
                let ScalarValue::Struct(variant_array) = scalar_variant else {
                    return exec_err!("expected struct array");
                };

                let variant_array = Arc::clone(variant_array) as ArrayRef;

                let variant_path = try_parse_string_scalar(variant_path)?
                    .map(|s| s.as_str())
                    .unwrap_or_default();

                let res = variant_get(
                    &variant_array,
                    build_get_options(VariantPath::try_from(variant_path)?, &type_field),
                )?;

                let scalar = ScalarValue::try_from_array(res.as_ref(), 0)?;
                ColumnarValue::Scalar(scalar)
            }
            (ColumnarValue::Array(variant_array), ColumnarValue::Array(variant_paths)) => {
                if variant_array.len() != variant_paths.len() {
                    return exec_err!(
                        "expected variant_array and variant paths to be of same length"
                    );
                }

                let variant_paths = try_parse_string_columnar(variant_paths)?;
                let variant_array = VariantArray::try_new(variant_array.as_ref())?;

                let mut out = Vec::with_capacity(variant_array.len());

                for (i, path) in variant_paths.iter().enumerate() {
                    let v = variant_array.value(i);
                    // todo: is there a better way to go from Variant -> VariantArray?
                    let singleton_variant_array: StructArray = VariantArray::from_iter([v]).into();

                    let arr = Arc::new(singleton_variant_array) as ArrayRef;

                    let res = variant_get(
                        &arr,
                        build_get_options(
                            VariantPath::try_from(path.unwrap_or_default())?,
                            &type_field,
                        ),
                    )?;

                    out.push(res);
                }

                let out_refs: Vec<&dyn Array> = out.iter().map(|a| a.as_ref()).collect();
                ColumnarValue::Array(concat(&out_refs)?)
            }
            (ColumnarValue::Scalar(scalar_variant), ColumnarValue::Array(variant_paths)) => {
                let ScalarValue::Struct(variant_array) = scalar_variant else {
                    return exec_err!("expected struct array");
                };

                let variant_array = Arc::clone(variant_array) as ArrayRef;
                let variant_paths = try_parse_string_columnar(variant_paths)?;

                let mut out = Vec::with_capacity(variant_paths.len());

                for path in variant_paths {
                    let path = path.unwrap_or_default();
                    let res = variant_get(
                        &variant_array,
                        build_get_options(VariantPath::try_from(path)?, &type_field),
                    )?;

                    out.push(res);
                }

                let out_refs: Vec<&dyn Array> = out.iter().map(|a| a.as_ref()).collect();
                ColumnarValue::Array(concat(&out_refs)?)
            }
        };

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, BinaryViewArray, Int64Array};
    use arrow_schema::{Field, Fields};
    use datafusion::logical_expr::{ReturnFieldArgs, ScalarFunctionArgs};
    use parquet_variant::Variant;
    use parquet_variant_compute::{VariantArrayBuilder, VariantType};
    use parquet_variant_json::JsonToVariant;

    use super::*;

    fn variant_scalar_from_json(json: serde_json::Value) -> ScalarValue {
        let mut builder = VariantArrayBuilder::new(1);
        builder.append_json(json.to_string().as_str()).unwrap();
        ScalarValue::Struct(Arc::new(builder.build().into()))
    }

    fn variant_array_from_json_rows(json_rows: &[serde_json::Value]) -> ArrayRef {
        let mut builder = VariantArrayBuilder::new(json_rows.len());
        for value in json_rows {
            builder.append_json(value.to_string().as_str()).unwrap();
        }
        let variant_array: StructArray = builder.build().into();
        Arc::new(variant_array) as ArrayRef
    }

    fn standard_arg_fields(with_type_hint: bool) -> Vec<FieldRef> {
        let mut fields = vec![
            Arc::new(
                Field::new("input", DataType::Struct(Fields::empty()), true)
                    .with_extension_type(VariantType),
            ),
            Arc::new(Field::new("path", DataType::Utf8, true)),
        ];
        if with_type_hint {
            fields.push(Arc::new(Field::new("type", DataType::Utf8, true)));
        }
        fields
    }

    fn get_return_field(
        udf: &VariantGetUdf,
        arg_fields: &[FieldRef],
        type_hint_value: Option<&ScalarValue>,
    ) -> FieldRef {
        let scalar_arguments: Vec<Option<&ScalarValue>> = if let Some(hint) = type_hint_value {
            vec![None, None, Some(hint)]
        } else {
            vec![]
        };

        udf.return_field_from_args(ReturnFieldArgs {
            arg_fields,
            scalar_arguments: &scalar_arguments,
        })
        .unwrap()
    }

    fn build_scalar_function_args(
        variant_input: ColumnarValue,
        path: &str,
        arg_fields: Vec<FieldRef>,
        return_field: FieldRef,
        type_hint: Option<ScalarValue>,
    ) -> ScalarFunctionArgs {
        let mut args = vec![
            variant_input,
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(path.to_string()))),
        ];
        if let Some(hint) = type_hint {
            args.push(ColumnarValue::Scalar(hint));
        }

        ScalarFunctionArgs {
            args,
            return_field,
            arg_fields,
            number_rows: Default::default(),
            config_options: Default::default(),
        }
    }

    #[test]
    fn test_get_variant_scalar() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50,
            "list": [false, true, ()]
        }));

        let udf = VariantGetUdf::default();
        let arg_fields = standard_arg_fields(false);
        let return_field = get_return_field(&udf, &arg_fields, None);

        let args = build_scalar_function_args(
            ColumnarValue::Scalar(variant_input),
            "name",
            arg_fields,
            return_field,
            None,
        );

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

    #[test]
    fn test_return_field_with_type_hint() {
        let udf = VariantGetUdf::default();
        let arg_fields = standard_arg_fields(true);
        let type_hint = ScalarValue::Utf8(Some("Int64".to_string()));
        let return_field = get_return_field(&udf, &arg_fields, Some(&type_hint));

        assert_eq!(return_field.data_type(), &DataType::Int64);
    }

    #[test]
    fn test_get_variant_scalar_with_type_hint() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50,
        }));

        let udf = VariantGetUdf::default();
        let arg_fields = standard_arg_fields(true);
        let type_hint = ScalarValue::Utf8(Some("Int64".to_string()));
        let return_field = get_return_field(&udf, &arg_fields, Some(&type_hint));

        let args = build_scalar_function_args(
            ColumnarValue::Scalar(variant_input),
            "age",
            arg_fields,
            return_field,
            Some(type_hint),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Int64(Some(value))) = result else {
            panic!("expected ScalarValue Int64");
        };

        assert_eq!(value, 50);
    }

    #[test]
    fn test_get_variant_array_with_type_hint() {
        let json_rows = vec![
            serde_json::json!({ "age": 50 }),
            serde_json::json!({ "age": 60 }),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);

        let udf = VariantGetUdf::default();
        let arg_fields = standard_arg_fields(true);
        let type_hint = ScalarValue::Utf8(Some("Int64".to_string()));
        let return_field = get_return_field(&udf, &arg_fields, Some(&type_hint));

        let args = build_scalar_function_args(
            ColumnarValue::Array(variant_array),
            "age",
            arg_fields,
            return_field,
            Some(type_hint),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(array) = result else {
            panic!("expected array output");
        };

        let values = array.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values.value(0), 50);
        assert_eq!(values.value(1), 60);
    }
}
