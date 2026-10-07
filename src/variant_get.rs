use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringViewArray},
    compute::{CastOptions, cast, concat},
};
use arrow_schema::{DataType, Field, FieldRef, Fields};
use datafusion::{
    common::{config::ConfigOptions, exec_datafusion_err, exec_err},
    error::{DataFusionError, Result},
    logical_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
        TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet_variant::{Variant, VariantPath};
use parquet_variant_compute::{
    GetOptions, VariantArray, VariantType, unshred_variant, variant_get,
};
use parquet_variant_json::VariantToJson;

use crate::impl_variant_get::impl_variant_get_typed;
use crate::shared::{
    invoke_variant_get_typed, try_field_as_variant_array, try_parse_string_columnar,
    try_parse_string_scalar,
};
use crate::sql_type::SqlTypeConfig;
use crate::variant_type::{is_variant, validate_getter_target};

fn type_hint_from_scalar(
    field_name: &str,
    scalar: &ScalarValue,
    config: &SqlTypeConfig,
) -> Result<FieldRef> {
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

    let field = config.resolve(type_name)?;
    validate_getter_target(&field)?;
    Ok(Arc::new(
        field
            .as_ref()
            .clone()
            .with_name(field_name)
            .with_nullable(true),
    ))
}

fn build_get_options<'a>(
    path: VariantPath<'a>,
    as_type: &Option<FieldRef>,
    cast_options: &CastOptions<'a>,
) -> GetOptions<'a> {
    GetOptions::new_with_path(path)
        .with_as_type(as_type.clone())
        .with_cast_options(cast_options.clone())
}

fn get_with_target(input: &ArrayRef, mut options: GetOptions<'_>) -> Result<ArrayRef> {
    let variant_target = options
        .as_type
        .as_ref()
        .filter(|field| is_variant(field))
        .cloned();
    if let Some(target) = variant_target {
        options.as_type = None;
        // Unshred before extraction: extracting a shredded field first can lose
        // the distinction between a missing field and an explicit Variant null.
        let unshredded = unshred_variant(&VariantArray::try_new(input.as_ref())?)?;
        let extracted = variant_get(&ArrayRef::from(unshredded), options)?;
        // Match SQL VARIANT's declared binary widths and child nullability.
        Ok(cast(extracted.as_ref(), target.data_type())?)
    } else {
        Ok(variant_get(input, options)?)
    }
}

/// Evaluate a Variant getter using Arrow paths and an explicit cast error policy.
fn invoke_variant_get(
    args: ScalarFunctionArgs,
    typed: bool,
    cast_options: CastOptions<'_>,
) -> Result<ColumnarValue> {
    let (variant_arg, variant_path) = match args.args.as_slice() {
        [variant_arg, variant_path] | [variant_arg, variant_path, _] => (variant_arg, variant_path),
        _ => return exec_err!("expected 2 or 3 arguments"),
    };

    let variant_field = args
        .arg_fields
        .first()
        .ok_or_else(|| exec_datafusion_err!("expected argument field"))?;

    try_field_as_variant_array(variant_field.as_ref())?;

    // Type hints are resolved during planning. Both explicit casts and string
    // hints use this same field; execution never reparses a type name.
    let type_field = (typed || args.args.len() == 3).then(|| Arc::clone(&args.return_field));

    let out = match (variant_arg, variant_path) {
        (ColumnarValue::Array(variant_array), ColumnarValue::Scalar(variant_path)) => {
            let variant_path = try_parse_string_scalar(variant_path)?
                .map(|s| s.as_str())
                .unwrap_or_default();

            let res = get_with_target(
                variant_array,
                build_get_options(
                    VariantPath::try_from(variant_path)?,
                    &type_field,
                    &cast_options,
                ),
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

            let res = get_with_target(
                &variant_array,
                build_get_options(
                    VariantPath::try_from(variant_path)?,
                    &type_field,
                    &cast_options,
                ),
            )?;

            let scalar = ScalarValue::try_from_array(res.as_ref(), 0)?;
            ColumnarValue::Scalar(scalar)
        }
        (ColumnarValue::Array(variant_array), ColumnarValue::Array(variant_paths)) => {
            if variant_array.len() != variant_paths.len() {
                return exec_err!("expected variant_array and variant paths to be of same length");
            }

            let variant_paths = try_parse_string_columnar(variant_paths)?;

            let mut out = Vec::with_capacity(variant_array.len());

            for (i, path) in variant_paths.iter().enumerate() {
                // Preserve SQL validity and shredded storage when selecting a row.
                let arr = variant_array.slice(i, 1);

                let res = get_with_target(
                    &arr,
                    build_get_options(
                        VariantPath::try_from(path.unwrap_or_default())?,
                        &type_field,
                        &cast_options,
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
                let res = get_with_target(
                    &variant_array,
                    build_get_options(VariantPath::try_from(path)?, &type_field, &cast_options),
                )?;

                out.push(res);
            }

            let out_refs: Vec<&dyn Array> = out.iter().map(|a| a.as_ref()).collect();
            ColumnarValue::Array(concat(&out_refs)?)
        }
    };

    Ok(out)
}

fn return_field_for_variant_get(
    name: &str,
    args: ReturnFieldArgs,
    config: &SqlTypeConfig,
) -> Result<Arc<Field>> {
    if let Some(maybe_scalar) = args.scalar_arguments.get(2) {
        let scalar = maybe_scalar.ok_or_else(|| {
            exec_datafusion_err!("type hint argument to {name} must be a literal")
        })?;
        return type_hint_from_scalar(name, scalar, config);
    }

    // Untyped extraction can retain nullable value storage. SQL VARIANT targets
    // instead materialize the canonical unshredded field from VariantArrayBuilder.
    let data_type = DataType::Struct(Fields::from(vec![
        Field::new("metadata", DataType::BinaryView, false),
        Field::new("value", DataType::BinaryView, true),
    ]));
    Ok(Arc::new(
        Field::new(name, data_type, true).with_extension_type(VariantType),
    ))
}

impl_variant_get_typed!(
    /// Extracts a string value from a Variant by path.
    ///
    /// `variant_get_str(variant, path)` returns the value at `path` as a UTF8 string.
    /// - String values are returned as-is (no JSON quotes)
    /// - Non-string values (numbers, booleans, objects, arrays) are JSON-serialized
    /// - Returns NULL if the path does not exist
    VariantGetStrUdf,
    "variant_get_str",
    DataType::Utf8View,
    ScalarValue::Utf8View,
    |values: Vec<Option<String>>| -> ArrayRef {
        let out: StringViewArray = values.into_iter().collect();
        Arc::new(out)
    },
    |value: Variant<'_, '_>| -> Result<Option<String>> {
        if let Some(s) = value.as_string() {
            Ok(Some(s.to_string()))
        } else {
            Ok(Some(value.to_json_string()?))
        }
    },
);

impl_variant_get_typed!(
    /// Extracts an integer value from a Variant by path.
    ///
    /// `variant_get_int(variant, path)` returns the value at `path` as an `INT64`.
    /// - Integer values are returned as-is (with widening when needed)
    /// - Non-integer values return NULL
    /// - Returns NULL if the path does not exist
    VariantGetIntUdf,
    "variant_get_int",
    DataType::Int64,
    ScalarValue::Int64,
    |values: Vec<Option<i64>>| -> ArrayRef {
        Arc::new(values.into_iter().collect::<Int64Array>())
    },
    |value: Variant<'_, '_>| -> Result<Option<i64>> { Ok(value.as_int64()) },
);

impl_variant_get_typed!(
    /// Extracts a floating-point value from a Variant by path.
    ///
    /// `variant_get_float(variant, path)` returns the value at `path` as a `FLOAT64`.
    /// - Float values are returned as-is
    /// - Integer values are returned as `FLOAT64` (large values may lose precision)
    /// - Non-numeric values return NULL
    /// - Returns NULL if the path does not exist
    VariantGetFloatUdf,
    "variant_get_float",
    DataType::Float64,
    ScalarValue::Float64,
    |values: Vec<Option<f64>>| -> ArrayRef {
        Arc::new(values.into_iter().collect::<Float64Array>())
    },
    |value: Variant<'_, '_>| -> Result<Option<f64>> {
        Ok(value
            .as_f64()
            .or_else(|| value.as_int64().map(|i| i as f64)))
    },
);

impl_variant_get_typed!(
    /// Extracts a boolean value from a Variant by path.
    ///
    /// `variant_get_bool(variant, path)` returns the value at `path` as a `BOOLEAN`.
    /// - Boolean values are returned as-is
    /// - Non-boolean values return NULL
    /// - Returns NULL if the path does not exist
    VariantGetBoolUdf,
    "variant_get_bool",
    DataType::Boolean,
    ScalarValue::Boolean,
    |values: Vec<Option<bool>>| -> ArrayRef {
        Arc::new(values.into_iter().collect::<BooleanArray>())
    },
    |value: Variant<'_, '_>| -> Result<Option<bool>> { Ok(value.as_boolean()) },
);

impl_variant_get_typed!(
    /// Extracts a value from a Variant by path and returns it as a JSON string.
    ///
    /// `variant_get_json(variant, path)` returns the value at `path` as a JSON string.
    /// - All values are JSON-serialized (strings include quotes, unlike `variant_get_str`)
    /// - Returns NULL if the path does not exist
    VariantGetJsonUdf,
    "variant_get_json",
    DataType::Utf8View,
    ScalarValue::Utf8View,
    |values: Vec<Option<String>>| -> ArrayRef {
        let out: StringViewArray = values.into_iter().collect();
        Arc::new(out)
    },
    |value: Variant<'_, '_>| -> Result<Option<String>> {
        Ok(Some(value.to_json_string()?))
    },
);

/// Extracts a Variant value, returning an error for failed result-type conversions.
///
/// `variant_get(value, path[, type])` accepts a SQL type string literal, such as
/// `BIGINT` or `DECIMAL(12,2)`. Use [`crate::register_variant_get_functions`] to
/// register with session settings and enable explicit casts of getters.
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantGetUdf {
    signature: Signature,
    config: SqlTypeConfig,
    target: Option<FieldRef>,
}

impl Default for VariantGetUdf {
    fn default() -> Self {
        Self::new_with_config(&ConfigOptions::default())
    }
}

impl VariantGetUdf {
    /// Create a getter using the session's SQL type-resolution settings.
    pub fn new_with_config(config: &ConfigOptions) -> Self {
        Self {
            config: config.into(),
            target: None,
            signature: Signature::new(
                TypeSignature::OneOf(vec![TypeSignature::Any(2), TypeSignature::Any(3)]),
                Volatility::Immutable,
            ),
        }
    }

    pub(crate) fn with_target(target: FieldRef) -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
            target: Some(Arc::new(target.as_ref().clone().with_nullable(true))),
            ..Self::default()
        }
    }

    pub(crate) fn is_untyped(&self) -> bool {
        self.target.is_none()
    }
}

impl ScalarUDFImpl for VariantGetUdf {
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

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        let mut updated = Self::new_with_config(config);
        updated.target = self.target.clone();
        updated.signature = self.signature.clone();
        Some(ScalarUDF::new_from_impl(updated))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        if let Some(target) = &self.target {
            return Ok(Arc::clone(target));
        }
        return_field_for_variant_get(self.name(), args, &self.config)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        invoke_variant_get(
            args,
            self.target.is_some(),
            CastOptions {
                safe: false,
                ..Default::default()
            },
        )
    }
}

/// Extracts a Variant value, returning NULL for failed result-type conversions.
/// Invalid paths and type hints still return errors.
///
/// `try_variant_get(value, path[, type])` accepts the same SQL type strings as
/// [`VariantGetUdf`].
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct TryVariantGetUdf {
    signature: Signature,
    config: SqlTypeConfig,
    target: Option<FieldRef>,
}

impl Default for TryVariantGetUdf {
    fn default() -> Self {
        Self::new_with_config(&ConfigOptions::default())
    }
}

impl TryVariantGetUdf {
    /// Create a getter using the session's SQL type-resolution settings.
    pub fn new_with_config(config: &ConfigOptions) -> Self {
        Self {
            config: config.into(),
            target: None,
            signature: Signature::new(
                TypeSignature::OneOf(vec![TypeSignature::Any(2), TypeSignature::Any(3)]),
                Volatility::Immutable,
            ),
        }
    }

    pub(crate) fn with_target(target: FieldRef) -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
            target: Some(Arc::new(target.as_ref().clone().with_nullable(true))),
            ..Self::default()
        }
    }

    pub(crate) fn is_untyped(&self) -> bool {
        self.target.is_none()
    }
}

impl ScalarUDFImpl for TryVariantGetUdf {
    fn name(&self) -> &str {
        "try_variant_get"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[arrow_schema::DataType]) -> Result<arrow_schema::DataType> {
        Err(DataFusionError::Internal(
            "implemented return_field_from_args instead".into(),
        ))
    }

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        let mut updated = Self::new_with_config(config);
        updated.target = self.target.clone();
        updated.signature = self.signature.clone();
        Some(ScalarUDF::new_from_impl(updated))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        if let Some(target) = &self.target {
            return Ok(Arc::clone(target));
        }
        return_field_for_variant_get(self.name(), args, &self.config)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        invoke_variant_get(
            args,
            self.target.is_some(),
            CastOptions {
                safe: true,
                ..Default::default()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{
        build_variant_get_args, standard_variant_get_arg_fields, variant_array_from_json_rows,
        variant_scalar_from_json,
    };
    use arrow::{
        array::{Array, BinaryViewArray, BooleanArray, Int64Array, ListArray},
        buffer::OffsetBuffer,
    };
    use arrow_schema::Field;
    use datafusion::logical_expr::{ReturnFieldArgs, ScalarFunctionArgs};
    use parquet_variant::Variant;

    fn standard_arg_fields(with_type_hint: bool) -> Vec<FieldRef> {
        let mut fields = standard_variant_get_arg_fields();
        if with_type_hint {
            fields.push(Arc::new(Field::new("type", DataType::Utf8, true)));
        }
        fields
    }

    fn get_return_field(
        udf: &dyn ScalarUDFImpl,
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
        let type_hint = ScalarValue::Utf8(Some("BIGINT".to_string()));
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
        let type_hint = ScalarValue::Utf8(Some("BIGINT".to_string()));
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
        let type_hint = ScalarValue::Utf8(Some("BIGINT".to_string()));
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

    #[test]
    fn test_str_scalar_string_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("name".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        assert_eq!(s, "norm");
    }

    #[test]
    fn test_str_scalar_numeric_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("age".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        assert_eq!(s, "50");
    }

    #[test]
    fn test_str_scalar_missing_path() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"name": "norm"}));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("missing".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(None)) = result else {
            panic!("expected NULL Utf8View scalar");
        };
    }

    #[test]
    fn test_str_scalar_nested_object() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "obj": {"a": 1, "b": 2}
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("obj".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        let json: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(json, serde_json::json!({"a": 1, "b": 2}));
    }

    #[test]
    fn test_str_scalar_boolean_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"flag": true}));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("flag".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        assert_eq!(s, "true");
    }

    #[test]
    fn test_str_scalar_null_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"key": null}));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("key".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        assert_eq!(s, "null");
    }

    #[test]
    fn test_str_array_variant_scalar_path() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "age": 30}),
            serde_json::json!({"name": "bob", "age": 40}),
            serde_json::json!({"age": 50}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("name".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let str_arr = arr.as_any().downcast_ref::<StringViewArray>().unwrap();
        assert_eq!(str_arr.len(), 3);
        assert_eq!(str_arr.value(0), "alice");
        assert_eq!(str_arr.value(1), "bob");
        assert!(str_arr.is_null(2));
    }

    #[test]
    fn test_str_array_variant_array_paths() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "age": 30}),
            serde_json::json!({"name": "bob", "age": 40}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);
        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec!["name", "age"]));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Array(path_array),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let str_arr = arr.as_any().downcast_ref::<StringViewArray>().unwrap();
        assert_eq!(str_arr.len(), 2);
        assert_eq!(str_arr.value(0), "alice");
        assert_eq!(str_arr.value(1), "40");
    }

    #[test]
    fn test_str_array_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "list": [1, 2, 3]
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("list".to_string()))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar");
        };

        let json: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(json, serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_int_scalar_integer_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50
        }));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("age".to_string()))),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Int64(Some(v))) = result else {
            panic!("expected Int64 scalar");
        };

        assert_eq!(v, 50);
    }

    #[test]
    fn test_int_scalar_non_integer_value_returns_null() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50.5
        }));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("name".to_string()))),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Int64(None)) = result else {
            panic!("expected NULL Int64 scalar");
        };
    }

    #[test]
    fn test_int_scalar_float_value_coerces() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "price": 10.5
        }));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("price".to_string()))),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        // parquet-variant 59 coerces numeric variants to the requested type,
        // truncating floats toward zero (10.5 -> 10) rather than returning NULL.
        let ColumnarValue::Scalar(ScalarValue::Int64(Some(10))) = result else {
            panic!("expected Int64(10) from float coercion, got {result:?}");
        };
    }

    #[test]
    fn test_int_scalar_missing_path() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"name": "norm"}));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("missing".to_string()))),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Int64(None)) = result else {
            panic!("expected NULL Int64 scalar");
        };
    }

    #[test]
    fn test_int_array_variant_scalar_path() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "age": 30}),
            serde_json::json!({"name": "bob", "age": 40}),
            serde_json::json!({"name": "charlie"}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("age".to_string()))),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let int_arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_arr.len(), 3);
        assert_eq!(int_arr.value(0), 30);
        assert_eq!(int_arr.value(1), 40);
        assert!(int_arr.is_null(2));
    }

    #[test]
    fn test_int_array_variant_array_paths() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "age": 30}),
            serde_json::json!({"name": "bob", "age": 40}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);
        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec!["age", "name"]));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Array(path_array),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let int_arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_arr.len(), 2);
        assert_eq!(int_arr.value(0), 30);
        assert!(int_arr.is_null(1));
    }

    #[test]
    fn test_int_scalar_variant_array_paths() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "alice",
            "age": 30
        }));

        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec!["age", "name", "missing"]));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Array(path_array),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let int_arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_arr.len(), 3);
        assert_eq!(int_arr.value(0), 30);
        assert!(int_arr.is_null(1));
        assert!(int_arr.is_null(2));
    }

    #[test]
    fn test_float_scalar_float_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "price": 50.5
        }));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("price".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Float64(Some(v))) = result else {
            panic!("expected Float64 scalar");
        };

        assert_eq!(v, 50.5);
    }

    #[test]
    fn test_float_scalar_integer_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50
        }));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("age".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Float64(Some(v))) = result else {
            panic!("expected Float64 scalar");
        };

        assert_eq!(v, 50.0);
    }

    #[test]
    fn test_float_scalar_large_integer_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "n": 9007199254740993_i64
        }));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("n".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Float64(Some(v))) = result else {
            panic!("expected Float64 scalar");
        };

        // `f64` cannot exactly represent all i64 values; this mirrors json_get_float behavior.
        assert_eq!(v, 9_007_199_254_740_992.0);
    }

    #[test]
    fn test_float_scalar_non_numeric_value_returns_null() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50.5
        }));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("name".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Float64(None)) = result else {
            panic!("expected NULL Float64 scalar");
        };
    }

    #[test]
    fn test_float_scalar_missing_path() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"name": "norm"}));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("missing".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Float64(None)) = result else {
            panic!("expected NULL Float64 scalar");
        };
    }

    #[test]
    fn test_float_array_variant_scalar_path() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "price": 30.25}),
            serde_json::json!({"name": "bob", "price": 40}),
            serde_json::json!({"name": "charlie"}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("price".to_string()))),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let float_arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(float_arr.len(), 3);
        assert_eq!(float_arr.value(0), 30.25);
        assert_eq!(float_arr.value(1), 40.0);
        assert!(float_arr.is_null(2));
    }

    #[test]
    fn test_float_array_variant_array_paths() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "price": 30.25}),
            serde_json::json!({"name": "bob", "price": 40}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);
        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec!["price", "name"]));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Array(path_array),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let float_arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(float_arr.len(), 2);
        assert_eq!(float_arr.value(0), 30.25);
        assert!(float_arr.is_null(1));
    }

    #[test]
    fn test_float_scalar_variant_array_paths() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "alice",
            "price": 30.25,
            "count": 3
        }));

        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec![
            "price", "count", "name", "missing",
        ]));

        let udf = VariantGetFloatUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Array(path_array),
            DataType::Float64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let float_arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(float_arr.len(), 4);
        assert_eq!(float_arr.value(0), 30.25);
        assert_eq!(float_arr.value(1), 3.0);
        assert!(float_arr.is_null(2));
        assert!(float_arr.is_null(3));
    }

    #[test]
    fn test_bool_scalar_bool_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "active": true
        }));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("active".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Boolean(Some(v))) = result else {
            panic!("expected Boolean scalar");
        };

        assert!(v);
    }

    #[test]
    fn test_bool_scalar_false_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "active": false
        }));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("active".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Boolean(Some(v))) = result else {
            panic!("expected Boolean scalar");
        };

        assert!(!v);
    }

    #[test]
    fn test_bool_scalar_non_bool_value_returns_null() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "norm",
            "age": 50
        }));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("name".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Boolean(None)) = result else {
            panic!("expected NULL Boolean scalar");
        };
    }

    #[test]
    fn test_bool_scalar_int_value_coerces() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "count": 1
        }));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("count".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        // parquet-variant 59 coerces numeric variants to boolean (nonzero -> true)
        // rather than returning NULL.
        let ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) = result else {
            panic!("expected Boolean(true) from int coercion, got {result:?}");
        };
    }

    #[test]
    fn test_bool_scalar_missing_path() {
        let variant_input = variant_scalar_from_json(serde_json::json!({"name": "norm"}));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("missing".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Boolean(None)) = result else {
            panic!("expected NULL Boolean scalar");
        };
    }

    #[test]
    fn test_bool_array_variant_scalar_path() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "active": true}),
            serde_json::json!({"name": "bob", "active": false}),
            serde_json::json!({"name": "charlie"}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("active".to_string()))),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let bool_arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(bool_arr.len(), 3);
        assert!(bool_arr.value(0));
        assert!(!bool_arr.value(1));
        assert!(bool_arr.is_null(2));
    }

    #[test]
    fn test_bool_array_variant_array_paths() {
        let json_rows = vec![
            serde_json::json!({"name": "alice", "active": true}),
            serde_json::json!({"name": "bob", "active": false}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);
        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec!["active", "name"]));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Array(path_array),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let bool_arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(bool_arr.len(), 2);
        assert!(bool_arr.value(0));
        assert!(bool_arr.is_null(1));
    }

    #[test]
    fn test_bool_scalar_variant_array_paths() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "name": "alice",
            "active": true,
            "count": 3
        }));

        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec![
            "active", "count", "name", "missing",
        ]));

        let udf = VariantGetBoolUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Array(path_array),
            DataType::Boolean,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };

        let bool_arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(bool_arr.len(), 4);
        assert!(bool_arr.value(0)); // active = true
        assert!(bool_arr.value(1)); // count = 3 coerces to true (parquet-variant 59)
        assert!(bool_arr.is_null(2)); // name = "alice" is not boolean-castable
        assert!(bool_arr.is_null(3)); // missing path
    }

    fn string_list_scalar(values: &[&str]) -> ScalarValue {
        let string_array = Arc::new(StringViewArray::from(values.to_vec())) as ArrayRef;

        ScalarValue::List(Arc::new(ListArray::new(
            Arc::new(Field::new_list_field(DataType::Utf8View, true)),
            OffsetBuffer::from_lengths([values.len()]),
            string_array,
            None,
        )))
    }

    #[test]
    fn test_get_str_list_path_dotted_key() {
        // List path should treat each element as a single field — no dot splitting
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "http.response.status_code": 200,
            "service.name": "my-service"
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(string_list_scalar(&["http.response.status_code"])),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar, got {result:?}");
        };
        assert_eq!(s, "200");
    }

    #[test]
    fn test_get_str_list_path_string_value() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "service.name": "my-service"
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(string_list_scalar(&["service.name"])),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar, got {result:?}");
        };
        assert_eq!(s, "my-service");
    }

    #[test]
    fn test_get_str_list_path_array_variant() {
        // Test array variant input with list path
        let json_rows = vec![
            serde_json::json!({"http.status": 200, "http.method": "GET"}),
            serde_json::json!({"http.status": 404, "http.method": "POST"}),
            serde_json::json!({"http.status": 500}),
        ];

        let variant_array = variant_array_from_json_rows(&json_rows);

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Array(variant_array),
            ColumnarValue::Scalar(string_list_scalar(&["http.status"])),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(arr) = result else {
            panic!("expected array output");
        };
        let str_arr = arr.as_any().downcast_ref::<StringViewArray>().unwrap();
        assert_eq!(str_arr.len(), 3);
        assert_eq!(str_arr.value(0), "200");
        assert_eq!(str_arr.value(1), "404");
        assert_eq!(str_arr.value(2), "500");
    }

    #[test]
    fn test_get_int_list_path_dotted_key() {
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "http.status": 200
        }));

        let udf = VariantGetIntUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(string_list_scalar(&["http.status"])),
            DataType::Int64,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Int64(Some(v))) = result else {
            panic!("expected Int64 scalar, got {result:?}");
        };
        assert_eq!(v, 200);
    }

    #[test]
    fn test_get_str_list_path_nested_traversal() {
        // List path with multiple elements should traverse nested objects
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "a": { "b": { "c": 42 } }
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(string_list_scalar(&["a", "b", "c"])),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) = result else {
            panic!("expected Utf8View scalar, got {result:?}");
        };
        assert_eq!(s, "42");
    }

    #[test]
    fn test_get_str_string_path_dot_notation_splits() {
        // String path uses dot notation — should return NULL for a dotted key
        // stored as a single field name
        let variant_input = variant_scalar_from_json(serde_json::json!({
            "http.response.status_code": 200
        }));

        let udf = VariantGetStrUdf::default();
        let args = build_variant_get_args(
            ColumnarValue::Scalar(variant_input),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                "http.response.status_code".to_string(),
            ))),
            DataType::Utf8View,
            standard_variant_get_arg_fields(),
        );

        let result = udf.invoke_with_args(args).unwrap();

        // Dot notation splits on dots, tries to traverse http -> response -> status_code
        // which doesn't exist, so returns NULL
        let ColumnarValue::Scalar(ScalarValue::Utf8View(None)) = result else {
            panic!("expected NULL (dot notation splits the key), got {result:?}");
        };
    }
}
