// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, StringViewArray},
    compute::{CastOptions, concat},
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
use datafusion_doc::Documentation;
use datafusion_macros::user_doc;
use parquet_variant::{Variant, VariantPath};
use parquet_variant_compute::{GetOptions, VariantType, variant_get};
use parquet_variant_json::VariantToJson;

use crate::impl_variant_get::impl_variant_get_typed;
use crate::shared::{
    invoke_variant_get_typed, try_field_as_variant_array, try_parse_string_columnar,
    try_parse_string_scalar,
};
use crate::sql_type::SqlTypeConfig;
use crate::variant_type::validate_getter_target;

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

/// Evaluate a Variant getter using Arrow paths and an explicit cast error policy.
fn invoke_variant_get(
    args: ScalarFunctionArgs,
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
    // Request the planned Variant output explicitly too, so Arrow can produce
    // its canonical storage schema instead of retaining shredded child types.
    let type_field = Some(Arc::clone(&args.return_field));

    let out = match (variant_arg, variant_path) {
        (ColumnarValue::Array(variant_array), ColumnarValue::Scalar(variant_path)) => {
            let variant_path = try_parse_string_scalar(variant_path)?
                .map(|s| s.as_str())
                .unwrap_or_default();

            let res = variant_get(
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

            let res = variant_get(
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

                let res = variant_get(
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
                let res = variant_get(
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

    // Request canonical Variant storage explicitly, matching Arrow 60 output.
    let data_type = DataType::Struct(Fields::from(vec![
        Field::new("metadata", DataType::BinaryView, false),
        Field::new("value", DataType::BinaryView, false),
    ]));
    Ok(Arc::new(
        Field::new(name, data_type, true).with_extension_type(VariantType),
    ))
}

impl_variant_get_typed!(
    /// Extracts a value as JSON, including quotes around string values.
    VariantGetJsonUdf, "variant_get_json", DataType::Utf8View,
    |args| invoke_variant_get_typed(args, ScalarValue::Utf8View,
        |values: Vec<Option<String>>| -> ArrayRef {
            Arc::new(values.into_iter().collect::<StringViewArray>())
        },
        |value: Variant<'_, '_>| -> Result<Option<String>> {
            Ok(Some(value.to_json_string()?))
        }),
);

/// Extracts a Variant value, returning an error for failed result-type conversions.
///
/// `variant_get(value, path[, type])` accepts a SQL type string literal, such as
/// `BIGINT` or `DECIMAL(12,2)`. Use [`crate::register_variant_get_functions`] to
/// register with session settings and enable explicit casts of getters.
#[user_doc(
    doc_section(label = "Variant Functions"),
    description = "Extracts the value at a path and optionally converts it to a SQL type.
Without a type hint, returns Variant. SQL NULL input and missing paths return SQL NULL.
Variant null remains a Variant null for Variant output and becomes SQL NULL for typed output.
Invalid paths, invalid type hints, and failed conversions raise errors.

Type names follow DataFusion SQL semantics, including session settings. For example,
`INT8` means BIGINT; Arrow type names such as `Int64` are not accepted.
Nested Variant target types are not supported.

Register with `register_variant_get_functions` to enable SQL casts of getters.
Also register `VariantExprPlanner` for `:` field access. A cast of an untyped
getter or colon access performs the same conversion as an explicit type hint.",
    syntax_example = "variant_get(value, path[, type])",
    alternative_syntax = "CAST(variant_get(value, path) AS type)",
    alternative_syntax = "value:field",
    alternative_syntax = "value:field::type",
    sql_example = r#"```sql
> SELECT variant_get(parse_json('{"a":1}'), 'a', 'BIGINT') AS value;
+-------+
| value |
+-------+
| 1     |
+-------+
> SELECT variant_get(parse_json('{"a":1}'), 'b', 'BIGINT') IS NULL AS missing;
+---------+
| missing |
+---------+
| true    |
+---------+
> SELECT to_json(variant_get(parse_json('[1,"hello"]'), '[1]')) AS json;
+---------+
| json    |
+---------+
| "hello" |
+---------+
> WITH input AS (SELECT parse_json('{"price":12.5}') AS v)
  SELECT variant_get(v, 'price', 'DECIMAL(12,2)') AS type_hint,
         CAST(variant_get(v, 'price') AS DECIMAL(12,2)) AS sql_cast,
         v:price::DECIMAL(12,2) AS colon_cast
  FROM input;
+-----------+----------+------------+
| type_hint | sql_cast | colon_cast |
+-----------+----------+------------+
| 12.50     | 12.50    | 12.50      |
+-----------+----------+------------+
```"#,
    argument(name = "value", description = "Variant expression to extract from."),
    argument(
        name = "path",
        description = r#"String expression selecting fields with dots and array elements with zero-based brackets, such as `a.b[0]`. Use `["a.b"]` for a literal key containing a dot. An empty string or SQL NULL selects the root. Use field paths without Spark's leading `$`; here `$` is treated as a field name. Keys containing `]` currently require escaping that bracket with a backslash, even inside quotes."#
    ),
    argument(
        name = "type",
        description = "Optional non-null SQL type string literal, such as `BIGINT`, `DECIMAL(12,2)`, or `ARRAY<INT>`. Defaults to `VARIANT`."
    ),
    related_udf(name = "try_variant_get"),
    related_udf(name = "is_variant_null")
)]
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
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
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
#[user_doc(
    doc_section(label = "Variant Functions"),
    description = "Extracts the value at a path and optionally converts it to a SQL type.
Uses the same paths, type hints, and null handling as `variant_get`, but failed
result-type conversions return SQL NULL. Without a type hint, returns Variant.
Invalid paths and invalid type hints still raise errors.

Register with `register_variant_get_functions` to enable `TRY_CAST` of getters.
Also register `VariantExprPlanner` for `:` field access. `TRY_CAST` of an untyped
getter or colon access uses the same conversion policy as `try_variant_get`.",
    syntax_example = "try_variant_get(value, path[, type])",
    alternative_syntax = "TRY_CAST(variant_get(value, path) AS type)",
    alternative_syntax = "TRY_CAST(value:field AS type)",
    sql_example = r#"```sql
> SELECT try_variant_get(parse_json('{"a":1}'), 'a', 'BIGINT') AS value;
+-------+
| value |
+-------+
| 1     |
+-------+
> SELECT try_variant_get(parse_json('[1,"hello"]'), '[1]', 'BIGINT') IS NULL AS failed_cast;
+-------------+
| failed_cast |
+-------------+
| true        |
+-------------+
> WITH input AS (SELECT parse_json('{"price":"invalid"}') AS v)
  SELECT try_variant_get(v, 'price', 'DECIMAL(12,2)') IS NULL AS type_hint,
         TRY_CAST(variant_get(v, 'price') AS DECIMAL(12,2)) IS NULL AS sql_cast,
         TRY_CAST(v:price AS DECIMAL(12,2)) IS NULL AS colon_cast
  FROM input;
+-----------+----------+------------+
| type_hint | sql_cast | colon_cast |
+-----------+----------+------------+
| true      | true     | true       |
+-----------+----------+------------+
```"#,
    argument(name = "value", description = "Variant expression to extract from."),
    argument(
        name = "path",
        description = "String expression using the path syntax described for `variant_get`."
    ),
    argument(
        name = "type",
        description = "Optional non-null SQL type string literal. Defaults to `VARIANT`; accepts the same types as `variant_get`."
    ),
    related_udf(name = "variant_get"),
    related_udf(name = "is_variant_null")
)]
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
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

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
        standard_variant_get_arg_fields, variant_array_from_json_rows, variant_scalar_from_json,
    };
    use arrow::array::{Array, BinaryViewArray, Int64Array};
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
}
