// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow::{array::StructArray, buffer::NullBuffer, compute::CastOptions};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::{
    common::{DFSchema, Result, config::ConfigOptions, plan_err, tree_node::Transformed},
    execution::FunctionRegistry,
    logical_expr::{
        ColumnarValue, Expr, ExprSchemable, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
        ScalarUDFImpl, Signature, Volatility, expr_rewriter::FunctionRewrite,
    },
    prelude::SessionContext,
    scalar::ScalarValue,
};
use datafusion_doc::Documentation;
use datafusion_macros::user_doc;
use parquet_variant_compute::cast_to_variant_with_options;

use crate::variant_type::{is_variant, variant_field};

/// Register `to_variant_object` and rewriting for `CAST` / `TRY_CAST` to VARIANT.
/// Install [`crate::VariantTypePlanner`] on the session for SQL VARIANT syntax.
pub fn register_variant_conversion_functions(ctx: &mut SessionContext) -> Result<()> {
    ctx.register_udf(Arc::new(ScalarUDF::new_from_impl(
        ToVariantObjectUdf::default(),
    )))?;
    ctx.register_function_rewrite(Arc::new(VariantConversionRewrite))
}

/// Convert arrays, structs and string-keyed maps to a Variant value.
///
/// Use [`register_variant_conversion_functions`] to also register SQL cast rewrites.
#[user_doc(
    doc_section(label = "Variant Functions"),
    description = "Converts an array, struct, or string-keyed map to VARIANT. Structs and maps become unordered objects. SQL NULL inputs remain SQL NULL; null array elements become Variant null.

Unlike Spark, null object fields are currently omitted and duplicate object keys are not rejected. Nested VARIANT inputs are not supported.

Use CAST(value AS VARIANT) for scalar inputs.",
    syntax_example = "to_variant_object(expression)",
    sql_example = r#"```sql
> SELECT to_json(to_variant_object(named_struct('a', 1))) AS json;
+---------+
| json    |
+---------+
| {"a":1} |
+---------+
```"#,
    argument(
        name = "expression",
        description = "An array, struct, or map. Map keys must be strings, including in nested maps."
    ),
    related_udf(name = "parse_json"),
    related_udf(name = "to_json")
)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct ToVariantObjectUdf {
    signature: Signature,
}

impl Default for ToVariantObjectUdf {
    fn default() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for ToVariantObjectUdf {
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
    fn name(&self) -> &str {
        "to_variant_object"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(variant_field(self.name()).data_type().clone())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [input] = args.arg_fields else {
            return plan_err!("to_variant_object expects one argument");
        };
        if is_variant(input)
            || !(input.data_type().is_list()
                || matches!(input.data_type(), DataType::Struct(_) | DataType::Map(_, _)))
        {
            return plan_err!("to_variant_object requires an array, struct, or map");
        }
        validate_source(input, true)?;
        Ok(Arc::new(
            variant_field(self.name()).with_nullable(input.is_nullable()),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        convert_to_variant(&args.args[0], false)
    }
}

// This UDF is embedded by the rewrite, not registered as another SQL function.
#[derive(Debug, Hash, PartialEq, Eq)]
struct ToVariantUdf {
    signature: Signature,
    safe: bool,
}

impl ToVariantUdf {
    fn new(safe: bool) -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
            safe,
        }
    }
}

impl ScalarUDFImpl for ToVariantUdf {
    fn name(&self) -> &str {
        if self.safe {
            "try_cast_to_variant"
        } else {
            "cast_to_variant"
        }
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(variant_field(self.name()).data_type().clone())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [input] = args.arg_fields else {
            return plan_err!("CAST to VARIANT expects one argument");
        };
        validate_source(input, false)?;
        Ok(Arc::new(
            variant_field(self.name()).with_nullable(self.safe || input.is_nullable()),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        convert_to_variant(&args.args[0], self.safe)
    }
}

// TODO: replace this rewrite with registered extension casts once DataFusion
// supports pluggable cast resolution: https://github.com/apache/datafusion/issues/20755
#[derive(Debug)]
struct VariantConversionRewrite;

impl FunctionRewrite for VariantConversionRewrite {
    fn name(&self) -> &str {
        "VariantConversionRewrite"
    }

    fn rewrite(
        &self,
        expr: Expr,
        schema: &DFSchema,
        _config: &ConfigOptions,
    ) -> Result<Transformed<Expr>> {
        let (input, target, safe) = match &expr {
            Expr::Cast(cast) => (&cast.expr, &cast.field, false),
            Expr::TryCast(cast) => (&cast.expr, &cast.field, true),
            _ => return Ok(Transformed::no(expr)),
        };
        if !is_variant(target) {
            return Ok(Transformed::no(expr));
        }
        if is_variant(&input.to_field(schema)?.1) {
            // Preserve storage (including shredding) and leave getters visible
            // to their rewrite, regardless of helper registration order.
            return Ok(Transformed::yes(input.as_ref().clone()));
        }
        Ok(Transformed::yes(
            ScalarUDF::from(ToVariantUdf::new(safe)).call(vec![input.as_ref().clone()]),
        ))
    }
}

// Spark 4.2 VariantGet.checkDataType: scalar/array casts, with structs and maps
// admitted only by to_variant_object. Arrow storage variants of SQL types are
// accepted too. Validate before execution so all-null batches cannot hide errors.
// TODO: reuse Arrow's type-support checks once exposed for planning, while
// retaining Spark's CAST/to_variant_object restrictions here.
// https://github.com/apache/arrow-rs/issues/11946
fn validate_source(field: &Field, allow_objects: bool) -> Result<()> {
    if is_variant(field) {
        // TODO(arrow-rs): teach cast_to_variant to honor nested Variant extension
        // fields instead of serializing their metadata/value storage as objects.
        // https://github.com/apache/arrow-rs/issues/11942
        return plan_err!("Nested VARIANT conversion requires Arrow kernel support");
    }
    match field.data_type() {
        data_type if data_type.is_numeric() || data_type.is_string() || data_type.is_binary() => {
            Ok(())
        }
        DataType::Null
        | DataType::Boolean
        | DataType::Date32
        | DataType::Date64
        | DataType::Timestamp(_, _)
        | DataType::Time32(_)
        | DataType::Time64(_) => Ok(()),
        DataType::List(child)
        | DataType::LargeList(child)
        | DataType::FixedSizeList(child, _)
        | DataType::ListView(child)
        | DataType::LargeListView(child) => validate_source(child, allow_objects),
        DataType::Struct(fields) if allow_objects => {
            for child in fields {
                validate_source(child, allow_objects)?;
            }
            Ok(())
        }
        DataType::Map(entries, _) if allow_objects => {
            let DataType::Struct(fields) = entries.data_type() else {
                return plan_err!("Invalid map entries type");
            };
            if fields.len() != 2 || !fields[0].data_type().is_string() {
                return plan_err!("to_variant_object requires string map keys");
            }
            validate_source(&fields[1], allow_objects)
        }
        DataType::Dictionary(_, values) => validate_source(
            &Field::new("", values.as_ref().clone(), true),
            allow_objects,
        ),
        DataType::Struct(_) | DataType::Map(_, _) => {
            plan_err!("CAST to VARIANT does not accept structs or maps; use to_variant_object")
        }
        data_type => plan_err!("Cannot convert {data_type} to VARIANT"),
    }
}

fn convert_to_variant(value: &ColumnarValue, safe: bool) -> Result<ColumnarValue> {
    let options = CastOptions {
        safe,
        ..Default::default()
    };
    let array = value.to_array(1)?;
    // TODO: pass input field metadata once Arrow supports UUID-aware conversion.
    // Currently arrow.uuid inputs become Variant binary through their FSB16 storage.
    // https://github.com/apache/arrow-rs/issues/11937
    let converted = cast_to_variant_with_options(array.as_ref(), &options)?;
    // Arrow encodes a failed safe scalar conversion as Variant null. TRY_CAST
    // must return SQL NULL instead. A successful typed conversion cannot yield
    // top-level Variant null; existing Variant inputs bypass this kernel.
    // TODO: remove this null-mask adjustment once our Arrow dependency includes
    // the fix for https://github.com/apache/arrow-rs/issues/11939.
    let nulls = safe.then(|| {
        NullBuffer::from_iter(
            converted
                .iter()
                .map(|value| value.is_some_and(|v| v.as_null().is_none())),
        )
    });
    let mut output: StructArray = converted.into();
    if let Some(nulls) = nulls {
        let (fields, columns, _) = output.into_parts();
        output = StructArray::new(fields, columns, Some(nulls));
    }
    Ok(match value {
        ColumnarValue::Scalar(_) => ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(output))),
        ColumnarValue::Array(_) => ColumnarValue::Array(Arc::new(output)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParseJsonUdf, VariantTypePlanner, register_variant_get_functions};
    use datafusion::execution::SessionStateBuilder;

    #[tokio::test]
    async fn getter_and_conversion_registration_order() -> Result<()> {
        for conversions_first in [true, false] {
            let state = SessionStateBuilder::new()
                .with_default_features()
                .with_type_planner(Arc::new(VariantTypePlanner))
                .build();
            let mut ctx = SessionContext::new_with_state(state);
            ctx.register_udf(ScalarUDF::from(ParseJsonUdf::default()));
            if conversions_first {
                register_variant_conversion_functions(&mut ctx)?;
                register_variant_get_functions(&mut ctx)?;
            } else {
                register_variant_get_functions(&mut ctx)?;
                register_variant_conversion_functions(&mut ctx)?;
            }
            let batches = ctx.sql(
                "SELECT CAST(CAST(variant_get(parse_json('{\"a\":7}'), 'a') AS VARIANT) AS INT),
                        CAST(CAST(7 AS VARIANT) AS INT)"
            ).await?.collect().await?;
            for column in batches[0].columns() {
                assert_eq!(
                    ScalarValue::try_from_array(column, 0)?,
                    ScalarValue::Int32(Some(7))
                );
            }
        }
        Ok(())
    }
}
