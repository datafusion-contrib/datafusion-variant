use std::sync::Arc;

use datafusion::{
    common::{DFSchema, Result, config::ConfigOptions, tree_node::Transformed},
    execution::FunctionRegistry,
    logical_expr::{
        Expr, ExprSchemable, ScalarUDF, expr::ScalarFunction, expr_rewriter::FunctionRewrite, lit,
    },
    prelude::SessionContext,
};

use crate::variant_type::{is_variant, validate_getter_target};
use crate::{TryVariantGetUdf, VariantGetUdf};

/// Register both generic getters with the session's configuration, plus the
/// rewrite for CAST / TRY_CAST (including casts after `:` field access).
pub fn register_variant_get_functions(ctx: &mut SessionContext) -> Result<()> {
    let config = ctx.copied_config();
    ctx.register_udf(Arc::new(ScalarUDF::new_from_impl(
        VariantGetUdf::new_with_config(config.options()),
    )))?;
    ctx.register_udf(Arc::new(ScalarUDF::new_from_impl(
        TryVariantGetUdf::new_with_config(config.options()),
    )))?;
    ctx.register_function_rewrite(Arc::new(VariantGetRewrite))
}

/// Lower casts of Variant-returning getters to the extraction/conversion kernel,
/// retaining any explicit intermediate conversion.
#[derive(Debug)]
pub struct VariantGetRewrite;

impl FunctionRewrite for VariantGetRewrite {
    fn name(&self) -> &str {
        "VariantGetRewrite"
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
        let mut input = input.as_ref();
        while let Expr::Alias(alias) = input {
            input = alias.expr.as_ref();
        }
        let Expr::ScalarFunction(function) = input else {
            return Ok(Transformed::no(expr));
        };
        let getter = function.func.inner().downcast_ref::<VariantGetUdf>();
        let try_getter = function.func.inner().downcast_ref::<TryVariantGetUdf>();
        if getter.is_none() && try_getter.is_none() {
            return Ok(Transformed::no(expr));
        }
        let untyped = getter.is_some_and(VariantGetUdf::is_untyped)
            || try_getter.is_some_and(TryVariantGetUdf::is_untyped);
        let args = if untyped && function.args.len() == 2 {
            function.args.clone()
        } else if is_variant(&input.to_field(schema)?.1) {
            // Preserve an explicit VARIANT conversion before casting its result.
            vec![input.clone(), lit("")]
        } else {
            return Ok(Transformed::no(expr));
        };
        validate_getter_target(target)?;
        let func = if safe {
            ScalarUDF::new_from_impl(TryVariantGetUdf::with_target(Arc::clone(target)))
        } else {
            ScalarUDF::new_from_impl(VariantGetUdf::with_target(Arc::clone(target)))
        };
        Ok(Transformed::yes(Expr::ScalarFunction(
            ScalarFunction::new_udf(Arc::new(func), args),
        )))
    }
}
