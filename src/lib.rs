#![warn(clippy::all)]

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod shared;
mod sql_type;

mod expr_planner;
mod impl_variant_get;
mod is_valid_variant;
mod is_variant_null;
mod parse_json;
mod schema_of_variant;
mod to_json;
mod variant_contains;
mod variant_conversion;
mod variant_explode;
mod variant_get;
mod variant_get_rewrite;
mod variant_list_construct;
mod variant_list_delete;
mod variant_list_insert;
mod variant_normalize;
mod variant_object_construct;
mod variant_object_delete;
mod variant_object_insert;
mod variant_object_keys;
mod variant_pretty;
mod variant_type;

pub use expr_planner::*;
pub use is_valid_variant::*;
pub use is_variant_null::*;
pub use parse_json::*;
pub use schema_of_variant::*;
pub use to_json::*;
pub use variant_contains::*;
pub use variant_conversion::*;
pub use variant_explode::*;
pub use variant_get::*;
pub use variant_get_rewrite::*;
pub use variant_list_construct::*;
pub use variant_list_delete::*;
pub use variant_list_insert::*;
pub use variant_normalize::*;
pub use variant_object_construct::*;
pub use variant_object_delete::*;
pub use variant_object_insert::*;
pub use variant_object_keys::*;
pub use variant_pretty::*;
pub use variant_type::*;

/// Register all Variant scalar, aggregate, and table functions, cast rewrites,
/// `:` field access, and SQL `VARIANT` type syntax.
///
/// Existing functions with the same names and the session's type planner are
/// replaced. Call during session setup, before preparing statements: installing
/// the type planner rebuilds the session state and clears prepared statements.
pub fn register_all(
    ctx: &mut datafusion::prelude::SessionContext,
) -> datafusion::error::Result<()> {
    use datafusion::execution::{FunctionRegistry, SessionStateBuilder};
    use datafusion::logical_expr::{AggregateUDF, ScalarUDF};
    use std::sync::Arc;

    {
        let state = ctx.state_ref();
        let mut state = state.write();
        *state = SessionStateBuilder::new_from_existing(state.clone())
            .with_session_id(state.session_id().to_owned())
            .with_type_planner(Arc::new(VariantTypePlanner))
            .build();
    }

    for udf in [
        ScalarUDF::new_from_impl(ParseJsonUdf::default()),
        ScalarUDF::new_from_impl(TryParseJsonUdf::default()),
        ScalarUDF::new_from_impl(VariantToJsonUdf::default()),
        ScalarUDF::new_from_impl(IsVariantNullUdf::default()),
        ScalarUDF::new_from_impl(IsValidVariantUdf::default()),
        ScalarUDF::new_from_impl(SchemaOfVariantUdf::default()),
        ScalarUDF::new_from_impl(VariantContainsUdf::default()),
        ScalarUDF::new_from_impl(VariantGetJsonUdf::default()),
        ScalarUDF::new_from_impl(VariantObjectConstruct::default()),
        ScalarUDF::new_from_impl(VariantObjectKeys::default()),
        ScalarUDF::new_from_impl(VariantObjectInsert::default()),
        ScalarUDF::new_from_impl(VariantObjectDelete::default()),
        ScalarUDF::new_from_impl(VariantListConstruct::default()),
        ScalarUDF::new_from_impl(VariantListInsert::default()),
        ScalarUDF::new_from_impl(VariantListDelete::default()),
        ScalarUDF::new_from_impl(VariantNormalizeUdf::default()),
        ScalarUDF::new_from_impl(VariantPretty::default()),
    ] {
        ctx.register_udf(Arc::new(udf))?;
    }
    ctx.register_udaf(Arc::new(AggregateUDF::new_from_impl(
        SchemaOfVariantAggUdf::default(),
    )))?;
    register_variant_get_functions(ctx)?;
    register_variant_conversion_functions(ctx)?;
    register_variant_explode_functions(ctx)?;
    ctx.register_expr_planner(Arc::new(VariantExprPlanner))
}
