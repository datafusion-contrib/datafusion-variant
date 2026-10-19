use std::sync::Arc;

use arrow_schema::{DataType, Field, FieldRef};
use datafusion::{
    common::{Result, plan_err},
    logical_expr::planner::TypePlanner,
    sql::sqlparser::ast::DataType as SqlDataType,
};
use parquet_variant_compute::{VariantArrayBuilder, VariantType};

/// Resolve SQL `VARIANT` to its Arrow storage field and extension metadata.
/// Register with `SessionStateBuilder::with_type_planner` for SQL type syntax.
#[derive(Debug, Default)]
pub struct VariantTypePlanner;

impl TypePlanner for VariantTypePlanner {
    fn plan_type_field(&self, sql_type: &SqlDataType) -> Result<Option<FieldRef>> {
        let SqlDataType::Custom(name, parameters) = sql_type else {
            return Ok(None);
        };
        let [part] = name.0.as_slice() else {
            return Ok(None);
        };
        if !part
            .as_ident()
            .is_some_and(|ident| ident.value.eq_ignore_ascii_case("variant"))
        {
            return Ok(None);
        }
        if !parameters.is_empty() {
            return plan_err!("VARIANT does not accept type parameters");
        }
        Ok(Some(Arc::new(variant_field(""))))
    }
}

pub(crate) fn variant_field(name: &str) -> Field {
    VariantArrayBuilder::new(0)
        .build()
        .field(name)
        .with_nullable(true)
}

pub(crate) fn is_variant(field: &Field) -> bool {
    field.has_valid_extension_type::<VariantType>()
}

// The upstream getter handles a top-level Variant through untyped extraction,
// but interprets nested extension fields as ordinary structs.
pub(crate) fn validate_getter_target(field: &Field) -> Result<()> {
    fn contains_variant(field: &Field) -> bool {
        is_variant(field)
            || match field.data_type() {
                DataType::List(child) | DataType::FixedSizeList(child, _) => {
                    contains_variant(child)
                }
                DataType::Struct(fields) => fields.iter().any(|child| contains_variant(child)),
                _ => false,
            }
    }
    if !is_variant(field) && contains_variant(field) {
        return plan_err!("Nested VARIANT getter targets are not yet supported");
    }
    Ok(())
}
