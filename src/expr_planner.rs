// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use datafusion::{
    common::{DFSchema, ScalarValue},
    error::DataFusionError,
    logical_expr::{
        ScalarUDF,
        expr::ScalarFunction,
        planner::{ExprPlanner, PlannerResult, RawBinaryExpr},
    },
    prelude::Expr,
    sql::sqlparser::ast::BinaryOperator,
};

use crate::VariantGetUdf;
use crate::variant_path::colon_path;

/// Custom [`ExprPlanner`] used to handle variant-specific syntax such as colon operator.
///
/// Currently implements:
/// - Colon operator: short-hand syntax for `variant_get`.
#[derive(Debug)]
pub struct VariantExprPlanner;

impl ExprPlanner for VariantExprPlanner {
    fn plan_binary_op(
        &self,
        expr: RawBinaryExpr,
        _schema: &DFSchema,
    ) -> Result<PlannerResult<RawBinaryExpr>, DataFusionError> {
        match &expr.op {
            BinaryOperator::Custom(s) if s == ":" => {
                let mut path = expr.right;
                if let Expr::Literal(value, _) = &mut path
                    && let Some(Some(text)) = value.try_as_str()
                {
                    *value = ScalarValue::Utf8(Some(colon_path(text)));
                }
                Ok(PlannerResult::Planned(Expr::ScalarFunction(
                    ScalarFunction::new_udf(
                        Arc::new(ScalarUDF::new_from_impl(VariantGetUdf::default())),
                        vec![expr.left, path],
                    ),
                )))
            }
            _ => Ok(PlannerResult::Original(expr)),
        }
    }
}
