use std::sync::Arc;

use datafusion::{
    common::DFSchema,
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

#[derive(Debug)]
pub struct VariantExprPlanner;

impl ExprPlanner for VariantExprPlanner {
    fn plan_binary_op(
        &self,
        expr: RawBinaryExpr,
        _schema: &DFSchema,
    ) -> Result<PlannerResult<RawBinaryExpr>, DataFusionError> {
        match &expr.op {
            BinaryOperator::Custom(s) if s == ":" => Ok(PlannerResult::Planned(
                Expr::ScalarFunction(ScalarFunction::new_udf(
                    Arc::new(ScalarUDF::new_from_impl(VariantGetUdf::default())),
                    vec![expr.left, expr.right],
                )),
            )),
            _ => Ok(PlannerResult::Original(expr)),
        }
    }
}
