//! Resolve type hints through DataFusion's public SQL cast planner.

use std::sync::Arc;

use arrow_schema::{DataType, FieldRef};
use datafusion::{
    common::{DFSchema, Result, TableReference, config::ConfigOptions, internal_err, plan_err},
    logical_expr::{
        AggregateUDF, Expr, HigherOrderUDF, ScalarUDF, TableSource, WindowUDF, planner::TypePlanner,
    },
    sql::{
        planner::{ContextProvider, PlannerContext, SqlToRel},
        sqlparser::{
            ast::{CastKind, Expr as SqlExpr, Value},
            dialect::dialect_from_str,
            parser::Parser,
            tokenizer::Token,
        },
    },
};

use crate::VariantTypePlanner;

/// Settings that affect parsing and resolving a SQL type. Keeping
/// them in the UDF makes return-type inference stable for an already-built plan.
/// `with_updated_config` supplies a new instance after SET / RESET.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SqlTypeConfig {
    dialect: String,
    recursion_limit: usize,
    normalize_identifiers: bool,
    varchar_lengths: bool,
    string_views: bool,
    time_zone: Option<String>,
}

impl From<&ConfigOptions> for SqlTypeConfig {
    fn from(config: &ConfigOptions) -> Self {
        Self {
            dialect: config.sql_parser.dialect.to_string(),
            recursion_limit: config.sql_parser.recursion_limit,
            normalize_identifiers: config.sql_parser.enable_ident_normalization,
            varchar_lengths: config.sql_parser.support_varchar_with_length,
            string_views: config.sql_parser.map_string_types_to_utf8view,
            time_zone: config.execution.time_zone.clone(),
        }
    }
}

impl SqlTypeConfig {
    pub(crate) fn resolve(&self, name: &str) -> Result<FieldRef> {
        let dialect = dialect_from_str(&self.dialect).ok_or_else(|| {
            datafusion::common::plan_datafusion_err!("Unknown dialect {}", self.dialect)
        })?;
        let mut parser = Parser::new(dialect.as_ref())
            .with_recursion_limit(self.recursion_limit)
            .try_with_sql(name)?;
        let data_type = parser.parse_data_type()?;
        if parser.peek_token().token != Token::EOF {
            return plan_err!("Expected end of SQL type, found {}", parser.peek_token());
        }

        // Construct an AST, not interpolated SQL. Only a complete type name is
        // accepted above; no expression or trailing SQL can escape this cast.
        let cast = SqlExpr::Cast {
            kind: CastKind::Cast,
            expr: Box::new(SqlExpr::Value(Value::Null.into())),
            data_type,
            format: None,
            array: false,
        };
        let mut config = ConfigOptions::default();
        config.sql_parser.enable_ident_normalization = self.normalize_identifiers;
        config.sql_parser.support_varchar_with_length = self.varchar_lengths;
        config.sql_parser.map_string_types_to_utf8view = self.string_views;
        config.execution.time_zone = self.time_zone.clone();
        let provider = TypeContext(config);
        let expr = SqlToRel::new(&provider).sql_to_expr(
            cast,
            &DFSchema::empty(),
            &mut PlannerContext::new(),
        )?;
        match expr {
            Expr::Cast(cast) => Ok(cast.field),
            _ => internal_err!("SQL type resolution expected a cast expression"),
        }
    }
}

// A cast of NULL needs only configuration, not catalogs or function lookup.
struct TypeContext(ConfigOptions);

impl ContextProvider for TypeContext {
    fn get_type_planner(&self) -> Option<Arc<dyn TypePlanner>> {
        Some(Arc::new(VariantTypePlanner))
    }

    fn options(&self) -> &ConfigOptions {
        &self.0
    }

    fn get_table_source(&self, _: TableReference) -> Result<Arc<dyn TableSource>> {
        internal_err!("Type resolution must not access tables")
    }

    fn get_function_meta(&self, _: &str) -> Option<Arc<ScalarUDF>> {
        None
    }

    fn get_higher_order_meta(&self, _: &str) -> Option<Arc<HigherOrderUDF>> {
        None
    }

    fn get_aggregate_meta(&self, _: &str) -> Option<Arc<AggregateUDF>> {
        None
    }

    fn get_window_meta(&self, _: &str) -> Option<Arc<WindowUDF>> {
        None
    }

    fn get_variable_type(&self, _: &[String]) -> Option<DataType> {
        None
    }

    fn udf_names(&self) -> Vec<String> {
        vec![]
    }

    fn higher_order_function_names(&self) -> Vec<String> {
        vec![]
    }

    fn udaf_names(&self) -> Vec<String> {
        vec![]
    }

    fn udwf_names(&self) -> Vec<String> {
        vec![]
    }
}
