// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow::array::{
    Array, Int32Array, ListArray, StringBuilder, StructArray, UInt64Array, cast::AsArray,
    new_empty_array,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::{cast, take};
use arrow_schema::{DataType, Field, FieldRef, Fields};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::utils::take_function_args;
use datafusion::common::{
    DFSchema, NullEquality, NullHandling, UnnestOptions, exec_datafusion_err, plan_err,
};
use datafusion::error::Result;
use datafusion::functions::core::expr_fn::get_field;
use datafusion::functions::utils::make_scalar_function;
use datafusion::logical_expr::planner::{
    PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning,
};
use datafusion::logical_expr::{
    ColumnarValue, Expr, JoinType, LogicalPlan, LogicalPlanBuilder, Operator, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility, binary_expr, col,
};
use datafusion::optimizer::analyzer::AnalyzerRule;
use datafusion::prelude::SessionContext;
use parquet_variant::Variant;
use parquet_variant_compute::{VariantArray, VariantValueArrayBuilder, unshred_variant};
use sqlparser::ast::{FunctionArg, FunctionArgExpr, TableFactor};

use crate::{
    shared::{try_field_as_variant_array, try_parse_binary_columnar},
    variant_type::variant_field,
};

/// Register `variant_explode` and `variant_explode_outer` table functions,
/// including their correlated `LATERAL` planning rule.
pub fn register_variant_explode_functions(ctx: &mut SessionContext) -> Result<()> {
    ctx.register_relation_planner(Arc::new(VariantExplodePlanner))?;
    ctx.add_analyzer_rule(Arc::new(VariantExplodePlanner));
    Ok(())
}

#[derive(Debug, Default)]
struct VariantExplodePlanner;

impl AnalyzerRule for VariantExplodePlanner {
    fn name(&self) -> &str {
        "variant_explode_lateral"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_up_with_subqueries(rewrite_lateral_explode)
            .data()
    }
}

// DataFusion decorrelates predicates in lateral subqueries, but cannot yet
// move a projection of outer references below an Unnest. For our generated
// plan, expand directly over the left input; no join keys or deduplication are
// needed, so repeated input rows retain their multiplicity.
// TODO: Remove this rewrite when upstream decorrelates outer projections through
// Unnest: https://github.com/apache/datafusion/issues/21201 and
// https://github.com/apache/datafusion/issues/11773.
fn rewrite_lateral_explode(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Join(join) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let right = strip_relation_aliases(&join.right);
    let LogicalPlan::Subquery(subquery) = right else {
        return Ok(Transformed::no(plan));
    };
    let LogicalPlan::Projection(output) = strip_relation_aliases(&subquery.subquery) else {
        return Ok(Transformed::no(plan));
    };
    let LogicalPlan::Unnest(unnest) = output.input.as_ref() else {
        return Ok(Transformed::no(plan));
    };
    let LogicalPlan::Projection(input) = unnest.input.as_ref() else {
        return Ok(Transformed::no(plan));
    };
    let [expr] = input.expr.as_slice() else {
        return Ok(Transformed::no(plan));
    };
    let Expr::ScalarFunction(call) = expr.clone().unalias() else {
        return Ok(Transformed::no(plan));
    };
    if !call.func.inner().is::<VariantExplodeEntries>()
        || !matches!(input.input.as_ref(), LogicalPlan::EmptyRelation(e) if e.produce_one_row)
        || join.right.schema().fields().len() != 3
        || output.expr != entry_projection(input.schema.field(0).name())
    {
        return Ok(Transformed::no(plan));
    }
    if join.join_type != JoinType::Inner {
        return plan_err!(
            "Correlated variant_explode supports CROSS/INNER JOIN LATERAL; use variant_explode_outer to preserve empty inputs"
        );
    }
    let expr = Expr::ScalarFunction(call)
        .transform_up(|expr| match expr {
            Expr::OuterReferenceColumn(_, ref column) if join.left.schema().has_column(column) => {
                Ok(Transformed::yes(Expr::Column(column.clone())))
            }
            _ => Ok(Transformed::no(expr)),
        })?
        .data;
    let mut entries = "__variant_explode_entries".to_string();
    while join
        .left
        .schema()
        .has_column_with_unqualified_name(&entries)
    {
        entries.push('_');
    }
    let left_columns: Vec<_> = join
        .left
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    let mut projection = left_columns.clone();
    projection.push(expr.alias(&entries));
    let mut output = left_columns;
    for (expr, (qualifier, field)) in entry_projection(&entries)
        .into_iter()
        .zip(join.right.schema().iter())
    {
        output.push(
            expr.unalias()
                .alias_qualified(qualifier.cloned(), field.name()),
        );
    }
    let mut result = LogicalPlanBuilder::from(Arc::clone(&join.left))
        .project(projection)?
        .unnest_column_with_options(entries, unnest.options.clone())?
        .project(output)?;
    for (left, right) in &join.on {
        let predicate = match join.null_equality {
            NullEquality::NullEqualsNothing => left.clone().eq(right.clone()),
            NullEquality::NullEqualsNull => {
                binary_expr(left.clone(), Operator::IsNotDistinctFrom, right.clone())
            }
        };
        result = result.filter(predicate)?;
    }
    if let Some(filter) = &join.filter {
        result = result.filter(filter.clone())?;
    }
    Ok(Transformed::yes(result.build()?))
}

// Strip only aliases and positional column renames, never a user projection
// that reorders, duplicates, drops, or computes columns.
fn strip_relation_aliases(mut plan: &LogicalPlan) -> &LogicalPlan {
    loop {
        match plan {
            LogicalPlan::SubqueryAlias(alias) => plan = &alias.input,
            LogicalPlan::Projection(projection)
                if projection.expr.len() == projection.input.schema().fields().len()
                    && projection
                        .expr
                        .iter()
                        .zip(projection.input.schema().columns())
                        .all(|(expr, column)| expr.clone().unalias() == Expr::Column(column)) =>
            {
                plan = &projection.input
            }
            _ => return plan,
        }
    }
}

fn entry_projection(entries: &str) -> [Expr; 3] {
    ["pos", "key", "value"].map(|name| get_field(col(entries), name).alias(name))
}

impl RelationPlanner for VariantExplodePlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> Result<RelationPlanning> {
        let (name, args, alias, with_ordinality) = match &relation {
            TableFactor::Table {
                name,
                args: Some(args),
                alias,
                with_ordinality,
                ..
            } => (name, &args.args, alias, *with_ordinality),
            TableFactor::Function {
                name,
                args,
                alias,
                with_ordinality,
                ..
            } => (name, args, alias, *with_ordinality),
            _ => return Ok(RelationPlanning::Original(Box::new(relation))),
        };
        let [part] = name.0.as_slice() else {
            return Ok(RelationPlanning::Original(Box::new(relation)));
        };
        let Some(ident) = part.as_ident() else {
            return Ok(RelationPlanning::Original(Box::new(relation)));
        };
        let name = context.normalize_ident(ident.clone());
        let outer = match name.as_str() {
            "variant_explode" => false,
            "variant_explode_outer" => true,
            _ => return Ok(RelationPlanning::Original(Box::new(relation))),
        };
        if with_ordinality {
            return plan_err!("{name} does not support WITH ORDINALITY; use pos");
        }
        if let TableFactor::Table {
            args: Some(args),
            with_hints,
            version,
            partitions,
            json_path,
            sample,
            index_hints,
            ..
        } = &relation
            && (args.settings.is_some()
                || !with_hints.is_empty()
                || version.is_some()
                || !partitions.is_empty()
                || json_path.is_some()
                || sample.is_some()
                || !index_hints.is_empty())
        {
            return plan_err!("{name} does not support table modifiers");
        }
        let [arg] = args.as_slice() else {
            return plan_err!("{name} requires exactly one argument: input");
        };
        let arg = match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
            FunctionArg::Named {
                name,
                arg: FunctionArgExpr::Expr(expr),
                ..
            } if context.normalize_ident(name.clone()) == "input" => expr,
            _ => return plan_err!("{name} requires an expression, optionally named input"),
        };
        let input = context.sql_to_expr(arg.clone(), &DFSchema::empty())?;
        let entries = "__variant_explode_entries";
        let udf = ScalarUDF::new_from_impl(VariantExplodeEntries::default());
        let null_handling = if outer {
            NullHandling::PreserveAndExpandEmpty
        } else {
            NullHandling::Drop
        };
        let plan = LogicalPlanBuilder::empty(true)
            .project(vec![udf.call(vec![input]).alias(entries)])?
            .unnest_column_with_options(
                entries,
                UnnestOptions::new().with_null_handling(null_handling),
            )?
            .project(entry_projection(entries))?
            .build()?;
        Ok(RelationPlanning::Planned(Box::new(PlannedRelation::new(
            plan,
            alias.clone(),
        ))))
    }
}

fn entry_fields() -> Fields {
    vec![
        Field::new("pos", DataType::Int32, false),
        Field::new("key", DataType::Utf8, true),
        variant_field("value").with_nullable(false),
    ]
    .into()
}

fn entry_field() -> FieldRef {
    Arc::new(Field::new("item", DataType::Struct(entry_fields()), false))
}

// Internal expression used by the relation planner, not registered as a SQL scalar function.
#[derive(Debug, Hash, PartialEq, Eq)]
struct VariantExplodeEntries {
    signature: Signature,
}

impl Default for VariantExplodeEntries {
    fn default() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for VariantExplodeEntries {
    fn name(&self) -> &str {
        "__variant_explode_entries"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(entry_field()))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [field] = take_function_args(self.name(), args.arg_fields)?;
        if field.data_type() != &DataType::Null {
            try_field_as_variant_array(field)?;
        }
        Ok(Arc::new(Field::new(
            self.name(),
            self.return_type(&[])?,
            false,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(
            |arrays| {
                let [array] = take_function_args(self.name(), arrays)?;
                let result = if array.data_type() == &DataType::Null {
                    ListArray::new(
                        entry_field(),
                        OffsetBuffer::new_zeroed(array.len()),
                        new_empty_array(&DataType::Struct(entry_fields())),
                        None,
                    )
                } else {
                    explode_entries(&VariantArray::try_new(array.as_ref())?)?
                };
                Ok(Arc::new(result))
            },
            vec![],
        )(&args.args)
    }
}

fn explode_entries(array: &VariantArray) -> Result<ListArray> {
    let unshredded;
    let array = if array
        .typed_value_column()
        .is_some_and(|typed| typed.data_type() != &DataType::Null)
    {
        // Inherits Arrow's unsupported-layout errors.
        // TODO: Adopt Arrow's fix for the missing-metadata-key panic when available:
        // https://github.com/apache/arrow-rs/issues/11069
        unshredded = unshred_variant(array)?;
        &unshredded
    } else {
        array
    };
    let metadata = cast(array.metadata_column(), &DataType::BinaryView)?;
    let values = try_parse_binary_columnar(array.value_column())?;
    let metadata = metadata.as_binary_view();
    let mut positions = Vec::new();
    let mut keys = StringBuilder::new();
    let mut children = VariantValueArrayBuilder::new(0);
    let mut parents = Vec::new();
    let mut offsets = vec![0i32];
    for (row, bytes) in values.into_iter().enumerate() {
        if !array.is_null(row) {
            if metadata.is_null(row) {
                return Err(exec_datafusion_err!(
                    "variant_explode: missing Variant metadata"
                ));
            }
            // Arrow's fallible decoder validates the entire value, including nested
            // children. This is stricter than Spark's lazy container traversal.
            // TODO: Use fallible shallow views to read and copy immediate children
            // without recursive validation once the API is available in our dependency:
            // https://github.com/apache/arrow-rs/pull/11511
            let value = Variant::try_new(metadata.value(row), bytes.unwrap_or(&[0]))?;
            let mut append =
                |pos: usize, key: Option<&str>, value: Variant<'_, '_>| -> Result<()> {
                    positions.push(i32::try_from(pos).map_err(|_| {
                        exec_datafusion_err!("variant_explode: position exceeds INT32")
                    })?);
                    keys.append_option(key);
                    children.append_value(value);
                    parents.push(row as u64);
                    Ok(())
                };
            match value {
                Variant::Object(object) => {
                    for (pos, (key, value)) in object.iter().enumerate() {
                        append(pos, Some(key), value)?;
                    }
                }
                Variant::List(list) => {
                    for (pos, value) in list.iter().enumerate() {
                        append(pos, None, value)?;
                    }
                }
                _ => {}
            }
        }
        offsets.push(
            i32::try_from(positions.len()).map_err(|_| {
                exec_datafusion_err!("variant_explode: output exceeds List capacity")
            })?,
        );
    }
    let variant_type = variant_field("value");
    let DataType::Struct(storage) = variant_type.data_type() else {
        unreachable!()
    };
    // Keep the parent dictionary: Arrow's value builder copies nested container
    // bytes without recursively rebuilding them or converting through JSON.
    let children = StructArray::try_new(
        storage.clone(),
        vec![
            take(metadata, &UInt64Array::from(parents), None)?,
            Arc::new(children.build()?),
        ],
        None,
    )?;
    let entries = StructArray::try_new(
        entry_fields(),
        vec![
            Arc::new(Int32Array::from(positions)),
            Arc::new(keys.finish()),
            Arc::new(children),
        ],
        None,
    )?;
    Ok(ListArray::try_new(
        entry_field(),
        OffsetBuffer::new(offsets.into()),
        Arc::new(entries),
        None,
    )?)
}
