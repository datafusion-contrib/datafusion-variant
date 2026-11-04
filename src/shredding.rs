// SPDX-License-Identifier: Apache-2.0

//! Explicit Variant storage layouts for Parquet COPY statements.

use std::{any::Any, collections::BTreeMap, sync::Arc};

use arrow::array::new_empty_array;
use arrow_schema::{DataType, FieldRef};
use datafusion::{
    common::{
        Column, Result,
        config::{ConfigEntry, ConfigExtension, ConfigOptions, ExtensionOptions},
        internal_err, plan_datafusion_err, plan_err,
        tree_node::{Transformed, TransformedResult},
        utils::take_function_args,
    },
    functions::utils::make_scalar_function,
    logical_expr::{
        ColumnarValue, Expr, LogicalPlan, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
        ScalarUDFImpl, Signature, Volatility, logical_plan::builder::project,
    },
    optimizer::AnalyzerRule,
};
use parquet_variant_compute::{VariantArray, shred_variant, unshred_variant};

use crate::{
    sql_type::SqlTypeConfig,
    variant_type::{contains_variant, is_variant},
};

const SHREDDING_OPTION: &str = "variant.shredding";

// datafusion-cli validates table options while registering the destination
// object store, before analyzer rules run. Declare the option in its own
// namespace; the COPY analyzer below owns its interpretation.
// TableOptions passes fully qualified keys, unlike the extensions_options!
// macro, which expects bare field names.
#[derive(Debug, Clone, Default)]
pub(crate) struct VariantWriteOptions {
    shredding: Option<String>,
}

impl ConfigExtension for VariantWriteOptions {
    const PREFIX: &'static str = "variant";
}

impl ExtensionOptions for VariantWriteOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> Result<()> {
        if key != SHREDDING_OPTION {
            return plan_err!("Unknown Variant write option '{key}'");
        }
        self.shredding = Some(value.to_owned());
        Ok(())
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        vec![ConfigEntry {
            key: SHREDDING_OPTION.to_owned(),
            value: self.shredding.clone(),
            description: "JSON map from COPY output columns to Variant shredding SQL types",
        }]
    }
}

/// Consume the COPY option before DataFusion validates Parquet format options.
#[derive(Debug)]
pub(crate) struct VariantShredding;

impl AnalyzerRule for VariantShredding {
    fn name(&self) -> &str {
        "variant_shredding"
    }

    fn analyze(&self, plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_up_with_subqueries(|mut plan| {
            let LogicalPlan::Copy(copy) = &mut plan else {
                return Ok(Transformed::no(plan));
            };
            let Some(option) = copy.options.remove(SHREDDING_OPTION) else {
                return Ok(Transformed::no(plan));
            };
            if !copy.file_type.get_ext().eq_ignore_ascii_case("parquet") {
                return plan_err!("SHREDDING is only supported for COPY to Parquet");
            }
            let layouts: BTreeMap<String, String> = serde_json::from_str(&option).map_err(|e| {
                plan_datafusion_err!("SHREDDING must be a JSON object mapping column names to SQL types: {e}")
            })?;
            let schema = copy.input.schema();
            let type_config = SqlTypeConfig::from(config);
            let mut expressions: Vec<_> = schema.columns().into_iter().map(Expr::Column).collect();
            for (name, sql_type) in layouts {
                // Resolve an unqualified name first so ambiguous output names are rejected.
                let field = schema.field_with_unqualified_name(&name)?;
                let index = schema.index_of_column(&Column::from_name(&name))?;
                if copy.partition_by.contains(&name) {
                    return plan_err!("SHREDDING column '{name}' cannot be a partition column");
                }
                if !is_variant(field) {
                    return plan_err!("SHREDDING column '{name}' must be VARIANT");
                }
                let target = type_config.resolve(&sql_type)?;
                // TODO: Allow Variant children once Arrow's shredding kernel supports them:
                // https://github.com/apache/arrow-rs/issues/12078
                if contains_variant(&target) {
                    return plan_err!("SHREDDING layout cannot contain VARIANT; omit fields to keep them in residual data");
                }
                let udf = ScalarUDF::from(ShredVariant {
                    signature: Signature::any(1, Volatility::Immutable),
                    target: target.data_type().clone(),
                });
                expressions[index] = udf.call(vec![expressions[index].clone()]).alias(name);
            }
            // The projection fixes the storage schema before any batches reach the sink.
            // Removing the option also makes repeated analysis idempotent.
            copy.input = Arc::new(project(copy.input.as_ref().clone(), expressions)?);
            Ok(Transformed::yes(plan))
        })
        .data()
    }
}

// Embedded by the analyzer only; this is not a new SQL scalar function.
#[derive(Debug, Hash, PartialEq, Eq)]
struct ShredVariant {
    signature: Signature,
    target: DataType,
}

impl ShredVariant {
    fn shred(&self, mut input: VariantArray) -> Result<VariantArray> {
        if input.typed_value_column().is_some() {
            // TODO: Avoid this binary reconstruction once Arrow supports direct reshredding:
            // https://github.com/apache/arrow-rs/issues/8153
            input = unshred_variant(&input)?;
        }
        Ok(shred_variant(&input, &self.target)?)
    }
}

impl ScalarUDFImpl for ShredVariant {
    fn name(&self) -> &str {
        "__shred_variant"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("__shred_variant requires field-aware return type inference")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [field] = take_function_args(self.name(), args.arg_fields)?;
        let input = VariantArray::try_new(new_empty_array(field.data_type()).as_ref())?;
        // The kernel validates input and target layouts and provides the output
        // schema even for empty input. Preserve extension metadata and nullability.
        let output = self.shred(input)?;
        Ok(Arc::new(
            field
                .as_ref()
                .clone()
                .with_data_type(output.data_type().clone()),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(
            |arrays| {
                let [array] = take_function_args(self.name(), arrays)?;
                let input = VariantArray::try_new(array.as_ref())?;
                Ok(self.shred(input)?.into())
            },
            vec![],
        )(&args.args)
    }
}
