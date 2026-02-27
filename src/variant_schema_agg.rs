use arrow::array::AsArray;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::{
    error::Result,
    logical_expr::{
        Accumulator, AggregateUDFImpl, Signature, TypeSignature, Volatility,
        function::{AccumulatorArgs, StateFieldsArgs},
        utils::format_state_name,
    },
    scalar::ScalarValue,
};
use parquet_variant_compute::VariantArray;
use std::sync::Arc;

use crate::{
    VariantSchema, merge_variant_schema_from, print_schema, schema_from_variant,
    shared::try_parse_binary_columnar,
};

/// Aggregate schema inference for VARIANT values across rows.
///
/// This function infers per-row schemas using `schema_from_variant` and merges
/// them into a single schema per group.
///
/// Semantics:
/// - Input: one VARIANT expression
/// - Output: one schema string per aggregate group
/// - Row filtering should be done via SQL `FILTER (WHERE ...)`
///
/// Use `variant_schema` for row-wise (non-aggregate) inference.
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantSchemaAggUDAF {
    signature: Signature,
}

impl Default for VariantSchemaAggUDAF {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(1), Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for VariantSchemaAggUDAF {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_schema_agg"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8View)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let fields = vec![Arc::new(Field::new(
            format_state_name(args.name, "variant_schema"),
            DataType::Binary,
            true,
        ))];

        Ok(fields
            .into_iter()
            .chain(args.ordering_fields.to_vec())
            .collect())
    }

    fn accumulator(
        &self,
        acc_args: datafusion::logical_expr::function::AccumulatorArgs,
    ) -> Result<Box<dyn datafusion::logical_expr::Accumulator>> {
        Ok(Box::new(VariantSchemaAccumulator::new(acc_args)))
    }
}

/// Accumulator state for `variant_schema_agg`.
#[derive(Debug)]
pub struct VariantSchemaAccumulator {
    schema: VariantSchema,
}

impl VariantSchemaAccumulator {
    fn new(_acc_args: AccumulatorArgs) -> Self {
        Self {
            schema: VariantSchema::Primitive(DataType::Null),
        }
    }
}

impl Accumulator for VariantSchemaAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Binary(Some(
            self.schema.to_state_bytes(),
        ))])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        // Return the schema as a Utf8 representation
        Ok(ScalarValue::Utf8View(Some(print_schema(&self.schema))))
    }

    fn update_batch(&mut self, values: &[arrow::array::ArrayRef]) -> Result<()> {
        if self.schema == VariantSchema::Variant {
            return Ok(());
        }

        // We're assuming the input is an array of variants
        for value in values {
            // Ensure we are dealing with VariantArray and extract the variant values
            let variant_array = VariantArray::try_new(value.as_struct())?;
            for variant in variant_array.iter().flatten() {
                let new_schema = schema_from_variant(&variant);
                // Merge the new schema with the current schema
                merge_variant_schema_from(&mut self.schema, &new_schema);
                if self.schema == VariantSchema::Variant {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[arrow::array::ArrayRef]) -> Result<()> {
        if self.schema == VariantSchema::Variant {
            return Ok(());
        }

        for state in states {
            for encoded_state in try_parse_binary_columnar(state)?.into_iter().flatten() {
                let new_schema = VariantSchema::from_state_bytes(encoded_state)?;
                merge_variant_schema_from(&mut self.schema, &new_schema);
                if self.schema == VariantSchema::Variant {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn size(&self) -> usize {
        // The size is essentially the number of variants processed, if needed
        1 // This could be expanded to return a more useful size
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow::array::ArrayRef;
    use arrow_schema::{DataType, Field, Fields, Schema};
    use datafusion::{
        logical_expr::{Accumulator, function::AccumulatorArgs},
        physical_expr::PhysicalSortExpr,
        physical_plan::expressions::col,
        scalar::ScalarValue,
    };
    use parquet_variant_compute::VariantType;

    use crate::{
        shared::build_variant_array_from_json_array, variant_schema_agg::VariantSchemaAccumulator,
    };

    #[test]
    fn test_merge_batch_from_state_roundtrip() {
        let schema = Schema::new(vec![
            Field::new(
                "b",
                DataType::Struct(Fields::from(vec![
                    Field::new("metadata", DataType::Binary, true),
                    Field::new("value", DataType::Binary, true),
                ])),
                true,
            )
            .with_extension_type(VariantType),
        ]);

        let b1 = build_variant_array_from_json_array(&[Some(serde_json::json!({"a": 1}))]);
        let b1: ArrayRef = Arc::new(b1.into_inner());

        let b2 = build_variant_array_from_json_array(&[Some(serde_json::json!({"a": 2.5}))]);
        let b2: ArrayRef = Arc::new(b2.into_inner());

        let expr = col("b", &schema).unwrap();
        let order_bys = vec![PhysicalSortExpr::new_default(Arc::clone(&expr))];
        let exprs = vec![expr];
        let expr_fields = vec![Arc::new(
            Field::new(
                "b",
                DataType::Struct(Fields::from(vec![
                    Field::new("metadata", DataType::Binary, true),
                    Field::new("value", DataType::Binary, true),
                ])),
                true,
            )
            .with_extension_type(VariantType),
        )];

        let acc1_args = AccumulatorArgs {
            return_field: Arc::new(Field::new("result", DataType::Utf8View, true)),
            schema: &schema,
            ignore_nulls: false,
            order_bys: &order_bys,
            is_reversed: false,
            name: "variant_schema_agg",
            is_distinct: false,
            exprs: &exprs,
            expr_fields: &expr_fields,
        };
        let acc2_args = AccumulatorArgs {
            return_field: Arc::new(Field::new("result", DataType::Utf8View, true)),
            schema: &schema,
            ignore_nulls: false,
            order_bys: &order_bys,
            is_reversed: false,
            name: "variant_schema_agg",
            is_distinct: false,
            exprs: &exprs,
            expr_fields: &expr_fields,
        };
        let merged_args = AccumulatorArgs {
            return_field: Arc::new(Field::new("result", DataType::Utf8View, true)),
            schema: &schema,
            ignore_nulls: false,
            order_bys: &order_bys,
            is_reversed: false,
            name: "variant_schema_agg",
            is_distinct: false,
            exprs: &exprs,
            expr_fields: &expr_fields,
        };

        let mut acc1 = VariantSchemaAccumulator::new(acc1_args);
        acc1.update_batch(&[Arc::clone(&b1)]).unwrap();
        let state_1 = acc1
            .state()
            .unwrap()
            .into_iter()
            .map(|s| s.to_array().unwrap())
            .collect::<Vec<_>>();

        let mut acc2 = VariantSchemaAccumulator::new(acc2_args);
        acc2.update_batch(&[Arc::clone(&b2)]).unwrap();
        let state_2 = acc2
            .state()
            .unwrap()
            .into_iter()
            .map(|s| s.to_array().unwrap())
            .collect::<Vec<_>>();

        let mut merged = VariantSchemaAccumulator::new(merged_args);
        merged.merge_batch(&state_1).unwrap();
        merged.merge_batch(&state_2).unwrap();

        assert_eq!(
            merged.evaluate().unwrap(),
            ScalarValue::Utf8View(Some("OBJECT<a: Float64>".to_string()))
        );
    }
}
