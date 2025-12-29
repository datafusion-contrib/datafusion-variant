use arrow::array::AsArray;
use arrow_schema::DataType;
use datafusion::{
    error::Result,
    logical_expr::{Accumulator, AggregateUDFImpl, Signature, TypeSignature, Volatility, function::AccumulatorArgs},
    scalar::ScalarValue,
};
use parquet_variant_compute::VariantArray;

use crate::{VariantSchema, merge_variant_schema, print_schema, schema_from_variant};

#[derive(Debug, Hash, PartialEq, Eq)]
struct VariantSchemaAggUDAF {
    signature: Signature,
}

impl Default for VariantSchemaAggUDAF {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::VariadicAny, Volatility::Immutable),
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

    fn accumulator(
        &self,
        acc_args: datafusion::logical_expr::function::AccumulatorArgs,
    ) -> Result<Box<dyn datafusion::logical_expr::Accumulator>> {
        Ok(Box::new(VariantSchemaAccumulator::new(acc_args)))
    }
}

#[derive(Debug)]
/// An accumulator to compute and store merged VariantSchame
pub struct VariantSchemaAccumulator {
    schema: VariantSchema, // This will store the current inferred schema
}

impl VariantSchemaAccumulator {
    fn new(_acc_args: AccumulatorArgs) -> Self {
        // Initialize with Variant as the starting schema
        Self {
            schema: VariantSchema::Primitive(DataType::Null),
        }
    }
}

impl Accumulator for VariantSchemaAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        // Return the current state (the inferred schema)
        Ok(vec![ScalarValue::Utf8View(Some(print_schema(
            &self.schema,
        )))])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        // Return the schema as a Utf8 representation
        Ok(ScalarValue::Utf8View(Some(print_schema(&self.schema))))
    }

    fn update_batch(&mut self, values: &[arrow::array::ArrayRef]) -> Result<()> {
        // We're assuming the input is an array of variants
        for value in values {
            // Ensure we are dealing with VariantArray and extract the variant values
            let variant_array = VariantArray::try_new(value.as_struct())?;
            for variant in variant_array.iter().flatten() {
                let new_schema = schema_from_variant(&variant);
                // Merge the new schema with the current schema
                self.schema = merge_variant_schema(self.schema.clone(), new_schema);
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[arrow::array::ArrayRef]) -> Result<()> {
        // Merge schemas from other states (batches)
        for state in states {
            let variant_array = VariantArray::try_new(state.as_struct())?;
            for variant in variant_array.iter().flatten() {
                let new_schema = schema_from_variant(&variant);
                self.schema = merge_variant_schema(self.schema.clone(), new_schema);
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

    use arrow::array::{ArrayRef};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        logical_expr::{
            Accumulator,
            function::AccumulatorArgs,
        },
        scalar::ScalarValue,
    };
    use parquet_variant_compute::VariantType;

    use crate::{
        shared::build_variant_array_from_json_array, variant_schema_agg::VariantSchemaAccumulator,
    };

    #[test]
    fn test_get_agg_variant_schema() {
        let b = build_variant_array_from_json_array(&[
            Some(serde_json::json!({"foo": "bar", "wing": {"ding": "dong"}})),
            None,
            Some(serde_json::json!({"wing": {"ding": "man"}})),
        ]);
        let b: ArrayRef = Arc::new(b.into_inner());

        let schema = Schema::new(vec![Field::new("b", DataType::Struct(_), true).with_extension_type(VariantType)]);

        let acc_args = AccumulatorArgs {
            return_field: Arc::new(Field::new("result", DataType::Utf8View, true)),
            schema: &Schema::empty(),
            ignore_nulls: false,
            order_bys:,
            is_reversed: false,
            name: "variant_schema_agg",
            is_distinct: false,
            exprs:,
        };

        let mut schema = VariantSchemaAccumulator::new(acc_args);
        schema.update_batch(&[Arc::clone(&b)]).unwrap();
        let final_schema = schema.evaluate().unwrap();
        assert_eq!(
            final_schema,
            ScalarValue::Utf8View(Some(
                "OBJECT<foo: Utf8, wing: OBJECT<ding: Utf8>>".to_string()
            ))
        )
    }

    // #[test]
    // fn test_get_array_variant_schema() {
    //     let udaf = VariantSchemaAggUDAF::default();
    //     let variant_array = build_variant_array_from_json_array(&[
    //         Some(serde_json::json!({"foo": "bar", "wing": {"ding": "dong"}})),
    //         None,
    //         Some(serde_json::json!({"wing": {"ding": "man"}})),
    //     ]);
    //     let struct_array = variant_array.into_inner();
    //     let args = build_array_udf_args(struct_array);
    //     let result = udaf.accumulator(acc_args)
    //     let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
    //         panic!()
    //     };
    //     assert_eq!(schema, "OBJECT<foo: Utf8, wing: OBJECT<ding: Utf8>>")
    // }

    // #[test]
    // fn test_get_array_variant_conflicting_schema() {
    //     let udf = VariantSchemaAggUDAF::default();
    //     let variant_array = build_variant_array_from_json_array(&[
    //         Some(serde_json::json!({"foo": "bar", "wing": {"ding": "dong"}})),
    //         // None,
    //         Some(serde_json::json!({"wing": 123})),
    //     ]);
    //     let struct_array = variant_array.into_inner();
    //     let args = build_array_udf_args(struct_array);
    //     let result = udf.invoke_with_args(args).unwrap();
    //     let ColumnarValue::Scalar(ScalarValue::Utf8View(Some(schema))) = result else {
    //         panic!()
    //     };
    //     assert_eq!(schema, "OBJECT<foo: Utf8, wing: VARIANT>")
    // }
}
