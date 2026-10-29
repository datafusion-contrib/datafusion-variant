// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow::array::{Array, BooleanArray, cast::AsArray};
use arrow::buffer::BooleanBuffer;
use arrow::compute::cast;
use arrow_schema::{ArrowError, DataType, Field};
use datafusion::common::utils::take_function_args;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};
use datafusion::scalar::ScalarValue;
use datafusion_doc::Documentation;
use datafusion_macros::user_doc;
use parquet_variant::Variant;
use parquet_variant_compute::{VariantArray, unshred_variant};

use crate::shared::try_field_as_variant_array;

/// Validates Variant values, preserving SQL NULL and returning false for malformed bytes.
///
/// Uses Arrow validation, which is stricter than Spark for metadata, UTF-8,
/// object-key ordering/uniqueness, container offsets, nesting, and date/time ranges.
/// Arrow also accepts time/nanosecond primitive types not recognized by the pinned
/// Spark revision. Invalid storage schemas and unsupported reconstruction remain errors.
/// Malformed shredded metadata can still panic in Arrow (see the TODO below).
#[user_doc(
    doc_section(label = "Variant Functions"),
    description = "Validates a Variant value using Arrow validation. Returns false for malformed bytes. SQL
NULL stays SQL NULL; Variant null is valid.

Invalid storage schemas and unsupported reconstruction raise errors. Arrow validation is
stricter than Spark for metadata, UTF-8, object keys, offsets, nesting, and date/time
ranges. Malformed shredded metadata can still panic in Arrow; see
[#11069](https://github.com/apache/arrow-rs/issues/11069).",
    syntax_example = "is_valid_variant(value)",
    sql_example = r#"```sql
> SELECT is_valid_variant(parse_json('null')) AS valid;
+-------+
| valid |
+-------+
| true  |
+-------+
```"#,
    argument(name = "value", description = "Variant expression to validate.")
)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct IsValidVariantUdf {
    signature: Signature,
}

impl Default for IsValidVariantUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(1), Volatility::Immutable),
        }
    }
}

fn validate_array(array: &VariantArray) -> Result<BooleanArray> {
    // Arrow normalizes binary widths and dictionary/run-end encoded metadata.
    let metadata = cast(array.metadata_column(), &DataType::BinaryView)?;
    let values = cast(array.value_column(), &DataType::BinaryView)?;
    let metadata = metadata.as_binary_view();
    let values = values.as_binary_view();
    // NullArray has no typed values and is unsupported by Arrow's unshred kernel.
    let shredded = array
        .typed_value_column()
        .is_some_and(|typed| typed.data_type() != &DataType::Null);
    let result = (0..array.len())
        .map(|i| {
            if array.is_null(i) || metadata.is_null(i) {
                return Ok(false);
            }
            let unshredded;
            let value = if shredded {
                // Reconstruct one row at a time so returned errors only affect that row.
                // TODO: reuse a fallible row decoder to avoid rebuilding arrays per row:
                // https://github.com/apache/arrow-rs/issues/11260
                // TODO: adopt Arrow's fix for missing shredded field names in metadata;
                // the kernel currently panics instead of returning an error here:
                // https://github.com/apache/arrow-rs/issues/11069
                unshredded = match unshred_variant(&array.slice(i, 1)) {
                    Ok(row) => row,
                    Err(error @ ArrowError::NotYetImplemented(_)) => return Err(error.into()),
                    Err(_) => return Ok(false),
                };
                // Arrow's unshred kernel retains metadata and emits BinaryView values.
                unshredded.value_column().as_binary_view().value(0)
            } else if values.is_null(i) {
                // The required-value fallback is Variant null, not SQL NULL.
                &[0]
            } else {
                values.value(i)
            };
            // Validate the final value too: nesting may span typed and encoded storage.
            Ok(Variant::try_new(metadata.value(i), value).is_ok())
        })
        .collect::<Result<BooleanBuffer>>()?;
    Ok(BooleanArray::new(result, array.nulls().cloned()))
}

impl ScalarUDFImpl for IsValidVariantUdf {
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn name(&self) -> &str {
        "is_valid_variant"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        let [field] = take_function_args(self.name(), args.arg_fields)?;
        if field.data_type() != &DataType::Null {
            try_field_as_variant_array(field)?;
        }
        Ok(Arc::new(Field::new(
            self.name(),
            DataType::Boolean,
            field.is_nullable() || field.data_type() == &DataType::Null,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [field] = take_function_args(self.name(), &args.arg_fields)?;
        let [arg] = take_function_args(self.name(), &args.args)?;
        if field.data_type() != &DataType::Null {
            try_field_as_variant_array(field)?;
        }
        let array = arg.to_array(1)?;
        let result = if array.data_type() == &DataType::Null {
            BooleanArray::new_null(array.len())
        } else {
            validate_array(&VariantArray::try_new(array.as_ref())?)?
        };
        if matches!(arg, ColumnarValue::Scalar(_)) {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(Arc::new(result)))
        }
    }
}
