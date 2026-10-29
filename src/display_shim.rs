// SPDX-License-Identifier: Apache-2.0

//! Temporary output adapter for top-level extension columns in CLI and SLT results.

use std::{fmt::Write, sync::Arc};

use datafusion::arrow::{
    array::StringBuilder,
    datatypes::DataType,
    util::display::{ArrayFormatter, ArrayFormatterFactory, FormatOptions as ArrowFormatOptions},
};
use datafusion::{
    catalog::Session,
    common::{Column, Result},
    config::FormatOptions,
    dataframe::DataFrame,
    functions::utils::make_scalar_function,
    logical_expr::{Expr, Volatility, create_udf, extension_types::DFArrayFormatterFactory},
};

// TODO: remove this output projection once datafusion-cli and the SLT runner
// use registered extension formatters directly. Upstream follow-up:
// https://github.com/apache/datafusion/pull/21714#discussion_r3160465335
pub fn format_for_display(df: DataFrame, options: FormatOptions) -> Result<DataFrame> {
    let (state, plan) = df.into_parts();
    let registry = state.extension_type_registry();
    let mut changed = false;
    let expressions = plan
        .schema()
        .iter()
        .map(|(qualifier, field)| {
            let column = Expr::Column(Column::new(qualifier.cloned(), field.name()));
            let Some(name) = field.extension_type_name() else {
                return column;
            };
            if registry.extension_type_registration(name).is_err() {
                return column;
            }

            changed = true;
            let factory = DFArrayFormatterFactory::new(Arc::clone(registry));
            let options = options.clone();
            let field = Arc::clone(field);
            let column_name = field.name().clone();
            let formatter = create_udf(
                "__extension_display",
                vec![field.data_type().clone()],
                DataType::Utf8,
                Volatility::Volatile,
                Arc::new(make_scalar_function(
                    move |args| {
                        let array = &args[0];
                        let options: ArrowFormatOptions = (&options).try_into()?;
                        let options = options.with_formatter_factory(Some(&factory));
                        let formatter = match factory.create_array_formatter(
                            array.as_ref(),
                            &options,
                            Some(field.as_ref()),
                        )? {
                            Some(formatter) => formatter,
                            None => ArrayFormatter::try_new(array.as_ref(), &options)?,
                        };
                        let mut values = StringBuilder::new();
                        for row in 0..array.len() {
                            let value = formatter.value(row);
                            if options.safe() {
                                write!(values, "{value}")?;
                            } else {
                                value.write(&mut values)?;
                            }
                            values.append_value("");
                        }
                        Ok(Arc::new(values.finish()))
                    },
                    vec![],
                )),
            );
            formatter
                .call(vec![column])
                .alias_qualified(qualifier.cloned(), column_name)
        })
        .collect::<Vec<_>>();
    let df = DataFrame::new(state, plan);
    if changed {
        df.select(expressions)
    } else {
        Ok(df)
    }
}
