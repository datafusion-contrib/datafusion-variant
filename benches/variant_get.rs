use std::{hint::black_box, sync::Arc};

use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringViewArray};
use arrow_schema::{DataType, Field};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use datafusion::{
    logical_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl},
    scalar::ScalarValue,
};
use datafusion_variant::{
    VariantGetBoolUdf, VariantGetFloatUdf, VariantGetIntUdf, VariantGetStrUdf, VariantGetUdf,
};
use parquet_variant::{Variant, VariantBuilderExt};
use parquet_variant_compute::{VariantArray, VariantArrayBuilder, VariantType};

const ROWS: usize = 8192;

fn make_args(
    udf: &dyn ScalarUDFImpl,
    input: &ArrayRef,
    type_hint: Option<&str>,
) -> ScalarFunctionArgs {
    let mut args = vec![
        ColumnarValue::Array(Arc::clone(input)),
        ColumnarValue::Scalar(ScalarValue::Utf8(Some("a".into()))),
    ];
    let mut arg_fields = vec![
        Arc::new(
            Field::new("v", input.data_type().clone(), false).with_extension_type(VariantType),
        ),
        Arc::new(Field::new("path", DataType::Utf8, false)),
    ];
    if let Some(type_hint) = type_hint {
        args.push(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
            type_hint.into(),
        ))));
        arg_fields.push(Arc::new(Field::new("type", DataType::Utf8, false)));
    }
    let scalar_arguments = args
        .iter()
        .map(|arg| match arg {
            ColumnarValue::Scalar(value) => Some(value),
            ColumnarValue::Array(_) => None,
        })
        .collect::<Vec<_>>();
    let return_field = udf
        .return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &scalar_arguments,
        })
        .unwrap();

    ScalarFunctionArgs {
        args,
        arg_fields,
        number_rows: ROWS,
        return_field,
        config_options: Default::default(),
    }
}

fn check_output(output: ColumnarValue, values: &[Variant<'_, '_>], expected: Option<&ArrayRef>) {
    let ColumnarValue::Array(output) = output else {
        panic!("expected array output");
    };
    assert_eq!(output.len(), values.len());
    assert_eq!(output.null_count(), 0);
    if let Some(expected) = expected {
        assert_eq!(output.to_data(), expected.to_data());
    } else {
        // Validate Variant storage as well as every extracted value.
        let output = VariantArray::try_new(output.as_ref()).unwrap();
        for (actual, expected) in output.iter().zip(values) {
            assert_eq!(actual.as_ref(), Some(expected));
        }
    }
}

fn bench_values(
    c: &mut Criterion,
    values: &[Variant<'_, '_>],
    expected: ArrayRef,
    helper: &dyn ScalarUDFImpl,
) {
    let mut builder = VariantArrayBuilder::new(ROWS);
    for value in values {
        builder.new_object().with_field("a", value.clone()).finish();
    }
    let input = ArrayRef::from(builder.build());
    let generic = VariantGetUdf::default();
    let type_hint = expected.data_type().to_string();
    let hinted_name = format!("variant_get_{}", type_hint.to_lowercase());
    let cases: [(&str, &dyn ScalarUDFImpl, Option<&str>); 3] = [
        ("variant_get", &generic, None),
        (&hinted_name, &generic, Some(&type_hint)),
        (helper.name(), helper, None),
    ];

    // Preserve the original integer benchmark IDs and saved baselines.
    let suffix = if expected.data_type() == &DataType::Int64 {
        String::new()
    } else {
        format!("/{type_hint}")
    };
    let mut group = c.benchmark_group(format!(
        "variant_get/unshredded_object/literal_a{suffix}/8192"
    ));
    group.throughput(Throughput::Elements(ROWS as u64));
    for (name, udf, type_hint) in cases {
        let args = make_args(udf, &input, type_hint);
        check_output(
            udf.invoke_with_args(args.clone()).unwrap(),
            values,
            (name != "variant_get").then_some(&expected),
        );
        group.bench_function(name, |b| {
            // Invocation consumes its arguments; cloning shares the input buffers.
            // Include argument cloning, path parsing, and output allocation/drop.
            b.iter(|| {
                drop(black_box(
                    udf.invoke_with_args(black_box(args.clone())).unwrap(),
                ));
            });
        });
    }
    group.finish();
}

fn variant_get(c: &mut Criterion) {
    // Alternate signs and use values outside the i32 range to exercise Int64 storage.
    let integers: Vec<i64> = (0..ROWS)
        .map(|i| {
            let value = (1_i64 << 40) + i as i64;
            if i % 2 == 0 { value } else { -value }
        })
        .collect();
    bench_values(
        c,
        &integers
            .iter()
            .copied()
            .map(Variant::from)
            .collect::<Vec<_>>(),
        Arc::new(Int64Array::from(integers.clone())),
        &VariantGetIntUdf::default(),
    );
    let floats: Vec<f64> = (0..ROWS)
        .map(|i| (i as f64 - ROWS as f64 / 2.0) / 4.0)
        .collect();
    bench_values(
        c,
        &floats
            .iter()
            .copied()
            .map(Variant::from)
            .collect::<Vec<_>>(),
        Arc::new(Float64Array::from(floats.clone())),
        &VariantGetFloatUdf::default(),
    );
    let booleans: Vec<bool> = (0..ROWS).map(|i| i % 2 == 0).collect();
    bench_values(
        c,
        &booleans
            .iter()
            .copied()
            .map(Variant::from)
            .collect::<Vec<_>>(),
        Arc::new(BooleanArray::from(booleans.clone())),
        &VariantGetBoolUdf::default(),
    );
    let strings: Vec<String> = (0..ROWS).map(|i| format!("value_{i}")).collect();
    bench_values(
        c,
        &strings
            .iter()
            .map(|s| Variant::from(s.as_str()))
            .collect::<Vec<_>>(),
        Arc::new(StringViewArray::from_iter_values(&strings)),
        &VariantGetStrUdf::default(),
    );
}

criterion_group!(benches, variant_get);
criterion_main!(benches);
