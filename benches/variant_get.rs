use std::{hint::black_box, sync::Arc};

use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringViewArray};
use arrow_schema::{DataType, Field};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use datafusion::{
    logical_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl},
    scalar::ScalarValue,
};
use datafusion_variant::{
    VariantGetBoolUdf, VariantGetFloatUdf, VariantGetIntUdf, VariantGetStrUdf, VariantGetUdf,
};
use parquet_variant::{Variant, VariantBuilderExt};
use parquet_variant_compute::{VariantArray, VariantArrayBuilder, VariantType, shred_variant};

const ROWS: usize = 8192;

fn make_args(
    udf: &dyn ScalarUDFImpl,
    input: &ArrayRef,
    path: &str,
    type_hint: Option<&str>,
) -> ScalarFunctionArgs {
    let mut args = vec![
        ColumnarValue::Array(Arc::clone(input)),
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(path.into()))),
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
    group: &mut BenchmarkGroup<'_, WallTime>,
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

    for (name, udf, hint) in cases {
        let args = make_args(udf, &input, "a", hint);
        check_output(
            udf.invoke_with_args(args.clone()).unwrap(),
            values,
            (name != "variant_get").then_some(&expected),
        );
        group.bench_function(BenchmarkId::new(format!("{type_hint}/{name}"), ROWS), |b| {
            // Invocation consumes its arguments; cloning shares the input buffers.
            // Include argument cloning, path parsing, and output allocation/drop.
            b.iter(|| {
                drop(black_box(
                    udf.invoke_with_args(black_box(args.clone())).unwrap(),
                ));
            });
        });
    }
}

fn bench_storage_layouts(c: &mut Criterion, values: &[Variant<'_, '_>]) {
    let mut builder = VariantArrayBuilder::new(ROWS);
    for (row, value) in values.iter().enumerate() {
        builder
            .new_object()
            .with_field("a", value.clone())
            .with_field("b", format!("value_{row}").as_str())
            .finish();
    }
    let unshredded = builder.build();
    let fully_shredded = shred_variant(
        &unshredded,
        &DataType::Struct(
            vec![
                Field::new("a", DataType::Int64, true),
                Field::new("b", DataType::Utf8, true),
            ]
            .into(),
        ),
    )
    .unwrap();
    // Shred only the sibling field, leaving a in binary storage to exercise
    // fallback within a partially shredded object.
    let partially_shredded = shred_variant(
        &unshredded,
        &DataType::Struct(vec![Field::new("b", DataType::Utf8, true)].into()),
    )
    .unwrap();

    let udf = VariantGetUdf::default();
    let mut group = c.benchmark_group("variant_get/storage_layout");
    group.throughput(Throughput::Elements(ROWS as u64));
    for (name, input, residual_nulls) in [
        ("unshredded", unshredded, 0),
        ("fully_shredded", fully_shredded, ROWS),
        ("partially_shredded", partially_shredded, 0),
    ] {
        // Verify storage layouts as well as the extracted values before timing.
        assert_eq!(input.typed_value_field().is_some(), name != "unshredded");
        assert_eq!(input.value_field().unwrap().null_count(), residual_nulls);
        let args = make_args(&udf, &ArrayRef::from(input), "a", None);
        check_output(udf.invoke_with_args(args.clone()).unwrap(), values, None);
        group.bench_function(BenchmarkId::new(name, ROWS), |b| {
            // Match the existing cases: include argument cloning and output drop.
            b.iter(|| {
                drop(black_box(
                    udf.invoke_with_args(black_box(args.clone())).unwrap(),
                ));
            });
        });
    }
    group.finish();
}

fn bench_path_traversal(c: &mut Criterion, values: &[Variant<'_, '_>]) {
    let mut builder = VariantArrayBuilder::new(ROWS);
    for value in values {
        // All paths select the same value from the same document. Distinct array
        // elements make the result checks sensitive to incorrect indexing.
        let mut row = builder.new_object();
        row.insert("top", value.clone());
        let mut object = row.new_object("obj");
        object
            .new_object("b")
            .with_field("c", value.clone())
            .finish();
        object.finish();
        let mut array = row.new_list("arr");
        array.new_list().with_value(0i64).with_value(1i64).finish();
        array
            .new_list()
            .with_value(2i64)
            .with_value(value.clone())
            .finish();
        array.finish();
        let mut mixed = row.new_list("mix");
        mixed.new_object().with_field("b", 0i64).finish();
        mixed.new_object().with_field("b", value.clone()).finish();
        mixed.finish();
        row.finish();
    }
    let input = ArrayRef::from(builder.build());
    let udf = VariantGetUdf::default();
    let mut group = c.benchmark_group("variant_get/path_traversal");
    group.throughput(Throughput::Elements(ROWS as u64));
    // The three nested paths each take three steps; top_level is a one-step control.
    for (name, path) in [
        ("top_level", "top"),
        ("nested_objects", "obj.b.c"),
        ("nested_arrays", "arr[1][1]"),
        ("mixed", "mix[1].b"),
    ] {
        let args = make_args(&udf, &input, path, None);
        check_output(udf.invoke_with_args(args.clone()).unwrap(), values, None);
        group.bench_function(BenchmarkId::new(name, ROWS), |b| {
            // Match the other groups: include argument cloning, path parsing,
            // and output allocation/drop, with preparation outside the timed loop.
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
    let mut group = c.benchmark_group("variant_get/output_types");
    group.throughput(Throughput::Elements(ROWS as u64));
    // Alternate signs and use values outside the i32 range to exercise Int64 storage.
    let integers: Vec<i64> = (0..ROWS)
        .map(|i| {
            let value = (1_i64 << 40) + i as i64;
            if i % 2 == 0 { value } else { -value }
        })
        .collect();
    let integer_values: Vec<_> = integers.iter().copied().map(Variant::from).collect();
    bench_values(
        &mut group,
        &integer_values,
        Arc::new(Int64Array::from(integers.clone())),
        &VariantGetIntUdf::default(),
    );
    let floats: Vec<f64> = (0..ROWS)
        .map(|i| (i as f64 - ROWS as f64 / 2.0) / 4.0)
        .collect();
    bench_values(
        &mut group,
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
        &mut group,
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
        &mut group,
        &strings
            .iter()
            .map(|s| Variant::from(s.as_str()))
            .collect::<Vec<_>>(),
        Arc::new(StringViewArray::from_iter_values(&strings)),
        &VariantGetStrUdf::default(),
    );
    group.finish();
    bench_storage_layouts(c, &integer_values);
    bench_path_traversal(c, &integer_values);
}

criterion_group!(benches, variant_get);
criterion_main!(benches);
