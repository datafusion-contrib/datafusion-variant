use std::{hint::black_box, sync::Arc};

use arrow::array::{ArrayRef, AsArray};
use arrow_schema::{DataType, Field};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use datafusion::{
    logical_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl},
    scalar::ScalarValue,
};
use datafusion_variant::VariantToJsonUdf;
use parquet_variant_compute::{VariantArray, VariantArrayBuilder, VariantType, shred_variant};
use parquet_variant_json::JsonToVariant;

const ROWS: usize = 8192;
const ESCAPED_STRING: &str = r#""line\n\"quoted\"\\tail""#;
const NESTED: &str = r#"{"a":[1,null,true],"b":{"c":"value"}}"#;

fn input(rows: &[Option<&str>]) -> ArrayRef {
    let mut builder = VariantArrayBuilder::new(rows.len());
    for row in rows {
        match row {
            Some(json) => builder.append_json(json).unwrap(),
            None => builder.append_null(),
        }
    }
    builder.build().into()
}

fn bench_input(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    value: ColumnarValue,
    expected: &[Option<&str>],
) {
    let udf = VariantToJsonUdf::default();
    let arg_fields = vec![Arc::new(
        Field::new("v", value.data_type(), true).with_extension_type(VariantType),
    )];
    let scalar_arguments = vec![match &value {
        ColumnarValue::Scalar(value) => Some(value),
        ColumnarValue::Array(_) => None,
    }];
    let return_field = udf
        .return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &scalar_arguments,
        })
        .unwrap();
    let scalar = matches!(&value, ColumnarValue::Scalar(_));
    let args = ScalarFunctionArgs {
        args: vec![value],
        arg_fields,
        number_rows: expected.len(),
        return_field,
        config_options: Default::default(),
    };

    // Validate shape, type, values, and SQL validity outside the timed loop.
    let output = udf.invoke_with_args(args.clone()).unwrap();
    assert_eq!(matches!(&output, ColumnarValue::Scalar(_)), scalar);
    assert_eq!(&output.data_type(), args.return_field.data_type());
    let output = output.into_array(expected.len()).unwrap();
    assert_eq!(output.as_string_view().iter().collect::<Vec<_>>(), expected);

    group.throughput(Throughput::Elements(expected.len() as u64));
    group.bench_function(BenchmarkId::new(name, expected.len()), |b| {
        // Include argument cloning and output allocation/drop; input construction
        // and JSON parsing are outside timing. Cloning shares the input buffers.
        b.iter(|| {
            drop(black_box(
                udf.invoke_with_args(black_box(args.clone())).unwrap(),
            ));
        });
    });
}

fn bench_storage_layouts(c: &mut Criterion) {
    let cases = [
        ("integer", "42", DataType::Int64),
        ("escaped_string", ESCAPED_STRING, DataType::Utf8View),
    ];
    for (shape, rows) in [("scalar", 1), ("array", ROWS)] {
        let mut group = c.benchmark_group(format!("variant_to_json/storage_layout/{shape}"));
        for (name, json, data_type) in &cases {
            let expected = vec![Some(*json); rows];
            let unshredded = VariantArray::try_new(input(&expected).as_ref()).unwrap();
            // Build both representations before timing and verify that shredding
            // moved every value into the typed column, without binary fallback.
            let shredded = shred_variant(&unshredded, data_type).unwrap();
            assert!(unshredded.typed_value_field().is_none());
            let typed = shredded.typed_value_field().unwrap();
            assert_eq!(typed.data_type(), data_type);
            assert_eq!(typed.null_count(), 0);
            assert!(
                shredded
                    .value_field()
                    .is_none_or(|value| value.null_count() == rows)
            );

            for (layout, input) in [("unshredded", unshredded), ("shredded", shredded)] {
                let input = ArrayRef::from(input);
                let value = if shape == "scalar" {
                    ColumnarValue::Scalar(ScalarValue::try_from_array(input.as_ref(), 0).unwrap())
                } else {
                    ColumnarValue::Array(input)
                };
                bench_input(&mut group, &format!("{name}/{layout}"), value, &expected);
            }
        }
        // Shredded object reconstruction is unsupported by the current serializer
        // (#71); add fully/partially shredded object cases with that fix.
        group.finish();
    }
}

fn variant_to_json(c: &mut Criterion) {
    let cases = [
        ("integer", "42"),
        ("variant_null", "null"),
        ("escaped_string", ESCAPED_STRING),
        ("nested", NESTED),
    ];
    let mut scalars = c.benchmark_group("variant_to_json/scalar");
    for (name, json) in cases {
        let expected = [Some(json)];
        let value = ScalarValue::try_from_array(input(&expected).as_ref(), 0).unwrap();
        bench_input(&mut scalars, name, ColumnarValue::Scalar(value), &expected);
    }
    let expected = [None];
    let value = ScalarValue::try_from_array(input(&expected).as_ref(), 0).unwrap();
    bench_input(
        &mut scalars,
        "sql_null",
        ColumnarValue::Scalar(value),
        &expected,
    );
    scalars.finish();

    let mut arrays = c.benchmark_group("variant_to_json/array");
    for (name, json) in cases {
        let expected = vec![Some(json); ROWS];
        let value = ColumnarValue::Array(input(&expected));
        bench_input(&mut arrays, name, value, &expected);
    }
    for (name, expected) in [
        (
            "nested_sql_nulls_25pct",
            (0..ROWS).map(|i| (i % 4 != 0).then_some(NESTED)).collect(),
        ),
        ("sql_nulls_100pct", vec![None; ROWS]),
    ] {
        let value = ColumnarValue::Array(input(&expected));
        bench_input(&mut arrays, name, value, &expected);
    }
    arrays.finish();
    bench_storage_layouts(c);
}

criterion_group!(benches, variant_to_json);
criterion_main!(benches);
