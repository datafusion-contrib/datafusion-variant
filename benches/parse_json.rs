// SPDX-License-Identifier: Apache-2.0

use std::{hint::black_box, sync::Arc};

use arrow::array::StringArray;
use arrow_schema::Field;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use datafusion::{
    logical_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl},
    scalar::ScalarValue,
};
use datafusion_variant::ParseJsonUdf;
use parquet_variant_compute::VariantArray;
use parquet_variant_json::VariantToJson;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const ROWS: usize = 8192;
const ESCAPED_STRING: &str = r#""line\n\"quoted\"\\tail""#;
const NESTED: &str = r#"{"a":[1,null,true],"b":{"c":"value"}}"#;

fn bench_input(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    value: ColumnarValue,
    expected: &[Option<&str>],
) {
    let udf = ParseJsonUdf::default();
    let arg_fields = vec![Arc::new(Field::new("json", value.data_type(), true))];
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
    let output = VariantArray::try_new(output.as_ref()).unwrap();
    assert_eq!(output.len(), expected.len());
    for (actual, expected) in output.iter().zip(expected) {
        let actual = actual.map(|value| value.to_json_string().unwrap());
        assert_eq!(actual.as_deref(), *expected);
    }

    group.throughput(Throughput::Elements(expected.len() as u64));
    group.bench_function(BenchmarkId::new(name, expected.len()), |b| {
        // Include argument cloning, JSON parsing, and output allocation/drop.
        // Input construction and validation are outside timing.
        b.iter(|| {
            drop(black_box(
                udf.invoke_with_args(black_box(args.clone())).unwrap(),
            ));
        });
    });
}

fn parse_json(c: &mut Criterion) {
    let cases = [
        ("integer", "42"),
        ("json_null", "null"),
        ("escaped_string", ESCAPED_STRING),
        ("nested", NESTED),
    ];
    let mut scalars = c.benchmark_group("parse_json/scalar");
    for (name, json) in cases {
        let expected = [Some(json)];
        let value = ScalarValue::Utf8(Some(json.into()));
        bench_input(&mut scalars, name, ColumnarValue::Scalar(value), &expected);
    }
    let expected = [None];
    let value = ScalarValue::Utf8(None);
    bench_input(
        &mut scalars,
        "sql_null",
        ColumnarValue::Scalar(value),
        &expected,
    );
    scalars.finish();

    let mut arrays = c.benchmark_group("parse_json/array");
    for (name, json) in cases {
        let expected = vec![Some(json); ROWS];
        let value = ColumnarValue::Array(Arc::new(StringArray::from(expected.clone())));
        bench_input(&mut arrays, name, value, &expected);
    }
    for (name, expected) in [
        (
            "nested_sql_nulls_25pct",
            (0..ROWS).map(|i| (i % 4 != 0).then_some(NESTED)).collect(),
        ),
        ("sql_nulls_100pct", vec![None; ROWS]),
    ] {
        let value = ColumnarValue::Array(Arc::new(StringArray::from(expected.clone())));
        bench_input(&mut arrays, name, value, &expected);
    }
    arrays.finish();
}

criterion_group!(benches, parse_json);
criterion_main!(benches);
