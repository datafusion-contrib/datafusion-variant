// SPDX-License-Identifier: Apache-2.0

use std::{hint::black_box, sync::Arc};

use arrow::array::{Array, ArrayRef, AsArray, NullArray, StringArray};
use arrow_schema::{DataType, Field, Schema};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use datafusion::{
    logical_expr::{
        AggregateUDF, ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    },
    physical_expr::{
        aggregate::{
            AggregateExprBuilder, AggregateFunctionExpr, utils::get_accum_scalar_values_as_arrays,
        },
        expressions::col,
    },
    scalar::ScalarValue,
};
use datafusion_variant::{SchemaOfVariantAggUdf, SchemaOfVariantUdf};
use parquet_variant_compute::{VariantArray, json_to_variant, shred_variant};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const ROWS: usize = 8192;
const PARTIALS: usize = 32;
const NESTED: &str = r#"{"a":[1,null,true],"b":{"c":"value"}}"#;
const NESTED_SCHEMA: &str = "OBJECT<a: ARRAY<VARIANT>, b: OBJECT<c: STRING>>";

fn input(json: StringArray) -> ArrayRef {
    let json: ArrayRef = Arc::new(json);
    json_to_variant(&json).unwrap().into()
}

fn field(input: &dyn Array) -> Field {
    if input.data_type() == &DataType::Null {
        Field::new("v", DataType::Null, true)
    } else {
        VariantArray::try_new(input).unwrap().field("v")
    }
}

fn bench_input(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    value: ColumnarValue,
    expected: Option<&str>,
) {
    let udf = SchemaOfVariantUdf::default();
    let input = value.to_array(1).unwrap();
    let rows = input.len();
    let arg_fields = vec![Arc::new(field(input.as_ref()))];
    let scalar = matches!(&value, ColumnarValue::Scalar(_));
    let scalar_arguments = [match &value {
        ColumnarValue::Scalar(value) => Some(value),
        ColumnarValue::Array(_) => None,
    }];
    let return_field = udf
        .return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &scalar_arguments,
        })
        .unwrap();
    let args = ScalarFunctionArgs {
        args: vec![value],
        arg_fields,
        number_rows: rows,
        return_field,
        config_options: Default::default(),
    };
    let output = udf.invoke_with_args(args.clone()).unwrap();
    assert_eq!(matches!(&output, ColumnarValue::Scalar(_)), scalar);
    assert_eq!(&output.data_type(), args.return_field.data_type());
    let output = output.into_array_of_size(rows).unwrap();
    assert!(
        output
            .as_string_view()
            .iter()
            .all(|actual| actual == expected)
    );

    group.throughput(Throughput::Elements(rows as u64));
    group.bench_function(BenchmarkId::new(name, rows), |b| {
        // Include argument cloning, inference, formatting, and output drop.
        b.iter(|| {
            drop(black_box(
                udf.invoke_with_args(black_box(args.clone())).unwrap(),
            ));
        });
    });
}

fn aggregate_expr(input: &ArrayRef) -> AggregateFunctionExpr {
    let schema = Arc::new(Schema::new(vec![field(input.as_ref())]));
    AggregateExprBuilder::new(
        Arc::new(AggregateUDF::new_from_impl(SchemaOfVariantAggUdf::default())),
        vec![col("v", &schema).unwrap()],
    )
    .schema(schema)
    .alias("schema_of_variant_agg(v)")
    .build()
    .unwrap()
}

fn bench_aggregate(
    c: &mut Criterion,
    name: &str,
    input: ArrayRef,
    expected: &str,
    batch_sizes: &[usize],
) {
    let expr = aggregate_expr(&input);
    let expected = ScalarValue::Utf8View(Some(expected.into()));
    let mut group = c.benchmark_group(format!("schema_of_variant_agg/update/{name}"));
    group.throughput(Throughput::Elements(input.len() as u64));
    for &batch_size in batch_sizes {
        let batches: Vec<_> = (0..input.len())
            .step_by(batch_size)
            .map(|offset| [input.slice(offset, batch_size.min(input.len() - offset))])
            .collect();
        let run = || {
            let mut accumulator = expr.create_accumulator().unwrap();
            for batch in &batches {
                accumulator.update_batch(black_box(batch)).unwrap();
            }
            accumulator.evaluate().unwrap()
        };
        assert_eq!(run(), expected);
        group.bench_function(BenchmarkId::new("batch_size", batch_size), |b| {
            // Fresh accumulator per iteration: include construction, all updates,
            // evaluation, and drop. Slicing and input construction are untimed.
            b.iter(|| drop(black_box(run())));
        });
    }
    group.finish();
}

fn bench_states(c: &mut Criterion, name: &str, input: ArrayRef, expected: &str) {
    let expr = aggregate_expr(&input);
    let expected = ScalarValue::Utf8View(Some(expected.into()));
    // Keep inference outside the state/merge measurements. Each partial sees a
    // different row of the repeating fixture, so unions/widening happen on merge.
    let mut accumulator = expr.create_accumulator().unwrap();
    accumulator
        .update_batch(std::slice::from_ref(&input))
        .unwrap();
    let partials = (0..PARTIALS).map(|i| {
        let mut partial = expr.create_accumulator().unwrap();
        partial.update_batch(&[input.slice(i, 1)]).unwrap();
        let [state]: [_; 1] = partial.state().unwrap().try_into().unwrap();
        state
    });
    let states = [ScalarValue::iter_to_array(partials).unwrap()];
    let merge = |states: &[ArrayRef]| {
        let mut final_accumulator = expr.create_accumulator().unwrap();
        final_accumulator.merge_batch(black_box(states)).unwrap();
        final_accumulator.evaluate().unwrap()
    };
    assert_eq!(merge(&states), expected);
    // Check the full-input serialized state independently as well.
    let state = get_accum_scalar_values_as_arrays(accumulator.as_mut()).unwrap();
    assert_eq!(merge(&state), expected);

    let mut group = c.benchmark_group(format!("schema_of_variant_agg/state/{name}"));
    group.throughput(Throughput::Elements(1));
    group.bench_function("1", |b| {
        b.iter(|| drop(black_box(accumulator.state().unwrap())));
    });
    group.finish();
    let mut group = c.benchmark_group(format!("schema_of_variant_agg/merge/{name}"));
    group.throughput(Throughput::Elements(PARTIALS as u64));
    group.bench_function(PARTIALS.to_string(), |b| {
        b.iter(|| drop(black_box(merge(&states))));
    });
    group.finish();
}

fn schema_of_variant(c: &mut Criterion) {
    let integer = input(StringArray::new_repeated("42", ROWS));
    let nested = input(StringArray::new_repeated(NESTED, ROWS));
    let sql_nulls = input(StringArray::new_null(ROWS));
    let untyped_null: ArrayRef = Arc::new(NullArray::new(ROWS));
    let variants = VariantArray::try_new(integer.as_ref()).unwrap();
    let shredded = shred_variant(&variants, &DataType::Int64).unwrap();
    assert_eq!(shredded.typed_value_column().unwrap().null_count(), 0);
    assert_eq!(shredded.value_column().null_count(), ROWS);

    // One primitive covers the scalar conversion wrapper; inference paths below
    // use arrays rather than repeating every fixture in both calling forms.
    let mut group = c.benchmark_group("schema_of_variant/scalar");
    bench_input(
        &mut group,
        "integer",
        ColumnarValue::Scalar(ScalarValue::try_from_array(integer.as_ref(), 0).unwrap()),
        Some("BIGINT"),
    );
    group.finish();

    let mut group = c.benchmark_group("schema_of_variant/array");
    for (name, input, expected) in [
        ("integer", integer.clone(), Some("BIGINT")),
        ("nested_object", nested.clone(), Some(NESTED_SCHEMA)),
        (
            "mixed_array",
            input(StringArray::new_repeated("[1,null,2.5,3]", ROWS)),
            Some("ARRAY<DOUBLE>"),
        ),
        (
            "variant_null",
            input(StringArray::new_repeated("null", ROWS)),
            Some("VOID"),
        ),
        ("sql_nulls_100pct", sql_nulls.clone(), None),
        ("untyped_null", untyped_null.clone(), None),
        ("shredded_integer", shredded.into(), Some("BIGINT")),
    ] {
        bench_input(&mut group, name, ColumnarValue::Array(input), expected);
    }
    group.finish();

    let object_union = input(StringArray::from_iter_values(
        [
            r#"{"a":1}"#,
            r#"{"b":true}"#,
            r#"{"a":2.5}"#,
            r#"{"c":[1]}"#,
        ]
        .into_iter()
        .cycle()
        .take(ROWS),
    ));
    let object_schema = "OBJECT<a: DOUBLE, b: BOOLEAN, c: ARRAY<BIGINT>>";
    // The second row makes the root VARIANT, with nested rows still to visit.
    let early_variant = input(StringArray::from_iter_values(
        ["42", "true", r#"{"a":[1,2,3]}"#, "null"]
            .into_iter()
            .cycle()
            .take(ROWS),
    ));
    // Only the primitive baseline needs a second batch size to expose per-batch overhead.
    bench_aggregate(c, "integer", integer.clone(), "BIGINT", &[256, ROWS]);
    for (name, input, expected) in [
        ("nested_object", nested, NESTED_SCHEMA),
        ("object_union", object_union.clone(), object_schema),
        ("early_variant", early_variant, "VARIANT"),
        ("sql_nulls_100pct", sql_nulls, "VOID"),
        ("untyped_null", untyped_null, "VOID"),
    ] {
        bench_aggregate(c, name, input, expected, &[ROWS]);
    }
    // State has no storage layout or SQL validity. Primitive and recursive
    // schemas cover serialization and equal-schema versus field-union merging.
    bench_states(c, "integer", integer, "BIGINT");
    bench_states(c, "object_union", object_union, object_schema);
}

criterion_group!(benches, schema_of_variant);
criterion_main!(benches);
