use std::{hint::black_box, sync::Arc};

use arrow::array::{ArrayRef, Int64Array};
use arrow_schema::{DataType, Field};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use datafusion::{
    logical_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl},
    scalar::ScalarValue,
};
use datafusion_variant::{VariantGetIntUdf, VariantGetUdf};
use parquet_variant::VariantBuilderExt;
use parquet_variant_compute::{VariantArray, VariantArrayBuilder, VariantType};

const ROWS: usize = 8192;

fn make_args(udf: &dyn ScalarUDFImpl, input: &ArrayRef, type_hint: bool) -> ScalarFunctionArgs {
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
    if type_hint {
        args.push(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
            "Int64".into(),
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

fn check_output(output: ColumnarValue, expected: &[i64], typed: bool) {
    let ColumnarValue::Array(output) = output else {
        panic!("expected array output");
    };
    assert_eq!(output.len(), expected.len());
    assert_eq!(output.null_count(), 0);
    if typed {
        assert_eq!(output.data_type(), &DataType::Int64);
        let output = output.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(output.values().as_ref(), expected);
    } else {
        // Validate Variant storage as well as every extracted integer.
        let output = VariantArray::try_new(output.as_ref()).unwrap();
        for (actual, expected) in output.iter().zip(expected) {
            assert_eq!(actual.unwrap().as_int64(), Some(*expected));
        }
    }
}

fn variant_get(c: &mut Criterion) {
    // Alternate signs and use values outside the i32 range to exercise Int64 storage.
    let expected: Vec<i64> = (0..ROWS)
        .map(|i| {
            let value = (1_i64 << 40) + i as i64;
            if i % 2 == 0 { value } else { -value }
        })
        .collect();
    let mut builder = VariantArrayBuilder::new(ROWS);
    for value in &expected {
        builder.new_object().with_field("a", *value).finish();
    }
    let input = ArrayRef::from(builder.build());
    let generic = VariantGetUdf::default();
    let integer = VariantGetIntUdf::default();
    let cases: [(&str, &dyn ScalarUDFImpl, bool, bool); 3] = [
        ("variant_get", &generic, false, false),
        ("variant_get_int64", &generic, true, true),
        ("variant_get_int", &integer, false, true),
    ];

    let mut group = c.benchmark_group("variant_get/unshredded_object/literal_a/8192");
    group.throughput(Throughput::Elements(ROWS as u64));
    for (name, udf, type_hint, typed) in cases {
        let args = make_args(udf, &input, type_hint);
        check_output(
            udf.invoke_with_args(args.clone()).unwrap(),
            &expected,
            typed,
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

criterion_group!(benches, variant_get);
criterion_main!(benches);
