// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryViewArray, StructArray, cast::AsArray, new_null_array};
use arrow::compute::cast;
use arrow_schema::{DataType, Field};
use datafusion::common::{exec_datafusion_err, exec_err};
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::scalar::ScalarValue;
use datafusion_doc::Documentation;
use datafusion_macros::user_doc;
use parquet_variant::{
    ObjectFieldBuilder, Variant, VariantBuilderExt, VariantPath, VariantPathElement,
};
use parquet_variant_compute::{VariantArray, VariantValueArrayBuilder, unshred_variant};

use crate::shared::{
    try_field_as_variant_array, try_parse_string_columnar, try_parse_string_scalar,
};
use crate::variant_type::variant_field;

#[user_doc(
    doc_section(label = "Variant Functions"),
    description = "Deletes object fields or array elements at one or more JSONPaths, applied left to right. Missing targets and type mismatches leave the value unchanged. SQL NULL input returns SQL NULL; NULL paths are skipped. Paths must start with `$` and cannot select the root alone.",
    syntax_example = "variant_delete(v, path1[, path2, ...])",
    sql_example = r#"```sql
> SELECT variant_delete(parse_json('{"a":[1,2,3]}'), '$.a[0]', '$.a[0]') AS v;
+-----------+
| v         |
+-----------+
| {"a":[3]} |
+-----------+
```"#,
    argument(name = "v", description = "Variant expression to update."),
    argument(
        name = "path",
        description = "String expression identifying a field or array element."
    )
)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantDeleteUdf {
    signature: Signature,
}

impl Default for VariantDeleteUdf {
    fn default() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for VariantDeleteUdf {
    fn name(&self) -> &str {
        "variant_delete"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn return_type(&self, _types: &[DataType]) -> Result<DataType> {
        Ok(variant_field(self.name()).data_type().clone())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        let [input, paths @ ..] = args.arg_fields else {
            return exec_err!("variant_delete requires a Variant and at least one path");
        };
        if paths.is_empty() {
            return exec_err!("variant_delete requires at least one path");
        }
        if input.data_type() != &DataType::Null {
            try_field_as_variant_array(input)?;
        }
        for path in paths {
            if !matches!(
                path.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View | DataType::Null
            ) {
                return exec_err!("variant_delete paths must be strings");
            }
        }
        Ok(Arc::new(variant_field(self.name()).with_nullable(
            input.is_nullable() || input.data_type() == &DataType::Null,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [input, path_args @ ..] = args.args.as_slice() else {
            return exec_err!("variant_delete requires a Variant and at least one path");
        };
        if path_args.is_empty() {
            return exec_err!("variant_delete requires at least one path");
        }
        // Parse constant paths once, including on empty/NULL input, matching Spark's
        // generated execution. Dynamic paths are read only for non-NULL input rows.
        let paths = path_args
            .iter()
            .map(|arg| match arg {
                ColumnarValue::Scalar(value) if value.is_null() => Ok(DeletePath::Constant(None)),
                ColumnarValue::Scalar(value) => Ok(DeletePath::Constant(
                    try_parse_string_scalar(value)?
                        .map(|s| parse_path(s))
                        .transpose()?,
                )),
                ColumnarValue::Array(array) if array.data_type() == &DataType::Null => {
                    Ok(DeletePath::Constant(None))
                }
                ColumnarValue::Array(array) => {
                    Ok(DeletePath::Column(try_parse_string_columnar(array)?))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let rows = arrays[0].len();
        let result: ArrayRef = if input.data_type() == DataType::Null {
            new_null_array(args.return_field.data_type(), rows)
        } else {
            let mut output = VariantValueArrayBuilder::new(rows);
            let input = VariantArray::try_new(arrays[0].as_ref())?;
            let unshredded;
            let input = if input
                .typed_value_column()
                .is_some_and(|v| v.data_type() != &DataType::Null)
            {
                // TODO: adopt Arrow's missing-metadata-key panic fix:
                // https://github.com/apache/arrow-rs/issues/11069
                unshredded = unshred_variant(&input)?;
                &unshredded
            } else {
                &input
            };
            let metadata = cast(input.metadata_column(), &DataType::BinaryView)?;
            let metadata_values = metadata.as_binary_view();
            let values = cast(input.value_column(), &DataType::BinaryView)?;
            let values = values.as_binary_view();
            for row in 0..input.len() {
                if input.is_null(row) {
                    output.append_value(Variant::Null);
                    continue;
                }
                if metadata_values.is_null(row) {
                    return exec_err!("variant_delete: missing Variant metadata");
                }
                // Full validation prevents panics in Arrow's infallible iterators.
                // Spark can skip decoding the removed subtree. TODO: use shallow
                // fallible views when available: https://github.com/apache/arrow-rs/pull/11511
                let bytes = if values.is_null(row) {
                    &[0]
                } else {
                    values.value(row)
                };
                let original = Variant::try_new(metadata_values.value(row), bytes)?;
                let mut current: Option<BinaryViewArray> = None;
                for (index, path) in paths.iter().enumerate() {
                    let dynamic;
                    let path = match path {
                        DeletePath::Constant(path) => path.as_ref(),
                        DeletePath::Column(paths) => {
                            dynamic = paths[row].map(parse_path).transpose()?;
                            dynamic.as_ref()
                        }
                    };
                    let value = match &current {
                        Some(value) => {
                            Variant::new_with_metadata(original.metadata().clone(), value.value(0))
                        }
                        None => original.clone(),
                    };
                    // Arrow handles missing fields, bounds and type mismatches.
                    // Existing Variant nulls are Some(Null), so remain deletable.
                    let path = path.filter(|path| value.get_path(path).is_some());
                    // Only intermediate deletions need a temporary value. Write
                    // the final result straight into Arrow's array builder.
                    if index + 1 == paths.len() {
                        if let Some(path) = path {
                            append_deleted(
                                &mut output.builder_ext(original.metadata()),
                                value,
                                path,
                            )?;
                        } else {
                            output.append_value(value);
                        }
                    } else if let Some(path) = path {
                        let mut builder = VariantValueArrayBuilder::new(1);
                        append_deleted(&mut builder.builder_ext(original.metadata()), value, path)?;
                        current = Some(builder.build()?);
                    }
                }
            }
            let DataType::Struct(storage) = args.return_field.data_type() else {
                unreachable!()
            };
            Arc::new(StructArray::try_new(
                storage.clone(),
                vec![metadata, Arc::new(output.build()?)],
                input.nulls().cloned(),
            )?)
        };
        if args
            .args
            .iter()
            .all(|arg| matches!(arg, ColumnarValue::Scalar(_)))
        {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(result))
        }
    }
}

enum DeletePath<'a> {
    Constant(Option<VariantPath<'a>>),
    Column(Vec<Option<&'a str>>),
}

// Spark VariantBuilder.appendWithDeletionImpl at 046f574fce3f:
// https://github.com/apache/spark/blob/046f574fce3ffd93925f7bcdd459221fa11a730d/common/variant/src/main/java/org/apache/spark/types/variant/VariantBuilder.java
// Deletion adds no keys, so every builder uses the original metadata dictionary.
// Unused field names can remain in that dictionary after deletion.
// Arrow's byte-copy APIs preserve untouched siblings without re-encoding them
// while the containing objects and lists are rebuilt.
fn append_deleted(
    builder: &mut impl VariantBuilderExt,
    value: Variant<'_, '_>,
    path: &[VariantPathElement<'_>],
) -> Result<()> {
    match (value, path.split_first()) {
        (Variant::Object(object), Some((VariantPathElement::Field { name }, rest))) => {
            let mut output = builder.try_new_object()?;
            for (key, value) in object.iter() {
                if key != name.as_ref() {
                    output.insert_bytes(key, value);
                } else if !rest.is_empty() {
                    append_deleted(&mut ObjectFieldBuilder::new(key, &mut output), value, rest)?;
                }
            }
            output.finish();
        }
        (Variant::List(list), Some((VariantPathElement::Index { index }, rest))) => {
            let mut output = builder.try_new_list()?;
            for (i, value) in list.iter().enumerate() {
                if i != *index {
                    output.append_value_bytes(value);
                } else if !rest.is_empty() {
                    append_deleted(&mut output, value, rest)?;
                }
            }
            output.finish();
        }
        (value, _) => builder.append_value(value),
    }
    Ok(())
}

// Spark VariantPathParser's default (non-legacy) grammar. Arrow's parser accepts
// rootless paths and has different dot/quoted-key escaping rules, so stripping
// `$` and delegating would silently select different fields. Keep Arrow's path
// representation; do not change the existing getters' path dialect here.
fn parse_path(path: &str) -> Result<VariantPath<'_>> {
    let invalid = || exec_datafusion_err!("variant_delete: invalid path {path:?}");
    let mut remaining = path.strip_prefix('$').ok_or_else(invalid)?;
    let mut elements = Vec::new();
    while !remaining.is_empty() {
        if let Some(key) = remaining.strip_prefix('.') {
            let end = key.find(['.', '[']).unwrap_or(key.len());
            if end == 0 {
                return Err(invalid());
            }
            elements.push(VariantPathElement::field(&key[..end]));
            remaining = &key[end..];
        } else if let Some(body) = remaining.strip_prefix('[') {
            if body.starts_with(['\'', '"']) {
                let quote = body.as_bytes()[0] as char;
                let mut chars = body[1..].char_indices();
                let end = loop {
                    let (i, c) = chars.next().ok_or_else(invalid)?;
                    if c == '\\' {
                        chars.next().ok_or_else(invalid)?;
                    } else if c == quote {
                        break i + 1;
                    }
                };
                remaining = body[end + 1..].strip_prefix(']').ok_or_else(invalid)?;
                elements.push(VariantPathElement::field(unescape_key(
                    &body[1..end],
                    quote,
                )));
            } else {
                let end = body.find(']').ok_or_else(invalid)?;
                let index = &body[..end];
                if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(invalid());
                }
                let index = index.parse::<i32>().map_err(|_| invalid())? as usize;
                elements.push(VariantPathElement::index(index));
                remaining = &body[end + 1..];
            }
        } else {
            return Err(invalid());
        }
    }
    if elements.is_empty() {
        return Err(invalid());
    }
    Ok(VariantPath::new(elements))
}

fn unescape_key(raw: &str, quote: char) -> String {
    let mut output = String::new();
    let mut remaining = raw;
    while let Some((before, after)) = remaining.split_once('\\') {
        output.push_str(before);
        let c = after
            .chars()
            .next()
            .expect("parser checked escape terminator");
        let mut consumed = c.len_utf8();
        match c {
            'b' => output.push('\x08'),
            'f' => output.push('\x0c'),
            'n' => output.push('\n'),
            'r' => output.push('\r'),
            't' => output.push('\t'),
            '/' | '\\' => output.push(c),
            c if c == quote => output.push(c),
            'u' => {
                // Let serde decode valid Unicode escapes (including surrogate
                // pairs). Spark preserves malformed escapes and lone surrogates.
                let decoded = [11, 5].into_iter().find_map(|n| {
                    let escape = after.get(..n)?;
                    if !escape.as_bytes()[1..5].iter().all(u8::is_ascii_hexdigit)
                        || (n == 11
                            && (!escape.as_bytes()[5..].starts_with(b"\\u")
                                || !escape.as_bytes()[7..].iter().all(u8::is_ascii_hexdigit)))
                    {
                        return None;
                    }
                    let value = serde_json::from_str::<String>(&format!("\"\\{escape}\"")).ok()?;
                    Some((n, value))
                });
                if let Some((n, value)) = decoded {
                    consumed = n;
                    output.push_str(&value);
                } else {
                    output.push_str("\\u");
                }
            }
            c => {
                output.push('\\');
                output.push(c);
            }
        }
        remaining = &after[consumed..];
    }
    output.push_str(remaining);
    output
}
