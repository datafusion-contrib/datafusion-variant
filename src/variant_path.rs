// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion::common::{Result, exec_datafusion_err};
use parquet_variant::{VariantPath, VariantPathElement};

/// Parse rooted or rootless paths; `allow_root` permits `$` and empty paths.
pub(crate) fn parse_variant_path(path: &str, allow_root: bool) -> Result<VariantPath<'_>> {
    // TODO(#166): unify path dialects.
    let parsed = if path.starts_with('$') {
        parse_rooted_path(path)?
    } else {
        VariantPath::try_from(path)?
    };
    if !allow_root && parsed.path().is_empty() {
        return Err(exec_datafusion_err!("invalid Variant path {path:?}"));
    }
    Ok(parsed)
}

// Spark 4.2: quoted keys retain backslashes.
// https://github.com/apache/spark/blob/v4.2.0/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/variant/variantExpressions.scala
// TODO(arrow-rs): drop this parser once `VariantPath` parses Spark-style paths.
// https://github.com/apache/arrow-rs/issues/11627
fn parse_rooted_path(path: &str) -> Result<VariantPath<'_>> {
    let invalid = || exec_datafusion_err!("invalid Variant path {path:?}");
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
                let (key, rest) = body[1..].split_once(quote).ok_or_else(invalid)?;
                remaining = rest.strip_prefix(']').ok_or_else(invalid)?;
                elements.push(VariantPathElement::field(key));
            } else {
                let (index, rest) = body.split_once(']').ok_or_else(invalid)?;
                if !index.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(invalid());
                }
                let index = index.parse::<i32>().map_err(|_| invalid())? as usize;
                elements.push(VariantPathElement::index(index));
                remaining = rest;
            }
        } else {
            return Err(invalid());
        }
    }
    Ok(VariantPath::new(elements))
}

/// Combine two path strings into one equivalent rooted path, so that
/// `variant_get(variant_get(v, inner), outer)` equals `variant_get(v, combined)`.
///
/// Returns `None` if either path is invalid or the combination can not be
/// written as a rooted path, in which case the nested calls must be kept.
pub(crate) fn concat_variant_paths(inner: &str, outer: &str) -> Option<String> {
    let inner = parse_variant_path(inner, true).ok()?;
    let outer = parse_variant_path(outer, true).ok()?;
    let mut combined = String::from("$");
    for element in inner.iter().chain(outer.iter()) {
        write_rooted_element(&mut combined, element)?;
    }
    Some(combined)
}

/// Append `element` in the syntax accepted by [`parse_rooted_path`].
fn write_rooted_element(out: &mut String, element: &VariantPathElement<'_>) -> Option<()> {
    match element {
        VariantPathElement::Field { name } => {
            // Quoted keys have no escapes, so a key with both quotes has no rooted form.
            if !name.is_empty() && !name.contains(['.', '[']) {
                out.push('.');
                out.push_str(name);
            } else if !name.contains('\'') {
                out.push_str(&format!("['{name}']"));
            } else if !name.contains('"') {
                out.push_str(&format!("[\"{name}\"]"));
            } else {
                return None;
            }
        }
        VariantPathElement::Index { index } => {
            // The rooted parser only accepts indexes that fit in an i32.
            let index = i32::try_from(*index).ok()?;
            out.push_str(&format!("[{index}]"));
        }
        _ => return None,
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concat_paths() {
        for (inner, outer, expected) in [
            ("a", "b", Some("$.a.b")),
            ("$.a", "$.b", Some("$.a.b")),
            ("a.b", "c[1]", Some("$.a.b.c[1]")),
            ("a[0]", "[1]", Some("$.a[0][1]")),
            ("", "a", Some("$.a")),
            ("a", "$", Some("$.a")),
            ("", "", Some("$")),
            ("['x.y']", "z", Some("$['x.y'].z")),
            ("$['a[0]']", "b", Some("$['a[0]'].b")),
            (r#"$["it's.x"]"#, "$['']", Some(r#"$["it's.x"]['']"#)),
            (r#"$['say "hi"']"#, "a b", Some(r#"$.say "hi".a b"#)),
            // invalid paths are left for execution to report
            ("a", "$bad", None),
            ("$[", "a", None),
            ("a", "[2147483648]", None),
        ] {
            let combined = concat_variant_paths(inner, outer);
            assert_eq!(combined.as_deref(), expected, "{inner:?} + {outer:?}");
            let Some(combined) = combined else { continue };
            // The combined path parses back to the elements of both inputs.
            let inner = parse_variant_path(inner, true).unwrap();
            let outer = parse_variant_path(outer, true).unwrap();
            let elements: Vec<_> = inner.iter().chain(outer.iter()).cloned().collect();
            assert_eq!(
                parse_variant_path(&combined, true).unwrap().path(),
                &elements
            );
        }
    }

    #[test]
    fn concat_paths_without_rooted_form() {
        let both_quotes = VariantPathElement::field(r#"a.'b"c"#);
        assert!(write_rooted_element(&mut String::new(), &both_quotes).is_none());
    }

    #[test]
    fn rooted_and_legacy_paths() {
        for (text, expected) in [
            ("$", vec![]),
            ("", vec![]),
            (
                "$.a[0]",
                vec![VariantPathElement::field("a"), VariantPathElement::index(0)],
            ),
            (
                "a[0]",
                vec![VariantPathElement::field("a"), VariantPathElement::index(0)],
            ),
            ("$['$']", vec![VariantPathElement::field("$")]),
            ("$['a]b']", vec![VariantPathElement::field("a]b")]),
            ("$['']", vec![VariantPathElement::field("")]),
            (r"$['\u0061']", vec![VariantPathElement::field(r"\u0061")]),
            (r"$['a\']", vec![VariantPathElement::field("a\\")]),
            (r#"$["a'b"]"#, vec![VariantPathElement::field("a'b")]),
        ] {
            assert_eq!(
                parse_variant_path(text, true).unwrap().path(),
                &expected,
                "{text}"
            );
        }
    }

    #[test]
    fn root_policy() {
        for text in ["$", ""] {
            assert!(parse_variant_path(text, false).is_err(), "{text:?}");
        }
        assert!(parse_variant_path("$.a", false).is_ok());
    }

    #[test]
    fn malformed_rooted_paths_do_not_fall_back() {
        for text in [
            "$bad",
            "$.",
            "$[",
            "$['a'",
            "$[]",
            "$[-1]",
            "$[2147483648]",
            "$[*]",
            "$..a",
            r"$['a\'b']",
        ] {
            assert!(parse_variant_path(text, true).is_err(), "{text}");
        }
    }
}
