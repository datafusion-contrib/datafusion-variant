// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion::common::{Result, exec_datafusion_err};
use parquet_variant::{VariantPath, VariantPathElement};

/// Parse a Spark 4.2 Variant path; `allow_root` permits `$` alone.
pub(crate) fn parse_variant_path(path: &str, allow_root: bool) -> Result<VariantPath<'_>> {
    if !path.starts_with('$') {
        return Err(exec_datafusion_err!(
            "invalid Variant path {path:?}: paths start with `$`, such as `$.a[0]`"
        ));
    }
    let parsed = parse_rooted_path(path)?;
    if !allow_root && parsed.path().is_empty() {
        return Err(exec_datafusion_err!("invalid Variant path {path:?}"));
    }
    Ok(parsed)
}

/// The path for `v:<text>`, built as Spark's `AstBuilder.visitSemiStructuredExtract` does.
pub(crate) fn colon_path(text: &str) -> String {
    if text.starts_with('[') {
        format!("${text}")
    } else {
        format!("$.{text}")
    }
}

/// The `<text>` of `v:<text>` that [`colon_path`] turns into `path`.
pub(crate) fn colon_text(path: &str) -> Option<&str> {
    path.strip_prefix("$.")
        .or_else(|| path.strip_prefix('$').filter(|text| text.starts_with('[')))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_paths() {
        for (text, expected) in [
            ("$", vec![]),
            (
                "$.a[0]",
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
        assert!(parse_variant_path("$", false).is_err());
        assert!(parse_variant_path("$.a", false).is_ok());
    }

    #[test]
    fn invalid_paths() {
        for text in [
            "",
            "a",
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

    #[test]
    fn colon_paths_round_trip() {
        for (text, path) in [("a.b[0]", "$.a.b[0]"), ("['a.b']", "$['a.b']")] {
            assert_eq!(colon_path(text), path);
            assert_eq!(colon_text(path), Some(text));
        }
        assert_eq!(colon_text("$"), None);
    }
}
