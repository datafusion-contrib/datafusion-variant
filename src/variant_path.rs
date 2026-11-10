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
                let end = body[1..].find(quote).ok_or_else(invalid)? + 1;
                remaining = body[end + 1..].strip_prefix(']').ok_or_else(invalid)?;
                elements.push(VariantPathElement::field(&body[1..end]));
            } else {
                let end = body.find(']').ok_or_else(invalid)?;
                let index = &body[..end];
                if !index.bytes().all(|b| b.is_ascii_digit()) {
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
    Ok(VariantPath::new(elements))
}

#[cfg(test)]
mod tests {
    use super::*;

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
