// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Included by multiple benchmark binaries; each target uses a different subset.
#![allow(dead_code)]

use std::fmt::Write as _;

pub(crate) fn escape_json(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => {
                write!(&mut escaped, "\\u{:04x}", ch as u32)
                    .expect("writing to String cannot fail");
            }
            ch => escaped.push(ch),
        }
    }
    escaped
}

pub(crate) fn escape_csv(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

pub(crate) fn json_f64_or_null(value: Option<f64>, decimals: usize) -> String {
    value
        .filter(|value| value.is_finite())
        .map_or_else(|| "null".to_string(), |value| format_f64(value, decimals))
}

pub(crate) fn json_f64_or_inf(value: Option<f64>, decimals: usize) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| {
            if value.is_finite() {
                format_f64(value, decimals)
            } else {
                "\"inf\"".to_string()
            }
        },
    )
}

pub(crate) fn csv_f64_or_blank(value: Option<f64>, decimals: usize) -> String {
    value
        .filter(|value| value.is_finite())
        .map_or_else(String::new, |value| format_f64(value, decimals))
}

pub(crate) fn csv_f64_or_inf(value: Option<f64>, decimals: usize) -> String {
    value.map_or_else(String::new, |value| {
        if value.is_finite() {
            format_f64(value, decimals)
        } else {
            "inf".to_string()
        }
    })
}

fn format_f64(value: f64, decimals: usize) -> String {
    format!("{value:.decimals$}")
}

#[cfg(test)]
mod tests {
    use super::{
        csv_f64_or_blank, csv_f64_or_inf, escape_csv, escape_json, json_f64_or_inf,
        json_f64_or_null,
    };

    #[test]
    fn escape_json_handles_quotes_slashes_and_control_chars() {
        assert_eq!(
            escape_json("tile \"a\"\\b\n\t\u{0007}"),
            "tile \\\"a\\\"\\\\b\\n\\t\\u0007"
        );
    }

    #[test]
    fn escape_csv_quotes_only_when_required() {
        assert_eq!(escape_csv("plain"), "plain");
        assert_eq!(escape_csv("tile, \"a\"\n"), "\"tile, \"\"a\"\"\n\"");
    }

    #[test]
    fn optional_json_f64_policies_preserve_existing_reports() {
        assert_eq!(json_f64_or_null(Some(42.25), 8), "42.25000000");
        assert_eq!(json_f64_or_null(None, 8), "null");
        assert_eq!(json_f64_or_null(Some(f64::INFINITY), 8), "null");

        assert_eq!(json_f64_or_inf(Some(42.25), 6), "42.250000");
        assert_eq!(json_f64_or_inf(None, 6), "null");
        assert_eq!(json_f64_or_inf(Some(f64::NAN), 6), "\"inf\"");
    }

    #[test]
    fn optional_csv_f64_policies_preserve_existing_reports() {
        assert_eq!(csv_f64_or_blank(Some(42.25), 6), "42.250000");
        assert_eq!(csv_f64_or_blank(None, 6), "");
        assert_eq!(csv_f64_or_blank(Some(f64::NEG_INFINITY), 6), "");

        assert_eq!(csv_f64_or_inf(Some(42.25), 6), "42.250000");
        assert_eq!(csv_f64_or_inf(None, 6), "");
        assert_eq!(csv_f64_or_inf(Some(f64::INFINITY), 6), "inf");
    }
}
