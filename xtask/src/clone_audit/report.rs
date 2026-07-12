// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::Path;

pub(super) fn validate_jscpd_report(path: &Path, threshold: f64) -> Result<(), String> {
    let source = fs::read_to_string(path)
        .map_err(|error| format!("read jscpd JSON report {}: {error}", path.display()))?;
    let report = serde_json::from_str::<serde_json::Value>(&source)
        .map_err(|error| format!("parse jscpd JSON report {}: {error}", path.display()))?;
    let total = report
        .get("statistics")
        .and_then(|statistics| statistics.get("total"))
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "jscpd JSON report has no statistics.total object".to_string())?;
    let lines = require_count(total, "lines")?;
    let tokens = require_count(total, "tokens")?;
    let sources = require_count(total, "sources")?;
    let _clones = require_count(total, "clones")?;
    let duplicated_lines = require_count(total, "duplicatedLines")?;
    let duplicated_tokens = require_count(total, "duplicatedTokens")?;
    let percentage = require_percentage(total, "percentage")?;
    let _percentage_tokens = require_percentage(total, "percentageTokens")?;
    if lines == 0 || tokens == 0 || sources == 0 {
        return Err("jscpd JSON report must describe non-empty staged Rust sources".to_string());
    }
    if duplicated_lines > lines || duplicated_tokens > tokens {
        return Err("jscpd JSON report duplicated totals exceed analyzed totals".to_string());
    }
    if percentage >= threshold {
        return Err(format!(
            "jscpd JSON report duplicated-line percentage {percentage} meets or exceeds {threshold}"
        ));
    }
    Ok(())
}

fn require_count(
    total: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<u64, String> {
    total
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            format!(
                "jscpd JSON report statistics.total.{key} is missing or not an unsigned integer"
            )
        })
}

fn require_percentage(
    total: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<f64, String> {
    total
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite() && (0.0..=100.0).contains(value))
        .ok_or_else(|| {
            format!("jscpd JSON report statistics.total.{key} is missing or outside 0..=100")
        })
}
