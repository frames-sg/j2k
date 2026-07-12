// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::Path;

use super::REPORT_RELATIVE;

pub(super) const DUPLICATED_LINE_THRESHOLD: f64 = 3.34;

pub(super) fn validate_clone_config(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path)
        .map_err(|error| format!("read clone-audit config {}: {error}", path.display()))?;
    let config = serde_json::from_str::<serde_json::Value>(&source)
        .map_err(|error| format!("parse clone-audit config {}: {error}", path.display()))?;
    require_exact_keys(
        &config,
        &[
            "format",
            "gitignore",
            "ignore",
            "maxLines",
            "maxSize",
            "minLines",
            "minTokens",
            "mode",
            "output",
            "reporters",
            "threshold",
        ],
    )?;
    require_number(&config, "threshold", DUPLICATED_LINE_THRESHOLD)?;
    require_number(&config, "minLines", 20.0)?;
    require_number(&config, "minTokens", 50.0)?;
    require_number(&config, "maxLines", 20_000.0)?;
    require_string(&config, "maxSize", "2mb")?;
    require_string(&config, "mode", "mild")?;
    require_string(&config, "output", REPORT_RELATIVE)?;
    require_bool(&config, "gitignore", false)?;
    require_string_array(&config, "format", &["rust"])?;
    require_string_array(&config, "reporters", &["console", "json"])?;
    require_string_array(
        &config,
        "ignore",
        &[
            "**/benches/**",
            "**/examples/**",
            "**/fuzz/**",
            "**/generated/**",
            "**/tests/**",
            "**/*_tests.rs",
            "**/tests.rs",
            "**/test_*.rs",
            "**/*_test.rs",
            "**/test_helpers.rs",
            "**/test_support.rs",
            "**/build.rs",
            "**/j2k-test-support/**",
            "**/j2k-transcode-test-support/**",
        ],
    )?;
    Ok(())
}

fn require_exact_keys(config: &serde_json::Value, expected: &[&str]) -> Result<(), String> {
    let mut actual = config
        .as_object()
        .ok_or_else(|| "clone-audit config root must be an object".to_string())?
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    actual.sort_unstable();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "clone-audit config keys must equal {expected:?}, found {actual:?}"
        ))
    }
}

fn require_number(config: &serde_json::Value, key: &str, expected: f64) -> Result<(), String> {
    let actual = config.get(key).and_then(serde_json::Value::as_f64);
    if actual == Some(expected) {
        Ok(())
    } else {
        Err(format!(
            "clone-audit config {key} must equal {expected}, found {actual:?}"
        ))
    }
}

fn require_string(config: &serde_json::Value, key: &str, expected: &str) -> Result<(), String> {
    let actual = config.get(key).and_then(serde_json::Value::as_str);
    if actual == Some(expected) {
        Ok(())
    } else {
        Err(format!(
            "clone-audit config {key} must equal {expected:?}, found {actual:?}"
        ))
    }
}

fn require_bool(config: &serde_json::Value, key: &str, expected: bool) -> Result<(), String> {
    let actual = config.get(key).and_then(serde_json::Value::as_bool);
    if actual == Some(expected) {
        Ok(())
    } else {
        Err(format!(
            "clone-audit config {key} must equal {expected}, found {actual:?}"
        ))
    }
}

fn require_string_array(
    config: &serde_json::Value,
    key: &str,
    expected: &[&str],
) -> Result<(), String> {
    let actual = config
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("clone-audit config {key} must be an array"))?;
    let actual = actual
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| format!("clone-audit config {key} contains a non-string"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "clone-audit config {key} must equal {expected:?}, found {actual:?}"
        ))
    }
}
