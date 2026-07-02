// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{path::Path, process::Command};

use j2k_compare::{parse_positive_usize, usize_to_f64};
pub(crate) use j2k_test_support::canonicalize_manifest_row_path;

pub(crate) fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
}

pub(crate) fn env_falsey(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off"))
}

pub(crate) fn combined_batch_sizes(
    case_batch_sizes: &[usize],
    mixed_batch_sizes: &[usize],
) -> Vec<usize> {
    let mut values = case_batch_sizes
        .iter()
        .chain(mixed_batch_sizes.iter())
        .copied()
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

pub(crate) fn parse_batch_sizes(value: &str, label: &str) -> Result<Vec<usize>, String> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| parse_positive_usize(part, label))
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(format!("{label} must include at least one batch size"));
    }
    Ok(values)
}

pub(crate) fn optional_manifest_column(headers: &[&str], name: &str) -> Option<usize> {
    headers.iter().position(|header| *header == name)
}

pub(crate) fn manifest_column(
    headers: &[&str],
    name: &str,
    manifest_label: &str,
) -> Result<usize, String> {
    optional_manifest_column(headers, name)
        .ok_or_else(|| format!("{manifest_label} manifest is missing required {name:?} column"))
}

pub(crate) fn manifest_field<'a>(
    fields: &'a [&str],
    index: usize,
    name: &str,
    row_number: usize,
    manifest_label: &str,
) -> Result<&'a str, String> {
    fields.get(index).copied().ok_or_else(|| {
        format!("{manifest_label} manifest row {row_number} is missing {name:?} field")
    })
}

pub(crate) fn manifest_required_value(
    fields: &[&str],
    index: usize,
    name: &str,
    row_number: usize,
    manifest_label: &str,
) -> Result<String, String> {
    let value = manifest_field(fields, index, name, row_number, manifest_label)?.trim();
    if value.is_empty() {
        return Err(format!(
            "{manifest_label} manifest row {row_number} has empty required {name:?} field"
        ));
    }
    validate_manifest_value(value, name, row_number, manifest_label)?;
    Ok(value.to_string())
}

pub(crate) fn manifest_optional_value(
    fields: &[&str],
    index: Option<usize>,
    name: &str,
    row_number: usize,
    manifest_label: &str,
) -> Result<Option<String>, String> {
    let Some(index) = index else {
        return Ok(None);
    };
    let value = manifest_field(fields, index, name, row_number, manifest_label)?.trim();
    if value.is_empty() {
        return Ok(None);
    }
    validate_manifest_value(value, name, row_number, manifest_label)?;
    Ok(Some(value.to_string()))
}

fn validate_manifest_value(
    value: &str,
    name: &str,
    row_number: usize,
    manifest_label: &str,
) -> Result<(), String> {
    if value.chars().any(char::is_control) {
        return Err(format!(
            "{manifest_label} manifest row {row_number} field {name:?} contains a control character"
        ));
    }
    Ok(())
}

pub(crate) fn sanitized_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("unnamed")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

pub(crate) fn build_profile_label() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release-like"
    }
}

pub(crate) fn git_revision_label() -> String {
    git_revision().unwrap_or_else(|error| format!("unavailable:{error}"))
}

pub(crate) fn git_revision() -> Result<String, String> {
    command_stdout("git", &["rev-parse", "--short=12", "HEAD"])
}

pub(crate) fn git_dirty_label() -> String {
    git_dirty_status().map_or_else(|error| format!("unavailable:{error}"), str::to_string)
}

pub(crate) fn git_dirty_status() -> Result<&'static str, String> {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map_err(|error| format!("git:{error}"))?;
    if !output.status.success() {
        return Err(format!("git:status:{}", output.status));
    }
    if output.stdout.is_empty() {
        Ok("clean")
    } else {
        Ok("dirty")
    }
}

pub(crate) fn host_hardware_label() -> String {
    host_hardware_from_platform().unwrap_or_else(|error| format!("unavailable:{error}"))
}

#[cfg(target_os = "macos")]
fn host_hardware_from_platform() -> Result<String, String> {
    command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"])
}

#[cfg(target_os = "linux")]
fn host_hardware_from_platform() -> Result<String, String> {
    let cpuinfo =
        std::fs::read_to_string("/proc/cpuinfo").map_err(|error| format!("cpuinfo:{error}"))?;
    cpuinfo
        .lines()
        .find_map(|line| line.strip_prefix("model name\t: "))
        .map(str::to_string)
        .ok_or_else(|| "cpuinfo:model-name-missing".to_string())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn host_hardware_from_platform() -> Result<String, String> {
    Err("unsupported-platform".to_string())
}

fn command_stdout(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("{program}:{error}"))?;
    if !output.status.success() {
        return Err(format!("{program}:status:{}", output.status));
    }
    let stdout = String::from_utf8(output.stdout).map_err(|error| format!("{program}:{error}"))?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        Err(format!("{program}:empty-output"))
    } else {
        Ok(trimmed.to_string())
    }
}

pub(crate) fn is_publishable_license_status(status: &str) -> bool {
    matches!(
        status.trim().to_ascii_lowercase().as_str(),
        "apache-2.0"
            | "bsd"
            | "bsd-2-clause"
            | "bsd-3-clause"
            | "cc-by"
            | "cc-by-4.0"
            | "cc0"
            | "mit"
            | "open-data"
            | "permissive"
            | "permissive-test-fixture"
            | "public-domain"
            | "redistributable"
            | "redistributable-with-attribution"
    )
}

pub(crate) fn default_batch_sizes_present(
    batch_sizes: &[usize],
    default_batch_sizes: &[usize],
) -> bool {
    default_batch_sizes
        .iter()
        .all(|required| batch_sizes.contains(required))
}

pub(crate) fn join_usizes(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn join_string_labels(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

pub(crate) fn append_na_columns(row: &mut Vec<String>, count: usize) {
    row.extend((0..count).map(|_| "NA".to_string()));
}

pub(crate) fn skipped_external_mixed_prefix(
    tool_label: &str,
    batch_name: &str,
    method_label: &str,
) -> Vec<String> {
    vec![
        tool_label.to_string(),
        batch_name.to_string(),
        method_label.to_string(),
        "skipped".to_string(),
        "external:mixed".to_string(),
    ]
}

pub(crate) fn append_batch_input_columns(
    row: &mut Vec<String>,
    batch_size: usize,
    repeats: usize,
    input_bytes_per_repeat: usize,
    input_digest: String,
) {
    row.push(batch_size.to_string());
    row.push(repeats.to_string());
    row.push(input_bytes_per_repeat.to_string());
    row.push(input_digest);
}

pub(crate) fn join_tsv_row(row: Vec<String>) -> String {
    row.join("\t")
}

pub(crate) fn mib_per_second(bytes: usize, elapsed_us: f64) -> f64 {
    if elapsed_us <= 0.0 {
        return 0.0;
    }
    (usize_to_f64(bytes) / (1024.0 * 1024.0)) / (elapsed_us / 1_000_000.0)
}
