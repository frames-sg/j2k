use std::{collections::BTreeSet, fs, path::Path};

use crate::perf_guard::{discover_estimates, BenchEstimate};

use super::runner::METAL_TRANSCODE_BENCH_FILTER;
use super::summary::{AdoptionStep, StepStatus};

pub(super) fn criterion_estimate_json(estimate: &BenchEstimate) -> serde_json::Value {
    serde_json::json!({
        "id": estimate.id,
        "median_ns": estimate.median_ns,
        "median_lower_ns": estimate.median_lower_ns,
        "median_upper_ns": estimate.median_upper_ns,
    })
}

pub(super) fn criterion_summary_json(steps: &[AdoptionStep]) -> serde_json::Value {
    let mut total_count = 0_usize;
    let mut step_summaries = Vec::new();
    let mut all_estimates = Vec::new();
    for step in steps {
        let Some(root) = &step.criterion_root else {
            continue;
        };
        if !matches!(&step.status, StepStatus::Ran) {
            continue;
        }
        if !root.exists() {
            step_summaries.push(serde_json::json!({
                "step": step.name,
                "root": root.display().to_string(),
                "count": 0,
                "note": "no Criterion output produced",
            }));
            continue;
        }
        match discover_estimates(root) {
            Ok(estimates) => {
                total_count += estimates.len();
                all_estimates.extend(estimates.iter().map(criterion_estimate_json));
                step_summaries.push(serde_json::json!({
                    "step": step.name,
                    "root": root.display().to_string(),
                    "count": estimates.len(),
                    "estimates": estimates.iter().map(criterion_estimate_json).collect::<Vec<_>>(),
                }));
            }
            Err(error) => step_summaries.push(serde_json::json!({
                "step": step.name,
                "root": root.display().to_string(),
                "error": error,
            })),
        }
    }

    serde_json::json!({
        "count": total_count,
        "steps": step_summaries,
        "estimates": all_estimates,
    })
}

pub(super) fn read_metal_decode_summary(path: &Path, steps: &[AdoptionStep]) -> serde_json::Value {
    let Some(step) = steps
        .iter()
        .find(|step| step.name == "metal-decode-benchmark")
    else {
        return serde_json::json!({
            "output": path.display().to_string(),
            "status": "missing-step",
        });
    };
    if let StepStatus::Skipped { reason } = &step.status {
        return serde_json::json!({
            "output": path.display().to_string(),
            "status": "skipped",
            "reason": reason,
        });
    }
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            return serde_json::json!({
                "output": path.display().to_string(),
                "status": "error",
                "error": format!("failed to read Metal decode benchmark output: {error}"),
            });
        }
    };

    let mut benches = Vec::new();
    let mut metadata = serde_json::Map::new();
    let mut skipped_cases = Vec::new();
    for line in text.lines() {
        if let Some(row) = parse_metal_decode_bench_line(line) {
            benches.push(row);
        } else if let Some((key, value)) = line.split_once('\t') {
            if key.starts_with("j2k_metal_decode_") {
                metadata.insert(
                    key.to_string(),
                    serde_json::Value::String(value.to_string()),
                );
            }
        } else if let Some(rest) = line.strip_prefix("j2k_metal_decode_skipped_case ") {
            skipped_cases.push(serde_json::Value::String(rest.to_string()));
        }
    }

    let skipped_benches = benches
        .iter()
        .filter(|row| {
            row.get("metal_resident_ms")
                .and_then(serde_json::Value::as_str)
                == Some("skipped")
                || row
                    .get("metal_readback_ms")
                    .and_then(serde_json::Value::as_str)
                    == Some("skipped")
        })
        .count();
    let verified_benches = benches
        .iter()
        .filter(|row| {
            row.get("cpu_ms")
                .and_then(serde_json::Value::as_f64)
                .is_some()
                && row
                    .get("metal_resident_ms")
                    .and_then(serde_json::Value::as_f64)
                    .is_some()
                && row
                    .get("metal_readback_ms")
                    .and_then(serde_json::Value::as_f64)
                    .is_some()
        })
        .count();

    serde_json::json!({
        "output": path.display().to_string(),
        "status": "ran",
        "bench_count": benches.len(),
        "skipped_bench_count": skipped_benches,
        "verified_bench_count": verified_benches,
        "skipped_case_count": skipped_cases.len(),
        "skipped_cases": skipped_cases,
        "metadata": metadata,
        "benches": benches,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "the Metal encode summary parser keeps one output schema and its counters together"
)]
pub(super) fn read_metal_encode_summary(path: &Path, steps: &[AdoptionStep]) -> serde_json::Value {
    let Some(step) = steps
        .iter()
        .find(|step| step.name == "metal-encode-auto-routing")
    else {
        return serde_json::json!({
            "output": path.display().to_string(),
            "status": "missing-step",
        });
    };
    if let StepStatus::Skipped { reason } = &step.status {
        return serde_json::json!({
            "output": path.display().to_string(),
            "status": "skipped",
            "reason": reason,
        });
    }
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            return serde_json::json!({
                "output": path.display().to_string(),
                "status": "error",
                "error": format!("failed to read Metal benchmark output: {error}"),
            });
        }
    };

    let mut auto_benches = Vec::new();
    let mut auto_probes = Vec::new();
    let mut stage_benches = Vec::new();
    let mut resident_benches = Vec::new();
    let mut metadata = serde_json::Map::new();
    for line in text.lines() {
        if let Some(row) = parse_metal_auto_bench_line(line) {
            auto_benches.push(row);
        } else if let Some(row) = parse_metal_auto_probe_line(line) {
            auto_probes.push(row);
        } else if let Some(row) = parse_metal_stage_bench_line(line) {
            stage_benches.push(row);
        } else if let Some(row) = parse_metal_resident_bench_line(line) {
            resident_benches.push(row);
        } else if let Some((key, value)) = line.split_once('\t') {
            if key.starts_with("j2k_metal_encode_") {
                metadata.insert(
                    key.to_string(),
                    serde_json::Value::String(value.to_string()),
                );
            }
        }
    }

    let skipped_auto_benches = auto_benches
        .iter()
        .filter(|row| row.get("auto_ms").and_then(serde_json::Value::as_str) == Some("skipped"))
        .count();
    let skipped_stage_benches = stage_benches
        .iter()
        .filter(|row| row.get("metal_ms").and_then(serde_json::Value::as_str) == Some("skipped"))
        .count();
    let skipped_resident_benches = resident_benches
        .iter()
        .filter(|row| {
            row.get("resident_host_ms")
                .and_then(serde_json::Value::as_str)
                == Some("skipped")
                || row
                    .get("resident_buffer_ms")
                    .and_then(serde_json::Value::as_str)
                    == Some("skipped")
        })
        .count();
    let resident_verified_benches = resident_benches
        .iter()
        .filter(|row| {
            row.get("packetization_used")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
                && row
                    .get("codestream_assembly_used")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
                && row
                    .get("resident_host_ms")
                    .and_then(serde_json::Value::as_f64)
                    .is_some()
                && row
                    .get("resident_buffer_ms")
                    .and_then(serde_json::Value::as_f64)
                    .is_some()
        })
        .count();
    let probe_errors = auto_probes
        .iter()
        .filter(|row| row.get("error").is_some())
        .count();

    serde_json::json!({
        "output": path.display().to_string(),
        "status": "ran",
        "auto_bench_count": auto_benches.len(),
        "auto_probe_count": auto_probes.len(),
        "stage_bench_count": stage_benches.len(),
        "resident_bench_count": resident_benches.len(),
        "skipped_auto_bench_count": skipped_auto_benches,
        "skipped_stage_bench_count": skipped_stage_benches,
        "skipped_resident_bench_count": skipped_resident_benches,
        "resident_verified_bench_count": resident_verified_benches,
        "probe_error_count": probe_errors,
        "metadata": metadata,
        "auto_benches": auto_benches,
        "auto_probes": auto_probes,
        "stage_benches": stage_benches,
        "resident_benches": resident_benches,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "the Metal transcode summary parser keeps stdout, stderr, and schema reconciliation together"
)]
pub(super) fn read_metal_transcode_summary(
    stdout_path: &Path,
    stderr_path: &Path,
    steps: &[AdoptionStep],
) -> serde_json::Value {
    let Some(step) = steps
        .iter()
        .find(|step| step.name == "metal-transcode-benchmark")
    else {
        return serde_json::json!({
            "stdout": stdout_path.display().to_string(),
            "stderr": stderr_path.display().to_string(),
            "status": "missing-step",
        });
    };
    if let StepStatus::Skipped { reason } = &step.status {
        return serde_json::json!({
            "stdout": stdout_path.display().to_string(),
            "stderr": stderr_path.display().to_string(),
            "status": "skipped",
            "reason": reason,
        });
    }
    let stdout = match fs::read_to_string(stdout_path) {
        Ok(text) => text,
        Err(error) => {
            return serde_json::json!({
                "stdout": stdout_path.display().to_string(),
                "stderr": stderr_path.display().to_string(),
                "status": "error",
                "error": format!("failed to read Metal transcode benchmark stdout: {error}"),
            });
        }
    };
    let stderr = match fs::read_to_string(stderr_path) {
        Ok(text) => text,
        Err(error) => {
            return serde_json::json!({
                "stdout": stdout_path.display().to_string(),
                "stderr": stderr_path.display().to_string(),
                "status": "error",
                "error": format!("failed to read Metal transcode benchmark stderr: {error}"),
            });
        }
    };

    let mut profiles = Vec::new();
    for line in stdout.lines().chain(stderr.lines()) {
        if let Some(row) = parse_metal_transcode_profile_line(line) {
            profiles.push(row);
        }
    }

    let cpu_contexts = profile_contexts(&profiles, "cpu", "cpu");
    let auto_metal_contexts = profile_contexts(&profiles, "metal_auto", "metal");
    let explicit_metal_contexts = profile_contexts(&profiles, "metal_explicit", "metal");
    let cpu_profiles = profiles
        .iter()
        .filter(|row| {
            row.get("request").and_then(serde_json::Value::as_str) == Some("cpu")
                && row
                    .get("transform_processor")
                    .and_then(serde_json::Value::as_str)
                    == Some("cpu")
        })
        .count();
    let auto_metal_profiles = profiles
        .iter()
        .filter(|row| {
            row.get("request").and_then(serde_json::Value::as_str) == Some("metal_auto")
                && row
                    .get("transform_processor")
                    .and_then(serde_json::Value::as_str)
                    == Some("metal")
        })
        .count();
    let explicit_metal_profiles = profiles
        .iter()
        .filter(|row| {
            row.get("request").and_then(serde_json::Value::as_str) == Some("metal_explicit")
                && row
                    .get("transform_processor")
                    .and_then(serde_json::Value::as_str)
                    == Some("metal")
        })
        .count();
    let mut metal_contexts = auto_metal_contexts.clone();
    metal_contexts.extend(explicit_metal_contexts.iter().cloned());
    let comparison_context_count = cpu_contexts.intersection(&metal_contexts).count();
    let verified_profiles = profiles
        .iter()
        .filter(|row| {
            row.get("transform_processor")
                .and_then(serde_json::Value::as_str)
                == Some("metal")
                && row
                    .get("accelerator_dispatches")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    > 0
                && row
                    .get("successful_tiles")
                    .and_then(serde_json::Value::as_u64)
                    == row.get("tile_count").and_then(serde_json::Value::as_u64)
        })
        .count();

    serde_json::json!({
        "stdout": stdout_path.display().to_string(),
        "stderr": stderr_path.display().to_string(),
        "status": "ran",
        "bench_filter": METAL_TRANSCODE_BENCH_FILTER,
        "profile_count": profiles.len(),
        "verified_profile_count": verified_profiles,
        "cpu_profile_count": cpu_profiles,
        "auto_metal_profile_count": auto_metal_profiles,
        "explicit_metal_profile_count": explicit_metal_profiles,
        "comparison_context_count": comparison_context_count,
        "profiles": profiles,
    })
}

pub(super) fn profile_contexts(
    profiles: &[serde_json::Value],
    request: &str,
    transform_processor: &str,
) -> BTreeSet<String> {
    profiles
        .iter()
        .filter(|row| {
            row.get("request").and_then(serde_json::Value::as_str) == Some(request)
                && row
                    .get("transform_processor")
                    .and_then(serde_json::Value::as_str)
                    == Some(transform_processor)
                && row
                    .get("successful_tiles")
                    .and_then(serde_json::Value::as_u64)
                    == row.get("tile_count").and_then(serde_json::Value::as_u64)
        })
        .filter_map(|row| {
            row.get("context")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .collect()
}

pub(super) fn parse_metal_decode_bench_line(line: &str) -> Option<serde_json::Value> {
    const PREFIX: &str = "j2k_metal_decode_bench ";
    let rest = line.strip_prefix(PREFIX)?;
    let fields = j2k_profile::parse_profile_key_value_fields(rest);
    let mut row = serde_json::json!({
        "case": required_field(&fields, "case")?,
        "source": required_field(&fields, "source")?,
        "codec": required_field(&fields, "codec")?,
        "container": required_field(&fields, "container")?,
        "operation": required_field(&fields, "operation")?,
        "fmt": required_field(&fields, "fmt")?,
        "size": required_field(&fields, "size")?,
        "cpu_ms": parse_decimal_or_label_field(&fields, "cpu_ms")?,
        "metal_resident_ms": parse_decimal_or_label_field(&fields, "metal_resident_ms")?,
        "metal_readback_ms": parse_decimal_or_label_field(&fields, "metal_readback_ms")?,
        "output_bytes": parse_integer_or_label_field(&fields, "output_bytes")?,
    });
    if let Some(error) = suffix_after_key(rest, "error") {
        row["error"] = serde_json::Value::String(error.to_string());
    }
    Some(row)
}

pub(super) fn parse_metal_transcode_profile_line(line: &str) -> Option<serde_json::Value> {
    let fields = j2k_profile::parse_profile_line(line)?;
    if fields.kind() != j2k_profile::ParsedProfileKind::Row
        || fields.get("codec")? != "transcode"
        || fields.get("op")? != "transcode_batch"
    {
        return None;
    }
    let required = |key: &str| fields.get(key).map(str::to_string);
    let integer = |key: &str| fields.get(key)?.parse::<u64>().ok();
    Some(serde_json::json!({
        "request": required("request")?,
        "path": required("path")?,
        "pipeline": required("pipeline")?,
        "context": required("context")?,
        "coefficient_path": required("coefficient_path")?,
        "extract_processor": required("extract_processor")?,
        "transform_processor": required("transform_processor")?,
        "encode_processor": required("encode_processor")?,
        "tile_count": integer("tile_count")?,
        "successful_tiles": integer("successful_tiles")?,
        "failed_tiles": integer("failed_tiles")?,
        "transformed_components": integer("transformed_components")?,
        "total_us": integer("total_us")?,
        "extract_us": integer("extract_us")?,
        "transform_us": integer("transform_us")?,
        "encode_us": integer("encode_us")?,
        "dct_to_wavelet_total_us": integer("dct_to_wavelet_total_us")?,
        "dct_to_wavelet_accelerator_us": integer("dct_to_wavelet_accelerator_us")?,
        "dct_to_wavelet_cpu_fallback_us": integer("dct_to_wavelet_cpu_fallback_us")?,
        "dwt97_batch_pack_upload_transfers": integer("dwt97_batch_pack_upload_transfers")?,
        "dwt97_batch_pack_upload_bytes": integer("dwt97_batch_pack_upload_bytes")?,
        "dwt97_batch_resident_dct_handoff_count": integer("dwt97_batch_resident_dct_handoff_count")?,
        "dwt97_batch_resident_dwt_handoff_count": integer("dwt97_batch_resident_dwt_handoff_count")?,
        "dwt97_batch_readback_transfers": integer("dwt97_batch_readback_transfers")?,
        "dwt97_batch_readback_bytes": integer("dwt97_batch_readback_bytes")?,
        "host_to_device_transfer_count": integer("host_to_device_transfer_count")?,
        "host_to_device_transfer_bytes": integer("host_to_device_transfer_bytes")?,
        "device_to_host_transfer_count": integer("device_to_host_transfer_count")?,
        "device_to_host_transfer_bytes": integer("device_to_host_transfer_bytes")?,
        "component_count": integer("component_count")?,
        "batch_count": integer("batch_count")?,
        "batch_jobs": integer("batch_jobs")?,
        "accelerator_attempts": integer("accelerator_attempts")?,
        "accelerator_jobs": integer("accelerator_jobs")?,
        "accelerator_dispatches": integer("accelerator_dispatches")?,
        "accelerator_dispatched_jobs": integer("accelerator_dispatched_jobs")?,
        "cpu_fallback_jobs": integer("cpu_fallback_jobs")?,
    }))
}

pub(super) fn parse_metal_auto_bench_line(line: &str) -> Option<serde_json::Value> {
    const PREFIX: &str = "j2k_metal_encode_auto_bench ";
    let fields = j2k_profile::parse_profile_key_value_fields(line.strip_prefix(PREFIX)?);
    let auto_ms = required_field(&fields, "auto_ms")?;
    Some(serde_json::json!({
        "mode": required_field(&fields, "mode")?,
        "codec": required_field(&fields, "codec")?,
        "components": required_field(&fields, "components")?,
        "size": required_field(&fields, "size")?,
        "cpu_ms": parse_decimal_field(&fields, "cpu_ms")?,
        "auto_ms": parse_optional_decimal(auto_ms)?,
    }))
}

pub(super) fn parse_metal_auto_probe_line(line: &str) -> Option<serde_json::Value> {
    const PREFIX: &str = "j2k_metal_encode_auto_probe ";
    let rest = line.strip_prefix(PREFIX)?;
    let fields = j2k_profile::parse_profile_key_value_fields(rest);
    let mut row = serde_json::json!({
        "mode": required_field(&fields, "mode")?,
        "codec": required_field(&fields, "codec")?,
        "components": required_field(&fields, "components")?,
        "size": required_field(&fields, "size")?,
    });
    if let Some(dispatch) = suffix_after_key(rest, "dispatch") {
        row["dispatch"] = serde_json::Value::String(dispatch.to_string());
    }
    if let Some(error) = suffix_after_key(rest, "error") {
        row["error"] = serde_json::Value::String(error.to_string());
    }
    Some(row)
}

pub(super) fn parse_metal_stage_bench_line(line: &str) -> Option<serde_json::Value> {
    const PREFIX: &str = "j2k_metal_encode_stage_bench ";
    let rest = line.strip_prefix(PREFIX)?;
    let fields = j2k_profile::parse_profile_key_value_fields(rest);
    let metal_ms = required_field(&fields, "metal_ms")?;
    let mut row = serde_json::json!({
        "stage": required_field(&fields, "stage")?,
        "size": required_field(&fields, "size")?,
        "cpu_ms": parse_decimal_field(&fields, "cpu_ms")?,
        "metal_ms": parse_optional_decimal(metal_ms)?,
    });
    if let Some(dispatch) = suffix_after_key(rest, "dispatch") {
        row["dispatch"] = serde_json::Value::String(dispatch.to_string());
    }
    if let Some(error) = suffix_after_key(rest, "error") {
        row["error"] = serde_json::Value::String(error.to_string());
    }
    Some(row)
}

pub(super) fn parse_metal_resident_bench_line(line: &str) -> Option<serde_json::Value> {
    const PREFIX: &str = "j2k_metal_encode_resident_bench ";
    let rest = line.strip_prefix(PREFIX)?;
    let fields = j2k_profile::parse_profile_key_value_fields(rest);
    let mut row = serde_json::json!({
        "mode": required_field(&fields, "mode")?,
        "codec": required_field(&fields, "codec")?,
        "components": required_field(&fields, "components")?,
        "size": required_field(&fields, "size")?,
        "batch_size": parse_integer_field(&fields, "batch_size")?,
        "fixture_count": parse_integer_field(&fields, "fixture_count")?,
        "cpu_ms": parse_decimal_or_label_field(&fields, "cpu_ms")?,
        "hybrid_cpu_packet_ms": parse_decimal_or_label_field(&fields, "hybrid_cpu_packet_ms")?,
        "resident_host_ms": parse_decimal_or_label_field(&fields, "resident_host_ms")?,
        "resident_buffer_ms": parse_decimal_or_label_field(&fields, "resident_buffer_ms")?,
        "packetization_used": parse_bool_field(&fields, "packetization_used")?,
        "codestream_assembly_used": parse_bool_field(&fields, "codestream_assembly_used")?,
        "host_readback_ms": parse_decimal_or_label_field(&fields, "host_readback_ms")?,
        "gpu_ms": parse_decimal_or_label_field(&fields, "gpu_ms")?,
        "encoded_host_bytes": parse_integer_or_label_field(&fields, "encoded_host_bytes")?,
        "encoded_buffer_bytes": parse_integer_or_label_field(&fields, "encoded_buffer_bytes")?,
    });
    if let Some(error) = suffix_after_key(rest, "error") {
        row["error"] = serde_json::Value::String(error.to_string());
    }
    if let Some(value) = optional_field(&fields, "resident_input_storage") {
        row["resident_input_storage"] = serde_json::Value::String(value.to_string());
    }
    if let Some(value) = optional_field(&fields, "resident_staging") {
        row["resident_staging"] = serde_json::Value::String(value.to_string());
    }
    Some(row)
}

pub(super) fn required_field(fields: &[(String, String)], key: &str) -> Option<String> {
    fields
        .iter()
        .find_map(|(field_key, value)| (field_key == key).then(|| value.clone()))
}

pub(super) fn optional_field<'a>(fields: &'a [(String, String)], key: &str) -> Option<&'a str> {
    fields
        .iter()
        .find_map(|(field_key, value)| (field_key == key).then_some(value.as_str()))
}

pub(super) fn parse_decimal_field(fields: &[(String, String)], key: &str) -> Option<f64> {
    required_field(fields, key)?.parse().ok()
}

pub(super) fn parse_integer_field(fields: &[(String, String)], key: &str) -> Option<u64> {
    required_field(fields, key)?.parse().ok()
}

pub(super) fn parse_bool_field(fields: &[(String, String)], key: &str) -> Option<bool> {
    match required_field(fields, key)?.as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

pub(super) fn parse_decimal_or_label_field(
    fields: &[(String, String)],
    key: &str,
) -> Option<serde_json::Value> {
    parse_decimal_or_label(required_field(fields, key)?)
}

pub(super) fn parse_integer_or_label_field(
    fields: &[(String, String)],
    key: &str,
) -> Option<serde_json::Value> {
    let value = required_field(fields, key)?;
    if let Ok(number) = value.parse::<u64>() {
        return Some(serde_json::Value::Number(number.into()));
    }
    Some(serde_json::Value::String(value))
}

pub(super) fn parse_decimal_or_label(value: String) -> Option<serde_json::Value> {
    if let Ok(number) = value.parse::<f64>() {
        return serde_json::Number::from_f64(number).map(serde_json::Value::Number);
    }
    Some(serde_json::Value::String(value))
}

pub(super) fn parse_optional_decimal(value: String) -> Option<serde_json::Value> {
    if value == "skipped" {
        return Some(serde_json::Value::String(value));
    }
    serde_json::Number::from_f64(value.parse().ok()?).map(serde_json::Value::Number)
}

pub(super) fn suffix_after_key<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!(" {key}=");
    let start = text.find(&needle)? + needle.len();
    Some(&text[start..])
}

pub(super) fn read_tsv_metadata(path: &Path, keys: &[&str]) -> Result<serde_json::Value, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let mut map = serde_json::Map::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('\t') else {
            continue;
        };
        if keys.contains(&key) {
            map.insert(
                key.to_string(),
                serde_json::Value::String(value.to_string()),
            );
        }
    }
    if map.is_empty() {
        return Err(format!("{} contained no fixture metadata", path.display()));
    }
    Ok(serde_json::Value::Object(map))
}
