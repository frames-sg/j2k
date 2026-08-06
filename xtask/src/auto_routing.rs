// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component as PathComponent, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::perf_guard::{discover_estimates, BenchEstimate};

const REQUIRED_OPERATIONS: [Operation; 6] = [
    Operation::FullDecode,
    Operation::RoiDecode,
    Operation::ScaledDecode,
    Operation::BatchDecode,
    Operation::LosslessEncode,
    Operation::LossyEncode,
];
const MAX_EVIDENCE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_ESTIMATE_BYTES: u64 = 1024 * 1024;
const MAX_CELLS: usize = 4_096;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Evidence {
    schema_version: u32,
    candidate_sha: String,
    backend: Backend,
    platform: Platform,
    external_manifest_sha256: String,
    external_case_count: usize,
    cells: Vec<Cell>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalManifest {
    schema_version: u32,
    corpus: String,
    source_url: String,
    cases: Vec<ExternalCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalCase {
    id: String,
    path: String,
    kind: WorkloadKind,
    pixel_format: WorkloadPixelFormat,
    sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum WorkloadKind {
    Decode,
    Encode,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum WorkloadPixelFormat {
    Gray8,
    Rgb8,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Backend {
    Cuda,
    Metal,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Platform {
    os: String,
    arch: String,
    hardware: String,
    driver: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cell {
    id: String,
    operation: Operation,
    source: String,
    workload: String,
    cpu: Route,
    hybrid: Route,
    strict_device_supported: bool,
    strict_device: Option<Route>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Operation {
    FullDecode,
    RoiDecode,
    ScaledDecode,
    BatchDecode,
    LosslessEncode,
    LossyEncode,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Route {
    criterion_id: String,
    execution: Execution,
    output_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum Execution {
    Cpu,
    Hybrid,
    DeviceNative,
}

#[derive(Debug)]
struct VerifiedEvidence {
    cell_count: usize,
    artifact_sha256: String,
    report_json: String,
}

pub(crate) fn auto_routing(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let Some(command) = args.next() else {
        return Err(usage());
    };
    if command != "verify" {
        return Err(format!(
            "unknown auto-routing command `{command}`\n{}",
            usage()
        ));
    }
    let mut evidence = None;
    let mut external_manifest = None;
    let mut criterion_root = None;
    let mut output = None;
    while let Some(argument) = args.next() {
        let destination = match argument.as_str() {
            "--evidence" => &mut evidence,
            "--external-manifest" => &mut external_manifest,
            "--criterion-root" => &mut criterion_root,
            "--out" => &mut output,
            _ => {
                return Err(format!(
                    "unknown auto-routing argument `{argument}`\n{}",
                    usage()
                ))
            }
        };
        let value = args
            .next()
            .ok_or_else(|| format!("{argument} requires a path"))?;
        *destination = Some(PathBuf::from(value));
    }
    let evidence = evidence.ok_or_else(usage)?;
    let external_manifest = external_manifest.ok_or_else(usage)?;
    let criterion_root = criterion_root.ok_or_else(usage)?;
    let output = output.ok_or_else(usage)?;
    let verified = verify_evidence(&evidence, &external_manifest, &criterion_root)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create Auto-routing report directory: {error}"))?;
    }
    fs::write(&output, &verified.report_json)
        .map_err(|error| format!("write Auto-routing report {}: {error}", output.display()))?;
    eprintln!(
        "verified {} Auto-routing workload cells; artifact SHA-256 {}; wrote {}",
        verified.cell_count,
        verified.artifact_sha256,
        output.display()
    );
    Ok(())
}

fn verify_evidence(
    evidence_path: &Path,
    external_manifest_path: &Path,
    criterion_root: &Path,
) -> Result<VerifiedEvidence, String> {
    let evidence = read_evidence(evidence_path)?;
    validate_header(&evidence)?;
    let external_cases = validate_external_manifest(external_manifest_path, &evidence)?;
    let estimates = discover_estimates(criterion_root)?;
    let estimates = estimates
        .into_iter()
        .map(|estimate| (estimate.id.clone(), estimate))
        .collect::<BTreeMap<_, _>>();

    let mut cell_ids = BTreeSet::new();
    let mut criterion_ids = BTreeSet::new();
    let mut operations = BTreeSet::new();
    let mut used_external_cases = BTreeSet::new();
    for cell in &evidence.cells {
        let expected_kind = match cell.operation {
            Operation::FullDecode
            | Operation::RoiDecode
            | Operation::ScaledDecode
            | Operation::BatchDecode => WorkloadKind::Decode,
            Operation::LosslessEncode | Operation::LossyEncode => WorkloadKind::Encode,
        };
        if cell.id.is_empty()
            || cell.workload.is_empty()
            || cell.source != "external"
            || external_cases.get(&cell.workload) != Some(&expected_kind)
            || !cell_ids.insert(cell.id.as_str())
        {
            return Err(format!(
                "Auto-routing cell {:?} must have a unique id and a typed workload from the external manifest",
                cell.id
            ));
        }
        used_external_cases.insert(cell.workload.as_str());
        operations.insert(cell.operation);
        validate_route(
            &cell.id,
            "CPU",
            &cell.cpu,
            Execution::Cpu,
            criterion_root,
            &estimates,
            &mut criterion_ids,
        )?;
        validate_route(
            &cell.id,
            "hybrid",
            &cell.hybrid,
            Execution::Hybrid,
            criterion_root,
            &estimates,
            &mut criterion_ids,
        )?;
        match (cell.strict_device_supported, &cell.strict_device) {
            (true, Some(route)) => validate_route(
                &cell.id,
                "strict-device",
                route,
                Execution::DeviceNative,
                criterion_root,
                &estimates,
                &mut criterion_ids,
            )?,
            (false, None) => {}
            _ => {
                return Err(format!(
                    "Auto-routing cell {} has inconsistent strict-device support",
                    cell.id
                ));
            }
        }
        validate_cell_metrics(cell)?;
    }
    let required = REQUIRED_OPERATIONS.into_iter().collect::<BTreeSet<_>>();
    if operations != required {
        return Err("Auto-routing evidence must cover full, ROI, scaled, batch, lossless encode, and lossy encode workloads".to_string());
    }
    if used_external_cases.len() != external_cases.len() {
        return Err("Auto-routing evidence must exercise every external manifest case".to_string());
    }
    let artifact_sha256 = artifact_sha256(
        evidence_path,
        external_manifest_path,
        criterion_root,
        &criterion_ids,
    )?;
    let report_json = verification_report(&evidence, &estimates, &artifact_sha256)?;
    Ok(VerifiedEvidence {
        cell_count: evidence.cells.len(),
        artifact_sha256,
        report_json,
    })
}

fn validate_external_manifest(
    path: &Path,
    evidence: &Evidence,
) -> Result<BTreeMap<String, WorkloadKind>, String> {
    let bytes = read_bounded_regular_file(path, MAX_EVIDENCE_BYTES, "external manifest")?;
    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if actual_sha256 != evidence.external_manifest_sha256 {
        return Err(format!(
            "external manifest SHA-256 mismatch: expected {}, found {actual_sha256}",
            evidence.external_manifest_sha256
        ));
    }
    let manifest: ExternalManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse external manifest {}: {error}", path.display()))?;
    if manifest.schema_version != 1
        || manifest.corpus.is_empty()
        || !manifest.source_url.starts_with("https://")
        || manifest.source_url["https://".len()..].is_empty()
        || manifest
            .source_url
            .bytes()
            .any(|byte| byte.is_ascii_whitespace())
        || manifest.cases.is_empty()
        || manifest.cases.len() > MAX_CELLS
        || manifest.cases.len() != evidence.external_case_count
    {
        return Err("external manifest identity or case inventory is invalid".to_string());
    }
    let mut cases = BTreeMap::new();
    for case in manifest.cases {
        if !is_safe_case_id(&case.id)
            || !is_safe_relative_path(&case.path)
            || !is_lower_hex(&case.sha256, 64)
            || cases.insert(case.id, case.kind).is_some()
        {
            return Err("external manifest cases must have unique ids, safe relative paths, typed pixel formats, and lowercase SHA-256 hashes".to_string());
        }
        let _ = case.pixel_format;
    }
    Ok(cases)
}

fn read_evidence(path: &Path) -> Result<Evidence, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("read Auto-routing evidence {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_EVIDENCE_BYTES {
        return Err("Auto-routing evidence must be a non-empty bounded regular file".to_string());
    }
    let text = fs::read_to_string(path)
        .map_err(|error| format!("read Auto-routing evidence {}: {error}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("parse Auto-routing evidence {}: {error}", path.display()))
}

fn validate_header(evidence: &Evidence) -> Result<(), String> {
    if evidence.schema_version != 1
        || !is_hex(&evidence.candidate_sha, 40)
        || !is_hex(&evidence.external_manifest_sha256, 64)
        || evidence.external_case_count == 0
        || evidence.cells.is_empty()
        || evidence.cells.len() > MAX_CELLS
    {
        return Err(
            "Auto-routing evidence header or external corpus inventory is invalid".to_string(),
        );
    }
    if [
        &evidence.platform.os,
        &evidence.platform.arch,
        &evidence.platform.hardware,
        &evidence.platform.driver,
    ]
    .into_iter()
    .any(String::is_empty)
    {
        return Err("Auto-routing platform identity must be complete".to_string());
    }
    let platform_matches = match evidence.backend {
        Backend::Cuda => evidence.platform.os == "linux" && evidence.platform.arch == "x86_64",
        Backend::Metal => evidence.platform.os == "macos" && evidence.platform.arch == "aarch64",
    };
    if !platform_matches {
        return Err("Auto-routing backend and platform identity do not match".to_string());
    }
    Ok(())
}

fn validate_route(
    cell_id: &str,
    label: &str,
    route: &Route,
    expected_execution: Execution,
    criterion_root: &Path,
    estimates: &BTreeMap<String, BenchEstimate>,
    criterion_ids: &mut BTreeSet<String>,
) -> Result<(), String> {
    if route.execution != expected_execution
        || !is_hex(&route.output_sha256, 64)
        || !is_safe_criterion_id(&route.criterion_id)
        || !criterion_ids.insert(route.criterion_id.clone())
    {
        return Err(format!(
            "Auto-routing cell {cell_id} has invalid or duplicate {label} route evidence"
        ));
    }
    let estimate = estimates.get(&route.criterion_id).ok_or_else(|| {
        format!(
            "Auto-routing cell {cell_id} is missing Criterion estimate {}",
            route.criterion_id
        )
    })?;
    validate_estimate(cell_id, label, estimate)?;
    validate_confidence_level(criterion_root, &route.criterion_id)?;
    Ok(())
}

fn validate_estimate(cell_id: &str, label: &str, estimate: &BenchEstimate) -> Result<(), String> {
    if !estimate.median_ns.is_finite()
        || !estimate.median_lower_ns.is_finite()
        || !estimate.median_upper_ns.is_finite()
        || estimate.median_lower_ns <= 0.0
        || estimate.median_lower_ns > estimate.median_ns
        || estimate.median_ns > estimate.median_upper_ns
    {
        return Err(format!(
            "Auto-routing cell {cell_id} has an invalid {label} Criterion median interval"
        ));
    }
    Ok(())
}

fn validate_confidence_level(root: &Path, criterion_id: &str) -> Result<(), String> {
    let path = root.join(criterion_id).join("new/estimates.json");
    let value: serde_json::Value = serde_json::from_slice(&read_bounded_estimate(&path)?)
        .map_err(|error| format!("parse Criterion estimate {}: {error}", path.display()))?;
    let level = value
        .pointer("/median/confidence_interval/confidence_level")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            format!(
                "Criterion estimate {} is missing its confidence level",
                path.display()
            )
        })?;
    if (level - 0.95).abs() > f64::EPSILON {
        return Err(format!(
            "Criterion estimate {} does not use a 95% confidence interval",
            path.display()
        ));
    }
    Ok(())
}

fn validate_cell_metrics(cell: &Cell) -> Result<(), String> {
    validate_output_parity(cell, &cell.cpu, &cell.hybrid)?;
    if let Some(strict) = &cell.strict_device {
        validate_output_parity(cell, &cell.cpu, strict)?;
    }
    Ok(())
}

fn validate_output_parity(cell: &Cell, expected: &Route, actual: &Route) -> Result<(), String> {
    if expected.output_sha256 != actual.output_sha256 {
        return Err(format!(
            "Auto-routing cell {} routes do not produce identical outputs",
            cell.id
        ));
    }
    Ok(())
}

fn is_qualifying_win(hybrid: &BenchEstimate, competitor: &BenchEstimate) -> bool {
    hybrid.median_ns <= competitor.median_ns * 0.9
        && hybrid.median_upper_ns < competitor.median_lower_ns
}

fn is_safe_criterion_id(value: &str) -> bool {
    if value.is_empty() || value.contains('\\') {
        return false;
    }
    Path::new(value)
        .components()
        .all(|component| matches!(component, PathComponent::Normal(segment) if !segment.is_empty()))
}

fn is_safe_case_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn is_safe_relative_path(value: &str) -> bool {
    !value.is_empty()
        && !value.contains('\\')
        && !Path::new(value).is_absolute()
        && Path::new(value).components().all(
            |component| matches!(component, PathComponent::Normal(segment) if !segment.is_empty()),
        )
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn artifact_sha256(
    evidence_path: &Path,
    external_manifest_path: &Path,
    criterion_root: &Path,
    criterion_ids: &BTreeSet<String>,
) -> Result<String, String> {
    let mut hasher = Sha256::new();
    hasher.update(b"j2k-auto-routing-evidence-v1\0");
    hash_artifact_file(
        &mut hasher,
        "evidence.json",
        &read_bounded_regular_file(evidence_path, MAX_EVIDENCE_BYTES, "Auto-routing evidence")?,
    )?;
    hash_artifact_file(
        &mut hasher,
        "external-manifest.json",
        &read_bounded_regular_file(
            external_manifest_path,
            MAX_EVIDENCE_BYTES,
            "external manifest",
        )?,
    )?;
    for criterion_id in criterion_ids {
        let path = criterion_root.join(criterion_id).join("new/estimates.json");
        hash_artifact_file(&mut hasher, criterion_id, &read_bounded_estimate(&path)?)?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_artifact_file(hasher: &mut Sha256, name: &str, bytes: &[u8]) -> Result<(), String> {
    let name_len = u64::try_from(name.len())
        .map_err(|_| format!("Auto-routing artifact name is too long: {name}"))?;
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_| format!("Auto-routing artifact is too large: {name}"))?;
    hasher.update(name_len.to_le_bytes());
    hasher.update(name.as_bytes());
    hasher.update(byte_len.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

fn read_bounded_estimate(path: &Path) -> Result<Vec<u8>, String> {
    read_bounded_regular_file(path, MAX_ESTIMATE_BYTES, "Criterion estimate")
}

fn read_bounded_regular_file(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>, String> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("read {label} {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > limit {
        return Err(format!(
            "{label} {} must be a non-empty bounded regular file",
            path.display()
        ));
    }
    fs::read(path).map_err(|error| format!("read {label} {}: {error}", path.display()))
}

fn verification_report(
    evidence: &Evidence,
    estimates: &BTreeMap<String, BenchEstimate>,
    artifact_sha256: &str,
) -> Result<String, String> {
    let cells = evidence
        .cells
        .iter()
        .map(|cell| {
            let cpu = &estimates[&cell.cpu.criterion_id];
            let hybrid = &estimates[&cell.hybrid.criterion_id];
            let promotes_hybrid = is_qualifying_win(hybrid, cpu)
                && cell
                    .strict_device
                    .as_ref()
                    .is_none_or(|route| is_qualifying_win(hybrid, &estimates[&route.criterion_id]));
            let strict = cell.strict_device.as_ref().map(|route| {
                let estimate = &estimates[&route.criterion_id];
                serde_json::json!({
                    "criterion_id": route.criterion_id,
                    "median_ns": estimate.median_ns,
                    "median_lower_ns": estimate.median_lower_ns,
                    "median_upper_ns": estimate.median_upper_ns,
                    "hybrid_speedup_percent": speedup_percent(hybrid, estimate),
                })
            });
            serde_json::json!({
                "id": cell.id,
                "operation": cell.operation,
                "source": cell.source,
                "workload": cell.workload,
                "decision": if promotes_hybrid { "promote-hybrid" } else { "retain-current" },
                "output_sha256": cell.hybrid.output_sha256,
                "cpu": estimate_json(&cell.cpu.criterion_id, cpu),
                "hybrid": estimate_json(&cell.hybrid.criterion_id, hybrid),
                "hybrid_speedup_vs_cpu_percent": speedup_percent(hybrid, cpu),
                "strict_device": strict,
                "status": if promotes_hybrid { "promoted" } else { "retained" },
            })
        })
        .collect::<Vec<_>>();
    let promoted_cell_count = evidence
        .cells
        .iter()
        .filter(|cell| {
            let hybrid = &estimates[&cell.hybrid.criterion_id];
            is_qualifying_win(hybrid, &estimates[&cell.cpu.criterion_id])
                && cell
                    .strict_device
                    .as_ref()
                    .is_none_or(|route| is_qualifying_win(hybrid, &estimates[&route.criterion_id]))
        })
        .count();
    let report = serde_json::json!({
        "schema_version": 1,
        "candidate_sha": evidence.candidate_sha,
        "backend": evidence.backend,
        "platform": evidence.platform,
        "external_manifest_sha256": evidence.external_manifest_sha256,
        "external_case_count": evidence.external_case_count,
        "criterion_confidence_level": 0.95,
        "minimum_hybrid_speedup_percent": 10.0,
        "artifact_sha256": artifact_sha256,
        "promoted_cell_count": promoted_cell_count,
        "cells": cells,
        "status": "pass",
    });
    let mut json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize Auto-routing verification report: {error}"))?;
    json.push('\n');
    Ok(json)
}

fn estimate_json(criterion_id: &str, estimate: &BenchEstimate) -> serde_json::Value {
    serde_json::json!({
        "criterion_id": criterion_id,
        "median_ns": estimate.median_ns,
        "median_lower_ns": estimate.median_lower_ns,
        "median_upper_ns": estimate.median_upper_ns,
    })
}

fn speedup_percent(hybrid: &BenchEstimate, competitor: &BenchEstimate) -> f64 {
    (1.0 - hybrid.median_ns / competitor.median_ns) * 100.0
}

fn usage() -> String {
    "usage: cargo xtask auto-routing verify --evidence FILE --external-manifest FILE --criterion-root DIR --out FILE".to_string()
}

#[cfg(test)]
mod tests;
