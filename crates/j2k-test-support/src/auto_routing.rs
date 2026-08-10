// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path},
};

use sha2::{Digest, Sha256};

mod identity;
mod schema;

pub use identity::validate_auto_routing_decode_identity;
pub use schema::{
    AutoRoutingBackend, AutoRoutingCell, AutoRoutingCodec, AutoRoutingContainer,
    AutoRoutingEvidence, AutoRoutingExecution, AutoRoutingManifest, AutoRoutingManifestCase,
    AutoRoutingOperation, AutoRoutingPixelFormat, AutoRoutingPlatform, AutoRoutingPnm,
    AutoRoutingRoute, AutoRoutingWorkload, AutoRoutingWorkloadKind, AutoRoutingWorkloadSet,
};

const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const MAX_CASE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_TOTAL_CASE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_CASES: usize = 4_096;

/// Load and hash-check every case named by an external Auto-routing manifest.
///
/// # Errors
///
/// Returns an error for malformed or oversized manifests, unsafe paths, duplicate
/// case IDs, unsupported layouts, non-regular files, or hash mismatches.
pub fn load_auto_routing_manifest(
    manifest_path: &Path,
    corpus_root: &Path,
) -> Result<AutoRoutingWorkloadSet, String> {
    let manifest_bytes =
        read_bounded_regular_file(manifest_path, MAX_MANIFEST_BYTES, "Auto-routing manifest")?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let manifest: AutoRoutingManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| {
            format!(
                "parse Auto-routing manifest {}: {error}",
                manifest_path.display()
            )
        })?;
    validate_manifest_header(&manifest)?;
    let canonical_root = corpus_root.canonicalize().map_err(|error| {
        format!(
            "canonicalize Auto-routing corpus root {}: {error}",
            corpus_root.display()
        )
    })?;
    if !canonical_root.is_dir() {
        return Err(format!(
            "Auto-routing corpus root {} is not a directory",
            corpus_root.display()
        ));
    }

    let mut ids = BTreeSet::new();
    let mut total_bytes = 0u64;
    let mut workloads = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        workloads.push(load_auto_routing_case(
            case,
            manifest.schema_version,
            &canonical_root,
            &mut ids,
            &mut total_bytes,
        )?);
    }

    Ok(AutoRoutingWorkloadSet {
        manifest,
        manifest_sha256,
        workloads,
    })
}

fn load_auto_routing_case(
    case: &AutoRoutingManifestCase,
    schema_version: u32,
    canonical_root: &Path,
    ids: &mut BTreeSet<String>,
    total_bytes: &mut u64,
) -> Result<AutoRoutingWorkload, String> {
    if !is_safe_id(&case.id) || !ids.insert(case.id.clone()) {
        return Err("Auto-routing manifest cases must have unique ids".to_string());
    }
    if !is_lower_hex(&case.sha256, 64) {
        return Err(format!(
            "Auto-routing case {} has an invalid SHA-256",
            case.id
        ));
    }
    let relative = Path::new(&case.path);
    if case.path.is_empty()
        || relative.is_absolute()
        || !relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "Auto-routing case {} must use a safe relative path",
            case.id
        ));
    }
    let unresolved = canonical_root.join(relative);
    let metadata = fs::symlink_metadata(&unresolved).map_err(|error| {
        format!(
            "read Auto-routing case {} at {}: {error}",
            case.id,
            unresolved.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(format!(
            "Auto-routing case {} must be a non-empty regular file",
            case.id
        ));
    }
    if metadata.len() > MAX_CASE_BYTES {
        return Err(format!("Auto-routing case {} is too large", case.id));
    }
    *total_bytes = total_bytes
        .checked_add(metadata.len())
        .ok_or_else(|| "Auto-routing corpus byte count overflow".to_string())?;
    if *total_bytes > MAX_TOTAL_CASE_BYTES {
        return Err("Auto-routing corpus exceeds its total byte limit".to_string());
    }
    let path = unresolved.canonicalize().map_err(|error| {
        format!(
            "canonicalize Auto-routing case {} at {}: {error}",
            case.id,
            unresolved.display()
        )
    })?;
    if !path.starts_with(canonical_root) {
        return Err(format!(
            "Auto-routing case {} escapes the corpus root",
            case.id
        ));
    }
    let bytes = fs::read(&path).map_err(|error| {
        format!(
            "read Auto-routing case {} at {}: {error}",
            case.id,
            path.display()
        )
    })?;
    let actual_sha256 = sha256_hex(&bytes);
    if actual_sha256 != case.sha256 {
        return Err(format!(
            "Auto-routing case {} SHA-256 mismatch: expected {}, found {actual_sha256}",
            case.id, case.sha256
        ));
    }
    let (codec, container) = case_format(case, schema_version)?;
    Ok(AutoRoutingWorkload {
        id: case.id.clone(),
        path,
        kind: case.kind,
        codec,
        container,
        pixel_format: case.pixel_format,
        bytes,
    })
}

fn case_format(
    case: &AutoRoutingManifestCase,
    schema_version: u32,
) -> Result<(AutoRoutingCodec, AutoRoutingContainer), String> {
    let format = match (schema_version, case.codec, case.container) {
        (1, None, None) => (AutoRoutingCodec::Jpeg2000Part1, AutoRoutingContainer::Codestream),
        (2, Some(codec), Some(container)) => (codec, container),
        _ => {
            return Err(format!(
                "Auto-routing case {} must use codec and container fields required by its schema version",
                case.id
            ))
        }
    };
    if !matches!(
        format,
        (
            AutoRoutingCodec::Jpeg2000Part1,
            AutoRoutingContainer::Codestream | AutoRoutingContainer::Jp2
        ) | (
            AutoRoutingCodec::Htj2kPart15,
            AutoRoutingContainer::Codestream | AutoRoutingContainer::Jph
        )
    ) {
        return Err(format!(
            "Auto-routing case {} has an invalid codec/container pair",
            case.id
        ));
    }
    Ok(format)
}

/// Serialize validated route evidence deterministically with a trailing newline.
///
/// # Errors
///
/// Returns an error when evidence identity, route labels, output hashes, or the
/// destination path are invalid.
pub fn write_auto_routing_evidence(
    output_path: &Path,
    evidence: &AutoRoutingEvidence,
) -> Result<(), String> {
    validate_evidence(evidence)?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "create Auto-routing evidence directory {}: {error}",
                parent.display()
            )
        })?;
    }
    let mut json = serde_json::to_string_pretty(evidence)
        .map_err(|error| format!("serialize Auto-routing evidence: {error}"))?;
    json.push('\n');
    fs::write(output_path, json).map_err(|error| {
        format!(
            "write Auto-routing evidence {}: {error}",
            output_path.display()
        )
    })
}

/// Return the lowercase SHA-256 digest for an output byte sequence.
#[must_use]
pub fn auto_routing_sha256(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}

/// Parse and cross-check one encode workload as a binary 8-bit PGM or PPM.
///
/// # Errors
///
/// Returns an error when the workload is not a codestream encode input, the
/// PNM is malformed, or its declared pixel layout disagrees with the payload.
pub fn load_auto_routing_pnm(workload: &AutoRoutingWorkload) -> Result<AutoRoutingPnm, String> {
    if workload.kind != AutoRoutingWorkloadKind::Encode {
        return Err(format!(
            "Auto-routing workload {} is not an encode input",
            workload.id
        ));
    }
    if workload.container != AutoRoutingContainer::Codestream {
        return Err(format!(
            "Auto-routing encode workload {} currently measures codestream output only",
            workload.id
        ));
    }
    let image = crate::read_pnm_image(&workload.path).map_err(|error| {
        format!(
            "read encode workload {} as binary PNM: {error}",
            workload.id
        )
    })?;
    let expected_components = match workload.pixel_format {
        AutoRoutingPixelFormat::Gray8 => 1,
        AutoRoutingPixelFormat::Rgb8 => 3,
    };
    if image.channels != expected_components {
        return Err(format!(
            "encode workload {} declares {:?} but its PNM has {} components",
            workload.id, workload.pixel_format, image.channels
        ));
    }
    Ok(AutoRoutingPnm {
        id: workload.id.clone(),
        codec: workload.codec,
        pixels: image.pixels,
        width: image.width,
        height: image.height,
        components: u16::try_from(image.channels)
            .map_err(|_| "PNM channel count does not fit u16".to_string())?,
    })
}

/// Build one CPU-versus-hybrid evidence cell with no device-native claim.
#[must_use]
pub fn auto_routing_route_cell(
    workload: &str,
    operation: AutoRoutingOperation,
    criterion_group_id: &str,
    output_sha256: String,
) -> AutoRoutingCell {
    AutoRoutingCell {
        id: format!("{}-{workload}", auto_routing_operation_label(operation)),
        operation,
        source: "external".to_string(),
        workload: workload.to_string(),
        cpu: AutoRoutingRoute {
            criterion_id: format!("{criterion_group_id}/cpu"),
            execution: AutoRoutingExecution::Cpu,
            output_sha256: output_sha256.clone(),
        },
        hybrid: AutoRoutingRoute {
            criterion_id: format!("{criterion_group_id}/hybrid"),
            execution: AutoRoutingExecution::Hybrid,
            output_sha256,
        },
        strict_device_supported: false,
        strict_device: None,
    }
}

/// Append one length-delimited route output to a batch parity buffer.
///
/// # Errors
///
/// Returns an error when the output length cannot be represented as `u64`.
pub fn append_auto_routing_output(output: &mut Vec<u8>, bytes: &[u8]) -> Result<(), String> {
    let len = u64::try_from(bytes.len()).map_err(|_| "route output is too large")?;
    output.extend_from_slice(&len.to_le_bytes());
    output.extend_from_slice(bytes);
    Ok(())
}

/// Stable operation label used by Criterion IDs and verification reports.
#[must_use]
pub const fn auto_routing_operation_label(operation: AutoRoutingOperation) -> &'static str {
    match operation {
        AutoRoutingOperation::FullDecode => "full-decode",
        AutoRoutingOperation::RoiDecode => "roi-decode",
        AutoRoutingOperation::ScaledDecode => "scaled-decode",
        AutoRoutingOperation::BatchDecode => "batch-decode",
        AutoRoutingOperation::LosslessEncode => "lossless-encode",
        AutoRoutingOperation::LossyEncode => "lossy-encode",
    }
}

fn validate_manifest_header(manifest: &AutoRoutingManifest) -> Result<(), String> {
    if !matches!(manifest.schema_version, 1 | 2)
        || manifest.corpus.is_empty()
        || manifest.cases.is_empty()
        || manifest.cases.len() > MAX_CASES
        || !manifest.source_url.starts_with("https://")
        || manifest.source_url["https://".len()..].is_empty()
        || manifest.source_url.chars().any(char::is_whitespace)
    {
        return Err("Auto-routing manifest identity or case inventory is invalid".to_string());
    }
    Ok(())
}

fn validate_evidence(evidence: &AutoRoutingEvidence) -> Result<(), String> {
    if !matches!(evidence.schema_version, 1 | 2)
        || !is_lower_hex(&evidence.candidate_sha, 40)
        || !is_lower_hex(&evidence.external_manifest_sha256, 64)
        || evidence.external_case_count == 0
        || evidence.cells.is_empty()
        || evidence.cells.len() > MAX_CASES
    {
        return Err("Auto-routing evidence identity is invalid".to_string());
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
    let expected_platform = match evidence.backend {
        AutoRoutingBackend::Cuda => ("linux", "x86_64"),
        AutoRoutingBackend::Metal => ("macos", "aarch64"),
    };
    if (
        evidence.platform.os.as_str(),
        evidence.platform.arch.as_str(),
    ) != expected_platform
    {
        return Err("Auto-routing backend and platform identity do not match".to_string());
    }

    let mut cell_ids = BTreeSet::new();
    let mut criterion_ids = BTreeSet::new();
    for cell in &evidence.cells {
        if !is_safe_id(&cell.id)
            || !is_safe_id(&cell.workload)
            || cell.source != "external"
            || !cell_ids.insert(cell.id.as_str())
        {
            return Err("Auto-routing cells must have unique safe external ids".to_string());
        }
        validate_route(
            "CPU route",
            &cell.cpu,
            AutoRoutingExecution::Cpu,
            &mut criterion_ids,
        )?;
        validate_route(
            "hybrid route",
            &cell.hybrid,
            AutoRoutingExecution::Hybrid,
            &mut criterion_ids,
        )?;
        match (cell.strict_device_supported, &cell.strict_device) {
            (false, None) => {}
            (true, Some(route)) => validate_route(
                "strict-device route",
                route,
                AutoRoutingExecution::DeviceNative,
                &mut criterion_ids,
            )?,
            _ => return Err("Auto-routing strict-device support is inconsistent".to_string()),
        }
        if cell.cpu.output_sha256 != cell.hybrid.output_sha256
            || cell
                .strict_device
                .as_ref()
                .is_some_and(|route| route.output_sha256 != cell.cpu.output_sha256)
        {
            return Err(format!(
                "Auto-routing cell {} routes do not produce identical outputs",
                cell.id
            ));
        }
    }
    Ok(())
}

fn validate_route(
    label: &str,
    route: &AutoRoutingRoute,
    expected_execution: AutoRoutingExecution,
    criterion_ids: &mut BTreeSet<String>,
) -> Result<(), String> {
    if route.execution != expected_execution
        || !is_safe_criterion_id(&route.criterion_id)
        || !is_lower_hex(&route.output_sha256, 64)
        || !criterion_ids.insert(route.criterion_id.clone())
    {
        return Err(format!("Auto-routing {label} is invalid or duplicated"));
    }
    Ok(())
}

fn read_bounded_regular_file(path: &Path, max_bytes: u64, label: &str) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("read {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > max_bytes
    {
        return Err(format!(
            "{label} {} must be a non-empty bounded regular file",
            path.display()
        ));
    }
    fs::read(path).map_err(|error| format!("read {label} {}: {error}", path.display()))
}

fn is_safe_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn is_safe_criterion_id(value: &str) -> bool {
    !value.is_empty()
        && !value.contains('\\')
        && Path::new(value)
            .components()
            .all(|component| matches!(component, Component::Normal(segment) if !segment.is_empty()))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests;
