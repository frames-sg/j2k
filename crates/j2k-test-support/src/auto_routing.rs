// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const MAX_CASE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_TOTAL_CASE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_CASES: usize = 4_096;

/// A pinned external workload manifest for Auto-routing benchmarks.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingManifest {
    pub schema_version: u32,
    pub corpus: String,
    pub source_url: String,
    pub cases: Vec<AutoRoutingManifestCase>,
}

/// One hash-pinned input in an Auto-routing workload manifest.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingManifestCase {
    pub id: String,
    pub path: String,
    pub kind: AutoRoutingWorkloadKind,
    pub pixel_format: AutoRoutingPixelFormat,
    pub sha256: String,
}

/// Whether a workload is a compressed decode input or an uncompressed encode input.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRoutingWorkloadKind {
    Decode,
    Encode,
}

/// Pixel layout used for route-parity comparisons.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRoutingPixelFormat {
    Gray8,
    Rgb8,
}

/// One validated, in-memory external workload.
#[derive(Clone, Debug)]
pub struct AutoRoutingWorkload {
    pub id: String,
    pub path: PathBuf,
    pub kind: AutoRoutingWorkloadKind,
    pub pixel_format: AutoRoutingPixelFormat,
    pub bytes: Vec<u8>,
}

/// A validated manifest, its exact hash, and the inputs it names.
#[derive(Clone, Debug)]
pub struct AutoRoutingWorkloadSet {
    pub manifest: AutoRoutingManifest,
    pub manifest_sha256: String,
    pub workloads: Vec<AutoRoutingWorkload>,
}

/// Validated 8-bit PGM/PPM input for an encode benchmark cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutoRoutingPnm {
    pub id: String,
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub components: u16,
}

/// Accelerator lane that produced route evidence.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRoutingBackend {
    Cuda,
    Metal,
}

/// Hardware and software identity for one benchmark lane.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingPlatform {
    pub os: String,
    pub arch: String,
    pub hardware: String,
    pub driver: String,
}

/// Workload class evaluated for a fixed Auto-routing decision.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRoutingOperation {
    FullDecode,
    RoiDecode,
    ScaledDecode,
    BatchDecode,
    LosslessEncode,
    LossyEncode,
}

/// Actual execution class of a measured route.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRoutingExecution {
    Cpu,
    Hybrid,
    DeviceNative,
}

/// Criterion result identity and exact output produced by one route.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingRoute {
    pub criterion_id: String,
    pub execution: AutoRoutingExecution,
    pub output_sha256: String,
}

/// CPU, hybrid, and optional device-native measurements for one workload class.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingCell {
    pub id: String,
    pub operation: AutoRoutingOperation,
    pub source: String,
    pub workload: String,
    pub cpu: AutoRoutingRoute,
    pub hybrid: AutoRoutingRoute,
    pub strict_device_supported: bool,
    pub strict_device: Option<AutoRoutingRoute>,
}

/// Versioned route evidence emitted beside Criterion estimates.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRoutingEvidence {
    pub schema_version: u32,
    pub candidate_sha: String,
    pub backend: AutoRoutingBackend,
    pub platform: AutoRoutingPlatform,
    pub external_manifest_sha256: String,
    pub external_case_count: usize,
    pub cells: Vec<AutoRoutingCell>,
}

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
    Ok(AutoRoutingWorkload {
        id: case.id.clone(),
        path,
        kind: case.kind,
        pixel_format: case.pixel_format,
        bytes,
    })
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
/// Returns an error when the workload is not an encode input, the PNM is
/// malformed, or its declared pixel layout disagrees with the payload.
pub fn load_auto_routing_pnm(workload: &AutoRoutingWorkload) -> Result<AutoRoutingPnm, String> {
    if workload.kind != AutoRoutingWorkloadKind::Encode {
        return Err(format!(
            "Auto-routing workload {} is not an encode input",
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
    if manifest.schema_version != 1
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
    if evidence.schema_version != 1
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
mod tests {
    use std::fs;

    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::{
        auto_routing_route_cell, load_auto_routing_manifest, load_auto_routing_pnm,
        write_auto_routing_evidence, AutoRoutingBackend, AutoRoutingEvidence, AutoRoutingExecution,
        AutoRoutingOperation, AutoRoutingPlatform,
    };

    #[test]
    fn manifest_loader_hashes_and_classifies_bounded_external_inputs() {
        let root = temp_dir("manifest");
        let decode = root.join("decode/sample.j2k");
        let encode = root.join("encode/sample.ppm");
        fs::create_dir_all(decode.parent().unwrap()).unwrap();
        fs::create_dir_all(encode.parent().unwrap()).unwrap();
        fs::write(&decode, b"decode bytes").unwrap();
        fs::write(&encode, b"P6\n1 1\n255\n\x01\x02\x03").unwrap();
        let manifest = root.join("manifest.json");
        let manifest_value = json!({
            "schema_version": 1,
            "corpus": "routing-fixture",
            "source_url": "https://example.invalid/routing-fixture",
            "cases": [
                {
                    "id": "decode-case",
                    "path": "decode/sample.j2k",
                    "kind": "decode",
                    "pixel_format": "rgb8",
                    "sha256": sha256(b"decode bytes")
                },
                {
                    "id": "encode-case",
                    "path": "encode/sample.ppm",
                    "kind": "encode",
                    "pixel_format": "rgb8",
                    "sha256": sha256(b"P6\n1 1\n255\n\x01\x02\x03")
                }
            ]
        });
        fs::write(
            &manifest,
            serde_json::to_vec_pretty(&manifest_value).unwrap(),
        )
        .unwrap();

        let loaded = load_auto_routing_manifest(&manifest, &root).unwrap();

        assert_eq!(loaded.manifest.corpus, "routing-fixture");
        assert_eq!(loaded.workloads.len(), 2);
        assert_eq!(loaded.workloads[0].bytes, b"decode bytes");
        assert_eq!(loaded.workloads[1].bytes, b"P6\n1 1\n255\n\x01\x02\x03");
        assert_eq!(
            loaded.manifest_sha256,
            sha256(&fs::read(&manifest).unwrap())
        );
        let pnm = load_auto_routing_pnm(&loaded.workloads[1]).unwrap();
        assert_eq!((pnm.width, pnm.height, pnm.components), (1, 1, 3));
        assert_eq!(pnm.pixels, [1, 2, 3]);
    }

    #[test]
    fn manifest_loader_rejects_escape_hash_mismatch_and_duplicate_ids() {
        let root = temp_dir("invalid");
        fs::write(root.join("sample.j2k"), b"sample").unwrap();
        let manifest = root.join("manifest.json");
        let base = json!({
            "schema_version": 1,
            "corpus": "routing-fixture",
            "source_url": "https://example.invalid/routing-fixture",
            "cases": [{
                "id": "case",
                "path": "../outside.j2k",
                "kind": "decode",
                "pixel_format": "gray8",
                "sha256": sha256(b"sample")
            }]
        });
        fs::write(&manifest, serde_json::to_vec(&base).unwrap()).unwrap();
        assert!(load_auto_routing_manifest(&manifest, &root)
            .unwrap_err()
            .contains("safe relative path"));

        let mut mismatch = base.clone();
        mismatch["cases"][0]["path"] = json!("sample.j2k");
        mismatch["cases"][0]["sha256"] = json!("0".repeat(64));
        fs::write(&manifest, serde_json::to_vec(&mismatch).unwrap()).unwrap();
        assert!(load_auto_routing_manifest(&manifest, &root)
            .unwrap_err()
            .contains("SHA-256 mismatch"));

        let mut duplicate = mismatch;
        duplicate["cases"][0]["sha256"] = json!(sha256(b"sample"));
        let repeated = duplicate["cases"][0].clone();
        duplicate["cases"].as_array_mut().unwrap().push(repeated);
        fs::write(&manifest, serde_json::to_vec(&duplicate).unwrap()).unwrap();
        assert!(load_auto_routing_manifest(&manifest, &root)
            .unwrap_err()
            .contains("unique ids"));
    }

    #[test]
    fn evidence_writer_is_deterministic_and_refuses_invalid_route_labels() {
        let root = temp_dir("evidence");
        let output = root.join("nested/evidence.json");
        let mut evidence = AutoRoutingEvidence {
            schema_version: 1,
            candidate_sha: "1".repeat(40),
            backend: AutoRoutingBackend::Metal,
            platform: AutoRoutingPlatform {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                hardware: "Apple M fixture".to_string(),
                driver: "fixture driver".to_string(),
            },
            external_manifest_sha256: "2".repeat(64),
            external_case_count: 2,
            cells: vec![auto_routing_route_cell(
                "decode-case",
                AutoRoutingOperation::FullDecode,
                "auto-routing_full-decode_decode-case",
                "3".repeat(64),
            )],
        };

        write_auto_routing_evidence(&output, &evidence).unwrap();
        let first = fs::read(&output).unwrap();
        write_auto_routing_evidence(&output, &evidence).unwrap();
        assert_eq!(fs::read(&output).unwrap(), first);
        assert!(first.ends_with(b"\n"));

        evidence.cells[0].hybrid.execution = AutoRoutingExecution::Cpu;
        assert!(write_auto_routing_evidence(&output, &evidence)
            .unwrap_err()
            .contains("hybrid route"));
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "j2k-test-support-auto-routing-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }
}
