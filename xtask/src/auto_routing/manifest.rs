// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use sha2::{Digest, Sha256};

use super::{
    is_lower_hex, is_safe_case_id, is_safe_relative_path, read_bounded_regular_file,
    schema::{
        Codec, Container, Evidence, ExternalCase, ExternalManifest, Operation, ValidatedManifest,
        WorkloadIdentity, WorkloadKind,
    },
    MAX_CELLS, MAX_EVIDENCE_BYTES, REQUIRED_OPERATIONS,
};

pub(super) fn validate_external_manifest(
    path: &Path,
    evidence: &Evidence,
) -> Result<ValidatedManifest, String> {
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
    if !matches!(manifest.schema_version, 1 | 2)
        || manifest.schema_version != evidence.schema_version
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
    let schema_version = manifest.schema_version;
    let mut cases = BTreeMap::new();
    for case in manifest.cases {
        let identity = workload_identity(&case, schema_version)?;
        if !is_safe_case_id(&case.id)
            || !is_safe_relative_path(&case.path)
            || !is_lower_hex(&case.sha256, 64)
            || cases.insert(case.id, identity).is_some()
        {
            return Err("external manifest cases must have unique ids, safe relative paths, typed pixel formats, and lowercase SHA-256 hashes".to_string());
        }
        let _ = case.pixel_format;
    }
    Ok(ValidatedManifest {
        schema_version,
        cases,
    })
}

fn workload_identity(case: &ExternalCase, schema_version: u32) -> Result<WorkloadIdentity, String> {
    let (codec, container) = match (schema_version, case.codec, case.container) {
        (1, None, None) => (Codec::Jpeg2000Part1, Container::Codestream),
        (2, Some(codec), Some(container)) => (codec, container),
        _ => {
            return Err(format!(
                "external manifest case {} must use codec and container fields required by its schema version",
                case.id
            ))
        }
    };
    if !matches!(
        (codec, container),
        (Codec::Jpeg2000Part1, Container::Codestream | Container::Jp2)
            | (Codec::Htj2kPart15, Container::Codestream | Container::Jph)
    ) {
        return Err(format!(
            "external manifest case {} has an invalid codec/container pair",
            case.id
        ));
    }
    Ok(WorkloadIdentity {
        kind: case.kind,
        codec,
        container,
    })
}

pub(super) fn validate_workload_coverage(
    manifest: &ValidatedManifest,
    operations: &BTreeSet<Operation>,
    workload_operations: &BTreeMap<&str, BTreeSet<Operation>>,
) -> Result<(), String> {
    let required = REQUIRED_OPERATIONS.into_iter().collect::<BTreeSet<_>>();
    if operations != &required {
        return Err("Auto-routing evidence must cover full, ROI, scaled, batch, lossless encode, and lossy encode workloads".to_string());
    }
    if manifest.schema_version == 1 {
        return Ok(());
    }
    let required_formats = [
        WorkloadIdentity {
            kind: WorkloadKind::Decode,
            codec: Codec::Htj2kPart15,
            container: Container::Codestream,
        },
        WorkloadIdentity {
            kind: WorkloadKind::Decode,
            codec: Codec::Htj2kPart15,
            container: Container::Jph,
        },
        WorkloadIdentity {
            kind: WorkloadKind::Encode,
            codec: Codec::Htj2kPart15,
            container: Container::Codestream,
        },
    ];
    if required_formats
        .iter()
        .any(|required| !manifest.cases.values().any(|identity| identity == required))
    {
        return Err(
            "schema-v2 Auto-routing evidence requires HTJ2K codestream, JPH, and HTJ2K encode workloads"
                .to_string(),
        );
    }
    for (workload, identity) in &manifest.cases {
        let expected = match identity.kind {
            WorkloadKind::Decode => REQUIRED_OPERATIONS[..4].iter().copied().collect(),
            WorkloadKind::Encode => REQUIRED_OPERATIONS[4..].iter().copied().collect(),
        };
        if workload_operations.get(workload.as_str()) != Some(&expected) {
            return Err(format!(
                "schema-v2 Auto-routing workload {workload} must cover every required operation"
            ));
        }
    }
    Ok(())
}
