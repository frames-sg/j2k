// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use crate::perf_guard::BenchEstimate;

use super::schema::{
    Codec, Container, Evidence, ValidatedManifest, WorkloadIdentity, WorkloadKind,
};

pub(super) fn verification_report(
    evidence: &Evidence,
    manifest: &ValidatedManifest,
    estimates: &BTreeMap<String, BenchEstimate>,
    artifact_sha256: &str,
) -> Result<String, String> {
    let cells = evidence
        .cells
        .iter()
        .map(|cell| {
            let identity = &manifest.cases[&cell.workload];
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
                "codec": identity.codec,
                "container": identity.container,
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
    let htj2k_codestream_count = workload_count(
        manifest,
        WorkloadIdentity {
            kind: WorkloadKind::Decode,
            codec: Codec::Htj2kPart15,
            container: Container::Codestream,
        },
    );
    let jph_count = workload_count(
        manifest,
        WorkloadIdentity {
            kind: WorkloadKind::Decode,
            codec: Codec::Htj2kPart15,
            container: Container::Jph,
        },
    );
    let htj2k_encode_count = workload_count(
        manifest,
        WorkloadIdentity {
            kind: WorkloadKind::Encode,
            codec: Codec::Htj2kPart15,
            container: Container::Codestream,
        },
    );
    let report = serde_json::json!({
        "schema_version": evidence.schema_version,
        "candidate_sha": evidence.candidate_sha,
        "backend": evidence.backend,
        "platform": evidence.platform,
        "external_manifest_sha256": evidence.external_manifest_sha256,
        "external_case_count": evidence.external_case_count,
        "criterion_confidence_level": 0.95,
        "minimum_hybrid_speedup_percent": 10.0,
        "artifact_sha256": artifact_sha256,
        "promoted_cell_count": promoted_cell_count,
        "workload_formats": {
            "htj2k-codestream": htj2k_codestream_count,
            "jph": jph_count,
            "htj2k-encode": htj2k_encode_count,
        },
        "cells": cells,
        "status": "pass",
    });
    let mut json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize Auto-routing verification report: {error}"))?;
    json.push('\n');
    Ok(json)
}

fn workload_count(manifest: &ValidatedManifest, expected: WorkloadIdentity) -> usize {
    manifest
        .cases
        .values()
        .filter(|identity| **identity == expected)
        .count()
}

fn is_qualifying_win(hybrid: &BenchEstimate, competitor: &BenchEstimate) -> bool {
    hybrid.median_ns <= competitor.median_ns * 0.9
        && hybrid.median_upper_ns < competitor.median_lower_ns
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
