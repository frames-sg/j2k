use std::{fs, path::Path, time::SystemTime};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::{super::verify_evidence, OPERATIONS};

pub(super) struct Fixture {
    pub(super) root: std::path::PathBuf,
    pub(super) evidence: std::path::PathBuf,
    pub(super) manifest: std::path::PathBuf,
    pub(super) criterion: std::path::PathBuf,
}

impl Fixture {
    pub(super) fn verify(&self) -> Result<super::super::VerifiedEvidence, String> {
        verify_evidence(&self.evidence, &self.manifest, &self.criterion)
    }
}

pub(super) fn write_fixture(label: &str, overlap: bool, slow_hybrid: bool) -> Fixture {
    let root = temp_dir(label);
    let criterion = root.join("criterion");
    let evidence_path = root.join("evidence.json");
    let manifest_path = root.join("external-manifest.json");
    let manifest = json!({
        "schema_version": 1,
        "corpus": "external-test-corpus",
        "source_url": "https://example.invalid/j2k-routing-corpus",
        "cases": [
            {
                "id": "decode-case",
                "path": "decode/decode-case.j2k",
                "kind": "decode",
                "pixel_format": "rgb8",
                "sha256": "d".repeat(64)
            },
            {
                "id": "encode-case",
                "path": "encode/encode-case.ppm",
                "kind": "encode",
                "pixel_format": "rgb8",
                "sha256": "e".repeat(64)
            }
        ]
    });
    let manifest_bytes = format!(
        "{}\n",
        serde_json::to_string_pretty(&manifest).expect("serialize manifest fixture")
    )
    .into_bytes();
    fs::create_dir_all(&root).expect("create fixture root");
    fs::write(&manifest_path, &manifest_bytes).expect("write manifest fixture");
    let manifest_sha256 = format!("{:x}", Sha256::digest(&manifest_bytes));

    let mut cells = Vec::new();
    for operation in OPERATIONS {
        let workload = if operation.ends_with("encode") {
            "encode-case"
        } else {
            "decode-case"
        };
        let base = format!("auto-routing/{operation}/{workload}");
        let cpu_id = format!("{base}/cpu");
        let hybrid_id = format!("{base}/hybrid");
        let strict_id = format!("{base}/strict-device");
        write_estimate(&criterion, &cpu_id, 100.0, 98.0, 102.0);
        let (median, lower, upper) = if slow_hybrid {
            (91.0, 89.0, 93.0)
        } else if overlap {
            (80.0, 78.0, 99.0)
        } else {
            (80.0, 78.0, 82.0)
        };
        write_estimate(&criterion, &hybrid_id, median, lower, upper);
        write_estimate(&criterion, &strict_id, 110.0, 108.0, 112.0);
        cells.push(json!({
            "id": format!("{operation}-{workload}"),
            "operation": operation,
            "source": "external",
            "workload": workload,
            "cpu": route(&cpu_id, "cpu"),
            "hybrid": route(&hybrid_id, "hybrid"),
            "strict_device_supported": true,
            "strict_device": route(&strict_id, "device-native")
        }));
    }
    let evidence = json!({
        "schema_version": 1,
        "candidate_sha": "a".repeat(40),
        "backend": "metal",
        "platform": {
            "os": "macos",
            "arch": "aarch64",
            "hardware": "Apple M4 Pro",
            "driver": "macOS Metal"
        },
        "external_manifest_sha256": manifest_sha256,
        "external_case_count": 2,
        "cells": cells
    });
    write_json(&evidence_path, &evidence);
    Fixture {
        root,
        evidence: evidence_path,
        manifest: manifest_path,
        criterion,
    }
}

pub(super) fn upgrade_fixture_to_part15(fixture: &Fixture) {
    let mut manifest = read_json(&fixture.manifest);
    manifest["schema_version"] = json!(2);
    for case in manifest["cases"].as_array_mut().expect("manifest cases") {
        case["codec"] = json!("htj2k-part-15");
        case["container"] = json!("codestream");
    }
    manifest["cases"]
        .as_array_mut()
        .expect("manifest cases")
        .push(json!({
            "id": "jph-case",
            "path": "decode/jph-case.jph",
            "kind": "decode",
            "codec": "htj2k-part-15",
            "container": "jph",
            "pixel_format": "rgb8",
            "sha256": "f".repeat(64)
        }));
    write_json(&fixture.manifest, &manifest);

    let mut evidence = read_json(&fixture.evidence);
    evidence["schema_version"] = json!(2);
    evidence["external_case_count"] = json!(3);
    let cells = evidence["cells"].as_array_mut().expect("evidence cells");
    for operation in &OPERATIONS[..4] {
        let base = format!("auto-routing/{operation}/jph-case");
        let cpu_id = format!("{base}/cpu");
        let hybrid_id = format!("{base}/hybrid");
        let strict_id = format!("{base}/strict-device");
        write_estimate(&fixture.criterion, &cpu_id, 100.0, 98.0, 102.0);
        write_estimate(&fixture.criterion, &hybrid_id, 80.0, 78.0, 82.0);
        write_estimate(&fixture.criterion, &strict_id, 110.0, 108.0, 112.0);
        cells.push(json!({
            "id": format!("{operation}-jph-case"),
            "operation": operation,
            "source": "external",
            "workload": "jph-case",
            "cpu": route(&cpu_id, "cpu"),
            "hybrid": route(&hybrid_id, "hybrid"),
            "strict_device_supported": true,
            "strict_device": route(&strict_id, "device-native")
        }));
    }
    write_json(&fixture.evidence, &evidence);
    rewrite_manifest_hash(fixture);
}

pub(super) fn rewrite_manifest_hash(fixture: &Fixture) {
    let manifest_bytes = fs::read(&fixture.manifest).expect("read altered manifest");
    let mut evidence = read_json(&fixture.evidence);
    evidence["external_manifest_sha256"] = json!(format!("{:x}", Sha256::digest(&manifest_bytes)));
    write_json(&fixture.evidence, &evidence);
}

fn route(criterion_id: &str, execution: &str) -> Value {
    json!({
        "criterion_id": criterion_id,
        "execution": execution,
        "output_sha256": "c".repeat(64)
    })
}

pub(super) fn write_estimate(root: &Path, id: &str, median: f64, lower: f64, upper: f64) {
    let path = root.join(id).join("new/estimates.json");
    fs::create_dir_all(path.parent().expect("estimate parent")).expect("create estimate path");
    let estimate = json!({
        "median": {
            "confidence_interval": {
                "confidence_level": 0.95,
                "lower_bound": lower,
                "upper_bound": upper
            },
            "point_estimate": median,
            "standard_error": 1.0
        }
    });
    write_json(&path, &estimate);
}

pub(super) fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("read JSON fixture")).expect("parse JSON fixture")
}

pub(super) fn write_json(path: &Path, value: &Value) {
    fs::write(
        path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(value).expect("serialize JSON fixture")
        ),
    )
    .expect("write JSON fixture");
}

fn temp_dir(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "j2k-auto-routing-{label}-{}-{nonce}",
        std::process::id()
    ))
}
