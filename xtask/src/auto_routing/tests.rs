use std::{fs, path::Path, time::SystemTime};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::{auto_routing, verify_evidence};

const OPERATIONS: [&str; 6] = [
    "full-decode",
    "roi-decode",
    "scaled-decode",
    "batch-decode",
    "lossless-encode",
    "lossy-encode",
];

#[test]
fn qualifying_external_hybrid_cells_pass() {
    let fixture = write_fixture("qualifying", false, false);

    let verified = fixture.verify().expect("qualifying evidence");

    assert_eq!(verified.cell_count, OPERATIONS.len());
    assert_eq!(verified.artifact_sha256.len(), 64);
    let report: Value = serde_json::from_str(&verified.report_json).expect("verified report JSON");
    assert_eq!(report["status"], "pass");
    assert_eq!(report["promoted_cell_count"], OPERATIONS.len());
    assert_eq!(
        report["cells"].as_array().map(Vec::len),
        Some(OPERATIONS.len())
    );
}

#[test]
fn overlapping_intervals_or_less_than_ten_percent_retain_the_current_route() {
    for (label, overlap, slow_hybrid) in [
        ("overlap", true, false),
        ("insufficient-speedup", false, true),
    ] {
        let fixture = write_fixture(label, overlap, slow_hybrid);

        let verified = fixture.verify().expect("valid nonqualifying evidence");
        let report: Value =
            serde_json::from_str(&verified.report_json).expect("verified report JSON");
        assert_eq!(report["promoted_cell_count"], 0, "{label}");
    }
}

#[test]
fn a_nonqualifying_measurement_can_retain_the_current_route() {
    let fixture = write_fixture("retained-current", false, true);

    let verified = fixture.verify().expect("retained routing evidence");
    let report: Value = serde_json::from_str(&verified.report_json).expect("report JSON");

    assert_eq!(report["promoted_cell_count"], 0);
    assert!(report["cells"]
        .as_array()
        .expect("report cells")
        .iter()
        .all(|cell| cell["decision"] == "retain-current"));
}

#[test]
fn routing_decisions_are_derived_from_measurements() {
    let fixture = write_fixture("derived-decision", false, false);
    let verified = fixture.verify().expect("qualifying routing evidence");
    let report: Value = serde_json::from_str(&verified.report_json).expect("report JSON");
    assert_eq!(report["promoted_cell_count"], OPERATIONS.len());

    let mut evidence = read_json(&fixture.evidence);
    evidence["cells"][0]["decision"] = json!("retain-current");
    write_json(&fixture.evidence, &evidence);

    let error = fixture
        .verify()
        .expect_err("raw evidence cannot override the measured decision");

    assert!(error.contains("unknown field `decision`"), "{error}");
}

#[test]
fn altered_external_manifest_fails_closed() {
    let fixture = write_fixture("altered-manifest", false, false);
    fs::write(&fixture.manifest, b"{}\n").expect("alter manifest");

    let error = fixture.verify().expect_err("must fail closed");

    assert!(error.contains("external manifest SHA-256"), "{error}");
}

#[test]
fn external_manifest_requires_safe_typed_workload_paths() {
    let fixture = write_fixture("unsafe-manifest-path", false, false);
    let mut manifest = read_json(&fixture.manifest);
    manifest["cases"][0]["path"] = json!("../outside.j2k");
    write_json(&fixture.manifest, &manifest);
    rewrite_manifest_hash(&fixture);

    let error = fixture
        .verify()
        .expect_err("unsafe workload path must fail");

    assert!(error.contains("safe relative paths"), "{error}");
}

#[test]
fn routes_require_identical_outputs_and_honest_execution_labels() {
    for (label, field, value, expected) in [
        (
            "output-mismatch",
            "output_sha256",
            Value::String("e".repeat(64)),
            "identical outputs",
        ),
        (
            "execution-mismatch",
            "execution",
            Value::String("cpu".to_string()),
            "invalid or duplicate hybrid route evidence",
        ),
    ] {
        let fixture = write_fixture(label, false, false);
        let mut evidence = read_json(&fixture.evidence);
        evidence["cells"][0]["hybrid"][field] = value;
        write_json(&fixture.evidence, &evidence);

        let error = fixture.verify().expect_err("must fail closed");

        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn every_workload_class_and_exact_confidence_level_are_required() {
    let missing = write_fixture("missing-operation", false, false);
    let mut evidence = read_json(&missing.evidence);
    evidence["cells"].as_array_mut().expect("cells").pop();
    write_json(&missing.evidence, &evidence);
    let error = missing.verify().expect_err("missing operation must fail");
    assert!(
        error.contains("must cover full, ROI, scaled, batch"),
        "{error}"
    );

    let confidence = write_fixture("wrong-confidence", false, false);
    let estimate_path = confidence
        .criterion
        .join("auto-routing/full-decode/decode-case/cpu/new/estimates.json");
    let mut estimate = read_json(&estimate_path);
    estimate["median"]["confidence_interval"]["confidence_level"] = json!(0.90);
    write_json(&estimate_path, &estimate);
    let error = confidence.verify().expect_err("wrong confidence must fail");
    assert!(
        error.contains("does not use a 95% confidence interval"),
        "{error}"
    );
}

#[test]
fn criterion_ids_cannot_escape_the_artifact_root() {
    let fixture = write_fixture("unsafe-path", false, false);
    let mut evidence = read_json(&fixture.evidence);
    evidence["cells"][0]["cpu"]["criterion_id"] = json!("../outside");
    write_json(&fixture.evidence, &evidence);

    let error = fixture.verify().expect_err("unsafe path must fail");

    assert!(
        error.contains("invalid or duplicate CPU route evidence"),
        "{error}"
    );
}

#[test]
fn unsupported_strict_device_cells_are_allowed_but_must_be_explicit() {
    let fixture = write_fixture("unsupported-strict-device", false, false);
    let mut evidence = read_json(&fixture.evidence);
    evidence["cells"][0]["strict_device_supported"] = json!(false);
    evidence["cells"][0]["strict_device"] = Value::Null;
    write_json(&fixture.evidence, &evidence);

    fixture
        .verify()
        .expect("explicit unsupported strict-device route");
}

#[test]
fn artifact_hash_covers_exact_criterion_estimates() {
    let fixture = write_fixture("artifact-hash", false, false);
    let before = fixture.verify().expect("initial evidence").artifact_sha256;
    let estimate_path = fixture
        .criterion
        .join("auto-routing/full-decode/decode-case/cpu/new/estimates.json");
    let mut estimate = read_json(&estimate_path);
    estimate["median"]["standard_error"] = json!(2.0);
    write_json(&estimate_path, &estimate);

    let after = fixture
        .verify()
        .expect("altered but valid evidence")
        .artifact_sha256;

    assert_ne!(before, after);
}

#[test]
fn command_writes_the_verified_report_and_requires_an_output() {
    let fixture = write_fixture("command", false, false);
    let output = fixture.root.join("report/verified.json");
    let args = vec![
        "verify".to_string(),
        "--evidence".to_string(),
        fixture.evidence.display().to_string(),
        "--external-manifest".to_string(),
        fixture.manifest.display().to_string(),
        "--criterion-root".to_string(),
        fixture.criterion.display().to_string(),
        "--out".to_string(),
        output.display().to_string(),
    ];

    auto_routing(args.into_iter()).expect("verify command");

    let report = read_json(&output);
    assert_eq!(report["status"], "pass");
    let error = auto_routing(
        [
            "verify",
            "--evidence",
            fixture.evidence.to_str().expect("UTF-8 path"),
        ]
        .into_iter()
        .map(str::to_string),
    )
    .expect_err("missing output must fail");
    assert!(error.contains("--out FILE"), "{error}");
}

struct Fixture {
    root: std::path::PathBuf,
    evidence: std::path::PathBuf,
    manifest: std::path::PathBuf,
    criterion: std::path::PathBuf,
}

impl Fixture {
    fn verify(&self) -> Result<super::VerifiedEvidence, String> {
        verify_evidence(&self.evidence, &self.manifest, &self.criterion)
    }
}

fn write_fixture(label: &str, overlap: bool, slow_hybrid: bool) -> Fixture {
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

fn rewrite_manifest_hash(fixture: &Fixture) {
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

fn write_estimate(root: &Path, id: &str, median: f64, lower: f64, upper: f64) {
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

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("read JSON fixture")).expect("parse JSON fixture")
}

fn write_json(path: &Path, value: &Value) {
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
