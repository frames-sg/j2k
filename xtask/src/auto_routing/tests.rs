use std::fs;

use serde_json::{json, Value};

use super::auto_routing;
use fixture::{
    read_json, rewrite_manifest_hash, upgrade_fixture_to_part15, write_fixture, write_json,
};

mod fixture;

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
fn schema_v2_requires_and_reports_htj2k_codestream_jph_and_encode_workloads() {
    let fixture = write_fixture("part15", false, false);
    upgrade_fixture_to_part15(&fixture);

    let verified = fixture.verify().expect("complete Part 15 route evidence");
    let report: Value = serde_json::from_str(&verified.report_json).expect("verified report JSON");

    assert_eq!(verified.cell_count, 10);
    assert_eq!(report["schema_version"], 2);
    assert_eq!(report["workload_formats"]["htj2k-codestream"], 1);
    assert_eq!(report["workload_formats"]["jph"], 1);
    assert_eq!(report["workload_formats"]["htj2k-encode"], 1);
    assert!(report["cells"]
        .as_array()
        .expect("report cells")
        .iter()
        .any(|cell| cell["codec"] == "htj2k-part-15" && cell["container"] == "jph"));
}

#[test]
fn schema_v2_rejects_missing_jph_and_partial_per_case_operation_coverage() {
    let missing_jph = write_fixture("part15-missing-jph", false, false);
    let mut manifest = read_json(&missing_jph.manifest);
    manifest["schema_version"] = json!(2);
    for case in manifest["cases"].as_array_mut().expect("manifest cases") {
        case["codec"] = json!("htj2k-part-15");
        case["container"] = json!("codestream");
    }
    write_json(&missing_jph.manifest, &manifest);
    let mut evidence = read_json(&missing_jph.evidence);
    evidence["schema_version"] = json!(2);
    write_json(&missing_jph.evidence, &evidence);
    rewrite_manifest_hash(&missing_jph);
    let error = missing_jph
        .verify()
        .expect_err("Part 15 evidence must include JPH");
    assert!(
        error.contains("HTJ2K codestream, JPH, and HTJ2K encode"),
        "{error}"
    );

    let partial = write_fixture("part15-partial", false, false);
    upgrade_fixture_to_part15(&partial);
    let mut evidence = read_json(&partial.evidence);
    evidence["cells"]
        .as_array_mut()
        .expect("evidence cells")
        .retain(|cell| cell["workload"] != "jph-case" || cell["operation"] != "scaled-decode");
    write_json(&partial.evidence, &evidence);
    let error = partial
        .verify()
        .expect_err("every decode case needs every decode operation");
    assert!(error.contains("every required operation"), "{error}");
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
