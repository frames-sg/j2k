// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::{
    jscpd_args, stage_production_sources, stage_test_sources, validate_clone_config,
    validate_jscpd_report, validate_test_clone_config, JSCPD_PACKAGE,
};

const INLINE_TEST_FIXTURE: &str = include_str!("../../tests/fixtures/clone_audit/inline_test_a.rs");

#[test]
fn repository_clone_config_is_pinned_and_source_staging_owned() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest parent");
    validate_clone_config(&root.join(".jscpd.json")).expect("valid repository clone config");
    validate_test_clone_config(&root.join(".jscpd-tests.json"))
        .expect("valid repository test clone config");
}

#[test]
fn clone_config_rejects_additional_scope_exclusions() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest parent");
    let source = fs::read_to_string(root.join(".jscpd.json")).expect("read clone config");
    let mut config =
        serde_json::from_str::<serde_json::Value>(&source).expect("parse clone config");
    config["ignore"]
        .as_array_mut()
        .expect("ignore array")
        .push(serde_json::Value::String("**/*.rs".to_string()));
    let temp = temp_dir("config-drift");
    fs::create_dir_all(&temp).expect("create config directory");
    let path = temp.join(".jscpd.json");
    fs::write(
        &path,
        serde_json::to_vec(&config).expect("serialize config"),
    )
    .expect("write changed clone config");

    let error = validate_clone_config(&path).expect_err("extra exclusion must fail closed");
    assert!(error.contains("ignore"));

    fs::remove_dir_all(temp).expect("remove config test directory");
}

#[test]
fn staged_clone_sources_preserve_paths_and_remove_inline_tests() {
    let temp = temp_dir("stage");
    let root = temp.join("repo");
    let source_path = root.join("crates/fixture/src/lib.rs");
    fs::create_dir_all(source_path.parent().expect("source parent")).expect("create source parent");
    fs::write(&source_path, INLINE_TEST_FIXTURE).expect("write source fixture");
    let stage = temp.join("stage");

    let summary = stage_production_sources(&root, &stage).expect("stage production source");
    let staged_path = stage.join("crates/fixture/src/lib.rs");
    let staged = fs::read_to_string(staged_path).expect("read staged source");

    assert_eq!(summary.files, 1);
    assert!(summary.masked_nodes > 0);
    assert!(staged.contains("alpha_production_value"));
    assert!(!staged.contains("repeated_test_clone"));
    assert_eq!(
        staged.bytes().filter(|byte| *byte == b'\n').count(),
        INLINE_TEST_FIXTURE
            .bytes()
            .filter(|byte| *byte == b'\n')
            .count()
    );

    fs::remove_dir_all(temp).expect("remove clone-audit test directory");
}

#[test]
fn test_clone_stage_includes_inline_and_physical_test_support() {
    let temp = temp_dir("test-stage");
    let root = temp.join("repo");
    let source_path = root.join("crates/fixture/src/lib.rs");
    let test_path = root.join("crates/fixture/tests/integration.rs");
    fs::create_dir_all(source_path.parent().expect("source parent")).expect("create source parent");
    fs::create_dir_all(test_path.parent().expect("test parent")).expect("create test parent");
    fs::write(&source_path, INLINE_TEST_FIXTURE).expect("write inline fixture");
    fs::write(
        &test_path,
        "#[test]\nfn physical_test() { assert_eq!(1, 1); }\n",
    )
    .expect("write physical fixture");
    let stage = temp.join("stage");

    let summary = stage_test_sources(&root, &stage).expect("stage test sources");
    let inline = fs::read_to_string(stage.join("crates/fixture/src/lib.rs"))
        .expect("read staged inline tests");
    let physical = fs::read_to_string(stage.join("crates/fixture/tests/integration.rs"))
        .expect("read staged physical test");

    assert_eq!(summary.files, 2);
    assert!(inline.contains("repeated_test_clone"));
    assert!(!inline.contains("alpha_production_value"));
    assert!(physical.contains("physical_test"));

    fs::remove_dir_all(temp).expect("remove test clone-audit directory");
}

#[test]
fn test_clone_stage_rejects_a_stage_root_that_is_not_a_directory() {
    let temp = temp_dir("test-stage-parent-error");
    let root = temp.join("repo");
    let source_path = root.join("crates/fixture/src/lib.rs");
    fs::create_dir_all(source_path.parent().expect("source parent")).expect("create source parent");
    fs::write(&source_path, INLINE_TEST_FIXTURE).expect("write inline fixture");
    let stage = temp.join("stage");
    fs::write(&stage, "not a directory").expect("write blocking stage file");

    let error = stage_test_sources(&root, &stage).expect_err("stage parent must fail closed");
    assert!(error.contains("create test clone-audit stage"));

    fs::remove_dir_all(temp).expect("remove test clone-audit directory");
}

#[test]
fn test_clone_stage_reports_staged_source_write_failures() {
    let temp = temp_dir("test-stage-write-error");
    let root = temp.join("repo");
    let relative = Path::new("crates/fixture/src/lib.rs");
    let source_path = root.join(relative);
    fs::create_dir_all(source_path.parent().expect("source parent")).expect("create source parent");
    fs::write(&source_path, INLINE_TEST_FIXTURE).expect("write inline fixture");
    let stage = temp.join("stage");
    fs::create_dir_all(stage.join(relative)).expect("create blocking staged directory");

    let error = stage_test_sources(&root, &stage).expect_err("source write must fail closed");
    assert!(error.contains("write staged test clone-audit source"));

    fs::remove_dir_all(temp).expect("remove test clone-audit directory");
}

#[test]
fn test_clone_stage_rejects_a_repository_without_test_sources() {
    let temp = temp_dir("test-stage-empty");
    let root = temp.join("repo");
    let source_path = root.join("crates/fixture/src/lib.rs");
    fs::create_dir_all(source_path.parent().expect("source parent")).expect("create source parent");
    fs::write(&source_path, "pub fn production_only() {}\n").expect("write production fixture");

    let error = stage_test_sources(&root, &temp.join("stage"))
        .expect_err("empty test stage must fail closed");
    assert!(error.contains("found no eligible sources"));

    fs::remove_dir_all(temp).expect("remove test clone-audit directory");
}

#[test]
fn jscpd_invocation_pins_package_config_output_and_silent_mode() {
    let arguments = jscpd_args(
        Path::new("/tmp/stage"),
        Path::new("/repo/.jscpd.json"),
        Path::new("/repo/target/clone-audit/report"),
    )
    .expect("UTF-8 paths");

    assert_eq!(arguments[0], "--yes");
    assert_eq!(arguments[1], JSCPD_PACKAGE);
    assert!(arguments
        .windows(2)
        .any(|pair| { pair == ["--config".to_string(), "/repo/.jscpd.json".to_string(),] }));
    assert!(arguments.windows(2).any(|pair| {
        pair == [
            "--output".to_string(),
            "/repo/target/clone-audit/report".to_string(),
        ]
    }));
    assert_eq!(arguments.last().map(String::as_str), Some("--silent"));
}

#[test]
fn jscpd_report_validation_fails_closed_on_incomplete_or_over_threshold_totals() {
    let temp = temp_dir("report");
    fs::create_dir_all(&temp).expect("create report directory");
    let valid = temp.join("valid.json");
    fs::write(
        &valid,
        r#"{"statistics":{"total":{"lines":10,"tokens":100,"sources":1,"clones":0,"duplicatedLines":0,"duplicatedTokens":0,"percentage":0,"percentageTokens":0}}}"#,
    )
    .expect("write valid report");
    validate_jscpd_report(&valid, 3.34).expect("valid report");

    let invalid = temp.join("invalid.json");
    fs::write(&invalid, r#"{"statistics":{"total":{"lines":1}}}"#).expect("write invalid report");
    let error = validate_jscpd_report(&invalid, 3.34).expect_err("incomplete report must fail");
    assert!(error.contains("tokens"));

    let threshold = temp.join("threshold.json");
    fs::write(
        &threshold,
        r#"{"statistics":{"total":{"lines":100,"tokens":1000,"sources":2,"clones":1,"duplicatedLines":4,"duplicatedTokens":40,"percentage":3.34,"percentageTokens":4}}}"#,
    )
    .expect("write threshold report");
    let error =
        validate_jscpd_report(&threshold, 3.34).expect_err("threshold report must fail closed");
    assert!(error.contains("meets or exceeds"));

    fs::remove_dir_all(temp).expect("remove clone-audit report directory");
}

fn temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "j2k-clone-audit-{label}-{}-{nonce}",
        std::process::id()
    ))
}
