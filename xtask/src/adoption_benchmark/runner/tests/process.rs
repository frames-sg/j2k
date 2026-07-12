// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{ffi::OsString, fs};

use super::super::{display_command, run_logged, run_logged_owned, skipped_step};
use super::support::{read, recording_program, temp_dir};
use crate::adoption_benchmark::summary::StepStatus;

#[test]
fn logged_process_records_command_output_environment_and_criterion_root() {
    let root = temp_dir("logged-success");
    let out_dir = root.join("out");
    let target_dir = root.join("target");
    fs::create_dir_all(&out_dir).expect("create output directory");
    let program = recording_program(&root);
    let args = ["first".to_string(), "second value".to_string()];
    let envs = [("CUSTOM".to_string(), "visible".to_string())];

    let step = run_logged_owned(
        "success",
        &program,
        &args,
        &envs,
        Some(&target_dir),
        &out_dir,
    )
    .expect("successful logged process");

    assert!(matches!(step.status, StepStatus::Ran));
    assert_eq!(step.criterion_root, Some(target_dir.join("criterion")));
    assert!(step.command.contains("CUSTOM=visible"));
    assert!(step.command.contains("CARGO_TARGET_DIR="));
    let stdout = read(&step.stdout);
    assert!(stdout.contains("arg=first\narg=second value\n"));
    assert!(stdout.contains("CUSTOM=visible"));
    assert!(stdout.contains(&format!("CARGO_TARGET_DIR={}", target_dir.display())));
    assert_eq!(read(&step.stderr), "stderr-custom=visible\n");
}

#[test]
fn logged_process_failure_preserves_both_output_artifacts() {
    let root = temp_dir("logged-failure");
    let out_dir = root.join("out");
    fs::create_dir_all(&out_dir).expect("create output directory");
    let program = recording_program(&root);
    let envs = [
        ("CUSTOM".to_string(), "failure".to_string()),
        ("EXIT_CODE".to_string(), "7".to_string()),
    ];

    let error = run_logged(
        "failure",
        program.into_os_string(),
        &["argument"],
        &envs,
        &out_dir,
    )
    .expect_err("nonzero process must fail");

    assert!(
        error.contains("exit status: 7"),
        "unexpected error: {error}"
    );
    assert!(error.contains("failure.out"));
    assert!(error.contains("failure.err"));
    assert!(read(&out_dir.join("failure.out")).contains("arg=argument"));
    assert_eq!(
        read(&out_dir.join("failure.err")),
        "stderr-custom=failure\n"
    );
}

#[test]
fn logged_process_start_and_output_creation_failures_are_contextual() {
    let root = temp_dir("logged-errors");
    let out_dir = root.join("out");
    fs::create_dir_all(&out_dir).expect("create output directory");
    let missing = root.join("missing-program");
    let error = run_logged_owned("missing", &missing, &[], &[], None, &out_dir)
        .expect_err("missing executable");
    assert!(error.contains("failed to start"));
    assert!(error.contains("missing-program"));
    assert!(out_dir.join("missing.out").is_file());
    assert!(out_dir.join("missing.err").is_file());

    let absent_out_dir = root.join("absent");
    let error = run_logged_owned(
        "no-output-dir",
        OsString::from("unused"),
        &[],
        &[],
        None,
        &absent_out_dir,
    )
    .expect_err("output creation failure");
    assert!(error.contains("failed to create"));
    assert!(error.contains("no-output-dir.out"));
}

#[test]
fn skipped_step_records_complete_nonexecution_evidence() {
    let root = temp_dir("skipped");
    let step = skipped_step("optional-step", "hardware not requested", &root);

    assert_eq!(step.name, "optional-step");
    assert_eq!(step.command, "not run");
    assert_eq!(step.stdout, root.join("optional-step.out"));
    assert_eq!(step.stderr, root.join("optional-step.err"));
    assert_eq!(step.criterion_root, None);
    assert!(matches!(
        step.status,
        StepStatus::Skipped { ref reason } if reason == "hardware not requested"
    ));
    assert!(!step.stdout.exists());
    assert!(!step.stderr.exists());
}

#[test]
fn displayed_command_orders_scrubs_overrides_target_program_and_arguments() {
    let command = display_command(
        std::ffi::OsStr::new("runner"),
        &["one".to_string(), "two".to_string()],
        &[("CUSTOM".to_string(), "value".to_string())],
        Some(std::path::Path::new("target-dir")),
    );

    let override_index = command.find("CUSTOM=value").expect("override");
    let target_index = command
        .find("CARGO_TARGET_DIR=target-dir")
        .expect("target directory");
    let program_index = command.find(" runner one two").expect("program and args");
    assert!(command.starts_with("env -u J2K_FIXTURE_COMPARE_MODE"));
    assert!(override_index < target_index && target_index < program_index);
}
