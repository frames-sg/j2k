// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::{
    command_output, command_output_allow_failure, command_output_os_detailed_with_env,
    passed_test_count, run_cargo_test_with_pass_floor, run_program,
    run_program_in_dir_owned_with_program, rust_sources, use_test_cargo_program,
};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn temp_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "j2k-command-support-{label}-{}-{}",
        std::process::id(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).expect("create command-support test directory");
    path
}

fn executable(path: &Path, source: &str) {
    fs::write(path, source).expect("write test executable");
    let mut permissions = fs::metadata(path)
        .expect("test executable metadata")
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(path, permissions).expect("make test executable runnable");
}

#[test]
fn detailed_output_preserves_success_failure_environment_and_non_utf8_errors() {
    let root = temp_dir("detailed-output");
    let program = root.join("output.sh");
    executable(
        &program,
        r#"#!/bin/sh
printf 'stdout:%s:%s\n' "$1" "${VISIBLE-unset}"
printf 'stderr:%s\n' "$1" >&2
if [ "${NON_UTF8-0}" = 1 ]; then printf '\377'; fi
exit "${EXIT_CODE-0}"
"#,
    );

    let success = command_output_os_detailed_with_env(
        program.clone().into_os_string(),
        &["argument"],
        &[("VISIBLE", "yes")],
    )
    .expect("successful detailed output");
    assert_eq!(success, "stdout:argument:yes");

    let error = command_output_os_detailed_with_env(
        program.clone().into_os_string(),
        &["failure"],
        &[("EXIT_CODE", "9")],
    )
    .expect_err("nonzero detailed output");
    assert!(error.contains("exit status: 9"));
    assert!(error.contains("stdout:failure:unset"));
    assert!(error.contains("stderr:failure"));

    let error =
        command_output_os_detailed_with_env(program.into_os_string(), &[], &[("NON_UTF8", "1")])
            .expect_err("non-UTF-8 output");
    assert!(error.contains("emitted non-UTF-8 stdout"));
}

#[test]
fn generic_program_helpers_preserve_args_env_cwd_and_failure_status() {
    let root = temp_dir("programs");
    let log = root.join("command.log");
    let program = root.join("record.sh");
    executable(
        &program,
        &format!(
            "#!/bin/sh\nprintf 'cwd=%s\\narg=%s\\nenv=%s\\n' \"$PWD\" \"$1\" \"${{VISIBLE-unset}}\" >> '{}'\nexit \"${{EXIT_CODE-0}}\"\n",
            log.display()
        ),
    );

    run_program(
        program.clone().into_os_string(),
        &["direct"],
        &[("VISIBLE", "direct-env")],
    )
    .expect("direct program");
    run_program_in_dir_owned_with_program(
        program.clone().into_os_string(),
        root.to_str().expect("UTF-8 root"),
        &["owned".to_string()],
        &[("VISIBLE", "owned-env")],
    )
    .expect("owned program");
    let error = run_program(program.into_os_string(), &[], &[("EXIT_CODE", "4")])
        .expect_err("nonzero program");
    assert!(error.contains("exit status: 4"));

    let log = fs::read_to_string(log).expect("command log");
    assert!(log.contains("arg=direct\nenv=direct-env"));
    let canonical_root = fs::canonicalize(&root).expect("canonical root");
    assert!(log.contains(&format!(
        "cwd={}\narg=owned\nenv=owned-env",
        canonical_root.display()
    )));
}

#[test]
fn cargo_test_pass_floor_counts_both_streams_and_rejects_failure_or_shortfall() {
    let root = temp_dir("pass-floor");
    let program = root.join("cargo.sh");
    executable(
        &program,
        r#"#!/bin/sh
printf 'test result: ok. %s passed; 0 failed;\n' "${STDOUT_PASSED-0}"
printf 'test result: ok. %s passed; 0 failed;\n' "${STDERR_PASSED-0}" >&2
exit "${EXIT_CODE-0}"
"#,
    );
    let _cargo = use_test_cargo_program(program.into_os_string());

    run_cargo_test_with_pass_floor(
        &["test", "-p", "fixture"],
        &[("STDOUT_PASSED", "2"), ("STDERR_PASSED", "3")],
        5,
        "fixture parity",
    )
    .expect("combined pass floor");
    let error =
        run_cargo_test_with_pass_floor(&["test"], &[("STDOUT_PASSED", "1")], 2, "fixture parity")
            .expect_err("pass floor shortfall");
    assert!(error.contains("executed 1 tests, expected at least 2"));
    let error =
        run_cargo_test_with_pass_floor(&["test"], &[("EXIT_CODE", "6")], 0, "fixture parity")
            .expect_err("cargo test failure");
    assert!(error.contains("exit status: 6"));
}

#[test]
fn output_helpers_and_test_summary_parser_keep_distinct_failure_contracts() {
    let root = temp_dir("output-helpers");
    let program = root.join("output.sh");
    executable(
        &program,
        r#"#!/bin/sh
printf 'stdout text\n'
printf 'stderr text\n' >&2
exit "${EXIT_CODE-0}"
"#,
    );
    let program = program.to_str().expect("UTF-8 program");
    assert_eq!(command_output(program, &[]), Ok("stdout text".to_string()));
    assert_eq!(
        command_output_allow_failure(program, &[]),
        Ok("stdout text\nstderr text".to_string())
    );
    assert_eq!(
        passed_test_count(
            "noise\ntest result: ok. 2 passed; 0 failed\ntest result: FAILED. 9 passed\ntest result: ok. nope passed\ntest result: ok. 3 passed"
        ),
        5
    );
}

#[test]
fn recursive_rust_source_inventory_includes_only_rs_files_and_reports_bad_roots() {
    let root = temp_dir("rust-sources");
    let nested = root.join("nested/deeper");
    fs::create_dir_all(&nested).expect("create source tree");
    fs::write(root.join("root.rs"), "fn root() {}\n").expect("root source");
    fs::write(nested.join("child.rs"), "fn child() {}\n").expect("child source");
    fs::write(nested.join("note.txt"), "not Rust\n").expect("non-Rust file");

    let mut sources = rust_sources(&root).expect("Rust source inventory");
    sources.sort();
    assert_eq!(sources, [nested.join("child.rs"), root.join("root.rs")]);
    let error = rust_sources(&root.join("missing")).expect_err("missing source root");
    assert!(error.contains("failed to read"));
}
