// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    clippy_suite_args, exact_suite_args, listed_rust_tests, passed_rust_tests,
    reject_cuda_skip_markers, require_cuda_host, runtime_suite_args, successful_test_summaries,
    validate_complete_test_run, validate_cuda_device_probe, validate_exact_named_run,
    CUDA_CLIPPY_SUITES, CUDA_RUNTIME_SUITES, EXACT_CUDA_SUITES, HTJ2K_ENCODE_PARITY_TESTS,
    TRANSCODE_PARITY_TESTS,
};

#[cfg(unix)]
use super::{
    release_cuda, require_cuda_device, run_cargo_captured, run_release_cuda,
    test_support::use_test_nvidia_smi_program, validate_exact_inventory, CUDA_RELEASE_ENV,
};
#[cfg(unix)]
use crate::{process::use_test_cargo_program, test_command::RecordingProgram};

#[test]
fn exact_cuda_inventories_are_unique_and_have_audited_sizes() {
    let ht = HTJ2K_ENCODE_PARITY_TESTS
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let transcode = TRANSCODE_PARITY_TESTS
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(ht.len(), 7);
    assert_eq!(ht.len(), HTJ2K_ENCODE_PARITY_TESTS.len());
    assert_eq!(transcode.len(), 8);
    assert_eq!(transcode.len(), TRANSCODE_PARITY_TESTS.len());
}

#[test]
fn release_commands_name_packages_features_and_non_benchmark_test_targets() {
    for suite in CUDA_RUNTIME_SUITES {
        let args = runtime_suite_args(suite);
        assert!(args.windows(2).any(|pair| pair == ["-p", suite.package]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--features", suite.features]));
        assert!(args.contains(&"--release"));
        assert!(args.contains(&"--lib"));
        assert!(args.contains(&"--bins"));
        assert!(args.contains(&"--tests"));
        assert!(args.contains(&"--examples"));
        assert!(!args.contains(&"--all-targets"));
        assert!(!args.contains(&"--benches"));
    }

    for suite in EXACT_CUDA_SUITES {
        let list_args = exact_suite_args(suite, true);
        let run_args = exact_suite_args(suite, false);
        for target in suite.test_targets {
            assert!(list_args.windows(2).any(|pair| pair == ["--test", target]));
        }
        assert!(list_args.contains(&"--list"));
        assert!(run_args.contains(&"--show-output"));
    }

    for suite in CUDA_CLIPPY_SUITES {
        let args = clippy_suite_args(suite);
        assert!(args.windows(2).any(|pair| pair == ["-p", suite.package]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--features", suite.features]));
        assert!(args.contains(&"--all-targets"));
        assert!(args.ends_with(&["--", "-D", "warnings"]));
    }
}

#[test]
fn cuda_host_and_device_probes_fail_closed() {
    assert!(require_cuda_host("linux", "x86_64").is_ok());
    assert!(require_cuda_host("macos", "aarch64").is_err());
    assert!(validate_cuda_device_probe("0, GPU-012345\n").is_ok());
    assert!(validate_cuda_device_probe("").is_err());
    assert!(validate_cuda_device_probe("No devices were found\n").is_err());
}

#[test]
fn cuda_skip_markers_and_equivalents_are_release_failures() {
    for output in [
        "J2K_GPU_TEST_SKIPPED gate=J2K_REQUIRE_CUDA_RUNTIME",
        "warning: skipping CUDA Oxide build",
        "CUDA runtime is unavailable",
        "reason=device-unavailable",
    ] {
        let error =
            reject_cuda_skip_markers(output, "CUDA").expect_err("every CUDA skip form must fail");
        assert!(error.contains("must fail rather than skip"));
    }
}

#[test]
fn complete_test_validation_rejects_missing_zero_and_partial_runs() {
    assert!(validate_complete_test_run("", "CUDA").is_err());
    assert!(validate_complete_test_run(
        "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out",
        "CUDA",
    )
    .is_err());
    assert!(validate_complete_test_run(
        "test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out",
        "CUDA",
    )
    .is_err());
    assert!(validate_complete_test_run(
        "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out",
        "CUDA",
    )
    .is_err());
    assert!(validate_complete_test_run("test result: CANCELLED.", "CUDA").is_err());
}

#[test]
fn exact_named_validation_rejects_partial_or_substituted_success() {
    let expected = ["alpha", "beta"];
    let complete = "\
test alpha ... ok
test beta ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
";
    validate_exact_named_run(complete, "CUDA", &expected).unwrap();

    let partial = "\
test alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
";
    assert!(validate_exact_named_run(partial, "CUDA", &expected).is_err());

    let substituted = "\
test alpha ... ok
test gamma ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
";
    assert!(validate_exact_named_run(substituted, "CUDA", &expected).is_err());
}

#[test]
fn rust_test_output_parsers_are_name_and_summary_exact() {
    let listed = listed_rust_tests("alpha: test\nbeta: test\n2 tests, 0 benchmarks\n");
    assert_eq!(listed.into_iter().collect::<Vec<_>>(), ["alpha", "beta"]);
    let passed = passed_rust_tests("test alpha ... ok\ntest beta ... ok\n");
    assert_eq!(passed.into_iter().collect::<Vec<_>>(), ["alpha", "beta"]);
    assert_eq!(
        successful_test_summaries(
            "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"
        )
        .len(),
        1
    );
    assert!(successful_test_summaries("Doc-tests j2k_cuda\nrunning 2 tests").is_empty());
}

#[cfg(unix)]
fn shell_lines(lines: &[&str], prefix: &str, suffix: &str) -> String {
    lines
        .iter()
        .map(|line| format!("printf '%s\\n' '{prefix}{line}{suffix}'"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(unix)]
fn recording_cuda_cargo() -> RecordingProgram {
    let ht_listed = shell_lines(HTJ2K_ENCODE_PARITY_TESTS, "", ": test");
    let transcode_listed = shell_lines(TRANSCODE_PARITY_TESTS, "", ": test");
    let ht_passed = shell_lines(HTJ2K_ENCODE_PARITY_TESTS, "test ", " ... ok");
    let transcode_passed = shell_lines(TRANSCODE_PARITY_TESTS, "test ", " ... ok");
    let script = format!(
        r#"case "$*" in
*"--list"*)
  case " $* " in
  *" -p j2k-cuda "*) {ht_listed} ;;
  *" -p j2k-transcode-cuda "*) {transcode_listed} ;;
  esac
  exit 0
  ;;
*" --test "*)
  case " $* " in
  *" -p j2k-cuda "*)
    {ht_passed}
    printf '%s\n' 'test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
    ;;
  *" -p j2k-transcode-cuda "*)
    {transcode_passed}
    printf '%s\n' 'test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
    ;;
  esac
  exit 0
  ;;
*"test --release"*)
  printf '%s\n' 'test smoke ... ok'
  printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
  ;;
esac"#
    );
    RecordingProgram::new("cuda-release-command-test", &script)
}

#[cfg(unix)]
#[test]
fn cuda_release_executes_the_complete_hermetic_command_plan() {
    let device = RecordingProgram::new("cuda-device-test", "printf '%s\\n' '0, GPU-HERMETIC'");
    let _device = use_test_nvidia_smi_program(device.program().as_os_str().to_owned());
    let cargo = recording_cuda_cargo();
    let _cargo = use_test_cargo_program(cargo.program().as_os_str().to_owned());

    let mut expected_runs = 1;
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        release_cuda().expect("platform wrapper should execute the hermetic CUDA release plan");
        expected_runs += 1;
    } else {
        assert!(release_cuda().is_err());
    }
    run_release_cuda("linux", "x86_64").expect("platform-independent CUDA release plan");

    let log = cargo.log();
    assert_eq!(log.lines().count(), expected_runs * 16);
    assert!(log.lines().all(|line| line.contains("RUST_TEST_THREADS=1")));
    assert_eq!(device.log().lines().count(), expected_runs);
}

#[cfg(unix)]
#[test]
fn cuda_device_override_is_nested_transactional_and_fail_closed() {
    let success = RecordingProgram::new("cuda-device-outer", "printf '%s\\n' '0, GPU-OUTER'");
    let _outer = use_test_nvidia_smi_program(success.program().as_os_str().to_owned());
    require_cuda_device().unwrap();

    {
        let failure = RecordingProgram::new("cuda-device-inner", "exit 9");
        let _inner = use_test_nvidia_smi_program(failure.program().as_os_str().to_owned());
        let error = require_cuda_device().unwrap_err();
        assert!(error.contains("exited with"));
    }

    require_cuda_device().unwrap();
    assert_eq!(success.log().lines().count(), 2);
}

#[cfg(unix)]
#[test]
fn exact_inventory_and_captured_cargo_report_subprocess_failures() {
    let mismatch = RecordingProgram::new(
        "cuda-inventory-mismatch",
        "printf '%s\\n' 'unexpected_test: test'",
    );
    {
        let _cargo = use_test_cargo_program(mismatch.program().as_os_str().to_owned());
        let error = validate_exact_inventory(&EXACT_CUDA_SUITES[0]).unwrap_err();
        assert!(error.contains("inventory mismatch"));
        assert!(error.contains("unexpected_test"));
    }

    let success = RecordingProgram::new(
        "cuda-captured-success",
        "printf '%s\\n' 'captured stdout'; printf '%s\\n' 'captured stderr' >&2",
    );
    {
        let _cargo = use_test_cargo_program(success.program().as_os_str().to_owned());
        let output = run_cargo_captured(&["check"], CUDA_RELEASE_ENV, "captured CUDA").unwrap();
        assert!(output.contains("captured stdout"));
        assert!(output.contains("captured stderr"));
    }

    let failure = RecordingProgram::new("cuda-captured-failure", "exit 7");
    let _cargo = use_test_cargo_program(failure.program().as_os_str().to_owned());
    let error = run_cargo_captured(&["check"], CUDA_RELEASE_ENV, "captured CUDA").unwrap_err();
    assert!(error.contains("exited with"));
    assert!(error.contains("captured CUDA"));
}
