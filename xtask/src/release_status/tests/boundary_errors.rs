// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    ffi::OsStr,
    os::unix::{ffi::OsStringExt, fs::symlink},
    process::Output,
};

use super::super::release_status;
use crate::test_command::RecordingProgram;

const CASE_ENV: &str = "XTASK_TEST_RELEASE_STATUS_BOUNDARY_CASE";
const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn release_status_result() -> Result<(), String> {
    release_status(["--sha", SHA].into_iter().map(str::to_string))
}

fn run_child(
    test_name: &str,
    case: &str,
    recording: &RecordingProgram,
    repository: Option<&OsStr>,
) -> Output {
    let program_dir = recording
        .program()
        .parent()
        .expect("recording program parent");
    symlink(recording.program(), program_dir.join("git")).expect("fake git symlink");
    symlink(recording.program(), program_dir.join("python3")).expect("fake python3 symlink");
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask workspace root");
    let mut command = std::process::Command::new(std::env::current_exe().expect("test binary"));
    command
        .arg(test_name)
        .arg("--exact")
        .arg("--nocapture")
        .current_dir(workspace)
        .env(CASE_ENV, case)
        .env("GITHUB_TOKEN", "test-token-placeholder")
        .env_remove("GH_TOKEN")
        .env("PATH", program_dir);
    if let Some(repository) = repository {
        command.env("GITHUB_REPOSITORY", repository);
    } else {
        command.env_remove("GITHUB_REPOSITORY");
    }
    command.output().expect("run release-status boundary child")
}

fn assert_child_success(output: &Output) {
    assert!(
        output.status.success(),
        "boundary child failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn repository_environment_empty_and_present_paths_execute_exact_contracts() {
    if std::env::var_os(CASE_ENV).is_some() {
        release_status_result().expect("hermetic environment repository release status");
        return;
    }

    for (case, repository, expected_repository, expected_commands) in [
        ("present", "environment/repo", "environment/repo", 1_usize),
        ("empty", "", "remote/repo", 2),
    ] {
        let recording = RecordingProgram::new(
            "release-status-environment-boundary",
            r#"case "${0##*/}" in
  git) printf '%s\n' 'git@example.invalid:remote/repo.git' ;;
  python3) exit 0 ;;
  *) exit 90 ;;
esac"#,
        );
        let output = run_child(
            "release_status::tests::boundary_errors::repository_environment_empty_and_present_paths_execute_exact_contracts",
            case,
            &recording,
            Some(OsStr::new(repository)),
        );
        assert_child_success(&output);

        let log = recording.log();
        let lines = log.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), expected_commands, "unexpected log: {log}");
        let verifier = lines.last().expect("verifier command");
        assert!(verifier.contains(&format!("--repository {expected_repository}")));
        assert!(verifier.contains("--token-env GITHUB_TOKEN"));
    }
}

#[test]
fn repository_discovery_rejects_process_and_unicode_failures() {
    if let Some(case) = std::env::var_os(CASE_ENV) {
        let error = release_status_result().expect_err("invalid repository discovery must reject");
        let expected = match case.to_str().expect("Unicode test case") {
            "git-failure" => "repository is required",
            "git-empty" => "remote.origin.url is empty",
            "git-non-unicode" => "remote.origin.url must contain valid Unicode",
            "env-non-unicode" => "GITHUB_REPOSITORY must contain valid Unicode",
            other => panic!("unknown boundary case {other}"),
        };
        assert!(error.contains(expected), "unexpected error: {error}");
        return;
    }

    for (case, script, repository) in [
        ("git-failure", "exit 7", None),
        ("git-empty", "exit 0", None),
        ("git-non-unicode", "printf '\\377'", None),
        (
            "env-non-unicode",
            "exit 90",
            Some(std::ffi::OsString::from_vec(vec![0xff])),
        ),
    ] {
        let recording = RecordingProgram::new("release-status-repository-error", script);
        let output = run_child(
            "release_status::tests::boundary_errors::repository_discovery_rejects_process_and_unicode_failures",
            case,
            &recording,
            repository.as_deref(),
        );
        assert_child_success(&output);
    }
}
