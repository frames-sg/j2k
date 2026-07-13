// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ffi::OsStr;

use super::{parse_options, repository_from_remote, resolve_repository, select_token_env};

#[cfg(unix)]
use super::release_status;

#[cfg(unix)]
const WORKSPACE_CHILD_ENV: &str = "XTASK_TEST_RELEASE_STATUS_WORKSPACE_CHILD";

#[test]
fn options_require_and_normalize_an_exact_sha() {
    let options = parse_options(
        ["--sha", &"A".repeat(40), "--repository", "frames-sg/j2k"]
            .into_iter()
            .map(str::to_string),
    )
    .unwrap();
    assert_eq!(options.sha, "a".repeat(40));
    assert_eq!(options.repository.as_deref(), Some("frames-sg/j2k"));

    for invalid in ["abc", &"g".repeat(40), &"a".repeat(41)] {
        assert!(parse_options(["--sha", invalid].into_iter().map(str::to_string)).is_err());
    }
}

#[test]
fn options_reject_missing_values_duplicates_help_and_unknown_arguments() {
    for (args, expected) in [
        (Vec::new(), "--sha is required"),
        (vec!["--sha"], "--sha` requires a value"),
        (vec!["--repository"], "--repository` requires a value"),
        (vec!["--help"], "usage: cargo xtask release-status"),
        (vec!["--unknown"], "unknown release-status argument"),
        (
            vec!["--sha", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "--sha"],
            "was provided more than once",
        ),
    ] {
        let error = parse_options(args.into_iter().map(str::to_string))
            .expect_err("invalid options must reject");
        assert!(error.contains(expected), "unexpected error: {error}");
    }

    let error = parse_options(
        [
            "--sha",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--repository",
            "invalid",
        ]
        .into_iter()
        .map(str::to_string),
    )
    .expect_err("malformed repository must reject");
    assert!(error.contains("owner/name"));
}

#[test]
fn repository_resolution_has_explicit_environment_remote_precedence() {
    assert_eq!(
        resolve_repository(
            Some("explicit/repo"),
            Some("environment/repo"),
            Some("git@example.invalid:remote/repo.git")
        )
        .unwrap(),
        "explicit/repo"
    );
    assert_eq!(
        resolve_repository(
            None,
            Some("environment/repo"),
            Some("git@example.invalid:remote/repo.git")
        )
        .unwrap(),
        "environment/repo"
    );
    assert_eq!(
        resolve_repository(None, None, Some("git@example.invalid:remote/repo.git")).unwrap(),
        "remote/repo"
    );
    assert!(resolve_repository(None, None, None).is_err());
}

#[test]
fn common_remote_url_forms_are_supported_fail_closed() {
    for remote in [
        "https://github.com/frames-sg/j2k.git",
        "ssh://git@github.com/frames-sg/j2k.git",
        "git@github.com:frames-sg/j2k.git",
        "frames-sg/j2k",
    ] {
        assert_eq!(repository_from_remote(remote).unwrap(), "frames-sg/j2k");
    }
    for remote in [
        "",
        "https://github.com/only-owner",
        "owner/repo/extra",
        "owner/repo?ref=x",
    ] {
        assert!(repository_from_remote(remote).is_err(), "accepted {remote}");
    }
}

#[test]
fn authentication_uses_named_environment_without_reading_token_text() {
    assert_eq!(
        select_token_env(Some(OsStr::new("secret")), None).unwrap(),
        "GH_TOKEN"
    );
    assert_eq!(
        select_token_env(None, Some(OsStr::new("secret"))).unwrap(),
        "GITHUB_TOKEN"
    );
    assert!(select_token_env(None, None).is_err());
    assert!(select_token_env(Some(OsStr::new("")), Some(OsStr::new(""))).is_err());
}

#[cfg(unix)]
#[test]
fn release_status_derives_remote_and_executes_exact_verifier_contract() {
    use std::os::unix::fs::symlink;

    use crate::test_command::RecordingProgram;

    if std::env::var_os(WORKSPACE_CHILD_ENV).is_some() {
        release_status(
            ["--sha", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"]
                .into_iter()
                .map(str::to_string),
        )
        .expect("hermetic release-status command");
        return;
    }

    let recording = RecordingProgram::new(
        "release-status-command-test",
        r#"case "${0##*/}" in
  git) printf '%s\n' 'git@example.invalid:frames-sg/j2k.git' ;;
  python3) exit 0 ;;
  *) exit 90 ;;
esac"#,
    );
    let program_dir = recording
        .program()
        .parent()
        .expect("recording program parent");
    symlink(recording.program(), program_dir.join("git")).expect("fake git symlink");
    symlink(recording.program(), program_dir.join("python3")).expect("fake python3 symlink");
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask workspace root");
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .arg("release_status::tests::release_status_derives_remote_and_executes_exact_verifier_contract")
        .arg("--exact")
        .arg("--nocapture")
        .current_dir(workspace)
        .env(WORKSPACE_CHILD_ENV, "1")
        .env("GH_TOKEN", "test-token-placeholder")
        .env_remove("GITHUB_TOKEN")
        .env_remove("GITHUB_REPOSITORY")
        .env("PATH", program_dir)
        .output()
        .expect("run release-status workspace child");
    assert!(
        output.status.success(),
        "workspace child failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let log = recording.log();
    let lines = log.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 2, "unexpected command log: {log}");
    assert!(lines[0].starts_with("config --get remote.origin.url|"));
    assert!(lines[1].contains("verify-candidate --repository frames-sg/j2k"));
    assert!(lines[1]
        .contains("--candidate-sha aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa --token-env GH_TOKEN"));
    assert!(lines[1].contains("--aggregate-job Release candidate aggregate"));
    assert!(lines[1].contains("--cuda-job CUDA API compatibility on x86_64"));
    assert!(lines[1].contains("--metal-job Metal validation on Apple Silicon"));
}
