// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path, process::Command};

use super::super::{
    is_archived_handoff, is_repo_lint_test_source, referenced_shell_scripts, repo_root,
    repo_text_files,
};

fn is_git_ignored_untracked(root: &Path, path: &Path) -> bool {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let status = Command::new("git")
        .args(["check-ignore", "--quiet", "--"])
        .arg(relative)
        .current_dir(root)
        .status()
        .unwrap_or_else(|err| panic!("run git check-ignore for {}: {err}", relative.display()));

    match status.code() {
        Some(0) => true,
        Some(1) => false,
        code => panic!(
            "git check-ignore for {} exited unexpectedly with {code:?}",
            relative.display()
        ),
    }
}

#[test]
fn home_path_scan_excludes_only_git_ignored_untracked_artifacts() {
    let root = repo_root();
    assert!(
        is_git_ignored_untracked(root, &root.join("coverage-never-created-summary.json")),
        "the generated coverage artifact pattern must be excluded even before generation"
    );
    assert!(
        !is_git_ignored_untracked(root, &root.join(".gitignore")),
        "tracked files must remain in the public home-path scan"
    );
}

#[test]
fn public_text_does_not_embed_local_user_home_paths() {
    let root = repo_root();
    let mut offenders = Vec::new();

    for path in repo_text_files(root) {
        if is_archived_handoff(&path) {
            continue;
        }
        if is_repo_lint_test_source(root, &path) {
            continue;
        }
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        if (source.contains("/Users/") || source.contains("C:\\Users\\"))
            && !is_git_ignored_untracked(root, &path)
        {
            offenders.push(
                path.strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            );
        }
    }

    assert!(
    offenders.is_empty(),
    "public text must not embed local user-home paths; use env vars or repo-relative defaults: {offenders:?}"
);
}

#[test]
fn referenced_shell_scripts_exist() {
    let root = repo_root();
    let mut missing = Vec::new();

    for path in repo_text_files(root) {
        if is_archived_handoff(&path) {
            continue;
        }
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for script in referenced_shell_scripts(&source) {
            let root_relative = root.join(&script);
            let file_relative = path.parent().expect("text file has parent").join(&script);
            if !root_relative.exists() && !file_relative.exists() {
                missing.push(format!(
                    "{} references missing script {script}",
                    path.strip_prefix(root).unwrap_or(&path).display()
                ));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "all referenced shell scripts must exist: {missing:?}"
    );
}
