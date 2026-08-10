// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path, process::Command};

use super::super::{
    is_archived_handoff, is_repo_lint_test_source, referenced_shell_scripts, repo_root,
    repo_text_files,
};

fn latest_release_version(root: &Path) -> String {
    let output = Command::new("git")
        .args(["tag", "--list", "v[0-9]*", "--sort=-v:refname"])
        .current_dir(root)
        .output()
        .expect("list release tags");
    assert!(
        output.status.success(),
        "list release tags: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tags = String::from_utf8(output.stdout).expect("release tags are UTF-8");
    tags.lines()
        .next()
        .and_then(|tag| tag.strip_prefix('v'))
        .filter(|version| !version.is_empty())
        .expect("repository has a versioned release tag")
        .to_owned()
}

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

#[test]
fn release_facing_docs_match_the_latest_release_tag() {
    let root = repo_root();
    let version = latest_release_version(root);
    let minor_line = version
        .rsplit_once('.')
        .map(|(line, _)| format!("{line}.x"))
        .expect("release version has a patch component");

    let checks = [
        (
            "README.md",
            format!("**Release status:** `{version}` is published and security-supported."),
        ),
        (
            "SECURITY.md",
            format!("| `{version}` | Latest published and security-supported release |"),
        ),
        (
            "docs/release.md",
            format!(
                "The `j2k` {version} public crate release is published and security-supported."
            ),
        ),
        (
            "docs/stable-api-1.0.md",
            format!("The currently published stable contract is the `{minor_line}` line."),
        ),
        (
            "docs/env-vars.md",
            format!("Stable: supported for the published v{minor_line} contract."),
        ),
    ];
    for (relative, expected) in checks {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|err| panic!("read {relative}: {err}"));
        assert!(
            source.contains(&expected),
            "{relative} must describe latest release tag v{version} with `{expected}`"
        );
    }
}

#[test]
fn static_site_release_status_uses_canonical_sources() {
    let root = repo_root();

    for relative in [
        "docs/index.html",
        "docs/cuda-jpeg2000-rust/index.html",
        "docs/gpu-jpeg2000-rust/index.html",
        "docs/htj2k-rust/index.html",
        "docs/metal-jpeg2000-rust/index.html",
        "docs/rust-jpeg2000-codec/index.html",
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|err| panic!("read {relative}: {err}"));
        assert!(
            source.contains("https://github.com/frames-sg/j2k/blob/main/docs/release.md")
                && source.contains("https://crates.io/crates/j2k"),
            "{relative} must point to the canonical release policy and crates.io"
        );
        assert!(
            !source.contains("<strong>Release status:</strong> j2k "),
            "{relative} must not duplicate a mutable release number"
        );
    }
}

#[test]
fn static_metal_page_documents_the_objc2_boundary() {
    let root = repo_root();
    let source = fs::read_to_string(root.join("docs/metal-jpeg2000-rust/index.html"))
        .expect("read Metal landing page");

    assert!(
        source.contains("<code>objc2-metal</code>")
            && source.contains("no longer depend on <code>metal-rs</code>"),
        "the Metal landing page must document the 0.9 expert-API dependency migration"
    );
}
