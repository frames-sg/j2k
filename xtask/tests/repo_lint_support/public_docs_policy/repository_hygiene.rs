// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{ffi::OsStr, fs};

use super::super::{
    is_allowed_legacy_name_history_reference, is_archived_handoff, is_repo_lint_test_source,
    referenced_shell_scripts, repo_root, repo_text_files,
};

#[test]
fn active_repo_text_does_not_reintroduce_signinum_names() {
    let root = repo_root();
    let mut offenders = Vec::new();

    for path in repo_text_files(root) {
        if is_archived_handoff(&path)
            || is_allowed_legacy_name_history_reference(root, &path)
            || is_repo_lint_test_source(root, &path)
        {
            continue;
        }
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for (line_idx, line) in source.lines().enumerate() {
            let lower = line.to_ascii_lowercase();
            if lower.contains("signinum") {
                offenders.push(format!(
                    "{}:{}:{}",
                    path.strip_prefix(root).unwrap_or(&path).display(),
                    line_idx + 1,
                    line
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "active repo text must not reintroduce signinum names after the j2k rename:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn public_repo_excludes_agent_private_artifacts() {
    let root = repo_root();
    let private_docs_name = ["super", "powers"].concat();
    let private_dir = ["docs", private_docs_name.as_str()].join("/");
    let migration_doc = ["MIGRATION", ".md"].concat();
    let migration_doc_lower = migration_doc.to_ascii_lowercase();
    let mut offenders = Vec::new();

    for path in repo_text_files(root) {
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let relative_text = relative.to_string_lossy();
        let file_name = path
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        if relative_text.starts_with(&private_dir) || file_name == migration_doc_lower {
            offenders.push(relative_text.to_string());
        }
    }

    assert!(
    offenders.is_empty(),
    "public repo must not track agent-private planning docs or migration scratch files: {offenders:?}"
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
        if source.contains("/Users/") || source.contains("C:\\Users\\") {
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
fn public_narrative_docs_do_not_carry_stale_zeiss_claims() {
    let root = repo_root();
    let mut offenders = Vec::new();

    for relative in [
        "README.md",
        "docs/architecture.md",
        "docs/release.md",
        "paper/paper.md",
        "paper/arxiv/main.tex",
    ] {
        let path = root.join(relative);
        let Ok(source) = fs::read_to_string(&path) else {
            if relative.starts_with("paper/") {
                continue;
            }
            panic!("read {}: missing required narrative doc", path.display());
        };
        if source.contains("Zeiss") {
            offenders.push(relative);
        }
    }

    assert!(
        offenders.is_empty(),
        "public narrative docs must not carry stale Zeiss integration claims: {offenders:?}"
    );
}
