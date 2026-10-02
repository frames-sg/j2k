// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

pub(crate) mod allocation_policy;
pub(crate) mod architecture_policy;
pub(crate) mod corpus_policy;
pub(crate) mod dependency_policy;
pub(crate) mod docs_and_workflows_policy;
pub(crate) mod phase_order_policy;
pub(crate) mod public_docs_policy;
pub(crate) mod rust_function_policy;
pub(crate) mod suppression_policy;
pub(crate) mod workflow_structure_policy;

pub(crate) fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
}

pub(crate) fn sha256_hex(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .or_else(|_| {
            Command::new("shasum")
                .args(["-a", "256"])
                .arg(path)
                .output()
        })
        .unwrap_or_else(|error| panic!("hash {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "hash command failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("missing hash output for {}", path.display()))
        .to_owned()
}

pub(crate) struct PatternCheck<'a> {
    source_name: &'a str,
    source: &'a str,
    required: &'a [&'a str],
}

impl<'a> PatternCheck<'a> {
    pub(crate) fn new(source_name: &'a str, source: &'a str) -> Self {
        Self {
            source_name,
            source,
            required: &[],
        }
    }

    pub(crate) fn required(mut self, required: &'a [&'a str]) -> Self {
        self.required = required;
        self
    }
}

pub(crate) fn assert_pattern_checks(checks: &[PatternCheck<'_>]) {
    assert!(!checks.is_empty(), "pattern check set must not be empty");
    for check in checks {
        assert!(
            !check.required.is_empty(),
            "{} pattern check must not be empty",
            check.source_name
        );
        for required in check.required {
            assert!(
                check.source.contains(required),
                "{} must contain `{required}`",
                check.source_name
            );
        }
    }
}

pub(crate) struct FilePatternCheck<'a> {
    relative_path: &'a str,
    required: &'a [&'a str],
}

impl<'a> FilePatternCheck<'a> {
    pub(crate) fn new(relative_path: &'a str) -> Self {
        Self {
            relative_path,
            required: &[],
        }
    }

    pub(crate) fn required(mut self, required: &'a [&'a str]) -> Self {
        self.required = required;
        self
    }
}

pub(crate) fn assert_file_pattern_checks(root: &Path, checks: &[FilePatternCheck<'_>]) {
    assert!(
        !checks.is_empty(),
        "file pattern check set must not be empty"
    );
    for check in checks {
        let source = fs::read_to_string(root.join(check.relative_path))
            .unwrap_or_else(|error| panic!("read {}: {error}", check.relative_path));
        assert_pattern_checks(&[
            PatternCheck::new(check.relative_path, &source).required(check.required)
        ]);
    }
}

pub(crate) fn rust_sources(directory: &Path) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    collect_rust_sources(directory, &mut sources);
    sources
}

fn collect_rust_sources(directory: &Path, sources: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
    {
        let path = entry.expect("read directory entry").path();
        if path.is_dir() {
            if !should_skip_repo_dir(&path) {
                collect_rust_sources(&path, sources);
            }
        } else if path.extension().and_then(OsStr::to_str) == Some("rs") {
            sources.push(path);
        }
    }
}

pub(crate) fn repo_text_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_repo_text_files(root, &mut files);
    files
}

fn collect_repo_text_files(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
    {
        let path = entry.expect("read directory entry").path();
        if path.is_dir() {
            if !should_skip_repo_dir(&path) {
                collect_repo_text_files(&path, files);
            }
        } else if is_repo_text_file(&path) {
            files.push(path);
        }
    }
}

fn should_skip_repo_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| matches!(name, ".codewhale" | ".git" | ".venv" | "target"))
}

fn is_repo_text_file(path: &Path) -> bool {
    if path.file_name().and_then(OsStr::to_str) == Some("Cargo.lock") {
        return true;
    }
    matches!(
        path.extension().and_then(OsStr::to_str),
        Some(
            "bib"
                | "c"
                | "cc"
                | "cpp"
                | "cu"
                | "h"
                | "hpp"
                | "json"
                | "lock"
                | "md"
                | "py"
                | "rs"
                | "sh"
                | "tex"
                | "toml"
                | "txt"
                | "yaml"
                | "yml"
        )
    )
}

pub(crate) fn is_archived_handoff(path: &Path) -> bool {
    path.file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| name.starts_with("HANDOFF-"))
}

pub(crate) fn is_repo_lint_test_source(root: &Path, path: &Path) -> bool {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let relative = relative.to_string_lossy().replace('\\', "/");
    relative == "xtask/tests/repo_lint.rs" || relative.starts_with("xtask/tests/repo_lint_support/")
}

pub(crate) fn referenced_shell_scripts(source: &str) -> Vec<String> {
    source
        .split(|character: char| {
            !(character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_' | '/'))
        })
        .filter(|token| {
            Path::new(token)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("sh"))
                && token.contains('/')
        })
        .filter(|token| !token.starts_with("http://") && !token.starts_with("https://"))
        .map(str::to_owned)
        .collect()
}
