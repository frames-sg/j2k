// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::BTreeSet,
    ffi::OsStr,
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
};

pub(crate) mod architecture_policy;
pub(crate) mod corpus_policy;
pub(crate) mod dependency_policy;
pub(crate) mod docs_and_workflows_policy;
pub(crate) mod j2k_ml_policy;
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
    forbidden: &'a [&'a str],
}

impl<'a> PatternCheck<'a> {
    pub(crate) fn new(source_name: &'a str, source: &'a str) -> Self {
        Self {
            source_name,
            source,
            required: &[],
            forbidden: &[],
        }
    }

    pub(crate) fn required(mut self, required: &'a [&'a str]) -> Self {
        self.required = required;
        self
    }

    pub(crate) fn forbidden(mut self, forbidden: &'a [&'a str]) -> Self {
        self.forbidden = forbidden;
        self
    }
}

pub(crate) fn assert_pattern_checks(checks: &[PatternCheck<'_>]) {
    assert!(!checks.is_empty(), "pattern check set must not be empty");
    for check in checks {
        assert!(
            !check.required.is_empty() || !check.forbidden.is_empty(),
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
        for forbidden in check.forbidden {
            assert!(
                !check.source.contains(forbidden),
                "{} must not contain `{forbidden}`",
                check.source_name
            );
        }
    }
}

pub(crate) struct FilePatternCheck<'a> {
    relative_path: &'a str,
    source_name: Option<&'a str>,
    required: &'a [&'a str],
    forbidden: &'a [&'a str],
}

impl<'a> FilePatternCheck<'a> {
    pub(crate) fn new(relative_path: &'a str) -> Self {
        Self {
            relative_path,
            source_name: None,
            required: &[],
            forbidden: &[],
        }
    }

    pub(crate) fn named(mut self, source_name: &'a str) -> Self {
        self.source_name = Some(source_name);
        self
    }

    pub(crate) fn required(mut self, required: &'a [&'a str]) -> Self {
        self.required = required;
        self
    }

    pub(crate) fn forbidden(mut self, forbidden: &'a [&'a str]) -> Self {
        self.forbidden = forbidden;
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
        assert_pattern_checks(&[PatternCheck::new(
            check.source_name.unwrap_or(check.relative_path),
            &source,
        )
        .required(check.required)
        .forbidden(check.forbidden)]);
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

pub(crate) fn j2k_env_tokens(source: &str) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    for line in source.lines() {
        let mut offset = 0;
        while let Some(relative_start) = line[offset..].find("J2K_") {
            let start = offset + relative_start;
            if start > 0 {
                let previous = line[..start].chars().next_back().expect("preceding char");
                if previous.is_ascii_alphanumeric() || previous == '_' {
                    offset = start + "J2K_".len();
                    continue;
                }
            }
            let end = line[start..]
                .find(|character: char| {
                    !(character.is_ascii_uppercase()
                        || character.is_ascii_digit()
                        || character == '_')
                })
                .map_or(line.len(), |end| start + end);
            tokens.insert(line[start..end].to_owned());
            offset = end;
        }
    }
    tokens
}

pub(crate) fn documented_j2k_env_vars(docs: &str) -> BTreeSet<String> {
    docs.lines()
        .filter_map(|line| {
            let line = line.trim_start();
            line.starts_with("| `J2K_")
                .then(|| line.split('`').nth(1).map(str::to_owned))
                .flatten()
        })
        .collect()
}

pub(crate) fn is_internal_j2k_token(token: &str) -> bool {
    token == "J2K_"
        || token == "J2K_ENCODE"
        || token.starts_with("J2K_SIGNPOST_")
        || token.starts_with("J2K_BATCH_")
        || token.starts_with("J2K_CLASSIC_")
        || token.starts_with("J2K_DECODE_")
        || token.starts_with("J2K_DEQUANTIZE")
        || token.starts_with("J2K_ENCODE_")
        || token.starts_with("J2K_FDWT97_")
        || token.starts_with("J2K_GPU_ENCODE_")
        || token.starts_with("J2K_HOST_")
        || token.starts_with("J2K_HT_")
        || token.starts_with("J2K_IDWT")
        || token.starts_with("J2K_KERNELS_")
        || token.starts_with("J2K_MCT_")
        || token.starts_with("J2K_NOT_")
        || token.starts_with("J2K_OUTPUT_")
        || token.starts_with("J2K_PACKET_")
        || token.starts_with("J2K_PLAN_")
        || token.starts_with("J2K_STATUS_")
        || token.starts_with("J2K_STORE_")
        || token.starts_with("J2K_UVLC_")
        || matches!(
            token,
            "J2K_JPEG_ZIGZAG"
                | "J2K_IMAGE_DIMENSION"
                | "J2K_LOSSY_97_QUANTIZATION_SCALE"
                | "J2K_PI"
                | "J2K_PLAN"
                | "J2K_PROFILE_TEST_STAGE_MODE"
                | "J2K_GPU_TEST_SKIPPED"
                | "J2K_REFINEMENT_FIXTURE"
                | "J2K_SPEC_COMPONENTS"
                | "J2K_TILE_COUNT"
                | "J2K_YCBCR"
        )
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

pub(crate) fn rust_include_paths(source: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for marker in ["include_bytes!(\"", "include_str!(\""] {
        let mut rest = source;
        while let Some(start) = rest.find(marker) {
            let after_marker = &rest[start + marker.len()..];
            let Some(end) = after_marker.find('"') else {
                break;
            };
            paths.push(after_marker[..end].to_owned());
            rest = &after_marker[end + 1..];
        }
    }
    paths
}

pub(crate) fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}
