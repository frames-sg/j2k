// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::BTreeSet,
    ffi::OsStr,
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

pub(crate) mod architecture_policy;
pub(crate) mod corpus_policy;
pub(crate) mod dependency_policy;
pub(crate) mod docs_and_workflows_policy;
pub(crate) mod gpu_adapter_policy;
pub(crate) mod public_docs_policy;
pub(crate) mod release_policy;
pub(crate) mod shader_policy;
pub(crate) mod source_policy;
pub(crate) mod workflow_policy;

pub(crate) fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
}

pub(crate) fn assert_contains_all(source_name: &str, source: &str, required: &[&str]) {
    assert!(
        !required.is_empty(),
        "{source_name} required-pattern set must not be empty"
    );
    for pattern in required {
        assert!(
            source.contains(pattern),
            "{source_name} must contain `{pattern}`"
        );
    }
}

pub(crate) fn assert_contains_all_normalized(source_name: &str, source: &str, required: &[&str]) {
    assert!(
        !required.is_empty(),
        "{source_name} normalized required-pattern set must not be empty"
    );
    let normalized_source = normalized_whitespace(source);
    for pattern in required {
        let normalized_pattern = normalized_whitespace(pattern);
        assert!(
            !normalized_pattern.is_empty(),
            "{source_name} normalized required pattern must not be empty"
        );
        assert!(
            normalized_source.contains(&normalized_pattern),
            "{source_name} must contain normalized `{pattern}`"
        );
    }
}

pub(crate) fn assert_not_contains_all(source_name: &str, source: &str, forbidden: &[&str]) {
    assert!(
        !forbidden.is_empty(),
        "{source_name} forbidden-pattern set must not be empty"
    );
    for pattern in forbidden {
        assert!(
            !source.contains(pattern),
            "{source_name} must not contain `{pattern}`"
        );
    }
}

pub(crate) fn assert_not_contains_all_normalized(
    source_name: &str,
    source: &str,
    forbidden: &[&str],
) {
    assert!(
        !forbidden.is_empty(),
        "{source_name} normalized forbidden-pattern set must not be empty"
    );
    let normalized_source = normalized_whitespace(source);
    for pattern in forbidden {
        let normalized_pattern = normalized_whitespace(pattern);
        assert!(
            !normalized_pattern.is_empty(),
            "{source_name} normalized forbidden pattern must not be empty"
        );
        assert!(
            !normalized_source.contains(&normalized_pattern),
            "{source_name} must not contain normalized `{pattern}`"
        );
    }
}

pub(crate) struct FilePatternCheck<'a> {
    relative_path: &'a str,
    source_name: Option<&'a str>,
    required: &'a [&'a str],
    forbidden: &'a [&'a str],
    normalized_required: &'a [&'a str],
    normalized_forbidden: &'a [&'a str],
}

pub(crate) struct PatternCheck<'a> {
    source_name: &'a str,
    source: &'a str,
    required: &'a [&'a str],
    forbidden: &'a [&'a str],
    normalized_required: &'a [&'a str],
    normalized_forbidden: &'a [&'a str],
}

pub(crate) struct RustSourceScanCheck<'a> {
    source_name: &'a str,
    relative_dirs: &'a [&'a str],
    forbidden: &'a [&'a str],
}

impl<'a> RustSourceScanCheck<'a> {
    pub(crate) fn new(source_name: &'a str, relative_dirs: &'a [&'a str]) -> Self {
        Self {
            source_name,
            relative_dirs,
            forbidden: &[],
        }
    }

    pub(crate) fn forbidden(mut self, forbidden: &'a [&'a str]) -> Self {
        self.forbidden = forbidden;
        self
    }

    fn has_scan_scope(&self) -> bool {
        !self.relative_dirs.is_empty() && !self.forbidden.is_empty()
    }
}

impl<'a> PatternCheck<'a> {
    pub(crate) fn new(source_name: &'a str, source: &'a str) -> Self {
        Self {
            source_name,
            source,
            required: &[],
            forbidden: &[],
            normalized_required: &[],
            normalized_forbidden: &[],
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

    pub(crate) fn normalized_required(mut self, normalized_required: &'a [&'a str]) -> Self {
        self.normalized_required = normalized_required;
        self
    }

    pub(crate) fn normalized_forbidden(mut self, normalized_forbidden: &'a [&'a str]) -> Self {
        self.normalized_forbidden = normalized_forbidden;
        self
    }

    fn has_patterns(&self) -> bool {
        has_any_pattern(
            self.required,
            self.forbidden,
            self.normalized_required,
            self.normalized_forbidden,
        )
    }
}

impl<'a> FilePatternCheck<'a> {
    pub(crate) fn new(relative_path: &'a str) -> Self {
        Self {
            relative_path,
            source_name: None,
            required: &[],
            forbidden: &[],
            normalized_required: &[],
            normalized_forbidden: &[],
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

    pub(crate) fn normalized_required(mut self, normalized_required: &'a [&'a str]) -> Self {
        self.normalized_required = normalized_required;
        self
    }

    pub(crate) fn normalized_forbidden(mut self, normalized_forbidden: &'a [&'a str]) -> Self {
        self.normalized_forbidden = normalized_forbidden;
        self
    }

    fn has_patterns(&self) -> bool {
        has_any_pattern(
            self.required,
            self.forbidden,
            self.normalized_required,
            self.normalized_forbidden,
        )
    }
}

fn has_any_pattern(
    required: &[&str],
    forbidden: &[&str],
    normalized_required: &[&str],
    normalized_forbidden: &[&str],
) -> bool {
    !required.is_empty()
        || !forbidden.is_empty()
        || !normalized_required.is_empty()
        || !normalized_forbidden.is_empty()
}

pub(crate) fn assert_pattern_checks(checks: &[PatternCheck<'_>]) {
    assert!(
        !checks.is_empty(),
        "source-pattern check set must not be empty"
    );
    for check in checks {
        assert!(
            check.has_patterns(),
            "{} source-pattern check must define at least one pattern",
            check.source_name
        );
        assert_pattern_sets(
            check.source_name,
            check.source,
            check.required,
            check.forbidden,
            check.normalized_required,
            check.normalized_forbidden,
        );
    }
}

pub(crate) fn assert_rust_source_scan_checks(root: &Path, checks: &[RustSourceScanCheck<'_>]) {
    assert!(
        !checks.is_empty(),
        "rust-source scan check set must not be empty"
    );
    for check in checks {
        assert!(
            check.has_scan_scope(),
            "{} rust-source scan must define at least one directory and forbidden pattern",
            check.source_name
        );

        let mut violations = Vec::new();
        for relative_dir in check.relative_dirs {
            let sources = rust_sources(&root.join(relative_dir));
            assert!(
                !sources.is_empty(),
                "{} rust-source scan found no sources under {relative_dir}",
                check.source_name
            );
            for path in sources {
                let source = fs::read_to_string(&path)
                    .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
                for forbidden in check.forbidden {
                    if source.contains(forbidden) {
                        violations.push(format!(
                            "{} contains `{forbidden}`",
                            path.strip_prefix(root).unwrap_or(&path).display()
                        ));
                    }
                }
            }
        }

        assert!(
            violations.is_empty(),
            "{} rust-source scan violations:\n{}",
            check.source_name,
            violations.join("\n")
        );
    }
}

pub(crate) fn assert_file_pattern_checks(root: &Path, checks: &[FilePatternCheck<'_>]) {
    assert!(
        !checks.is_empty(),
        "file-pattern check set must not be empty"
    );
    for check in checks {
        let source_name = check.source_name.unwrap_or(check.relative_path);
        assert!(
            check.has_patterns(),
            "{source_name} file-pattern check must define at least one pattern"
        );

        let source = fs::read_to_string(root.join(check.relative_path))
            .unwrap_or_else(|err| panic!("read {}: {err}", check.relative_path));
        assert_pattern_sets(
            source_name,
            &source,
            check.required,
            check.forbidden,
            check.normalized_required,
            check.normalized_forbidden,
        );
    }
}

fn assert_pattern_sets(
    source_name: &str,
    source: &str,
    required: &[&str],
    forbidden: &[&str],
    normalized_required: &[&str],
    normalized_forbidden: &[&str],
) {
    if !required.is_empty() {
        assert_contains_all(source_name, source, required);
    }
    if !forbidden.is_empty() {
        assert_not_contains_all(source_name, source, forbidden);
    }
    if !normalized_required.is_empty() {
        assert_contains_all_normalized(source_name, source, normalized_required);
    }
    if !normalized_forbidden.is_empty() {
        assert_not_contains_all_normalized(source_name, source, normalized_forbidden);
    }
}

pub(crate) fn contains_normalized(source: &str, pattern: &str) -> bool {
    let normalized_pattern = normalized_whitespace(pattern);
    !normalized_pattern.is_empty() && normalized_whitespace(source).contains(&normalized_pattern)
}

fn normalized_whitespace(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn workflow_job<'a>(workflow: &'a str, job_name: &str) -> &'a str {
    let marker = format!("  {job_name}:");
    let start = workflow
        .find(&marker)
        .unwrap_or_else(|| panic!("missing workflow job {job_name}"));
    let rest = &workflow[start..];
    let mut search_start = marker.len();
    let mut end = rest.len();
    while let Some(relative) = rest[search_start..].find("\n  ") {
        let candidate = search_start + relative + 1;
        if !rest[candidate..].starts_with("    ") {
            end = candidate;
            break;
        }
        search_start = candidate + 1;
    }
    &rest[..end]
}

pub(crate) fn publishable_crate_dirs(root: &Path) -> Vec<PathBuf> {
    let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
    const_array_values(&xtask, "PUBLISHABLE_PACKAGES")
        .into_iter()
        .map(|package| {
            let dir = root.join("crates").join(&package);
            assert!(
                dir.exists(),
                "PUBLISHABLE_PACKAGES entry `{package}` must resolve to an existing crate dir"
            );
            dir
        })
        .collect()
}

pub(crate) fn const_array_values(source: &str, name: &str) -> Vec<String> {
    let values = const_array_block(source, name)
        .lines()
        .filter_map(|line| {
            let value = line.trim().trim_matches([',', '"']);
            if value.is_empty()
                || value.starts_with("const ")
                || value.starts_with(']')
                || value.starts_with('&')
            {
                None
            } else {
                Some(value.to_string())
            }
        })
        .collect::<Vec<_>>();
    assert!(
        !values.is_empty(),
        "const array {name} must contain at least one parsed value"
    );
    values
}

pub(crate) fn const_array_block<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source
        .find(&format!("const {name}:"))
        .unwrap_or_else(|| panic!("missing const {name}"));
    let rest = &source[start..];
    let end = rest
        .find("];")
        .unwrap_or_else(|| panic!("unterminated const {name}"));
    &rest[..end]
}

const CARGO_PUBLIC_API_VERSION: &str = "0.52.0";

static CARGO_METADATA_WORKSPACE_EDGES: OnceLock<BTreeSet<(String, String)>> = OnceLock::new();

pub(crate) fn cargo_metadata_workspace_edges(root: &Path) -> BTreeSet<(String, String)> {
    CARGO_METADATA_WORKSPACE_EDGES
        .get_or_init(|| load_cargo_metadata_workspace_edges(root))
        .clone()
}

fn load_cargo_metadata_workspace_edges(root: &Path) -> BTreeSet<(String, String)> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version=1"])
        .current_dir(root)
        .output()
        .expect("run cargo metadata");
    assert!(
        output.status.success(),
        "cargo metadata failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let metadata =
        serde_json::from_slice::<serde_json::Value>(&output.stdout).expect("parse cargo metadata");
    let workspace_members = metadata["workspace_members"]
        .as_array()
        .expect("metadata workspace_members array")
        .iter()
        .map(|id| {
            id.as_str()
                .expect("workspace member id is string")
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    let workspace_packages = metadata["packages"]
        .as_array()
        .expect("metadata packages array")
        .iter()
        .filter(|package| {
            package["id"]
                .as_str()
                .is_some_and(|id| workspace_members.contains(id))
        })
        .filter_map(|package| package["name"].as_str())
        .collect::<BTreeSet<_>>();

    let mut edges = BTreeSet::new();
    for package in metadata["packages"]
        .as_array()
        .expect("metadata packages array")
        .iter()
        .filter(|package| {
            package["id"]
                .as_str()
                .is_some_and(|id| workspace_members.contains(id))
        })
    {
        let source = package["name"].as_str().expect("package name");
        for dependency in package["dependencies"]
            .as_array()
            .expect("package dependencies array")
            .iter()
            .filter(|dependency| dependency["kind"].is_null())
            .filter(|dependency| dependency["source"].is_null())
            .filter_map(|dependency| dependency["name"].as_str())
            .filter(|dependency| workspace_packages.contains(dependency))
        {
            edges.insert((source.to_owned(), dependency.to_owned()));
        }
    }
    edges
}

pub(crate) fn architecture_doc_dependency_edges(docs: &str) -> BTreeSet<(String, String)> {
    let graph_section = docs
        .split("## Crate dependency graph")
        .nth(1)
        .expect("architecture dependency graph section");
    let graph_block = graph_section
        .split("```")
        .nth(1)
        .expect("architecture dependency graph code block");
    let mut edges = BTreeSet::new();

    for line in graph_block.lines().filter(|line| line.contains("->")) {
        let (source, dependencies) = line.split_once("->").expect("graph edge line");
        let source = source.trim();
        for dependency in dependencies.split(',') {
            let dependency = dependency
                .split_whitespace()
                .next()
                .expect("graph dependency token");
            edges.insert((source.to_owned(), dependency.to_owned()));
        }
    }

    edges
}

pub(crate) fn format_edge(edge: &(String, String)) -> String {
    format!("{} -> {}", edge.0, edge.1)
}

pub(crate) fn cargo_public_api_required(root: &Path, package: &str) -> String {
    let version = Command::new("cargo")
        .args(["public-api", "--version"])
        .current_dir(root)
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "run cargo public-api --version: {err}; install cargo-public-api {CARGO_PUBLIC_API_VERSION}"
            )
        });
    let stdout = String::from_utf8_lossy(&version.stdout);
    let stderr = String::from_utf8_lossy(&version.stderr);
    let version_text = format!("{stdout}{stderr}");
    assert!(
        version.status.success(),
        "cargo public-api --version failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        version_text.contains(CARGO_PUBLIC_API_VERSION),
        "cargo-public-api version must be {CARGO_PUBLIC_API_VERSION}; found `{version_text}`"
    );

    let output = Command::new("cargo")
        .args(["public-api", "-p", package, "--all-features"])
        .current_dir(root)
        .output()
        .unwrap_or_else(|err| panic!("run cargo public-api for {package}: {err}"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo public-api failed for package {package}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    format!("{stdout}{stderr}")
}

pub(crate) fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_rust_sources(dir, &mut out);
    out
}

fn collect_rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|err| panic!("read {}: {err}", dir.display())) {
        let entry = entry.expect("read directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

pub(crate) fn repo_text_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_repo_text_files(root, &mut out);
    out
}

fn collect_repo_text_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|err| panic!("read {}: {err}", dir.display())) {
        let entry = entry.expect("read directory entry");
        let path = entry.path();
        if path.is_dir() {
            if should_skip_repo_dir(&path) {
                continue;
            }
            collect_repo_text_files(&path, out);
            continue;
        }
        if is_repo_text_file(&path) {
            out.push(path);
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
    let relative_text = relative.to_string_lossy().replace('\\', "/");
    relative_text == "xtask/tests/repo_lint.rs"
        || relative_text.starts_with("xtask/tests/repo_lint_support/")
}

pub(crate) fn is_allowed_legacy_name_history_reference(root: &Path, path: &Path) -> bool {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let relative_text = relative.to_string_lossy().replace('\\', "/");
    relative_text == "CHANGELOG.md"
        || relative_text.contains("/migration")
        || relative_text.contains("migration/")
}

pub(crate) fn j2k_env_tokens(source: &str) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    for line in source.lines() {
        let mut offset = 0;
        while let Some(relative_start) = line[offset..].find("J2K_") {
            let token_start = offset + relative_start;
            if token_start > 0 {
                let previous = line[..token_start].chars().next_back().unwrap();
                if previous.is_ascii_alphanumeric() || previous == '_' {
                    offset = token_start + "J2K_".len();
                    continue;
                }
            }
            let token_end = line[token_start..]
                .find(|ch: char| !(ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_'))
                .map_or(line.len(), |end| token_start + end);
            tokens.insert(line[token_start..token_end].to_string());
            offset = token_end;
        }
    }
    tokens
}

pub(crate) fn documented_j2k_env_vars(docs: &str) -> BTreeSet<String> {
    docs.lines()
        .filter_map(|line| {
            let line = line.trim_start();
            if !line.starts_with("| `J2K_") {
                return None;
            }
            line.split('`').nth(1).map(str::to_string)
        })
        .collect()
}

#[test]
fn j2k_env_tokens_ignore_embedded_htj2k_suffixes() {
    let embedded_token = ["J2", "K_HTJ2", "K_DECODE_FIXTURE"].concat();
    let source = format!(r#"println!("cargo:rerun-if-env-changed={embedded_token}");"#);
    let shorter_token = ["J2", "K_DECODE"].concat();
    let tokens = j2k_env_tokens(&source);
    assert!(tokens.contains(&embedded_token));
    assert!(!tokens.contains(&shorter_token));
}

#[test]
fn normalized_match_helpers_ignore_whitespace_only_drift() {
    let source = "fn example(\n    arg: u32,\n) {\n    call(arg);\n}";
    assert!(contains_normalized(source, "fn example( arg: u32, )"));
    assert_contains_all_normalized("sample source", source, &["fn example( arg: u32, )"]);
    assert_not_contains_all_normalized("sample source", source, &["fn different( arg: u32, )"]);
}

#[test]
fn file_pattern_runner_checks_files_and_rejects_empty_pattern_rows() {
    let missing_marker = ["definitely_missing", "_repo_lint_helper_marker"].concat();
    let missing_fn = ["fn definitely_missing", "_repo_lint_helper_marker()"].concat();
    let helper_source = include_str!("mod.rs");
    assert_pattern_checks(
        &[PatternCheck::new("repo lint helper source", helper_source)
            .required(&["pub(crate) struct PatternCheck"])
            .forbidden(&[missing_marker.as_str()])
            .normalized_required(&[
                "fn file_pattern_runner_checks_files_and_rejects_empty_pattern_rows()",
            ])
            .normalized_forbidden(&[missing_fn.as_str()])],
    );

    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("xtask/tests/repo_lint_support/mod.rs")
                .named("repo lint helper module")
                .required(&["pub(crate) struct FilePatternCheck"])
                .forbidden(&[missing_marker.as_str()])
                .normalized_required(&[
                    "fn file_pattern_runner_checks_files_and_rejects_empty_pattern_rows()",
                ])
                .normalized_forbidden(&[missing_fn.as_str()]),
        ],
    );

    assert_rust_source_scan_checks(
        repo_root(),
        &[RustSourceScanCheck::new(
            "repo lint helper source scan",
            &["xtask/tests/repo_lint_support"],
        )
        .forbidden(&[missing_marker.as_str()])],
    );

    assert!(!PatternCheck::new("empty source-pattern row", helper_source).has_patterns());
    assert!(!RustSourceScanCheck::new("empty rust-source scan", &[]).has_scan_scope());
    assert!(!FilePatternCheck::new("xtask/tests/repo_lint_support/mod.rs").has_patterns());
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
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_' | '/')))
        .filter(|token| {
            Path::new(token)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("sh"))
                && token.contains('/')
        })
        .filter(|token| !token.starts_with("http://") && !token.starts_with("https://"))
        .map(str::to_string)
        .collect()
}

pub(crate) fn rust_include_paths(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    for marker in ["include_bytes!(\"", "include_str!(\""] {
        let mut rest = source;
        while let Some(start) = rest.find(marker) {
            let after_marker = &rest[start + marker.len()..];
            let Some(end) = after_marker.find('"') else {
                break;
            };
            out.push(after_marker[..end].to_string());
            rest = &after_marker[end + 1..];
        }
    }
    out
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
