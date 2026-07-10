// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs, path::Path};

use super::{repo_root, repo_text_files, rust_sources};

const REVIEWED_SOURCE_ALLOWS: &[(&str, &str)] = &[
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
        "clippy::manual_div_ceil",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
        "clippy::too_many_arguments",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
        "clippy::too_many_lines",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "clippy::manual_div_ceil",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "clippy::manual_is_multiple_of",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "clippy::too_many_arguments",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "static_mut_refs",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_idwt/simt/src/main.rs",
        "static_mut_refs",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::cast_possible_truncation",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::cast_sign_loss",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::many_single_char_names",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::too_many_arguments",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_simt_prelude.rs",
        "dead_code",
    ),
];

const REVIEWED_WILDCARD_EXPORT_FILES: &[&str] = &[
    "crates/j2k-test-support/src/jpeg_fixtures.rs",
    "crates/j2k-test-support/src/lib.rs",
];

const REVIEWED_DEVICE_INCLUDE_FILES: &[&str] = &[
    "crates/j2k-cuda-runtime/src/cuda_oxide_copy_u8/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_decode/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_decode_store/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_dequantize/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_idwt/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_decode/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_transcode/simt/src/main.rs",
];

const NEVER_EXPECT_LINTS: &[&str] = &[
    "clippy::undocumented_unsafe_blocks",
    "clippy::uninit_assumed_init",
    "clippy::uninit_vec",
    "invalid_value",
    "unsafe_op_in_unsafe_fn",
];

// This is a ceiling, not a required inventory: deleting an override does not
// require replacing it. Adding or moving one requires an explicit policy edit.
const REVIEWED_MANIFEST_ALLOWS: &[(&str, &str)] = &[
    ("Cargo.toml", "missing_errors_doc"),
    ("Cargo.toml", "missing_panics_doc"),
    ("Cargo.toml", "module_name_repetitions"),
    ("Cargo.toml", "must_use_candidate"),
    ("crates/j2k-core/Cargo.toml", "unsafe_code"),
    ("crates/j2k/Cargo.toml", "missing_errors_doc"),
    ("crates/j2k/Cargo.toml", "module_name_repetitions"),
    ("crates/j2k/Cargo.toml", "must_use_candidate"),
    ("crates/j2k-compare/Cargo.toml", "missing_errors_doc"),
    ("crates/j2k-compare/Cargo.toml", "must_use_candidate"),
    ("crates/j2k-cuda-runtime/Cargo.toml", "missing_errors_doc"),
    ("crates/j2k-cuda-runtime/Cargo.toml", "must_use_candidate"),
    ("crates/j2k-cuda-runtime/Cargo.toml", "unsafe_code"),
    ("crates/j2k-cuda/Cargo.toml", "missing_errors_doc"),
    ("crates/j2k-cuda/Cargo.toml", "must_use_candidate"),
    ("crates/j2k-cuda/Cargo.toml", "unsafe_code"),
    ("crates/j2k-jpeg-cuda/Cargo.toml", "missing_errors_doc"),
    ("crates/j2k-jpeg-cuda/Cargo.toml", "must_use_candidate"),
    ("crates/j2k-jpeg-cuda/Cargo.toml", "unsafe_code"),
    ("crates/j2k-jpeg-metal/Cargo.toml", "missing_errors_doc"),
    ("crates/j2k-jpeg-metal/Cargo.toml", "missing_panics_doc"),
    ("crates/j2k-jpeg-metal/Cargo.toml", "must_use_candidate"),
    ("crates/j2k-jpeg-metal/Cargo.toml", "unsafe_code"),
    ("crates/j2k-jpeg/Cargo.toml", "similar_names"),
    ("crates/j2k-jpeg/Cargo.toml", "unsafe_code"),
    ("crates/j2k-metal-support/Cargo.toml", "unsafe_code"),
    ("crates/j2k-metal/Cargo.toml", "missing_errors_doc"),
    ("crates/j2k-metal/Cargo.toml", "must_use_candidate"),
    ("crates/j2k-metal/Cargo.toml", "unsafe_code"),
    (
        "crates/j2k-test-support/Cargo.toml",
        "module_name_repetitions",
    ),
    ("crates/j2k-test-support/Cargo.toml", "must_use_candidate"),
    (
        "crates/j2k-transcode-test-support/Cargo.toml",
        "missing_errors_doc",
    ),
    (
        "crates/j2k-transcode-test-support/Cargo.toml",
        "missing_panics_doc",
    ),
    (
        "crates/j2k-transcode-test-support/Cargo.toml",
        "module_name_repetitions",
    ),
    (
        "crates/j2k-transcode-test-support/Cargo.toml",
        "must_use_candidate",
    ),
    ("crates/j2k-transcode/Cargo.toml", "missing_errors_doc"),
    ("crates/j2k-transcode/Cargo.toml", "missing_panics_doc"),
    ("crates/j2k-transcode/Cargo.toml", "module_name_repetitions"),
    ("crates/j2k-transcode/Cargo.toml", "must_use_candidate"),
    ("crates/j2k-transcode-cuda/Cargo.toml", "missing_errors_doc"),
    ("crates/j2k-transcode-cuda/Cargo.toml", "missing_panics_doc"),
    (
        "crates/j2k-transcode-cuda/Cargo.toml",
        "module_name_repetitions",
    ),
    ("crates/j2k-transcode-cuda/Cargo.toml", "must_use_candidate"),
    (
        "crates/j2k-transcode-metal/Cargo.toml",
        "missing_errors_doc",
    ),
    (
        "crates/j2k-transcode-metal/Cargo.toml",
        "missing_panics_doc",
    ),
    (
        "crates/j2k-transcode-metal/Cargo.toml",
        "module_name_repetitions",
    ),
    (
        "crates/j2k-transcode-metal/Cargo.toml",
        "must_use_candidate",
    ),
    ("crates/j2k-transcode-metal/Cargo.toml", "unsafe_code"),
];

#[test]
fn source_lint_suppressions_stay_in_reviewed_device_generation_scopes() {
    let root = repo_root();
    let reviewed = REVIEWED_SOURCE_ALLOWS
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut sources = rust_sources(&root.join("crates"));
    sources.extend(rust_sources(&root.join("xtask")));
    sources.sort();

    let mut unreviewed = Vec::new();
    let mut file_expectations = Vec::new();
    let mut dangerous_expectations = Vec::new();
    let mut unexplained_expectations = Vec::new();
    for path in sources {
        let relative = relative_path(root, &path);
        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let lines = source.lines().collect::<Vec<_>>();
        for (index, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            let direct_allow = trimmed.starts_with("#[allow(") || trimmed.starts_with("#![allow(");
            let direct_expect =
                trimmed.starts_with("#[expect(") || trimmed.starts_with("#![expect(");
            let conditional_attribute =
                trimmed.starts_with("#[cfg_attr(") || trimmed.starts_with("#![cfg_attr(");
            let block = suppression_attribute_block(&lines, index);
            let contains_allow = direct_allow
                || block
                    .as_deref()
                    .is_some_and(|attribute| attribute.contains("allow("));

            if contains_allow {
                let attribute = block.as_deref().unwrap_or(trimmed);
                assert!(
                    attribute.contains("reason ="),
                    "reviewed source allowance {relative}:{} must state its device-specific reason",
                    index + 1
                );
                let attribute_lints = source_allow_lints(attribute);
                assert!(
                    !attribute_lints.is_empty(),
                    "source allowance {relative}:{} must name at least one lint",
                    index + 1
                );
                for lint in attribute_lints {
                    if !reviewed.contains(&(relative.as_str(), lint)) {
                        unreviewed.push(format!("{relative}:{} `{lint}`", index + 1));
                    }
                }
            }

            let contains_expect = direct_expect
                || (conditional_attribute
                    && block
                        .as_deref()
                        .is_some_and(|attribute| attribute.contains("expect(")));
            if contains_expect {
                let attribute = block.as_deref().unwrap_or(trimmed);
                if !attribute.contains("reason =") {
                    unexplained_expectations.push(format!("{relative}:{}", index + 1));
                }
                for lint in NEVER_EXPECT_LINTS {
                    if attribute.contains(lint) {
                        dangerous_expectations.push(format!("{relative}:{} `{lint}`", index + 1));
                    }
                }
            }

            let direct_file_expect = trimmed.starts_with("#![expect(");
            let conditional_file_expect = trimmed.starts_with("#![cfg_attr(")
                && block
                    .as_deref()
                    .is_some_and(|attribute| attribute.contains("expect("));
            if direct_file_expect || conditional_file_expect {
                file_expectations.push(format!("{relative}:{}", index + 1));
            }
        }
    }

    assert!(
        unreviewed.is_empty(),
        "host or unreviewed source lint allowances are forbidden: {unreviewed:?}"
    );
    assert!(
        file_expectations.is_empty(),
        "file-level lint expectations hide future findings; localize them to items: {file_expectations:?}"
    );
    assert!(
        unexplained_expectations.is_empty(),
        "lint expectations must explain the preserved contract or boundary: {unexplained_expectations:?}"
    );
    assert!(
        dangerous_expectations.is_empty(),
        "memory-safety lint expectations are forbidden; fix the unsafe boundary: {dangerous_expectations:?}"
    );
}

#[test]
fn manifest_allowances_stay_inside_the_reviewed_ceiling() {
    let root = repo_root();
    let reviewed = REVIEWED_MANIFEST_ALLOWS
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut unreviewed = Vec::new();

    for path in repo_text_files(root).into_iter().filter(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == "Cargo.toml")
    }) {
        let relative = relative_path(root, &path);
        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        for (line_index, line) in source.lines().enumerate() {
            let Some((raw_key, raw_value)) = line.split_once('=') else {
                continue;
            };
            let key = raw_key.trim();
            let compact_value = raw_value.split_whitespace().collect::<String>();
            let is_allow =
                compact_value == "\"allow\"" || compact_value.contains("level=\"allow\"");
            if is_allow && !reviewed.contains(&(relative.as_str(), key)) {
                unreviewed.push(format!("{relative}:{} `{key}`", line_index + 1));
            }
        }
    }

    assert!(
        unreviewed.is_empty(),
        "manifest lint allowances outside the reviewed API/unsafe ceiling are forbidden: {unreviewed:?}"
    );
}

#[test]
fn production_includes_and_wildcard_exports_stay_in_reviewed_scopes() {
    let root = repo_root();
    let reviewed_includes = REVIEWED_DEVICE_INCLUDE_FILES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let reviewed_wildcards = REVIEWED_WILDCARD_EXPORT_FILES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut sources = rust_sources(&root.join("crates"));
    sources.extend(rust_sources(&root.join("xtask")));
    sources.sort();

    let mut unreviewed_includes = Vec::new();
    let mut unreviewed_wildcards = Vec::new();
    for path in sources {
        let relative = relative_path(root, &path);
        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let lines = source.lines().collect::<Vec<_>>();
        for (line_index, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("include!(")
                && (trimmed != "include!(\"../../../cuda_oxide_simt_prelude.rs\");"
                    || !reviewed_includes.contains(relative.as_str()))
            {
                unreviewed_includes.push(format!("{relative}:{}", line_index + 1));
            }

            let public_use =
                trimmed.starts_with("pub use ") || trimmed.starts_with("pub(crate) use ");
            if public_use {
                let statement = statement_block(&lines, line_index);
                if statement.contains('*') && !reviewed_wildcards.contains(relative.as_str()) {
                    unreviewed_wildcards.push(format!("{relative}:{}", line_index + 1));
                }
            }
        }
    }

    assert!(
        unreviewed_includes.is_empty(),
        "host-production include seams are forbidden: {unreviewed_includes:?}"
    );
    assert!(
        unreviewed_wildcards.is_empty(),
        "unreviewed production wildcard re-exports are forbidden: {unreviewed_wildcards:?}"
    );
}

fn attribute_block(lines: &[&str], start: usize) -> String {
    let mut block = String::new();
    for line in lines.iter().skip(start).take(32) {
        block.push_str(line);
        block.push('\n');
        if line.trim_end().ends_with(")]") {
            break;
        }
    }
    block
}

fn suppression_attribute_block(lines: &[&str], start: usize) -> Option<String> {
    let trimmed = lines.get(start)?.trim_start();
    let suppression_attribute = trimmed.starts_with("#[allow(")
        || trimmed.starts_with("#![allow(")
        || trimmed.starts_with("#[expect(")
        || trimmed.starts_with("#![expect(")
        || trimmed.starts_with("#[cfg_attr(")
        || trimmed.starts_with("#![cfg_attr(");
    suppression_attribute.then(|| attribute_block(lines, start))
}

fn source_allow_lints(attribute: &str) -> Vec<&str> {
    let Some((_, after_allow)) = attribute.split_once("allow(") else {
        return Vec::new();
    };
    let Some((lint_list, _)) = after_allow.split_once("reason =") else {
        return Vec::new();
    };
    lint_list
        .split(',')
        .map(str::trim)
        .filter(|lint| !lint.is_empty())
        .collect()
}

#[test]
fn suppression_attribute_block_captures_multiline_expect_reasons() {
    let lines = [
        "#[expect(",
        "    dead_code,",
        "    reason = \"shared target-specific fixture helpers\"",
        ")]",
        "mod fixture;",
    ];
    let attribute = suppression_attribute_block(&lines, 0).expect("lint attribute");
    assert!(attribute.contains("dead_code"));
    assert!(attribute.contains("reason ="));
    assert!(suppression_attribute_block(&lines, 4).is_none());
}

#[test]
fn source_allow_lints_extracts_the_exact_registered_ceiling() {
    let attribute = "#![allow(\n    clippy::cast_possible_truncation,\n    clippy::cast_sign_loss,\n    reason = \"bounded device ABI narrowing\"\n)]";
    assert_eq!(
        source_allow_lints(attribute),
        ["clippy::cast_possible_truncation", "clippy::cast_sign_loss"]
    );
    assert!(source_allow_lints("#[allow(dead_code)]").is_empty());
}

fn statement_block(lines: &[&str], start: usize) -> String {
    let mut block = String::new();
    for line in lines.iter().skip(start).take(32) {
        block.push_str(line);
        block.push('\n');
        if line.contains(';') {
            break;
        }
    }
    block
}

#[test]
fn statement_block_captures_multiline_public_globs() {
    let lines = [
        "pub use fixtures::{",
        "    Builder,",
        "    *,",
        "};",
        "fn unrelated() {}",
    ];
    let statement = statement_block(&lines, 0);
    assert!(statement.contains('*'));
    assert!(!statement.contains("unrelated"));
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
