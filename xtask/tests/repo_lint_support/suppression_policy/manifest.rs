// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs};

use super::relative_path;
use crate::repo_lint_support::{repo_root, repo_text_files};

// Safety suppressions are an exact inventory. Removing one from a manifest must
// remove it here too, so stale policy entries cannot silently re-authorize it.
const REVIEWED_SAFETY_ALLOWS: &[(&str, &str, &str)] = &[
    ("crates/j2k-core/Cargo.toml", "rust", "unsafe_code"),
    ("crates/j2k-cuda-runtime/Cargo.toml", "rust", "unsafe_code"),
    ("crates/j2k-cuda/Cargo.toml", "rust", "unsafe_code"),
    ("crates/j2k-jpeg-cuda/Cargo.toml", "rust", "unsafe_code"),
    ("crates/j2k-jpeg-metal/Cargo.toml", "rust", "unsafe_code"),
    ("crates/j2k-jpeg/Cargo.toml", "rust", "unsafe_code"),
    ("crates/j2k-metal-support/Cargo.toml", "rust", "unsafe_code"),
    ("crates/j2k-metal/Cargo.toml", "rust", "unsafe_code"),
];

// Non-safety API/documentation suppressions remain a ceiling: deleting one is
// progress and does not require a replacement. Adding or moving one requires a
// deliberate policy edit.
const REVIEWED_ALLOW_CEILING: &[(&str, &str, &str)] = &[
    ("Cargo.toml", "clippy", "missing_errors_doc"),
    ("Cargo.toml", "clippy", "missing_panics_doc"),
    ("Cargo.toml", "clippy", "module_name_repetitions"),
    ("Cargo.toml", "clippy", "must_use_candidate"),
    ("crates/j2k/Cargo.toml", "clippy", "missing_errors_doc"),
    ("crates/j2k/Cargo.toml", "clippy", "module_name_repetitions"),
    ("crates/j2k/Cargo.toml", "clippy", "must_use_candidate"),
    (
        "crates/j2k-compare/Cargo.toml",
        "clippy",
        "missing_errors_doc",
    ),
    (
        "crates/j2k-compare/Cargo.toml",
        "clippy",
        "must_use_candidate",
    ),
    (
        "crates/j2k-cuda-runtime/Cargo.toml",
        "clippy",
        "missing_errors_doc",
    ),
    (
        "crates/j2k-cuda-runtime/Cargo.toml",
        "clippy",
        "must_use_candidate",
    ),
    ("crates/j2k-cuda/Cargo.toml", "clippy", "missing_errors_doc"),
    ("crates/j2k-cuda/Cargo.toml", "clippy", "must_use_candidate"),
    (
        "crates/j2k-jpeg-cuda/Cargo.toml",
        "clippy",
        "missing_errors_doc",
    ),
    (
        "crates/j2k-jpeg-cuda/Cargo.toml",
        "clippy",
        "must_use_candidate",
    ),
    (
        "crates/j2k-jpeg-metal/Cargo.toml",
        "clippy",
        "missing_errors_doc",
    ),
    (
        "crates/j2k-jpeg-metal/Cargo.toml",
        "clippy",
        "missing_panics_doc",
    ),
    (
        "crates/j2k-jpeg-metal/Cargo.toml",
        "clippy",
        "must_use_candidate",
    ),
    ("crates/j2k-jpeg/Cargo.toml", "clippy", "similar_names"),
    (
        "crates/j2k-metal/Cargo.toml",
        "clippy",
        "missing_errors_doc",
    ),
    (
        "crates/j2k-metal/Cargo.toml",
        "clippy",
        "must_use_candidate",
    ),
    (
        "crates/j2k-test-support/Cargo.toml",
        "clippy",
        "module_name_repetitions",
    ),
    (
        "crates/j2k-test-support/Cargo.toml",
        "clippy",
        "must_use_candidate",
    ),
    (
        "crates/j2k-transcode-test-support/Cargo.toml",
        "clippy",
        "missing_errors_doc",
    ),
    (
        "crates/j2k-transcode-test-support/Cargo.toml",
        "clippy",
        "missing_panics_doc",
    ),
    (
        "crates/j2k-transcode-test-support/Cargo.toml",
        "clippy",
        "module_name_repetitions",
    ),
    (
        "crates/j2k-transcode-test-support/Cargo.toml",
        "clippy",
        "must_use_candidate",
    ),
    (
        "crates/j2k-transcode/Cargo.toml",
        "clippy",
        "missing_errors_doc",
    ),
    (
        "crates/j2k-transcode/Cargo.toml",
        "clippy",
        "missing_panics_doc",
    ),
    (
        "crates/j2k-transcode/Cargo.toml",
        "clippy",
        "module_name_repetitions",
    ),
    (
        "crates/j2k-transcode/Cargo.toml",
        "clippy",
        "must_use_candidate",
    ),
    (
        "crates/j2k-transcode-cuda/Cargo.toml",
        "clippy",
        "missing_errors_doc",
    ),
    (
        "crates/j2k-transcode-cuda/Cargo.toml",
        "clippy",
        "missing_panics_doc",
    ),
    (
        "crates/j2k-transcode-cuda/Cargo.toml",
        "clippy",
        "module_name_repetitions",
    ),
    (
        "crates/j2k-transcode-cuda/Cargo.toml",
        "clippy",
        "must_use_candidate",
    ),
    (
        "crates/j2k-transcode-metal/Cargo.toml",
        "clippy",
        "missing_errors_doc",
    ),
    (
        "crates/j2k-transcode-metal/Cargo.toml",
        "clippy",
        "missing_panics_doc",
    ),
    (
        "crates/j2k-transcode-metal/Cargo.toml",
        "clippy",
        "module_name_repetitions",
    ),
    (
        "crates/j2k-transcode-metal/Cargo.toml",
        "clippy",
        "must_use_candidate",
    ),
];

const SAFETY_LINTS: &[(&str, &str)] = &[
    ("clippy", "undocumented_unsafe_blocks"),
    ("clippy", "uninit_assumed_init"),
    ("clippy", "uninit_vec"),
    ("rust", "invalid_value"),
    ("rust", "static_mut_refs"),
    ("rust", "unsafe_code"),
    ("rust", "unsafe_op_in_unsafe_fn"),
];

#[test]
fn allowances_stay_inside_the_reviewed_inventory() {
    let root = repo_root();
    let reviewed_safety = REVIEWED_SAFETY_ALLOWS
        .iter()
        .map(|&(path, tool, lint)| (path.to_owned(), tool.to_owned(), lint.to_owned()))
        .collect::<BTreeSet<_>>();
    let reviewed_ceiling = REVIEWED_ALLOW_CEILING
        .iter()
        .map(|&(path, tool, lint)| (path.to_owned(), tool.to_owned(), lint.to_owned()))
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();

    for path in repo_text_files(root).into_iter().filter(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == "Cargo.toml")
    }) {
        let relative = relative_path(root, &path);
        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let allowances = allow_lints(&source)
            .unwrap_or_else(|error| panic!("parse {relative} as TOML: {error}"));
        for (tool, lint) in allowances {
            actual.insert((relative.clone(), tool, lint));
        }
    }

    let actual_safety = actual
        .iter()
        .filter(|(_, tool, lint)| is_safety_lint(tool, lint))
        .cloned()
        .collect::<BTreeSet<_>>();
    let unexpected_safety = actual_safety
        .difference(&reviewed_safety)
        .cloned()
        .collect::<Vec<_>>();
    let missing_safety = reviewed_safety
        .difference(&actual_safety)
        .cloned()
        .collect::<Vec<_>>();
    let unreviewed_non_safety = actual
        .difference(&actual_safety)
        .filter(|allowance| !reviewed_ceiling.contains(*allowance))
        .cloned()
        .collect::<Vec<_>>();

    assert!(
        unexpected_safety.is_empty() && missing_safety.is_empty(),
        "manifest safety allowances must match the reviewed inventory exactly; unexpected: {unexpected_safety:?}; missing: {missing_safety:?}"
    );
    assert!(
        unreviewed_non_safety.is_empty(),
        "manifest lint allowances outside the reviewed API/documentation ceiling are forbidden: {unreviewed_non_safety:?}"
    );
}

fn allow_lints(source: &str) -> Result<BTreeSet<(String, String)>, toml::de::Error> {
    let document = toml::from_str::<toml::Value>(source)?;
    let mut allowances = BTreeSet::new();
    if let Some(lints) = document.get("lints") {
        collect_allow_lints(lints, &mut allowances);
    }
    if let Some(lints) = document
        .get("workspace")
        .and_then(|workspace| workspace.get("lints"))
    {
        collect_allow_lints(lints, &mut allowances);
    }
    Ok(allowances)
}

fn collect_allow_lints(lints: &toml::Value, allowances: &mut BTreeSet<(String, String)>) {
    let Some(tools) = lints.as_table() else {
        return;
    };
    for (tool, lint_table) in tools {
        let Some(lint_table) = lint_table.as_table() else {
            continue;
        };
        for (lint, configuration) in lint_table {
            if lint_level(configuration) == Some("allow") {
                allowances.insert((tool.clone(), lint.clone()));
            }
        }
    }
}

fn lint_level(configuration: &toml::Value) -> Option<&str> {
    configuration.as_str().or_else(|| {
        configuration
            .as_table()
            .and_then(|table| table.get("level"))
            .and_then(toml::Value::as_str)
    })
}

fn is_safety_lint(tool: &str, lint: &str) -> bool {
    SAFETY_LINTS.contains(&(tool, lint))
}

#[test]
fn parser_handles_cargo_toml_forms_and_comments() {
    let source = r#"
[lints.rust]
unsafe_code = 'allow' # a single-quoted level with a trailing comment
unused_variables = "warn" # level = "allow" is comment text, not configuration
invalid_value = { priority = 0, level = "allow" }

[lints.clippy]
uninit_vec = { level = 'allow', priority = 0 }

[workspace.lints.clippy]
missing_errors_doc = "allow" # a double-quoted level with a comment
"#;

    assert_eq!(
        allow_lints(source).expect("valid Cargo TOML"),
        BTreeSet::from([
            ("clippy".to_owned(), "missing_errors_doc".to_owned()),
            ("clippy".to_owned(), "uninit_vec".to_owned()),
            ("rust".to_owned(), "invalid_value".to_owned()),
            ("rust".to_owned(), "unsafe_code".to_owned()),
        ])
    );
}
