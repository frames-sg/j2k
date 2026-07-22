// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use syn::{spanned::Spanned, Item};

use crate::repo_lint_support::repo_root;

const MAX_FOCUSED_FUNCTION_LINES: usize = 75;

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn assert_focused_functions(relative: &str, source: &str) {
    let syntax =
        syn::parse_file(source).unwrap_or_else(|error| panic!("parse {relative}: {error}"));
    for function in syntax.items.iter().filter_map(|item| match item {
        Item::Fn(function) => Some(function),
        _ => None,
    }) {
        let start = function.span().start().line;
        let end = function.span().end().line;
        let line_count = end.saturating_sub(start).saturating_add(1);
        assert!(
            line_count <= MAX_FOCUSED_FUNCTION_LINES,
            "{relative}::{} spans {line_count} lines; focused functions must stay at or below {MAX_FOCUSED_FUNCTION_LINES}",
            function.sig.ident
        );
    }
}

#[test]
fn native_color_batch_uses_focused_lifecycle_prepare_store_submission_owners() {
    let execution_relative = "crates/j2k-cuda/src/decoder/color_batch/native_batch/execution.rs";
    let prepare_relative = "crates/j2k-cuda/src/decoder/color_batch/native_batch/prepare.rs";
    let store_relative = "crates/j2k-cuda/src/decoder/color_batch/native_batch/store.rs";
    let submission_relative =
        "crates/j2k-cuda/src/decoder/color_batch/native_batch/store/submission.rs";
    let execution = read(execution_relative);
    let prepare = read(prepare_relative);
    let store = read(store_relative);
    let submission = read(submission_relative);

    for (relative, shell, module, forbidden_owner) in [
        (
            execution_relative,
            execution.as_str(),
            "lifecycle",
            "fn enqueue_native_color_entropy",
        ),
        (
            prepare_relative,
            prepare.as_str(),
            "input",
            "fn prepare_native_color_input",
        ),
        (
            store_relative,
            store.as_str(),
            "targets",
            "fn build_store_targets",
        ),
    ] {
        assert!(
            shell.contains(&format!("mod {module};")),
            "{relative} must declare the real {module} owner module"
        );
        assert!(
            !shell.contains(forbidden_owner),
            "{relative} must delegate {forbidden_owner}"
        );
    }
    for module in ["external", "owned"] {
        assert!(
            submission.contains(&format!("mod {module};")),
            "{submission_relative} must declare the real {module} submission owner"
        );
    }
    assert!(submission.contains("fn store_targets("));

    let owned_sources = [
        (execution_relative, execution),
        (
            "crates/j2k-cuda/src/decoder/color_batch/native_batch/execution/lifecycle.rs",
            read("crates/j2k-cuda/src/decoder/color_batch/native_batch/execution/lifecycle.rs"),
        ),
        (prepare_relative, prepare),
        (
            "crates/j2k-cuda/src/decoder/color_batch/native_batch/prepare/input.rs",
            read("crates/j2k-cuda/src/decoder/color_batch/native_batch/prepare/input.rs"),
        ),
        (store_relative, store),
        (
            "crates/j2k-cuda/src/decoder/color_batch/native_batch/store/targets.rs",
            read("crates/j2k-cuda/src/decoder/color_batch/native_batch/store/targets.rs"),
        ),
        (submission_relative, submission),
        (
            "crates/j2k-cuda/src/decoder/color_batch/native_batch/store/submission/external.rs",
            read(
                "crates/j2k-cuda/src/decoder/color_batch/native_batch/store/submission/external.rs",
            ),
        ),
        (
            "crates/j2k-cuda/src/decoder/color_batch/native_batch/store/submission/owned.rs",
            read("crates/j2k-cuda/src/decoder/color_batch/native_batch/store/submission/owned.rs"),
        ),
    ];
    for (relative, source) in &owned_sources {
        assert!(!source.contains("clippy::too_many_lines"));
        assert!(!source.lines().any(|line| line.trim() == "use super::*;"));
        assert!(!source
            .lines()
            .any(|line| line.trim_start().starts_with("include!(")));
        assert_focused_functions(relative, source);
    }

    assert!(owned_sources[1]
        .1
        .contains("fn enqueue_native_color_entropy"));
    assert!(owned_sources[3].1.contains("fn prepare_native_color_input"));
    assert!(owned_sources[5].1.contains("fn build_store_targets"));
    assert!(owned_sources[7].1.contains("fn enqueue_external_store"));
    assert!(owned_sources[8].1.contains("fn enqueue_owned_store"));
    assert!(owned_sources[8].1.contains("fn finish_owned_store"));
}
