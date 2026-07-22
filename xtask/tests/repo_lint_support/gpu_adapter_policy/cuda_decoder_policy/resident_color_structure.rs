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
        let line_count = function
            .span()
            .end()
            .line
            .saturating_sub(function.span().start().line)
            .saturating_add(1);
        assert!(
            line_count <= MAX_FOCUSED_FUNCTION_LINES,
            "{relative}::{} spans {line_count} lines; focused functions must stay at or below {MAX_FOCUSED_FUNCTION_LINES}",
            function.sig.ident
        );
    }
}

fn assert_module_hygiene(source: &str) {
    assert!(!source.contains("clippy::too_many_lines"));
    assert!(!source.lines().any(|line| line.trim() == "use super::*;"));
    assert!(!source
        .lines()
        .any(|line| line.trim_start().starts_with("include!(")));
}

fn assert_explicit_module(relative: &str, source: &str) {
    assert_module_hygiene(source);
    assert_focused_functions(relative, source);
}

fn assert_named_function_focused(relative: &str, source: &str, name: &str) {
    let syntax =
        syn::parse_file(source).unwrap_or_else(|error| panic!("parse {relative}: {error}"));
    let function = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(function) => Some(function),
            _ => None,
        })
        .find(|function| function.sig.ident == name)
        .unwrap_or_else(|| panic!("{relative} must define {name}"));
    let line_count = function
        .span()
        .end()
        .line
        .saturating_sub(function.span().start().line)
        .saturating_add(1);
    assert!(
        line_count <= MAX_FOCUSED_FUNCTION_LINES,
        "{relative}::{name} spans {line_count} lines; focused functions must stay at or below {MAX_FOCUSED_FUNCTION_LINES}"
    );
}

#[test]
fn resident_color_batch_is_split_by_prepare_execute_and_complete_phases() {
    let shell_relative = "crates/j2k-cuda/src/decoder/color_batch/batch_execution.rs";
    let shell = read(shell_relative);
    for module in ["preparation", "execution", "completion"] {
        assert!(shell.contains(&format!("mod {module};")));
    }
    assert!(shell.contains("fn decode_color_cuda_resident_batch_surfaces_with_profile("));
    assert!(!shell.contains("fn prepare_color_cuda_resident_batch("));
    assert!(!shell.contains("fn enqueue_color_cuda_resident_batch("));
    assert!(!shell.contains("fn complete_color_cuda_resident_batch("));
    assert_explicit_module(shell_relative, &shell);

    for (module, owned_function) in [
        ("preparation", "fn prepare_color_cuda_resident_batch("),
        ("execution", "fn enqueue_color_cuda_resident_batch("),
        ("completion", "fn complete_color_cuda_resident_batch("),
    ] {
        let relative =
            format!("crates/j2k-cuda/src/decoder/color_batch/batch_execution/{module}.rs");
        let source = read(&relative);
        assert!(source.contains(owned_function));
        assert_explicit_module(&relative, &source);
    }
}

#[test]
fn referenced_color_planning_has_route_owned_short_tile_builders() {
    let shell_relative = "crates/j2k-cuda/src/decoder/plan/color_referenced.rs";
    let shell = read(shell_relative);
    for module in ["classic", "ht"] {
        assert!(shell.contains(&format!("mod {module};")));
    }
    for builder in [
        "fn build_cuda_htj2k_color_plans_from_referenced_with_profile(",
        "fn build_cuda_classic_color_plans_from_referenced_with_profile(",
    ] {
        assert!(!shell.contains(builder));
    }
    assert_explicit_module(shell_relative, &shell);

    for (module, batch_builder, tile_builder) in [
        (
            "ht",
            "fn build_cuda_htj2k_color_plans_from_referenced_with_profile(",
            "fn build_referenced_ht_color_tile(",
        ),
        (
            "classic",
            "fn build_cuda_classic_color_plans_from_referenced_with_profile(",
            "fn build_referenced_classic_color_tile(",
        ),
    ] {
        let relative = format!("crates/j2k-cuda/src/decoder/plan/color_referenced/{module}.rs");
        let source = read(&relative);
        assert!(source.contains(batch_builder));
        assert!(source.contains(tile_builder));
        assert_explicit_module(&relative, &source);
    }
}

#[test]
fn chunk_enqueue_has_selected_payload_target_and_kernel_phase_owners() {
    let shell_relative = "crates/j2k-cuda/src/decoder/resident/chunked_cleanup/enqueue.rs";
    let shell = read(shell_relative);
    for module in ["kernel", "materialize", "targets"] {
        assert!(shell.contains(&format!("mod {module};")));
    }
    assert!(shell.contains("fn enqueue_one_chunk("));
    for moved_function in [
        "fn select_chunk_jobs(",
        "fn materialize_chunk_payload(",
        "fn build_chunk_targets(",
        "fn enqueue_chunk_kernel(",
    ] {
        assert!(!shell.contains(moved_function));
    }
    assert_module_hygiene(&shell);
    assert_named_function_focused(shell_relative, &shell, "enqueue_one_chunk");

    for (module, owned_functions) in [
        (
            "materialize",
            &["fn select_chunk_jobs(", "fn materialize_chunk_payload("][..],
        ),
        ("targets", &["fn build_chunk_targets"][..]),
        ("kernel", &["fn enqueue_chunk_kernel("][..]),
    ] {
        let relative =
            format!("crates/j2k-cuda/src/decoder/resident/chunked_cleanup/enqueue/{module}.rs");
        let source = read(&relative);
        for owned_function in owned_functions {
            assert!(source.contains(owned_function));
        }
        assert!(!source.contains("struct EnqueueOneChunkState"));
        assert_explicit_module(&relative, &source);
    }
}
