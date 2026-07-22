// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use crate::repo_lint_support::repo_root;
use syn::spanned::Spanned;

fn assert_top_level_function_below(
    relative: &str,
    source: &str,
    function_name: &str,
    max_lines: usize,
) {
    let syntax =
        syn::parse_file(source).unwrap_or_else(|error| panic!("parse {relative}: {error}"));
    let function = syntax
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == function_name => Some(function),
            _ => None,
        })
        .unwrap_or_else(|| panic!("find {function_name} in {relative}"));
    let span = function.span();
    let lines = span.end().line.saturating_sub(span.start().line) + 1;
    assert!(
        lines < max_lines,
        "{function_name} in {relative} must stay below {max_lines} lines, found {lines}"
    );
}

pub(super) fn assert_direct_executor_line_ratchets(root: &Path) {
    let source_root = root.join("crates/j2k-metal/src");
    for (relative, max_lines) in [
        ("compute/direct_commands.rs", 160),
        ("compute/direct_grayscale_execute.rs", 500),
        ("compute/direct_grayscale_execute/allocation.rs", 150),
        (
            "compute/direct_grayscale_execute/color_batch_completion.rs",
            125,
        ),
        ("compute/direct_grayscale_execute/single.rs", 350),
        ("compute/direct_grayscale_execute/component_plane.rs", 100),
        (
            "compute/direct_grayscale_execute/component_plane/execution.rs",
            450,
        ),
        (
            "compute/direct_grayscale_execute/component_plane/execution/final_plane.rs",
            150,
        ),
        ("compute/direct_stacked_batch/repeated_grayscale.rs", 100),
        (
            "compute/direct_stacked_batch/repeated_grayscale/execution.rs",
            600,
        ),
    ] {
        let path = source_root.join(relative);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source.as_str(), |(production, _)| production);
        assert!(
            production.lines().count() < max_lines,
            "{relative} must stay below its direct-executor line-count ratchet"
        );
    }
}

#[test]
fn direct_color_batch_execution_has_explicit_phases() {
    let source = fs::read_to_string(
        repo_root().join("crates/j2k-metal/src/compute/direct_grayscale_execute.rs"),
    )
    .expect("read Metal direct color batch execution");
    let completion =
        fs::read_to_string(repo_root().join(
            "crates/j2k-metal/src/compute/direct_grayscale_execute/color_batch_completion.rs",
        ))
        .expect("read Metal direct color batch completion");
    let (_, orchestrator) = source
        .split_once("pub(super) fn execute_direct_color_plan_batch_with_tier1_options")
        .expect("find direct color batch orchestrator");

    for phase in [
        "fn allocate_direct_color_batch_execution(",
        "fn encode_direct_color_batch_routes(",
    ] {
        assert!(source.contains(phase), "missing concrete phase {phase}");
    }
    for phase in [
        "fn complete_direct_color_batch_command(",
        "fn complete_split_direct_color_batch_command(",
        "fn retire_direct_color_batch_resources(",
    ] {
        assert!(completion.contains(phase), "missing concrete phase {phase}");
    }
    assert!(source.contains("mod color_batch_completion;"));
    assert!(source.contains("use self::color_batch_completion::{"));
    for delegated in [
        "allocate_direct_color_batch_execution(",
        "encode_direct_color_batch_routes(",
    ] {
        assert!(
            orchestrator.contains(delegated),
            "orchestrator must delegate {delegated}"
        );
    }
    for implementation_detail in [
        "checked_count_sum(",
        "direct_ht_job_count(",
        "try_encode_stacked_mct_rgb8_direct_color_batch(",
        "retire_direct_status_checks(",
        "recycle_scratch_buffers(",
    ] {
        assert!(
            !orchestrator.contains(implementation_detail),
            "orchestrator must not inline {implementation_detail}"
        );
    }

    let (_, allocation_tail) = source
        .split_once("fn allocate_direct_color_batch_execution(")
        .expect("find direct color allocation phase");
    let (allocation, _) = allocation_tail
        .split_once("\nfn ")
        .expect("isolate direct color allocation phase");
    for accounting in [
        "tier1_mode == DirectTier1Mode::Metal",
        "direct_ht_job_count(",
        ".flat_map(|plan| plan.component_plans.iter())",
        "J2K Metal direct color batch HT jobs",
    ] {
        assert!(
            allocation.contains(accounting),
            "allocation phase must preserve {accounting}"
        );
    }
    assert!(orchestrator.lines().count() < 45);
    assert!(!source.contains("clippy::too_many_lines"));
    assert!(!completion.contains("clippy::too_many_lines"));
    assert!(!completion.contains("use super::*;"));
    assert!(source.lines().count() < 500);
    assert!(completion.lines().count() < 125);
}

#[test]
fn direct_prepare_hotspots_have_focused_owners_and_short_orchestrators() {
    let source_root = repo_root().join("crates/j2k-metal/src/compute/direct_prepare");
    let sources = [
        ("ht.rs", 100),
        ("ht/referenced.rs", 250),
        ("ht/grouped.rs", 275),
        ("classic.rs", 40),
        ("classic/payload.rs", 200),
        ("classic/sub_band.rs", 225),
        ("classic/grouped.rs", 275),
    ]
    .into_iter()
    .map(|(relative, max_lines)| {
        let path = source_root.join(relative);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its direct-prepare line-count ratchet"
        );
        assert!(
            !source.lines().any(|line| line.trim() == "use super::*;"),
            "{relative} must use explicit imports"
        );
        assert!(
            !source.contains("clippy::too_many_lines"),
            "{relative} must keep orchestration functions focused"
        );
        (relative, source)
    })
    .collect::<std::collections::HashMap<_, _>>();

    let ht = &sources["ht.rs"];
    for ownership in [
        "mod grouped;",
        "mod referenced;",
        "use self::grouped::prepare_ht_sub_band_groups;",
        "use self::referenced::prepare_referenced_ht_sub_band;",
    ] {
        assert!(ht.contains(ownership), "ht.rs must declare {ownership}");
    }
    assert!(!ht.contains("fn prepare_referenced_ht_sub_band("));
    assert!(!ht.contains("fn prepare_ht_sub_band_group("));

    let classic = &sources["classic.rs"];
    for ownership in [
        "mod grouped;",
        "mod payload;",
        "mod sub_band;",
        "use self::grouped::{",
        "prepare_classic_sub_band_groups, prepare_sub_band_groups,",
        "use self::payload::{",
        "prepare_referenced_classic_sub_band, ReferencedClassicPayloadCursor,",
        "use self::sub_band::prepare_classic_sub_band;",
    ] {
        assert!(
            classic.contains(ownership),
            "classic.rs must declare {ownership}"
        );
    }
    for misplaced in [
        "struct ReferencedClassicPayloadCursor",
        "fn prepare_classic_sub_band_with_payloads(",
        "fn prepare_classic_sub_band_group(",
    ] {
        assert!(
            !classic.contains(misplaced),
            "classic.rs must not own {misplaced}"
        );
    }

    for (relative, function_name) in [
        ("ht/referenced.rs", "prepare_referenced_ht_sub_band"),
        ("ht/grouped.rs", "prepare_ht_sub_band_group"),
        (
            "classic/sub_band.rs",
            "prepare_classic_sub_band_with_payloads",
        ),
        ("classic/grouped.rs", "prepare_classic_sub_band_group"),
    ] {
        assert_top_level_function_below(relative, &sources[relative], function_name, 80);
    }
}
