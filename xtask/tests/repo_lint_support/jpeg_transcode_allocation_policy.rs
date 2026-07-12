// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation integrity for CPU JPEG-to-HTJ2K staging and reference storage.

use super::rust_function_policy::FunctionCalls;
use super::{assert_pattern_checks, contains_normalized, PatternCheck};

mod source;
use self::source::JpegTranscodeSources;

mod dct_transform_policy;
mod live_budget_policy;
mod metrics_policy;
mod progressive_policy;
mod workspace_policy;

fn calls(sources: &JpegTranscodeSources, label: &str, function: &str) -> FunctionCalls {
    FunctionCalls::parse_many(label, &sources.production(), function)
}

#[test]
fn jpeg_transcode_reference_and_group_storage_remains_fallible_and_bounded() {
    let sources = JpegTranscodeSources::read();
    let integer_samples = calls(
        &sources,
        "integer reference sample storage",
        "idct_component_samples_i32",
    );
    integer_samples.assert_ordered(
        "integer reference sample allocation",
        &[
            "validate_component_block_grid",
            "try_vec_filled",
            "checked_product",
        ],
    );
    integer_samples.assert_propagated(
        "integer reference sample allocation",
        &["try_vec_filled", "checked_product"],
    );
    for (function, operation) in [
        ("checked_product", "checked_mul"),
        ("checked_sum", "checked_add"),
    ] {
        calls(&sources, "integer reference checked arithmetic", function).assert_contains(
            "integer reference overflow mapping",
            &[operation, "ok_or_else"],
        );
    }
    for function in ["rounded_wavelet_i32", "rounded_wavelet97_i32"] {
        calls(&sources, "float reference rounding output", function).assert_ordered(
            "float reference exact output reservation",
            &["try_vec_with_capacity", "append_rounded_i32"],
        );
    }
    calls(
        &sources,
        "float reference coefficient count",
        "wavelet_coefficient_count",
    )
    .assert_contains(
        "float reference coefficient overflow mapping",
        &["checked_add", "ok_or_else"],
    );
    for function in ["f64_to_f32", "i32_to_f32"] {
        calls(&sources, "reference coefficient conversion", function).assert_ordered(
            "fallible reference coefficient storage",
            &["try_vec_with_capacity", "extend"],
        );
    }
}

#[test]
fn jpeg_transcode_group_reference_storage_remains_fallible_and_bounded() {
    let sources = JpegTranscodeSources::read();
    let component_count = calls(
        &sources,
        "JPEG batch component count",
        "batch_component_count",
    );
    component_count.assert_ordered(
        "JPEG batch checked component count",
        &["try_fold", "checked_add"],
    );
    let group_budget = calls(
        &sources,
        "JPEG batch group workspace",
        "validate_group_workspace",
    );
    group_budget.assert_count(
        "JPEG group outer and reference bytes",
        "checked_allocation_bytes",
        2,
    );
    group_budget.assert_contains(
        "JPEG aggregate group bytes",
        &["checked_add_allocation_bytes"],
    );
    calls(&sources, "JPEG next group length", "next_group_len")
        .assert_contains("JPEG group growth overflow mapping", &["checked_add"]);

    for function in ["batch_component_groups", "float97_batch_component_groups"] {
        let grouping = calls(&sources, "JPEG component grouping", function);
        grouping.assert_ordered(
            "JPEG grouping preflight before reference storage",
            &[
                "batch_component_count",
                "validate_group_workspace",
                "try_vec_with_capacity",
            ],
        );
        grouping.assert_count("JPEG fallible group vectors", "try_vec_with_capacity", 2);
        grouping.assert_propagated(
            "JPEG fallible group reference storage",
            &[
                "batch_component_count",
                "validate_group_workspace",
                "try_vec_with_capacity",
                "next_group_len",
                "try_vec_reserve_len",
            ],
        );
    }

    let production = sources.combined();
    assert!(
        contains_normalized(
            &production,
            "checked_allocation_bytes::<Vec<BatchComponentRef>>(component_count)?",
        ) && contains_normalized(
            &production,
            "checked_allocation_bytes::<BatchComponentRef>(component_count)?",
        ) && contains_normalized(
            &production,
            "try_vec_reserve_len(group, next_group_len(group.len())?)?",
        ) && contains_normalized(
            &production,
            "left.checked_mul(right).ok_or_else(cap_overflow)",
        ) && contains_normalized(
            &production,
            "count.checked_add(band_len).ok_or_else(cap_overflow)?",
        ),
        "JPEG reference/group storage must retain checked sizing and fallible growth"
    );
}

#[test]
fn jpeg_transcode_allocation_policy_stays_focused() {
    assert!(
        include_str!("jpeg_transcode_allocation_policy.rs")
            .lines()
            .count()
            < 250,
        "JPEG transcode allocation policy must stay focused"
    );
    assert!(
        include_str!("jpeg_transcode_allocation_policy/source.rs")
            .lines()
            .count()
            < 75,
        "JPEG transcode source inventory must stay focused"
    );
    assert!(
        include_str!("jpeg_transcode_allocation_policy/workspace_policy.rs")
            .lines()
            .count()
            < 125,
        "JPEG transcode workspace policy must stay focused"
    );
}
