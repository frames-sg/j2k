// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::rust_function_policy::FunctionCalls;
use super::JpegEncodeAllocationSources;

mod contracts;

fn calls(source_name: &str, source: &str, function_name: &str) -> FunctionCalls {
    FunctionCalls::parse(source_name, source, function_name)
}

fn assert_focused_modules(sources: &JpegEncodeAllocationSources) {
    assert!(
        include_str!("encoder_checks.rs").lines().count() < 230,
        "JPEG encode allocation policy must remain a focused module"
    );
    for (relative, source, max_lines) in [
        ("encoded_output.rs", sources.encoded_output.as_str(), 160),
        ("encoder/entropy.rs", sources.entropy.as_str(), 260),
        (
            "encoder/entropy/restart.rs",
            sources.entropy_restart.as_str(),
            420,
        ),
        (
            "encoder/entropy/workspace.rs",
            sources.entropy_workspace.as_str(),
            130,
        ),
        (
            "baseline_encode/allocation.rs",
            sources.encode_allocation.as_str(),
            320,
        ),
        ("baseline_encode/frame.rs", sources.frame.as_str(), 220),
        (
            "baseline_encode/orchestrate.rs",
            sources.orchestrate.as_str(),
            190,
        ),
        (
            "baseline_encode/orchestrate/batch.rs",
            sources.orchestrate_batch_owner.as_str(),
            190,
        ),
        (
            "baseline_encode/orchestrate/batch/group.rs",
            sources.orchestrate_batch_group.as_str(),
            130,
        ),
        (
            "baseline_encode/planning.rs",
            sources.planning_owner.as_str(),
            260,
        ),
        (
            "baseline_encode/planning/batch.rs",
            sources.planning_batch.as_str(),
            120,
        ),
        ("baseline_encode/types.rs", sources.types.as_str(), 300),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "JPEG encode allocation module {relative} exceeded its focused line-count ratchet"
        );
        assert!(
            !source.contains("include!("),
            "{relative} must be a real module"
        );
    }
}

fn assert_cpu_and_transcode_preflight_before_materialization(
    sources: &JpegEncodeAllocationSources,
) {
    let cpu = calls(
        "JPEG CPU encoder",
        &sources.encoder,
        "encode_jpeg_baseline_cpu",
    );
    let cpu_calls = [
        "checked_cpu_encode_capacity_plan",
        "component_planes",
        "component_plane_capacity_bytes",
        "checked_encode_host_live_bytes",
        "encode_entropy",
        "checked_jpeg_baseline_frame_capacity",
        "assemble_jpeg_baseline_frame",
    ];
    cpu.assert_ordered("JPEG CPU output lifecycle", &cpu_calls);
    cpu.assert_propagated("JPEG CPU fallible allocation path", &cpu_calls);

    let plan = calls(
        "JPEG CPU encoder",
        &sources.encoder_planning,
        "checked_cpu_encode_capacity_plan",
    );
    let plan_calls = [
        "jpeg_baseline_entropy_capacity_bytes",
        "entropy_host_workspace_bytes",
        "checked_cpu_encode_live_bytes",
        "cpu_owned_plane_capacity_limit",
    ];
    plan.assert_ordered("JPEG CPU aggregate capacity plan", &plan_calls);
    plan.assert_propagated("JPEG CPU aggregate capacity plan", &plan_calls);

    calls(
        "shared JPEG allocation owner",
        &sources.shared_allocation,
        "try_new_vec_with_live_budget",
    )
    .assert_contains(
        "JPEG CPU shared fallible plane allocation",
        &["try_reserve_for_len_with_live_budget_typed"],
    );
    calls(
        "shared JPEG allocation owner",
        &sources.shared_allocation,
        "try_reserve_for_len_with_live_budget_typed",
    )
    .assert_contains(
        "shared actual-capacity live-budget enforcement",
        &[
            "include_vec",
            "ensure_budget_bytes",
            "try_reserve_for_len_with_budget",
            "capacity",
        ],
    );
    calls(
        "JPEG CPU encoder",
        &sources.encoder_planning,
        "component_plane_capacity_bytes",
    )
    .assert_contains(
        "JPEG CPU retained plane capacity",
        &["capacity", "checked_encode_host_live_bytes"],
    );

    let transcode = calls(
        "JPEG DCT transcode encoder",
        &sources.transcode,
        "encode_baseline_dct_image",
    );
    let transcode_calls = [
        "validate_baseline_dct_image",
        "jpeg_baseline_entropy_capacity_bytes",
        "validate_dct_reemission_live_bytes",
        "encode_dct_entropy",
        "checked_jpeg_baseline_frame_capacity",
        "checked_encode_host_live_bytes",
        "assemble_jpeg_baseline_frame_with_quant_tables",
    ];
    transcode.assert_ordered("JPEG DCT transcode output lifecycle", &transcode_calls);
    transcode.assert_propagated("JPEG DCT transcode fallible output path", &transcode_calls);
}

pub(super) fn assert_policy(sources: &JpegEncodeAllocationSources) {
    assert_focused_modules(sources);
    assert_cpu_and_transcode_preflight_before_materialization(sources);
    super::entropy_checks::assert_policy(sources);
    contracts::assert_regressions_and_static_errors_remain_owned(sources);
}
