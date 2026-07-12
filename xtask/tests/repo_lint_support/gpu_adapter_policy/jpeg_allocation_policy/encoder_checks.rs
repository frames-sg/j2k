// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::rust_function_policy::FunctionCalls;
use super::JpegEncodeAllocationSources;

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
        &sources.encoder,
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
        "JPEG CPU encoder",
        &sources.encoder,
        "try_vec_with_live_budget",
    )
    .assert_contains(
        "JPEG CPU actual plane capacity",
        &["try_reserve_exact", "capacity", "checked_add"],
    );
    calls(
        "JPEG CPU encoder",
        &sources.encoder,
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

fn assert_regressions_and_static_errors_remain_owned(sources: &JpegEncodeAllocationSources) {
    let regressions = [
        sources.encoder.as_str(),
        sources.encoded_output_tests.as_str(),
        sources.entropy_restart.as_str(),
        sources.encode_allocation_tests.as_str(),
        sources.adapter_tests.as_str(),
    ]
    .concat();
    for regression in [
        "restart_one_rejects_cap_valid_geometry_before_sample_or_entropy_allocation",
        "geometric_growth_counts_retained_and_replacement_storage",
        "allocator_reported_capacity_is_checked_after_reservation",
        "restart_chunk_preallocation_stops_before_the_next_writer_exceeds_the_cap",
        "gpu_runtime_params_and_returned_outer_metadata_share_the_group_cap",
        "second_gpu_source_group_keeps_prior_actual_frames_live",
        "gpu_batch_rejects_adapter_outer_capacity_above_the_tile_count",
        "gpu_batch_rejects_retained_frames_across_source_groups_before_submission",
    ] {
        assert!(
            regressions.contains(regression),
            "JPEG output allocation regression `{regression}` must remain explicit"
        );
    }

    assert!(
        sources
            .encoder
            .contains("InternalInvariant {\n        /// Static invariant description.\n        reason: &'static str,"),
        "baseline encode static invariant errors must remain allocation-free"
    );
    let (encoded_prefix, _) = sources
        .encoder
        .split_once("pub struct EncodedJpeg")
        .expect("EncodedJpeg declaration");
    let encoded_derive = encoded_prefix
        .rsplit("#[derive(")
        .next()
        .expect("EncodedJpeg derive");
    assert!(
        encoded_derive.starts_with("Debug, PartialEq, Eq)]"),
        "EncodedJpeg must remain move-only because its codestream can approach the shared frame cap"
    );
    assert!(
        !sources.encoder.contains("Internal(String)")
            && !sources.encoder.contains("JpegEncodeError::Internal("),
        "the unowned stringly JPEG encode error must not return"
    );
    for (label, source) in [
        ("CPU encoder", sources.encoder.as_str()),
        ("entropy orchestration", sources.entropy.as_str()),
        ("restart entropy", sources.entropy_restart.as_str()),
        ("entropy workspace", sources.entropy_workspace.as_str()),
        ("encode allocation", sources.encode_allocation.as_str()),
        ("frame assembly", sources.frame.as_str()),
        ("GPU orchestration", sources.orchestrate.as_str()),
        (
            "GPU batch orchestration",
            sources.orchestrate_batch.as_str(),
        ),
        ("GPU planning", sources.planning.as_str()),
    ] {
        assert!(
            !source.contains("JpegEncodeError::Internal("),
            "{label} must use allocation-free InternalInvariant for static diagnostics"
        );
    }
}

pub(super) fn assert_policy(sources: &JpegEncodeAllocationSources) {
    assert_focused_modules(sources);
    assert_cpu_and_transcode_preflight_before_materialization(sources);
    super::entropy_checks::assert_policy(sources);
    assert_regressions_and_static_errors_remain_owned(sources);
}
