// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::JpegEncodeAllocationSources;

pub(super) fn assert_regressions_and_static_errors_remain_owned(
    sources: &JpegEncodeAllocationSources,
) {
    let regressions = [
        sources.encoder_tests.as_str(),
        sources.encoded_output_tests.as_str(),
        sources.entropy_restart.as_str(),
        sources.encode_allocation_tests.as_str(),
        sources.adapter_tests.as_str(),
    ]
    .concat();
    assert!(
        sources
            .encoder_tests
            .contains("restart_one_rejects_cap_valid_geometry_before_sample_or_entropy_allocation"),
        "JPEG encoder test inventory must retain allocation regressions"
    );
    assert!(
        !sources
            .encoder
            .contains("restart_one_rejects_cap_valid_geometry_before_sample_or_entropy_allocation"),
        "JPEG encoder production inventory must exclude test-only regression sources"
    );
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
            .encoder_contract
            .contains("InternalInvariant {\n        /// Static invariant description.\n        reason: &'static str,"),
        "baseline encode static invariant errors must remain allocation-free"
    );
    let (encoded_prefix, _) = sources
        .encoder_contract
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
        !sources.encoder_contract.contains("Internal(String)")
            && !sources
                .encoder_contract
                .contains("JpegEncodeError::Internal("),
        "the unowned stringly JPEG encode error must not return"
    );
    for (label, source) in [
        ("CPU encoder", sources.encoder.as_str()),
        ("CPU encoder contract", sources.encoder_contract.as_str()),
        ("CPU encoder planning", sources.encoder_planning.as_str()),
        ("shared baseline entropy", sources.baseline_entropy.as_str()),
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
