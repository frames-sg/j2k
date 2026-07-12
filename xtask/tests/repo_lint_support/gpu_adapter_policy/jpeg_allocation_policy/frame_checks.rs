// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::rust_function_policy::FunctionCalls;
use super::JpegEncodeAllocationSources;

fn calls(source_name: &str, source: &str, function_name: &str) -> FunctionCalls {
    FunctionCalls::parse(source_name, source, function_name)
}

fn assert_capped_builder_is_fallible(sources: &JpegEncodeAllocationSources) {
    let initial = calls(
        "JPEG capped output builder",
        &sources.encoded_output,
        "try_with_capacity",
    );
    initial.assert_contains(
        "JPEG capped output initialization",
        &[
            "try_host_vec_with_capacity",
            "capacity",
            "ensure_capacity_within_limit",
            "memory_cap_error",
        ],
    );
    initial.assert_propagated(
        "JPEG capped output initialization",
        &["try_host_vec_with_capacity", "ensure_capacity_within_limit"],
    );

    let growth = calls(
        "JPEG capped output builder",
        &sources.encoded_output,
        "reserve_additional",
    );
    growth.assert_contains(
        "JPEG capped output growth",
        &[
            "checked_add",
            "memory_cap_error",
            "try_reserve_exact",
            "capacity",
            "ensure_capacity_within_limit",
        ],
    );
    growth.assert_propagated(
        "JPEG capped output growth",
        &["try_reserve_exact", "ensure_capacity_within_limit"],
    );
    assert!(
        sources
            .encoded_output
            .contains("let transient_peak = retained_capacity")
            && sources
                .encoded_output
                .contains("let actual_peak = retained_capacity")
            && sources
                .encoded_output
                .contains("ensure_capacity_within_limit(bytes.capacity(), max_len)?;"),
        "JPEG encoded output must precheck replacement growth and postcheck allocator capacity"
    );
}

fn assert_frame_assembly_is_shared_and_fallible(sources: &JpegEncodeAllocationSources) {
    for function_name in [
        "assemble_jpeg_baseline_frame",
        "assemble_jpeg_baseline_frame_with_quant_tables",
    ] {
        let frame = calls(
            "JPEG baseline frame assembly",
            &sources.frame,
            function_name,
        );
        frame.assert_ordered(
            function_name,
            &[
                "checked_jpeg_baseline_frame_capacity",
                "CappedBytes::try_with_capacity",
                "extend_from_slice",
                "into_vec",
            ],
        );
        frame.assert_propagated(
            function_name,
            &[
                "checked_jpeg_baseline_frame_capacity",
                "CappedBytes::try_with_capacity",
                "extend_from_slice",
            ],
        );
    }

    calls(
        "JPEG CPU encoder",
        &sources.encoder,
        "encode_jpeg_baseline_cpu",
    )
    .assert_contains("CPU JPEG frame route", &["assemble_jpeg_baseline_frame"]);
    calls(
        "JPEG DCT transcode encoder",
        &sources.transcode,
        "encode_baseline_dct_image",
    )
    .assert_contains(
        "transcode JPEG frame route",
        &["assemble_jpeg_baseline_frame_with_quant_tables"],
    );
    for (source, function_name) in [
        (
            sources.orchestrate_batch_group.as_str(),
            "encode_same_source_group",
        ),
        (
            sources.orchestrate.as_str(),
            "encode_jpeg_baseline_gpu_tile_with_tables",
        ),
    ] {
        calls("shared GPU JPEG orchestration", source, function_name)
            .assert_contains(function_name, &["assemble_jpeg_baseline_frame"]);
    }
}

fn assert_gpu_entropy_plans_use_the_frame_cap(sources: &JpegEncodeAllocationSources) {
    let tile = calls(
        "GPU JPEG entropy planning",
        &sources.planning,
        "jpeg_baseline_gpu_entropy_capacity_bytes",
    );
    tile.assert_ordered(
        "GPU JPEG tile output preflight",
        &[
            "jpeg_baseline_entropy_capacity_bytes",
            "checked_jpeg_baseline_frame_capacity",
        ],
    );
    tile.assert_propagated(
        "GPU JPEG tile output preflight",
        &[
            "jpeg_baseline_entropy_capacity_bytes",
            "checked_jpeg_baseline_frame_capacity",
        ],
    );

    let batch = calls(
        "GPU JPEG entropy planning",
        &sources.planning_batch,
        "jpeg_baseline_gpu_encode_batch_plan_with_live_bytes",
    );
    batch.assert_ordered(
        "GPU JPEG batch aggregate output preflight",
        &[
            "try_encode_metadata_vec",
            "jpeg_baseline_gpu_encode_tile_plan",
            "checked_add",
            "checked_jpeg_baseline_frame_capacity",
            "push",
        ],
    );
    batch.assert_propagated(
        "GPU JPEG batch aggregate output preflight",
        &[
            "try_encode_metadata_vec",
            "jpeg_baseline_gpu_encode_tile_plan",
            "checked_jpeg_baseline_frame_capacity",
        ],
    );
}

pub(super) fn assert_policy(sources: &JpegEncodeAllocationSources) {
    assert!(
        include_str!("frame_checks.rs").lines().count() < 190,
        "JPEG frame allocation policy must remain a focused module"
    );
    assert_capped_builder_is_fallible(sources);
    assert_frame_assembly_is_shared_and_fallible(sources);
    assert_gpu_entropy_plans_use_the_frame_cap(sources);
}
