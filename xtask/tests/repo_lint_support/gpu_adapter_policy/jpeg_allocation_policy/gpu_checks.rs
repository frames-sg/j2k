// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::rust_function_policy::FunctionCalls;
use super::JpegEncodeAllocationSources;

fn calls(source_name: &str, source: &str, function_name: &str) -> FunctionCalls {
    FunctionCalls::parse(source_name, source, function_name)
}

fn assert_actual_metadata_and_payload_capacity(sources: &JpegEncodeAllocationSources) {
    let metadata = calls(
        "GPU JPEG encode allocation",
        &sources.encode_allocation,
        "try_encode_metadata_vec",
    );
    metadata.assert_ordered(
        "GPU JPEG actual metadata allocation",
        &[
            "checked_element_capacity_bytes",
            "checked_encode_host_live_bytes",
            "try_host_vec_with_capacity",
            "checked_element_capacity_bytes",
            "capacity",
            "checked_encode_host_live_bytes",
        ],
    );
    metadata.assert_propagated(
        "GPU JPEG actual metadata allocation",
        &[
            "checked_element_capacity_bytes",
            "checked_encode_host_live_bytes",
            "try_host_vec_with_capacity",
        ],
    );

    calls(
        "GPU JPEG encode allocation",
        &sources.encode_allocation,
        "byte_chunk_capacity_bytes",
    )
    .assert_contains(
        "GPU JPEG returned entropy ownership",
        &[
            "checked_element_capacity_bytes",
            "checked_encode_host_live_bytes",
        ],
    );
    assert!(
        sources.encode_allocation.contains(".map(Vec::capacity)"),
        "GPU JPEG returned entropy ownership must count every chunk's actual capacity"
    );
    assert!(
        sources
            .encode_allocation
            .contains("GpuBatchAllocationBudget::with_fixed_metadata_bytes")
            && sources
                .encode_allocation
                .contains("fixed_metadata_bytes: usize")
            && sources.encode_allocation.contains("self.retained_frames"),
        "whole-batch preflight must carry actual fixed metadata and retained frames"
    );
}

fn assert_gpu_orchestration_carries_every_live_owner(sources: &JpegEncodeAllocationSources) {
    let batch = calls(
        "shared GPU JPEG orchestration",
        &sources.orchestrate_batch,
        "encode_jpeg_baseline_gpu_batch_with_external_live",
    );
    batch.assert_contains(
        "GPU JPEG whole-batch lifecycle",
        &[
            "try_encode_metadata_vec",
            "checked_gpu_batch_live_bytes",
            "encoded_frame_capacity_bytes",
            "checked_encode_host_live_bytes",
            "encode_same_source_group",
        ],
    );
    batch.assert_propagated(
        "GPU JPEG whole-batch lifecycle",
        &[
            "encoded_frame_capacity_bytes",
            "checked_encode_host_live_bytes",
        ],
    );
    let group = calls(
        "shared GPU JPEG group orchestration",
        &sources.orchestrate_batch_group,
        "encode_same_source_group",
    );
    group.assert_contains(
        "GPU JPEG same-source group lifecycle",
        &[
            "jpeg_baseline_gpu_encode_batch_plan_with_live_bytes",
            "capacity",
            "checked_gpu_group_runtime_live_bytes",
            "ensure_entropy_output_within_plan",
            "byte_chunk_capacity_bytes",
            "assemble_jpeg_baseline_frame",
        ],
    );
    group.assert_propagated(
        "GPU JPEG same-source group lifecycle",
        &[
            "checked_gpu_group_runtime_live_bytes",
            "ensure_entropy_output_within_plan",
            "byte_chunk_capacity_bytes",
            "checked_encode_host_live_bytes",
            "assemble_jpeg_baseline_frame",
            "encoded_frame_capacity_bytes",
        ],
    );
    assert!(
        sources.orchestrate_batch_group.contains(
            "ensure_entropy_output_within_plan(entropy_chunks.capacity(), tiles.len())?;"
        ) && sources
            .orchestrate_batch_group
            .contains("encoded_frame_capacity_bytes(&encoded[group_start..])?")
            && sources
                .orchestrate_batch_group
                .contains("frame.data.capacity()"),
        "GPU batch runtime must postcheck returned outer metadata and every retained frame"
    );

    let tile = calls(
        "shared GPU JPEG orchestration",
        &sources.orchestrate,
        "encode_jpeg_baseline_gpu_tile_with_tables",
    );
    tile.assert_contains(
        "GPU JPEG actual tile lifecycle",
        &[
            "checked_gpu_tile_live_bytes",
            "checked_encode_host_live_bytes",
            "capacity",
            "checked_jpeg_baseline_frame_capacity",
            "assemble_jpeg_baseline_frame",
        ],
    );
}

fn assert_adapter_contract_and_regressions(sources: &JpegEncodeAllocationSources) {
    for contract in [
        "capacity at most `tiles.len()`",
        "shared driver assumes the",
        "moved `plan` is dropped before this method returns",
        "adapter-owned live memory",
    ] {
        assert!(
            sources.types.contains(contract),
            "GPU JPEG adapter ownership contract must retain `{contract}`"
        );
    }
    for regression in [
        "gpu_batch_rejects_adapter_outer_capacity_above_the_tile_count",
        "gpu_batch_rejects_retained_frames_across_source_groups_before_submission",
        "gpu_batch_validates_every_adapter_capacity_before_frame_copy",
    ] {
        assert!(
            sources.adapter_tests.contains(regression),
            "GPU JPEG allocation regression `{regression}` must remain explicit"
        );
    }
}

pub(super) fn assert_policy(sources: &JpegEncodeAllocationSources) {
    assert!(
        include_str!("gpu_checks.rs").lines().count() < 180,
        "GPU JPEG allocation policy must remain a focused module"
    );
    assert_actual_metadata_and_payload_capacity(sources);
    assert_gpu_orchestration_carries_every_live_owner(sources);
    assert_adapter_contract_and_regressions(sources);
}
