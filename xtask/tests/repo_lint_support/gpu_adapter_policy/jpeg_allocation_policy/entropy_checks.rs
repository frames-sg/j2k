// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::contains_normalized;
use super::super::super::rust_function_policy::{
    assert_struct_field_type, assert_usize_const, FunctionCalls,
};
use super::JpegEncodeAllocationSources;

fn calls(source_name: &str, source: &str, function_name: &str) -> FunctionCalls {
    FunctionCalls::parse(source_name, source, function_name)
}

fn assert_entropy_storage_is_preallocated_before_parallel_fill(
    sources: &JpegEncodeAllocationSources,
) {
    assert_capped_entropy_writer(sources);
    assert_usize_const(
        "JPEG restart entropy",
        &sources.entropy_restart,
        "MAX_PARALLEL_ENTROPY_CHUNKS",
        64,
    );
    let restart = calls(
        "JPEG restart entropy",
        &sources.entropy_restart,
        "encode_entropy_restart_segments",
    );
    restart.assert_contains(
        "bounded JPEG restart entropy",
        &[
            "restart_entropy_workspace_plan",
            "prepare_restart_jobs",
            "par_iter_mut",
            "encode_entropy_restart_chunk_into",
            "CappedBytes::try_with_capacity",
            "extend_from_slice",
            "into_vec",
        ],
    );
    restart.assert_absent(
        "JPEG restart entropy segment fanout",
        &["collect", "into_par_iter"],
    );
    restart.assert_propagated(
        "JPEG restart entropy allocation",
        &[
            "restart_entropy_workspace_plan",
            "prepare_restart_jobs",
            "CappedBytes::try_with_capacity",
            "extend_from_slice",
        ],
    );
    restart.assert_ordered(
        "JPEG restart allocation before parallel fill",
        &[
            "prepare_restart_jobs",
            "par_iter_mut",
            "CappedBytes::try_with_capacity",
        ],
    );
    assert_restart_job_preallocation(sources);
    assert_restart_parallel_fill_has_no_allocations(sources);
}

fn assert_capped_entropy_writer(sources: &JpegEncodeAllocationSources) {
    assert_struct_field_type(
        "JPEG CPU encoder",
        &sources.encoder,
        "BitWriter",
        "bytes",
        "CappedBytes",
    );
    calls("JPEG CPU encoder", &sources.encoder, "try_with_max_bytes").assert_contains(
        "JPEG BitWriter storage",
        &["CappedBytes::try_with_capacity"],
    );
    assert!(
        contains_normalized(
            &sources.encoded_output,
            "#[cfg(test)] pub(crate) fn new(max_len: usize)",
        ),
        "zero-capacity CappedBytes construction must remain test-only"
    );
}

fn assert_restart_job_preallocation(sources: &JpegEncodeAllocationSources) {
    let prepare = calls(
        "JPEG restart job preparation",
        &sources.entropy_restart,
        "prepare_restart_jobs",
    );
    prepare.assert_ordered(
        "JPEG restart fallible serial preparation",
        &[
            "try_host_vec_with_capacity",
            "checked_restart_chunk_preallocation_live_bytes",
            "BitWriter::try_with_max_bytes",
        ],
    );
    prepare.assert_propagated(
        "JPEG restart serial allocation propagation",
        &[
            "try_host_vec_with_capacity",
            "checked_restart_chunk_preallocation_live_bytes",
            "BitWriter::try_with_max_bytes",
        ],
    );
}

fn assert_restart_parallel_fill_has_no_allocations(sources: &JpegEncodeAllocationSources) {
    let parallel_start = sources
        .entropy_restart
        .find("jobs.par_iter_mut()")
        .expect("restart parallel fill");
    let output_start = sources.entropy_restart[parallel_start..]
        .find("CappedBytes::try_with_capacity")
        .map(|offset| parallel_start + offset)
        .expect("restart aggregate output allocation");
    let parallel_fill = &sources.entropy_restart[parallel_start..output_start];
    assert!(
        !parallel_fill.contains("try_with_max_bytes")
            && !parallel_fill.contains("try_host_vec_with_capacity")
            && !parallel_fill.contains("Vec::with_capacity"),
        "Rayon restart workers must only fill writers preallocated by the serial owner"
    );

    let chunk_count = calls(
        "JPEG restart entropy",
        &sources.entropy_restart,
        "parallel_entropy_chunk_count",
    );
    chunk_count.assert_contains(
        "JPEG restart parallel chunk bound",
        &["usize::try_from", "clamp"],
    );
    assert!(
        contains_normalized(
            &sources.entropy_restart,
            ".clamp(1, MAX_PARALLEL_ENTROPY_CHUNKS)",
        ) && !sources.entropy_restart.contains("Vec<Vec<u8>>"),
        "restart work must remain bounded by the explicit chunk count"
    );
    assert!(
        sources
            .entropy_workspace
            .contains("size_of::<super::restart::RestartChunkJob>()")
            && sources.entropy_workspace.contains("chunk_capacity_bytes")
            && sources.entropy_workspace.contains("plan.live_bytes()?;"),
        "restart preflight must describe the same concrete owners used at runtime"
    );
}

fn assert_serial_and_transcode_entropy(sources: &JpegEncodeAllocationSources) {
    let serial = calls(
        "JPEG entropy orchestration",
        &sources.entropy,
        "encode_entropy_serial",
    );
    serial.assert_ordered(
        "serial JPEG entropy storage",
        &[
            "checked_jpeg_baseline_frame_capacity",
            "BitWriter::try_with_max_bytes",
            "checked_encode_host_live_bytes",
            "encode_entropy_mcu_range_into",
            "into_bytes",
        ],
    );
    serial.assert_propagated(
        "serial JPEG entropy storage",
        &[
            "checked_jpeg_baseline_frame_capacity",
            "BitWriter::try_with_max_bytes",
            "checked_encode_host_live_bytes",
            "encode_entropy_mcu_range_into",
        ],
    );

    let transcode_entropy = calls(
        "JPEG DCT transcode encoder",
        &sources.transcode,
        "encode_dct_entropy",
    );
    transcode_entropy.assert_ordered(
        "transcode JPEG entropy storage",
        &[
            "BitWriter::try_with_max_bytes",
            "encode_block",
            "into_bytes",
        ],
    );
    transcode_entropy.assert_propagated(
        "transcode JPEG entropy storage",
        &["BitWriter::try_with_max_bytes", "encode_block"],
    );
}

pub(super) fn assert_policy(sources: &JpegEncodeAllocationSources) {
    assert!(
        include_str!("entropy_checks.rs").lines().count() < 230,
        "JPEG entropy allocation policy must remain a focused module"
    );
    assert_entropy_storage_is_preallocated_before_parallel_fill(sources);
    assert_serial_and_transcode_entropy(sources);
}
