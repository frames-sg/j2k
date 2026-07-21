// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

use j2k_core::BatchInfrastructureError;

use super::{
    allocate_distinct_classic_metadata, encode_distinct_classic_batches_to_buffer_in_encoder,
    DistinctClassicBatch, Error, J2kClassicCleanupBatchJob, J2kClassicSegment,
};
use crate::batch_allocation::BatchMetadataBudget;
use crate::compute::{
    checked_buffer_slice, commit_and_wait_metal, new_command_buffer, new_compute_command_encoder,
    new_shared_buffer_with_slice, validate_direct_status, with_runtime,
};

#[test]
fn distinct_classic_zero_fill_is_barriered_before_cleanup_dispatch() {
    let source = include_str!("distinct_batch.rs");
    let body = source
        .split_once("fn encode_distinct_classic_batches_to_buffer_in_encoder")
        .expect("distinct classic batch encoder")
        .1;
    let zero_fill = body
        .find("dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output")
        .expect("distinct classic zero-fill dispatch");
    let barrier = body
        .find("encoder.memory_barrier_with_resources(&[output]);")
        .expect("distinct classic zero-fill resource barrier");
    let cleanup = body
        .find("dispatch_classic_cleanup_batched_in_encoder(")
        .expect("distinct classic cleanup dispatch");
    assert!(zero_fill < barrier && barrier < cleanup);
}

#[test]
fn distinct_classic_dispatch_adopts_the_prebuilt_source_table() {
    let source = include_str!("distinct_batch.rs");
    let body = source
        .split_once("fn encode_distinct_classic_batches_to_buffer_in_encoder")
        .expect("distinct classic batch encoder")
        .1;
    let dispatch = body
        .split_once("dispatch_classic_cleanup_batched_in_encoder(")
        .expect("distinct classic cleanup dispatch")
        .1
        .split_once(")?;")
        .expect("distinct classic cleanup dispatch end")
        .0;

    assert!(
        dispatch.contains("Some(source_indices)"),
        "distinct dispatch must transfer its existing source table into status ownership"
    );
    assert!(
        !body.contains("set_classic_sources"),
        "distinct dispatch must not replace a second job-sized source allocation"
    );
}

#[test]
fn distinct_classic_metadata_honors_exact_cap_and_one_byte_over() {
    let coded_len = 11;
    let job_count = 3;
    let segment_count = 5;
    let exact_cap = coded_len
        + job_count * size_of::<J2kClassicCleanupBatchJob>()
        + segment_count * size_of::<J2kClassicSegment>()
        + job_count * size_of::<usize>();
    let owners = allocate_distinct_classic_metadata(
        coded_len,
        job_count,
        segment_count,
        BatchMetadataBudget::with_cap(
            "classic J2K MetalDirect distinct color submission",
            exact_cap,
        ),
    )
    .expect("exact distinct classic metadata cap");
    assert_eq!(owners.coded_data.capacity(), coded_len);
    assert_eq!(owners.jobs.capacity(), job_count);
    assert_eq!(owners.segments.capacity(), segment_count);
    assert_eq!(owners.source_indices.capacity(), job_count);

    assert!(matches!(
        allocate_distinct_classic_metadata(
            coded_len,
            job_count,
            segment_count,
            BatchMetadataBudget::with_cap(
                "classic J2K MetalDirect distinct color submission",
                exact_cap - 1,
            ),
        ),
        Err(Error::BatchInfrastructure(
            BatchInfrastructureError::AllocationTooLarge {
                requested,
                cap,
                ..
            }
        )) if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn distinct_classic_batches_honor_empty_and_zero_fill_output_semantics() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    with_runtime(|runtime| {
        let output = new_shared_buffer_with_slice(&runtime.device, &[7.0_f32; 8])?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        let batches = [
            DistinctClassicBatch {
                coded_data: &[],
                jobs: &[],
                segments: &[],
                output_base: 0,
                output_len: 4,
                zero_fill: false,
            },
            DistinctClassicBatch {
                coded_data: &[],
                jobs: &[],
                segments: &[],
                output_base: 4,
                output_len: 4,
                zero_fill: false,
            },
        ];
        let mut scratch_buffers = Vec::new();
        let (retained, status) = encode_distinct_classic_batches_to_buffer_in_encoder(
            runtime,
            &encoder,
            batches.into_iter(),
            &output,
            &mut scratch_buffers,
        )?;
        encoder.end_encoding();
        commit_and_wait_metal(&command_buffer)?;
        validate_direct_status(runtime, status)?;
        assert_eq!(
            checked_buffer_slice::<f32>(&output, 8, "distinct classic empty batch output")?,
            vec![0.0; 8]
        );
        drop(retained);
        drop(scratch_buffers);

        let zero_pass_job = J2kClassicCleanupBatchJob {
            coded_offset: 0,
            coded_len: 0,
            segment_offset: 0,
            segment_count: 0,
            width: 4,
            height: 1,
            output_stride: 4,
            output_offset: 0,
            missing_msbs: 0,
            total_bitplanes: 1,
            roi_shift: 0,
            number_of_coding_passes: 0,
            sub_band_type: 0,
            style_flags: 1,
            strict: 1,
            dequantization_step: 1.0,
        };
        let output = new_shared_buffer_with_slice(&runtime.device, &[11.0_f32; 4])?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        let batches = [DistinctClassicBatch {
            coded_data: &[],
            jobs: core::slice::from_ref(&zero_pass_job),
            segments: &[],
            output_base: 0,
            output_len: 4,
            zero_fill: true,
        }];
        let mut scratch_buffers = Vec::new();
        let (retained, status) = encode_distinct_classic_batches_to_buffer_in_encoder(
            runtime,
            &encoder,
            batches.into_iter(),
            &output,
            &mut scratch_buffers,
        )?;
        encoder.end_encoding();
        commit_and_wait_metal(&command_buffer)?;
        validate_direct_status(runtime, status)?;
        assert_eq!(
            checked_buffer_slice::<f32>(&output, 4, "distinct classic zero-fill output")?,
            vec![0.0; 4]
        );
        drop(retained);
        drop(scratch_buffers);
        Ok(())
    })
    .expect("distinct empty classic batches");
}

#[test]
fn distinct_classic_device_failure_keeps_nonzero_source_identity() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    with_runtime(|runtime| {
        let valid = J2kClassicCleanupBatchJob {
            coded_offset: 0,
            coded_len: 0,
            segment_offset: 0,
            segment_count: 0,
            width: 4,
            height: 1,
            output_stride: 4,
            output_offset: 0,
            missing_msbs: 0,
            total_bitplanes: 1,
            roi_shift: 0,
            number_of_coding_passes: 0,
            sub_band_type: 0,
            style_flags: 1,
            strict: 1,
            dequantization_step: 1.0,
        };
        let invalid = J2kClassicCleanupBatchJob {
            total_bitplanes: 0,
            number_of_coding_passes: 1,
            ..valid
        };
        let output = new_shared_buffer_with_slice(&runtime.device, &[0.0_f32; 8])?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        let batches = [
            DistinctClassicBatch {
                coded_data: &[],
                jobs: core::slice::from_ref(&valid),
                segments: &[],
                output_base: 0,
                output_len: 4,
                zero_fill: true,
            },
            DistinctClassicBatch {
                coded_data: &[],
                jobs: core::slice::from_ref(&invalid),
                segments: &[],
                output_base: 4,
                output_len: 4,
                zero_fill: true,
            },
        ];
        let mut scratch_buffers = Vec::new();
        let (retained, mut status) = encode_distinct_classic_batches_to_buffer_in_encoder(
            runtime,
            &encoder,
            batches.into_iter(),
            &output,
            &mut scratch_buffers,
        )?;
        status.remap_sources(&[3, 9])?;
        encoder.end_encoding();
        commit_and_wait_metal(&command_buffer)?;
        let error = validate_direct_status(runtime, status)
            .expect_err("invalid second classic source must fail");
        let message = error.to_string();
        assert!(message.contains("source 9"), "{message}");
        assert!(!message.contains("source 3"), "{message}");
        drop(retained);
        drop(scratch_buffers);
        Ok(())
    })
    .expect("distinct classic status attribution");
}
