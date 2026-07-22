// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;
use std::sync::Arc;

use j2k_native::{DecodeSettings, DecoderContext, EncodeOptions, Image};
use metal::foreign_types::ForeignType;

use super::super::{
    default_metal_ht_chunk_limits, encode_metal_ht_batches_in_encoder,
    encode_repeated_metal_ht_batch_in_command_buffer, HtBatchInput,
};
use crate::compute::{
    checked_buffer_slice, commit_and_wait_metal, decode_prepared_ht_sub_band_group_on_cpu_profile,
    new_command_buffer, new_compute_command_encoder, new_shared_buffer,
    prepare_direct_grayscale_plan, validate_direct_status, wait_for_completion_metal,
    DirectStatusCheck, MetalRuntime, PreparedHtExecutionOwner,
};

#[test]
fn completed_ht_status_storage_returns_to_the_shared_pool() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let pixels = (0..64_u8).rev().collect::<Vec<_>>();
    let bytes = j2k_native::encode_htj2k(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .expect("encode status-pool fixture");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("fixture image");
    let mut context = DecoderContext::default();
    let direct = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct fixture plan");
    let prepared = prepare_direct_grayscale_plan(&direct).expect("prepared fixture plan");
    let group = prepared.ht_groups.first().expect("prepared HT group");
    let input = HtBatchInput {
        source_index: 0,
        payload: group.payload_source.as_ht_payload_source(),
        jobs: &group.jobs,
        output_base: 0,
        execution_owner: &group.execution_owner,
    };
    let runtime = MetalRuntime::new().expect("isolated Metal runtime");

    for submission in 0..2 {
        let output =
            new_shared_buffer(&runtime.device, group.total_coefficients * size_of::<f32>())
                .expect("status-pool output");
        let command_buffer = new_command_buffer(&runtime.queue).expect("status-pool command");
        let encoder = new_compute_command_encoder(&command_buffer).expect("status-pool encoder");
        let (retained, status) = encode_metal_ht_batches_in_encoder(
            &runtime,
            &encoder,
            &[input],
            &output,
            group.total_coefficients,
            default_metal_ht_chunk_limits(),
        )
        .expect("status-pool encode");
        encoder.end_encoding();
        commit_and_wait_metal(&command_buffer).expect("status-pool completion");
        validate_direct_status(&runtime, status).expect("status-pool validation");
        drop(retained);

        let diagnostics = runtime
            .buffer_pool_diagnostics()
            .expect("status-pool diagnostics");
        assert_eq!(
            diagnostics.shared.cached_buffers, 1,
            "submission {submission} must retire exactly one reusable HT status buffer"
        );
    }
}

#[test]
fn reused_ht_status_storage_is_overwritten_by_every_dispatched_job() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let image = Image::new(
        j2k_test_support::openhtj2k_refinement_fixture(),
        &DecodeSettings::default(),
    )
    .expect("refinement status-overwrite fixture image");
    let mut context = DecoderContext::default();
    let direct = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct fixture plan");
    let prepared = prepare_direct_grayscale_plan(&direct).expect("prepared fixture plan");
    let group = prepared.ht_groups.first().expect("prepared HT group");
    assert!(
        group.jobs.iter().any(|job| job.number_of_coding_passes > 1),
        "status-overwrite fixture must exercise refinement jobs"
    );
    let mut invalid_jobs = group.jobs.clone();
    for job in &mut invalid_jobs {
        job.num_bitplanes = 0;
    }
    let invalid_owner = Arc::new(PreparedHtExecutionOwner);
    let runtime = MetalRuntime::new().expect("isolated Metal runtime");

    let invalid_output =
        new_shared_buffer(&runtime.device, group.total_coefficients * size_of::<f32>())
            .expect("invalid status output");
    let invalid_command = new_command_buffer(&runtime.queue).expect("invalid status command");
    let invalid_encoder =
        new_compute_command_encoder(&invalid_command).expect("invalid status encoder");
    let (_, invalid_status) = encode_metal_ht_batches_in_encoder(
        &runtime,
        &invalid_encoder,
        &[HtBatchInput {
            source_index: 0,
            payload: group.payload_source.as_ht_payload_source(),
            jobs: &invalid_jobs,
            output_base: 0,
            execution_owner: &invalid_owner,
        }],
        &invalid_output,
        group.total_coefficients,
        default_metal_ht_chunk_limits(),
    )
    .expect("invalid status encode");
    invalid_encoder.end_encoding();
    commit_and_wait_metal(&invalid_command).expect("invalid status completion");
    let DirectStatusCheck::Ht {
        buffer: invalid_status_buffer,
        ..
    } = &invalid_status
    else {
        panic!("invalid distinct submission must retain HT status")
    };
    let invalid_status_ptr = invalid_status_buffer.as_ptr();
    assert!(
        validate_direct_status(&runtime, invalid_status).is_err(),
        "invalid first submission must seed every status slot with a failure"
    );

    let valid_output =
        new_shared_buffer(&runtime.device, group.total_coefficients * size_of::<f32>())
            .expect("valid status output");
    let valid_command = new_command_buffer(&runtime.queue).expect("valid status command");
    let valid_encoder = new_compute_command_encoder(&valid_command).expect("valid status encoder");
    let (_, valid_status) = encode_metal_ht_batches_in_encoder(
        &runtime,
        &valid_encoder,
        &[HtBatchInput {
            source_index: 0,
            payload: group.payload_source.as_ht_payload_source(),
            jobs: &group.jobs,
            output_base: 0,
            execution_owner: &group.execution_owner,
        }],
        &valid_output,
        group.total_coefficients,
        default_metal_ht_chunk_limits(),
    )
    .expect("valid status encode");
    valid_encoder.end_encoding();
    commit_and_wait_metal(&valid_command).expect("valid status completion");
    let DirectStatusCheck::Ht {
        buffer: valid_status_buffer,
        ..
    } = &valid_status
    else {
        panic!("valid distinct submission must retain HT status")
    };
    assert_eq!(
        valid_status_buffer.as_ptr(),
        invalid_status_ptr,
        "valid distinct dispatch must overwrite the recycled failure-status allocation"
    );
    validate_direct_status(&runtime, valid_status)
        .expect("every valid dispatch must overwrite its recycled failure status");
}

#[test]
fn reused_repeated_ht_status_storage_is_overwritten_by_every_dispatched_job() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let image = Image::new(
        j2k_test_support::openhtj2k_refinement_fixture(),
        &DecodeSettings::default(),
    )
    .expect("repeated refinement status fixture image");
    let mut context = DecoderContext::default();
    let direct = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("repeated refinement direct fixture plan");
    let prepared = prepare_direct_grayscale_plan(&direct).expect("prepared refinement plan");
    let group = prepared.ht_groups.first().expect("prepared HT group");
    assert!(
        group.jobs.iter().any(|job| job.number_of_coding_passes > 1),
        "repeated status-overwrite fixture must exercise refinement jobs"
    );
    let mut invalid_jobs = group.jobs.clone();
    for job in &mut invalid_jobs {
        job.num_bitplanes = 0;
    }
    let invalid_owner = Arc::new(PreparedHtExecutionOwner);
    let runtime = MetalRuntime::new().expect("isolated repeated Metal runtime");
    let count = 2;
    let output_words = group
        .total_coefficients
        .checked_mul(count)
        .expect("repeated status output word count");

    let invalid_output = new_shared_buffer(&runtime.device, output_words * size_of::<f32>())
        .expect("invalid repeated status output");
    let invalid_command =
        new_command_buffer(&runtime.queue).expect("invalid repeated status command");
    let (_, invalid_status) = encode_repeated_metal_ht_batch_in_command_buffer(
        &runtime,
        &invalid_command,
        HtBatchInput {
            source_index: 0,
            payload: group.payload_source.as_ht_payload_source(),
            jobs: &invalid_jobs,
            output_base: 0,
            execution_owner: &invalid_owner,
        },
        count,
        group.total_coefficients,
        &invalid_output,
        default_metal_ht_chunk_limits(),
    )
    .expect("invalid repeated status encode");
    commit_and_wait_metal(&invalid_command).expect("invalid repeated status completion");
    let DirectStatusCheck::Ht {
        buffer: invalid_status_buffer,
        ..
    } = &invalid_status
    else {
        panic!("invalid repeated submission must retain HT status")
    };
    let invalid_status_ptr = invalid_status_buffer.as_ptr();
    assert!(
        validate_direct_status(&runtime, invalid_status).is_err(),
        "invalid repeated submission must seed every status slot with a failure"
    );

    let valid_output = new_shared_buffer(&runtime.device, output_words * size_of::<f32>())
        .expect("valid repeated status output");
    let valid_command = new_command_buffer(&runtime.queue).expect("valid repeated status command");
    let (_, valid_status) = encode_repeated_metal_ht_batch_in_command_buffer(
        &runtime,
        &valid_command,
        HtBatchInput {
            source_index: 0,
            payload: group.payload_source.as_ht_payload_source(),
            jobs: &group.jobs,
            output_base: 0,
            execution_owner: &group.execution_owner,
        },
        count,
        group.total_coefficients,
        &valid_output,
        default_metal_ht_chunk_limits(),
    )
    .expect("valid repeated status encode");
    commit_and_wait_metal(&valid_command).expect("valid repeated status completion");
    let DirectStatusCheck::Ht {
        buffer: valid_status_buffer,
        ..
    } = &valid_status
    else {
        panic!("valid repeated submission must retain HT status")
    };
    assert_eq!(
        valid_status_buffer.as_ptr(),
        invalid_status_ptr,
        "valid repeated dispatch must overwrite the recycled failure-status allocation"
    );
    validate_direct_status(&runtime, valid_status)
        .expect("every valid repeated dispatch must overwrite recycled failure status");
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the overlap-lifetime scenario keeps both pending submissions and ownership assertions visible"
)]
fn overlapping_prepared_ht_submissions_keep_distinct_status_owners() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let pixels = (0..64_u8).collect::<Vec<_>>();
    let bytes = j2k_native::encode_htj2k(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .expect("encode overlapping-status fixture");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("fixture image");
    let mut context = DecoderContext::default();
    let direct = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct fixture plan");
    let prepared = prepare_direct_grayscale_plan(&direct).expect("prepared fixture plan");
    let group = prepared.ht_groups.first().expect("prepared HT group");
    let input = HtBatchInput {
        source_index: 0,
        payload: group.payload_source.as_ht_payload_source(),
        jobs: &group.jobs,
        output_base: 0,
        execution_owner: &group.execution_owner,
    };
    let cpu = decode_prepared_ht_sub_band_group_on_cpu_profile(group, None)
        .expect("overlapping CPU coefficient oracle");
    let runtime = MetalRuntime::new().expect("isolated Metal runtime");

    let first_output =
        new_shared_buffer(&runtime.device, group.total_coefficients * size_of::<f32>())
            .expect("first overlapping output");
    let first_command = new_command_buffer(&runtime.queue).expect("first overlapping command");
    let first_encoder =
        new_compute_command_encoder(&first_command).expect("first overlapping encoder");
    let (first_retained, first_status) = encode_metal_ht_batches_in_encoder(
        &runtime,
        &first_encoder,
        &[input],
        &first_output,
        group.total_coefficients,
        default_metal_ht_chunk_limits(),
    )
    .expect("first overlapping encode");
    first_encoder.end_encoding();
    first_command.commit();

    let second_output =
        new_shared_buffer(&runtime.device, group.total_coefficients * size_of::<f32>())
            .expect("second overlapping output");
    let second_command = new_command_buffer(&runtime.queue).expect("second overlapping command");
    let second_encoder =
        new_compute_command_encoder(&second_command).expect("second overlapping encoder");
    let (second_retained, second_status) = encode_metal_ht_batches_in_encoder(
        &runtime,
        &second_encoder,
        &[input],
        &second_output,
        group.total_coefficients,
        default_metal_ht_chunk_limits(),
    )
    .expect("second overlapping encode");
    second_encoder.end_encoding();
    second_command.commit();

    let DirectStatusCheck::Ht {
        buffer: first_buffer,
        ..
    } = &first_status
    else {
        panic!("first prepared HT submission must retain HT status")
    };
    let DirectStatusCheck::Ht {
        buffer: second_buffer,
        ..
    } = &second_status
    else {
        panic!("second prepared HT submission must retain HT status")
    };
    assert_ne!(
        first_buffer.as_ptr(),
        second_buffer.as_ptr(),
        "overlapping submissions must not alias in-flight status storage"
    );

    wait_for_completion_metal(&first_command).expect("first overlapping completion");
    wait_for_completion_metal(&second_command).expect("second overlapping completion");
    validate_direct_status(&runtime, first_status).expect("first overlapping status");
    validate_direct_status(&runtime, second_status).expect("second overlapping status");
    let first_coefficients = checked_buffer_slice::<f32>(
        &first_output,
        group.total_coefficients,
        "first overlapping HT output",
    )
    .expect("read first overlapping HT output");
    let second_coefficients = checked_buffer_slice::<f32>(
        &second_output,
        group.total_coefficients,
        "second overlapping HT output",
    )
    .expect("read second overlapping HT output");
    assert_eq!(first_coefficients, cpu);
    assert_eq!(second_coefficients, cpu);
    drop((first_retained, second_retained));
}
