// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

use j2k_native::{DecodeSettings, DecoderContext, EncodeOptions, Image};

use super::super::{
    default_metal_ht_chunk_limits, encode_metal_ht_batches_in_encoder, HtBatchInput,
};
use crate::compute::test_counters::{
    ht_immutable_job_uploads_for_test, ht_immutable_payload_uploads_for_test,
    reset_ht_immutable_job_uploads_for_test, reset_ht_immutable_payload_uploads_for_test,
};
use crate::compute::{
    checked_buffer_slice, commit_and_wait_metal, decode_prepared_ht_sub_band_group_on_cpu_profile,
    new_command_buffer, new_compute_command_encoder, new_shared_buffer,
    prepare_direct_grayscale_plan, validate_direct_status, with_runtime,
};

mod referenced;

#[test]
fn second_prepared_ht_submission_reuses_immutable_gpu_arenas() {
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
    .expect("encode prepared-reuse fixture");
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
        .expect("prepared-reuse CPU coefficient oracle");

    with_runtime(|runtime| {
        reset_ht_immutable_payload_uploads_for_test();
        reset_ht_immutable_job_uploads_for_test();
        for submission in 0..2 {
            if submission == 1 {
                assert!(ht_immutable_payload_uploads_for_test() > 0);
                assert!(ht_immutable_job_uploads_for_test() > 0);
                reset_ht_immutable_payload_uploads_for_test();
                reset_ht_immutable_job_uploads_for_test();
            }
            let output =
                new_shared_buffer(&runtime.device, group.total_coefficients * size_of::<f32>())?;
            let command_buffer = new_command_buffer(&runtime.queue)?;
            let encoder = new_compute_command_encoder(&command_buffer)?;
            let (retained, status) = encode_metal_ht_batches_in_encoder(
                runtime,
                &encoder,
                &[input],
                &output,
                group.total_coefficients,
                default_metal_ht_chunk_limits(),
            )?;
            encoder.end_encoding();
            commit_and_wait_metal(&command_buffer)?;
            validate_direct_status(runtime, status)?;
            let coefficients = checked_buffer_slice::<f32>(
                &output,
                group.total_coefficients,
                "prepared HT cache reuse parity",
            )?;
            assert_eq!(
                coefficients, cpu,
                "submission {submission} must decode the cached immutable arenas exactly"
            );
            drop(retained);
        }
        assert_eq!(ht_immutable_payload_uploads_for_test(), 0);
        assert_eq!(ht_immutable_job_uploads_for_test(), 0);
        Ok(())
    })
    .expect("prepared HT arena reuse");
}
