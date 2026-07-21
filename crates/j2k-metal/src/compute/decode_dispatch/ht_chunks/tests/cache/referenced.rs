// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;
use std::sync::Arc;

use super::super::super::{
    default_metal_ht_chunk_limits, encode_metal_ht_batches_in_encoder, HtBatchInput,
};
use crate::compute::test_counters::{
    ht_immutable_job_uploads_for_test, ht_immutable_payload_uploads_for_test,
    reset_ht_immutable_job_uploads_for_test, reset_ht_immutable_payload_uploads_for_test,
};

#[test]
fn second_referenced_prepared_submission_uploads_no_immutable_arenas() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let runtime = crate::compute::MetalRuntime::new().expect("isolated Metal runtime");
    let pixels = (0..64_u8).collect::<Vec<_>>();
    let bytes = Arc::<[u8]>::from(
        j2k_native::encode_htj2k(
            &pixels,
            8,
            8,
            1,
            8,
            false,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode referenced prepared fixture"),
    );
    let image = j2k_native::Image::new(&bytes, &j2k_native::DecodeSettings::strict())
        .expect("referenced prepared image");
    let mut context = j2k_native::DecoderContext::default();
    let referenced = image
        .build_referenced_htj2k_plan_region_with_context(&mut context, (0, 0, 8, 8))
        .expect("referenced prepared plan");
    let prepared = crate::compute::prepare_referenced_htj2k_grayscale_plan(&referenced, &bytes)
        .expect("Metal referenced prepared plan");
    let group = prepared.ht_groups.first().expect("prepared HT group");
    let input = [HtBatchInput {
        source_index: 0,
        payload: group.payload_source.as_ht_payload_source(),
        jobs: &group.jobs,
        output_base: 0,
        execution_owner: &group.execution_owner,
    }];

    reset_ht_immutable_payload_uploads_for_test();
    reset_ht_immutable_job_uploads_for_test();
    for submission in 0..2 {
        if submission == 1 {
            assert_eq!(ht_immutable_payload_uploads_for_test(), 1);
            assert_eq!(ht_immutable_job_uploads_for_test(), 1);
            reset_ht_immutable_payload_uploads_for_test();
            reset_ht_immutable_job_uploads_for_test();
        }
        let output = crate::compute::new_shared_buffer(
            &runtime.device,
            group.total_coefficients * size_of::<f32>(),
        )
        .expect("referenced prepared output");
        let command_buffer = crate::compute::new_command_buffer(&runtime.queue)
            .expect("referenced prepared command buffer");
        let encoder = crate::compute::new_compute_command_encoder(&command_buffer)
            .expect("referenced prepared encoder");
        let (retained, status) = encode_metal_ht_batches_in_encoder(
            &runtime,
            &encoder,
            &input,
            &output,
            group.total_coefficients,
            default_metal_ht_chunk_limits(),
        )
        .expect("submit referenced prepared execution");
        encoder.end_encoding();
        crate::compute::commit_and_wait_metal(&command_buffer)
            .expect("complete referenced prepared execution");
        crate::compute::validate_direct_status(&runtime, status)
            .expect("validate referenced prepared execution");
        drop(retained);
    }

    assert_eq!(ht_immutable_payload_uploads_for_test(), 0);
    assert_eq!(ht_immutable_job_uploads_for_test(), 0);
}
