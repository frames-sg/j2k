// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use core::{mem::size_of, num::NonZeroUsize};

use j2k_core::HtGpuJobChunkLimits;
use j2k_native::{DecodeSettings, DecoderContext, EncodeOptions, Image};

use super::super::{
    encode_metal_ht_batches_in_encoder, plan_metal_ht_chunks, HtBatchInput, J2kHtCleanupBatchJob,
};
use crate::engine::{
    checked_buffer_slice, commit_and_wait_metal, decode_prepared_ht_sub_band_group_on_cpu_profile,
    new_command_buffer, new_compute_command_encoder, new_shared_buffer,
    prepare_direct_grayscale_plan, validate_direct_status, with_runtime,
};

#[test]
fn forced_multi_chunk_metal_output_matches_cpu_coefficients() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(64)
        .expect("fixture pixel allocation");
    pixels.extend(0..64_u8);
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
    .expect("encode forced-chunk fixture");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("fixture image");
    let mut context = DecoderContext::default();
    let direct = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct fixture plan");
    let prepared = prepare_direct_grayscale_plan(&direct).expect("prepared fixture plan");
    let group = prepared.ht_groups.first().expect("prepared HT group");
    let max_payload = group
        .jobs
        .iter()
        .map(|job| job.coded_len as usize)
        .max()
        .unwrap_or(0);
    let tiny_limits = HtGpuJobChunkLimits::new(
        NonZeroUsize::MIN,
        max_payload,
        size_of::<J2kHtCleanupBatchJob>(),
    );
    let input = HtBatchInput {
        source_index: 4,
        payload: group.payload_source.as_ht_payload_source(),
        jobs: &group.jobs,
        output_base: 0,
        execution_owner: &group.execution_owner,
    };
    let chunk_plan = plan_metal_ht_chunks(&[input], tiny_limits).expect("forced chunk plan");
    assert!(
        chunk_plan.chunk_count() > 1,
        "fixture must split into chunks"
    );
    let cpu = decode_prepared_ht_sub_band_group_on_cpu_profile(group, None)
        .expect("CPU coefficient oracle");

    let gpu = with_runtime(|runtime| {
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
            tiny_limits,
        )?;
        encoder.endEncoding();
        commit_and_wait_metal(&command_buffer)?;
        validate_direct_status(runtime, status)?;
        let coefficients =
            checked_buffer_slice::<f32>(&output, group.total_coefficients, "chunk parity")?;
        drop(retained);
        Ok(coefficients)
    })
    .expect("forced multi-chunk Metal decode");

    assert_eq!(gpu, cpu);
}
