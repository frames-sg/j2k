// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::J2kBlockCodingMode;

use super::{
    LosslessDeviceEncodePlan, MetalEncodeInputStaging, ResidentLosslessBufferEncodeMetadata,
};
use crate::compute;

pub(super) fn checked_add_bytes(lhs: usize, rhs: usize) -> usize {
    lhs.saturating_add(rhs)
}

pub(super) fn checked_mul_bytes(lhs: usize, rhs: usize) -> usize {
    lhs.saturating_mul(rhs)
}

pub(super) fn estimate_resident_lossless_encode_peak_bytes(
    metadata: &ResidentLosslessBufferEncodeMetadata,
    coefficient_count: usize,
    staging: MetalEncodeInputStaging,
) -> usize {
    let pixels = checked_mul_bytes(
        metadata.tile.output_width as usize,
        metadata.tile.output_height as usize,
    )
    .max(1);
    let plane_bytes = checked_mul_bytes(pixels, core::mem::size_of::<f32>());
    let code_block_count = metadata.plan.code_blocks.len().max(1);
    let packet_count = metadata
        .packet_descriptors
        .len()
        .max(metadata.plan.resolutions.len())
        .max(1);
    let input_bytes = checked_mul_bytes(
        checked_mul_bytes(metadata.tile.width as usize, metadata.tile.height as usize),
        metadata.bytes_per_pixel,
    );
    let staged_input_bytes = if matches!(staging, MetalEncodeInputStaging::CopyAndPad) {
        checked_mul_bytes(pixels, metadata.bytes_per_pixel)
    } else {
        0
    };
    let coefficient_bytes =
        checked_mul_bytes(coefficient_count.max(1), core::mem::size_of::<i32>());
    let plane_buffers = checked_mul_bytes(3, plane_bytes);
    let scratch_buffers = checked_mul_bytes(usize::from(metadata.components), plane_bytes);
    let code_block_tables = checked_mul_bytes(code_block_count, 256);
    let tier1_output = estimated_tier1_output_bytes(&metadata.plan);
    let packet_header = checked_add_bytes(checked_mul_bytes(code_block_count, 256), 4096);
    let packet_output = checked_add_bytes(
        checked_add_bytes(tier1_output, checked_mul_bytes(packet_header, packet_count)),
        1024,
    );
    let codestream_capacity = checked_add_bytes(
        packet_output,
        checked_add_bytes(4096, checked_mul_bytes(pixels, metadata.bytes_per_pixel)),
    );
    let validation_bytes = checked_mul_bytes(pixels, metadata.bytes_per_pixel).saturating_mul(
        usize::from(metadata.plan.write_tlm || metadata.plan.use_mct || metadata.components > 0),
    );

    [
        input_bytes / 4,
        staged_input_bytes,
        plane_buffers,
        scratch_buffers,
        coefficient_bytes,
        code_block_tables,
        tier1_output,
        packet_output,
        codestream_capacity,
        validation_bytes,
        4 * 1024 * 1024,
    ]
    .into_iter()
    .fold(0usize, checked_add_bytes)
}

pub(super) fn estimated_tier1_output_bytes(plan: &LosslessDeviceEncodePlan) -> usize {
    fn estimated_ht_output_capacity(width: usize, height: usize) -> usize {
        const HT_MAX_SAMPLES: usize = 16_384;
        const HT_MEL_SIZE: usize = 192;
        const HT_VLC_SIZE: usize = 3072 - HT_MEL_SIZE;
        const HT_MS_SIZE: usize = (HT_MAX_SAMPLES * 16).div_ceil(15);
        const HT_MS_BYTES_PER_SAMPLE_FLOOR: usize = 5;

        let samples = checked_mul_bytes(width, height).min(HT_MAX_SAMPLES);
        let scaled_ms = checked_mul_bytes(HT_MS_SIZE, samples)
            .div_ceil(HT_MAX_SAMPLES)
            .max(1);
        let ms_floor = checked_mul_bytes(samples, HT_MS_BYTES_PER_SAMPLE_FLOOR);
        let ms_size = scaled_ms.max(ms_floor).min(HT_MS_SIZE);
        let fixed_entropy = checked_add_bytes(HT_MEL_SIZE, HT_VLC_SIZE);
        checked_add_bytes(ms_size, fixed_entropy)
    }

    plan.code_blocks
        .iter()
        .map(|block| match plan.block_coding_mode {
            J2kBlockCodingMode::HighThroughput => {
                estimated_ht_output_capacity(block.width as usize, block.height as usize)
            }
            J2kBlockCodingMode::Classic => {
                let samples = checked_mul_bytes(block.width as usize, block.height as usize);
                checked_add_bytes(
                    checked_mul_bytes(samples, usize::from(block.total_bitplanes).max(1)),
                    4097,
                )
                .max(4097)
            }
        })
        .fold(0usize, checked_add_bytes)
        .max(1)
}

pub(super) fn resident_codestream_assembly_job_for_metadata(
    metadata: &ResidentLosslessBufferEncodeMetadata,
) -> compute::J2kLosslessCodestreamAssemblyJob {
    compute::J2kLosslessCodestreamAssemblyJob {
        width: metadata.tile.output_width,
        height: metadata.tile.output_height,
        component_count: metadata.plan.components,
        bit_depth: metadata.plan.bit_depth,
        signed: false,
        num_decomposition_levels: metadata.plan.num_decomposition_levels,
        use_mct: metadata.plan.use_mct,
        guard_bits: metadata.plan.guard_bits,
        code_block_width_exp: metadata.plan.code_block_width_exp,
        code_block_height_exp: metadata.plan.code_block_height_exp,
        progression_order: metadata.plan.progression_order,
        write_tlm: metadata.plan.write_tlm,
        block_coding_mode: match metadata.plan.block_coding_mode {
            J2kBlockCodingMode::Classic => compute::J2kLosslessCodestreamBlockCodingMode::Classic,
            J2kBlockCodingMode::HighThroughput => {
                compute::J2kLosslessCodestreamBlockCodingMode::HighThroughput
            }
        },
    }
}

pub(super) fn resident_classic_batch_encode_should_retry_conservative(
    error: &crate::Error,
) -> bool {
    error.is_conservative_retry_candidate(crate::MetalKernelRetryClass::ResidentClassicBatch)
}

pub(super) fn resident_ht_batch_encode_should_retry_conservative(error: &crate::Error) -> bool {
    error.is_conservative_retry_candidate(crate::MetalKernelRetryClass::ResidentHtBatch)
}
