// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{J2kBlockCodingMode, J2kLosslessEncodeOptions};
use j2k_native::{J2kHtj2kTileEncodeJob, J2kPacketizationEncodeJob};

use super::{
    compute, cpu_packetization_resolutions_from_lossless_device_plan,
    packet_descriptors_for_lossless_device_order, packetization_progression_order,
};

pub(super) struct ResidentLossyHtTile {
    pub(super) data: Vec<u8>,
    pub(super) required_magnitude_bound: u8,
    pub(super) code_block_count: usize,
}

fn supports_resident_lossy_job(job: J2kHtj2kTileEncodeJob<'_>) -> bool {
    let samples = u64::from(job.width) * u64::from(job.height);
    !(job.reversible
        || job.signed
        || job.bit_depth != 8
        || !matches!(job.num_components, 1 | 3)
        || job.use_mct != (job.num_components == 3)
        || !matches!(job.code_block_width, 32 | 64)
        || job.code_block_height != job.code_block_width
        || samples == 0
        || samples > 16 * 1024 * 1024
        || job.quantization_steps.len() != 1 + usize::from(job.num_decomposition_levels) * 3
        || job.component_sampling.len() != usize::from(job.num_components)
        || job
            .component_sampling
            .iter()
            .any(|sampling| *sampling != (1, 1))
        || samples * u64::from(job.num_components) != job.pixels.len() as u64
        || job.num_decomposition_levels
            > j2k_types::encode_geometry::maximum_decomposition_levels(job.width, job.height))
}

pub(super) fn encode_resident_lossy_ht_tile(
    job: J2kHtj2kTileEncodeJob<'_>,
) -> Result<Option<ResidentLossyHtTile>, crate::Error> {
    if !supports_resident_lossy_job(job) {
        return Ok(None);
    }
    let options = J2kLosslessEncodeOptions::default()
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_progression(
            super::packet_plan::lossless_progression_from_packetization_order(
                job.progression_order,
            ),
        );
    // Both transforms have the same origin-zero band and code-block geometry.
    // Only this layout is reused; irreversible bitplanes come from the caller's
    // quantization metadata below, and native assembly retains its 9/7 headers.
    let Some(mut plan) = super::plan::device_encode_plan_with_levels(
        (job.width, job.height),
        u8::try_from(job.num_components).map_err(|_| plan_error("component count"))?,
        job.bit_depth,
        options,
        (job.code_block_width, job.code_block_height),
        job.num_decomposition_levels,
    )?
    else {
        return Ok(None);
    };
    let mut steps =
        crate::batch_allocation::try_vec(plan.code_blocks.len(), "Metal lossy quantization jobs")?;
    steps.resize(plan.code_blocks.len(), (0, 0, 0));
    for (index, resolution) in plan.resolutions.iter().enumerate() {
        let resolution_index = index / usize::from(job.num_components);
        for (band_index, band) in resolution.subbands.iter().enumerate() {
            let quant_index = if resolution_index == 0 {
                0
            } else {
                1 + (resolution_index - 1) * 3 + band_index
            };
            let (exponent, mantissa) = job.quantization_steps[quant_index];
            let total_bitplanes = u16::from(job.guard_bits)
                .checked_add(exponent)
                .and_then(|value| value.checked_sub(1));
            if exponent > 31
                || mantissa > 2047
                || !total_bitplanes.is_some_and(|bits| (1..=31).contains(&bits))
            {
                return Ok(None);
            }
            let range = band.code_block_start..band.code_block_start + band.code_block_count;
            for (block, step) in plan.code_blocks[range.clone()]
                .iter_mut()
                .zip(&mut steps[range])
            {
                block.total_bitplanes = u8::try_from(total_bitplanes.unwrap_or(0))
                    .map_err(|_| plan_error("bitplanes"))?;
                *step = (
                    exponent,
                    mantissa,
                    if resolution_index == 0 {
                        job.num_decomposition_levels
                    } else {
                        job.num_decomposition_levels + 1
                            - u8::try_from(resolution_index)
                                .map_err(|_| plan_error("resolution index"))?
                    },
                );
            }
        }
    }
    let (blocks, required_magnitude_bound) =
        compute::encode_resident_lossy_ht_blocks(job, &plan.code_blocks, &steps)?;
    let data = packetize_lossy(&plan, &blocks)?;
    Ok(Some(ResidentLossyHtTile {
        data,
        required_magnitude_bound,
        code_block_count: blocks.len(),
    }))
}

fn packetize_lossy(
    plan: &super::plan::LosslessDeviceEncodePlan,
    blocks: &[j2k_native::EncodedHtJ2kCodeBlock],
) -> Result<Vec<u8>, crate::Error> {
    let resolutions =
        cpu_packetization_resolutions_from_lossless_device_plan(plan, blocks.len(), blocks)?;
    let descriptors = packet_descriptors_for_lossless_device_order(
        plan.resolutions.len(),
        plan.components,
        plan.progression_order,
    )?;
    let packet_job = J2kPacketizationEncodeJob {
        resolution_count: u32::try_from(plan.resolutions.len())
            .map_err(|_| plan_error("resolution count"))?,
        num_layers: 1,
        num_components: u16::from(plan.components),
        code_block_count: u32::try_from(blocks.len()).map_err(|_| plan_error("block count"))?,
        progression_order: packetization_progression_order(plan.progression_order),
        packet_descriptors: &descriptors,
        resolutions: &resolutions,
    };
    // The explicit device route requires real Metal packetization as well as
    // transforms and entropy coding. Reuse its validated packet encoder.
    compute::encode_tier2_packetization(packet_job)
}

fn plan_error(field: &str) -> crate::Error {
    crate::Error::MetalKernel {
        message: format!("Metal resident lossy {field} exceeds GPU metadata limits"),
    }
}
