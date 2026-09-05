// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{J2kBlockCodingMode, J2kLosslessEncodeOptions};
use j2k_native::{J2kHtj2kTileEncodeJob, J2kPacketizationEncodeJob};

use super::{compute, lossless_device_encode_plan,
    cpu_packetization_resolutions_from_lossless_device_plan,
    packet_descriptors_for_lossless_device_order, packetization_progression_order};

pub(super) struct ResidentLossyHtTile {
    pub(super) data: Vec<u8>,
    pub(super) required_magnitude_bound: u8,
}

pub(super) fn encode_resident_lossy_ht_tile(
    job: J2kHtj2kTileEncodeJob<'_>,
) -> Result<Option<ResidentLossyHtTile>, crate::Error> {
    let samples = u64::from(job.width) * u64::from(job.height);
    if job.reversible || job.signed || job.bit_depth != 8
        || !matches!(job.num_components, 1 | 3)
        || job.use_mct != (job.num_components == 3)
        || !matches!(job.code_block_width, 32 | 64)
        || job.code_block_height != job.code_block_width
        || samples == 0 || samples > 16 * 1024 * 1024
        || job.quantization_steps.len() != 1 + usize::from(job.num_decomposition_levels) * 3
        || job.component_sampling.len() != usize::from(job.num_components)
        || job.component_sampling.iter().any(|sampling| *sampling != (1,1))
        || samples * u64::from(job.num_components) != job.pixels.len() as u64
    { return Ok(None); }
    let options = J2kLosslessEncodeOptions::default()
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(job.num_decomposition_levels))
        .with_progression(super::packet_plan::lossless_progression_from_packetization_order(job.progression_order));
    // Both transforms have the same origin-zero band and code-block geometry.
    // Only this layout is reused; irreversible bitplanes come from the caller's
    // quantization metadata below, and native assembly retains its 9/7 headers.
    let Some(mut plan) = lossless_device_encode_plan(job.width,job.height,job.num_components as u8,
        job.bit_depth,options,job.code_block_width,job.code_block_height)? else { return Ok(None); };
    if plan.num_decomposition_levels != job.num_decomposition_levels { return Ok(None); }
    let mut steps = crate::batch_allocation::try_vec(plan.code_blocks.len(), "Metal lossy quantization jobs")?;
    steps.resize(plan.code_blocks.len(), (0,0));
    for (index,resolution) in plan.resolutions.iter().enumerate() {
        let resolution_index = index / usize::from(job.num_components);
        for (band_index,band) in resolution.subbands.iter().enumerate() {
            let quant_index = if resolution_index == 0 { 0 } else { 1+(resolution_index-1)*3+band_index };
            let (exponent,mantissa) = job.quantization_steps[quant_index];
            let total_bitplanes = u16::from(job.guard_bits).checked_add(exponent).and_then(|value| value.checked_sub(1));
            if exponent > 31 || mantissa > 2047 || !total_bitplanes.is_some_and(|bits| (1..=31).contains(&bits)) {
                return Ok(None);
            }
            for index in band.code_block_start..band.code_block_start+band.code_block_count {
                plan.code_blocks[index].total_bitplanes = u8::try_from(total_bitplanes.unwrap_or(0))
                    .map_err(|_| crate::Error::MetalKernel { message: "Metal lossy bitplanes exceed u8".to_owned() })?;
                steps[index] = (exponent,mantissa);
            }
        }
    }
    let (blocks, required_magnitude_bound) = compute::encode_resident_lossy_ht_blocks(job,&plan.code_blocks,&steps)?;
    let resolutions = cpu_packetization_resolutions_from_lossless_device_plan(&plan,blocks.len(),&blocks)?;
    let descriptors = packet_descriptors_for_lossless_device_order(plan.resolutions.len(),plan.components,plan.progression_order)?;
    let packet_job = J2kPacketizationEncodeJob {
        resolution_count: plan.resolutions.len() as u32, num_layers:1,
        num_components:job.num_components, code_block_count:blocks.len() as u32,
        progression_order:packetization_progression_order(plan.progression_order),
        packet_descriptors:&descriptors, resolutions:&resolutions,
    };
    let data = j2k_native::encode_j2k_packetization_scalar(packet_job)
        .map_err(|source| crate::Error::MetalKernel { message:format!("Metal lossy packetization: {source}") })?;
    Ok(Some(ResidentLossyHtTile { data,required_magnitude_bound }))
}
