// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::prelude::*;

use j2k_native::J2kSubBandType;

use super::{
    copied_slice_buffer, label_compute_encoder, lossless_deinterleave_params,
    lossless_prepare_sizes, new_command_buffer, new_compute_command_encoder, size_of,
    take_recyclable_private_buffer, with_runtime_for_session, zeroed_shared_buffer, Error,
    J2kFusedDwt53Band, J2kFusedDwt53LevelParams, J2kLosslessDeviceCodeBlock,
    J2kLosslessDevicePrepareJob, J2kMctStatus, J2kPreparedLosslessDeviceCodeBlocks,
};

/// Input samples per side of a `j2k_forward_dwt53_fused_level` tile.
const FUSED_DWT53_TILE: u32 = 32;
const FUSED_DWT53_THREADS: usize = 256;

/// The `(width, height)` of each 5/3 level the forward transform runs.
pub(super) fn forward_dwt53_levels(
    job: J2kLosslessDevicePrepareJob<'_>,
) -> Result<Vec<(u32, u32)>, Error> {
    let level_count = usize::from(job.num_decomposition_levels);
    let mut levels = crate::batch_allocation::try_vec(level_count, "J2K Metal 5/3 levels")?;
    let (mut width, mut height) = (job.output_width, job.output_height);
    while levels.len() < level_count && (width >= 2 || height >= 2) {
        levels.push((width, height));
        (width, height) = (width.div_ceil(2), height.div_ceil(2));
    }
    Ok(levels)
}

/// Runs the RCT and every 5/3 level with one fused dispatch per level, which
/// writes the coefficients straight into their code blocks.
#[cfg(target_os = "macos")]
pub(super) fn prepare_fused_lossless_device_code_blocks(
    session: &crate::MetalBackendSession,
    job: J2kLosslessDevicePrepareJob<'_>,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    levels: &[(u32, u32)],
) -> Result<J2kPreparedLosslessDeviceCodeBlocks, Error> {
    let sizes = lossless_prepare_sizes(job)?;
    let (bands, block_width_log2, block_height_log2) =
        fused_dwt53_bands(job, levels, &code_blocks)?;
    let (input_byte_offset, input) = lossless_deinterleave_params(job)?;
    // Levels alternate between two LL planes; the second holds level 1's.
    let low_plane_bytes = |level: usize| {
        levels.get(level).map_or(1, |&(width, height)| {
            let low_samples = width.div_ceil(2) as usize * height.div_ceil(2) as usize;
            low_samples * usize::from(job.component_count) * size_of::<f32>()
        })
    };

    with_runtime_for_session(session, |runtime| {
        let pipeline = &runtime.encode()?.fdwt53_fused_level;
        let mut recyclable_private_buffers = Vec::new();
        let mut low_planes = crate::batch_allocation::try_vec(2, "J2K Metal 5/3 LL planes")?;
        for level in 0..2 {
            low_planes.push(take_recyclable_private_buffer(
                runtime,
                low_plane_bytes(level),
                &mut recyclable_private_buffers,
            )?);
        }
        let coefficient_buffer = take_recyclable_private_buffer(
            runtime,
            sizes.coefficient_bytes,
            &mut recyclable_private_buffers,
        )?;
        let band_buffer = copied_slice_buffer(&runtime.device, &bands)?;
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>())?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let threads = FUSED_DWT53_THREADS.min(pipeline.maxTotalThreadsPerThreadgroup());
        let mut first_band = 0;
        for (level, &(width, height)) in levels.iter().enumerate() {
            let params = J2kFusedDwt53LevelParams {
                input,
                current_width: width,
                current_height: height,
                code_block_width_log2: block_width_log2,
                code_block_height_log2: block_height_log2,
                first_band,
                from_samples: u32::from(level == 0),
                low_to_coefficients: u32::from(level + 1 == levels.len()),
            };
            first_band += 4 * input.components;
            let encoder = new_compute_command_encoder(&command_buffer)?;
            label_compute_encoder(&encoder, "J2K coefficient prep fused DWT 5/3 level");
            encoder.setComputePipelineState(pipeline);
            encoder.set_buffer(0, Some(job.input), input_byte_offset);
            encoder.set_buffer(1, Some(&low_planes[(level + 1) % 2]), 0);
            encoder.set_buffer(2, Some(&low_planes[level % 2]), 0);
            encoder.set_buffer(3, Some(&coefficient_buffer), 0);
            encoder.set_buffer(4, Some(&band_buffer), 0);
            encoder.set_bytes::<J2kFusedDwt53LevelParams>(5, &params);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                j2k_metal_support::mtl_size(
                    u64::from(width.div_ceil(FUSED_DWT53_TILE)),
                    u64::from(height.div_ceil(FUSED_DWT53_TILE)),
                    1,
                ),
                j2k_metal_support::mtl_size(threads as u64, 1, 1),
            );
            encoder.endEncoding();
        }
        command_buffer.commit();
        Ok(J2kPreparedLosslessDeviceCodeBlocks {
            coefficient_buffer,
            coefficient_byte_offset: 0,
            coefficient_byte_len: sizes.coefficient_bytes,
            coefficient_buffer_is_batch_shared: false,
            code_blocks,
            recyclable_private_buffers,
            _prepare_command_buffer: command_buffer,
            _prepare_deinterleave_rct_command_buffer: None,
            _prepare_dwt53_command_buffer: None,
            _prepare_dwt53_vertical_command_buffers: Vec::new(),
            _prepare_dwt53_horizontal_command_buffers: Vec::new(),
            _prepare_coefficient_extract_command_buffer: None,
            _deinterleave_status_buffer: status_buffer,
            _plane_buffers: low_planes,
            _scratch_buffers: Vec::new(),
            _coefficient_job_buffer: band_buffer,
        })
    })
}

/// Each level's LL, HL, LH and HH band per component, and the code-block size
/// exponents. Fails unless every code block sits where the kernel writes its
/// samples and every band the kernel writes fits the coefficient buffer.
fn fused_dwt53_bands(
    job: J2kLosslessDevicePrepareJob<'_>,
    levels: &[(u32, u32)],
    code_blocks: &[J2kLosslessDeviceCodeBlock],
) -> Result<(Vec<J2kFusedDwt53Band>, u32, u32), Error> {
    let components = usize::from(job.component_count);
    let mut bands =
        crate::batch_allocation::try_vec(levels.len() * components * 4, "J2K Metal 5/3 bands")?;
    for &(width, height) in levels {
        let (low_width, low_height) = (width.div_ceil(2), height.div_ceil(2));
        for _ in 0..components {
            for (width, height) in [
                (low_width, low_height),
                (width - low_width, low_height),
                (low_width, height - low_height),
                (width - low_width, height - low_height),
            ] {
                bands.push(J2kFusedDwt53Band {
                    coefficient_offset: u32::MAX,
                    width,
                    height,
                });
            }
        }
    }
    // Only a band's last block column or row is narrower than the nominal size.
    let nominal = |size: fn(&J2kLosslessDeviceCodeBlock) -> u32| {
        code_blocks
            .iter()
            .map(size)
            .max()
            .unwrap_or(1)
            .next_power_of_two()
    };
    let (block_width, block_height) = (nominal(|b| b.width), nominal(|b| b.height));
    let invalid = || Error::MetalStateInvariant {
        state: "J2K Metal fused 5/3 coefficient prep",
        reason: "code blocks do not follow the sub-band raster layout",
    };
    for block in code_blocks {
        let (level, band) = fused_dwt53_band(levels, block)
            .filter(|_| (block.component as usize) < components)
            .ok_or_else(invalid)?;
        let band = &mut bands[(level * components + block.component as usize) * 4 + band];
        if block.block_x == 0 && block.block_y == 0 {
            band.coefficient_offset = block.coefficient_offset;
        }
        let in_place = band.coefficient_offset != u32::MAX
            && block.block_x % block_width == 0
            && block.block_y % block_height == 0
            && block.block_x < band.width
            && block.block_y < band.height
            && block.width == block_width.min(band.width - block.block_x)
            && block.height == block_height.min(band.height - block.block_y)
            && u64::from(block.coefficient_offset)
                == u64::from(band.coefficient_offset)
                    + u64::from(block.block_y) * u64::from(band.width)
                    + u64::from(block.block_x) * u64::from(block.height);
        if !in_place {
            return Err(invalid());
        }
    }
    // The kernel writes every detail band and the last level's LL band.
    let last_level = levels.len().saturating_sub(1) * components * 4;
    for (index, band) in bands.iter().enumerate() {
        let samples = u64::from(band.width) * u64::from(band.height);
        let written = samples != 0 && (index % 4 != 0 || index >= last_level);
        if written && u64::from(band.coefficient_offset) + samples > job.coefficient_count as u64 {
            return Err(invalid());
        }
    }
    Ok((
        bands,
        block_width.trailing_zeros(),
        block_height.trailing_zeros(),
    ))
}

/// The level and band (LL 0, HL 1, LH 2, HH 3) in which the transform writes
/// a code block's sub-band; only the last level writes its LL band.
fn fused_dwt53_band(
    levels: &[(u32, u32)],
    block: &J2kLosslessDeviceCodeBlock,
) -> Option<(usize, usize)> {
    let band = match block.sub_band_type {
        J2kSubBandType::LowLow => 0,
        J2kSubBandType::HighLow => 1,
        J2kSubBandType::LowHigh => 2,
        J2kSubBandType::HighHigh => 3,
    };
    let holds_band = |&(width, height): &(u32, u32)| {
        let origin = (
            u32::from(band & 1 == 1) * width.div_ceil(2),
            u32::from(band >= 2) * height.div_ceil(2),
        );
        origin == (block.subband_x, block.subband_y)
            && (band & 1 == 0 || width >= 2)
            && (band < 2 || height >= 2)
    };
    let level = if band == 0 {
        levels
            .len()
            .checked_sub(1)
            .filter(|&last| holds_band(&levels[last]))
    } else {
        levels.iter().position(holds_band)
    };
    level.map(|level| (level, band))
}
