// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    compute, cpu_packetization_resolutions_from_lossless_device_plan, lossless_device_encode_plan,
    lossless_sample_shape, packet_descriptors_for_lossless_device_order,
    packetization_progression_order, validate_metal_encode_tile,
    validate_padded_contiguous_metal_encode_tile, EncodeBackendPreference, J2kBlockCodingMode,
    J2kLosslessEncodeOptions, J2kPacketizationEncodeJob, MetalEncodeInputStaging,
    MetalLosslessEncodeTile, PixelFormat, ReversibleTransform,
};

#[cfg(target_os = "macos")]
pub(super) struct ResidentHybridHtTileBody {
    pub(super) tile_data: Vec<u8>,
    pub(super) code_block_count: usize,
    pub(super) num_decomposition_levels: u8,
    pub(super) used_fused_rct: bool,
    pub(super) forward_dwt53_dispatches: usize,
    pub(super) ht_code_block_dispatches: usize,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_resident_ht_tile_body_with_cpu_packetization(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    code_block_width: u32,
    code_block_height: u32,
) -> Result<Option<ResidentHybridHtTileBody>, crate::Error> {
    if !should_try_resident_lossless_ht_cpu_packetization(tile, options, staging) {
        return Ok(None);
    }
    validate_metal_encode_tile(tile)?;
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    let bytes_per_sample =
        u8::try_from(tile.format.bytes_per_sample()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal resident hybrid bytes per sample exceeds u8".to_string(),
        })?;
    validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    let Some(plan) = lossless_device_encode_plan(
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        options,
        code_block_width,
        code_block_height,
    )?
    else {
        return Ok(None);
    };
    if plan.block_coding_mode != J2kBlockCodingMode::HighThroughput {
        return Ok(None);
    }

    let coefficient_count = lossless_device_coefficient_count(&plan.code_blocks)?;
    let prepared = compute::prepare_lossless_device_code_blocks(
        session,
        compute::J2kLosslessDevicePrepareJob {
            input: tile.buffer,
            input_byte_offset: tile.byte_offset,
            input_width: tile.width,
            input_height: tile.height,
            input_pitch_bytes: tile.pitch_bytes,
            output_width: tile.output_width,
            output_height: tile.output_height,
            component_count: components,
            bytes_per_sample,
            bit_depth,
            num_decomposition_levels: plan.num_decomposition_levels,
            coefficient_count,
        },
        plan.code_blocks.clone(),
    )?;
    let resident_tier1 =
        compute::encode_ht_prepared_device_code_blocks_resident(session, prepared)?;
    let encoded_blocks = compute::read_resident_ht_tier1_code_blocks_for_cpu_packetization(
        session,
        &resident_tier1,
    )?;
    let packetization_resolutions =
        cpu_packetization_resolutions_from_lossless_device_plan(&plan, &encoded_blocks)?;
    let packet_descriptors = packet_descriptors_for_lossless_device_order(
        plan.resolutions.len(),
        plan.components,
        plan.progression_order,
    )?;
    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: u32::try_from(plan.resolutions.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident hybrid resolution count exceeds u32".to_string(),
            }
        })?,
        num_layers: 1,
        num_components: u16::from(plan.components),
        code_block_count: u32::try_from(plan.code_blocks.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident hybrid code-block count exceeds u32".to_string(),
            }
        })?,
        progression_order: packetization_progression_order(plan.progression_order),
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let tile_data =
        j2k_native::encode_j2k_packetization_scalar(packetization_job).map_err(|reason| {
            crate::Error::MetalKernel {
                message: format!("J2K Metal resident hybrid CPU packetization failed: {reason}"),
            }
        })?;

    Ok(Some(ResidentHybridHtTileBody {
        tile_data,
        code_block_count: plan.code_blocks.len(),
        num_decomposition_levels: plan.num_decomposition_levels,
        used_fused_rct: plan.use_mct && tile.format == PixelFormat::Rgb8,
        forward_dwt53_dispatches: if plan.num_decomposition_levels > 0 {
            usize::from(plan.components)
        } else {
            0
        },
        ht_code_block_dispatches: usize::from(!plan.code_blocks.is_empty()),
    }))
}

#[cfg(target_os = "macos")]
pub(super) fn lossless_device_coefficient_count(
    code_blocks: &[compute::J2kLosslessDeviceCodeBlock],
) -> Result<usize, crate::Error> {
    let mut count = 0usize;
    for block in code_blocks {
        let offset =
            usize::try_from(block.coefficient_offset).map_err(|_| crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient offset exceeds usize".to_string(),
            })?;
        let block_count = (block.width as usize)
            .checked_mul(block.height as usize)
            .ok_or_else(|| crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient count overflow".to_string(),
            })?;
        count = count.max(offset.checked_add(block_count).ok_or_else(|| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient count overflow".to_string(),
            }
        })?);
    }
    Ok(count)
}

#[cfg(target_os = "macos")]
fn should_try_resident_lossless_ht_cpu_packetization(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    staging: MetalEncodeInputStaging,
) -> bool {
    options.backend == EncodeBackendPreference::Auto
        && options.block_coding_mode == J2kBlockCodingMode::HighThroughput
        && matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous)
        && matches!(tile.format, PixelFormat::Gray8 | PixelFormat::Rgb8)
        && (tile.format == PixelFormat::Gray8
            || options.reversible_transform == ReversibleTransform::Rct53)
}
