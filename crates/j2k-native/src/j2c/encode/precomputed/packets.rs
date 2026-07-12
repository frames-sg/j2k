// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible move-or-copy construction for legacy 9/7 packet inputs.

use super::super::PreparedCodeBlockCoefficients;
use super::allocation::ConstructionTracker;
use super::{
    internal_sub_band_type, BlockCodingMode, NativeEncodePipelineError, NativeEncodePipelineResult,
    PreencodedHtj2k97CodeBlock, PreencodedHtj2k97Component, PreencodedHtj2k97Image,
    PreencodedHtj2k97Subband, PreparedEncodeCodeBlock, PreparedEncodeSubband,
    PreparedResolutionPacket, PrequantizedHtj2k97Component, PrequantizedHtj2k97Subband, Range, Vec,
};

pub(in crate::j2c::encode) fn try_prepared_packets_from_prequantized_component(
    component_idx: usize,
    component: &PrequantizedHtj2k97Component,
    tracker: &mut ConstructionTracker<'_, '_>,
) -> NativeEncodePipelineResult<Vec<PreparedResolutionPacket>> {
    let component_idx = u16::try_from(component_idx).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("component index exceeds u16")
    })?;
    let mut packets = tracker.try_vec::<PreparedResolutionPacket>(
        component.resolutions.len(),
        "prequantized 9/7 prepared resolutions",
    )?;
    for (resolution_idx, resolution) in component.resolutions.iter().enumerate() {
        let mut subbands = tracker.try_vec::<PreparedEncodeSubband>(
            resolution.subbands.len(),
            "prequantized 9/7 prepared subbands",
        )?;
        for subband in &resolution.subbands {
            subbands.push(try_prepared_subband_from_prequantized(subband, tracker)?);
        }
        packets.push(PreparedResolutionPacket {
            component: component_idx,
            resolution: u32::try_from(resolution_idx).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("resolution index exceeds u32")
            })?,
            precinct: 0,
            subbands,
        });
    }
    Ok(packets)
}

fn try_prepared_subband_from_prequantized(
    subband: &PrequantizedHtj2k97Subband,
    tracker: &mut ConstructionTracker<'_, '_>,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let mut code_blocks = tracker.try_vec::<PreparedEncodeCodeBlock>(
        subband.code_blocks.len(),
        "prequantized 9/7 prepared code blocks",
    )?;
    for block in &subband.code_blocks {
        code_blocks.push(PreparedEncodeCodeBlock {
            coefficients: PreparedCodeBlockCoefficients::I32(
                tracker
                    .try_copy_slice(&block.coefficients, "prequantized 9/7 copied coefficients")?,
            ),
            width: block.width,
            height: block.height,
        });
    }
    prepared_subband_metadata(
        subband.num_cbs_x,
        subband.num_cbs_y,
        subband.total_bitplanes,
        subband.sub_band_type,
        code_blocks,
        None,
    )
}

pub(in crate::j2c::encode) fn try_prepared_packets_from_preencoded_component(
    component_idx: usize,
    component: &PreencodedHtj2k97Component,
    tracker: &mut ConstructionTracker<'_, '_>,
) -> NativeEncodePipelineResult<Vec<PreparedResolutionPacket>> {
    let component_idx = u16::try_from(component_idx).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("component index exceeds u16")
    })?;
    let mut packets = tracker.try_vec::<PreparedResolutionPacket>(
        component.resolutions.len(),
        "preencoded 9/7 prepared resolutions",
    )?;
    for (resolution_idx, resolution) in component.resolutions.iter().enumerate() {
        let mut subbands = tracker.try_vec::<PreparedEncodeSubband>(
            resolution.subbands.len(),
            "preencoded 9/7 prepared subbands",
        )?;
        for subband in &resolution.subbands {
            subbands.push(try_prepared_subband_from_preencoded(subband, tracker)?);
        }
        packets.push(PreparedResolutionPacket {
            component: component_idx,
            resolution: u32::try_from(resolution_idx).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("resolution index exceeds u32")
            })?,
            precinct: 0,
            subbands,
        });
    }
    Ok(packets)
}

fn try_prepared_subband_from_preencoded(
    subband: &PreencodedHtj2k97Subband,
    tracker: &mut ConstructionTracker<'_, '_>,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let mut code_blocks = tracker.try_vec::<PreparedEncodeCodeBlock>(
        subband.code_blocks.len(),
        "preencoded 9/7 prepared code-block shapes",
    )?;
    let mut encoded = tracker.try_vec::<crate::EncodedHtJ2kCodeBlock>(
        subband.code_blocks.len(),
        "preencoded 9/7 copied payload owners",
    )?;
    for block in &subband.code_blocks {
        code_blocks.push(empty_code_block_shape(block));
        encoded.push(crate::EncodedHtJ2kCodeBlock {
            data: tracker.try_copy_slice(&block.encoded.data, "preencoded 9/7 copied payload")?,
            cleanup_length: block.encoded.cleanup_length,
            refinement_length: block.encoded.refinement_length,
            num_coding_passes: block.encoded.num_coding_passes,
            num_zero_bitplanes: block.encoded.num_zero_bitplanes,
        });
    }
    prepared_subband_from_shapes(subband, code_blocks, Some(encoded))
}

/// Allocate and reconcile the complete destination owner graph while the
/// source image remains borrowed. Payload vectors are left empty but reserved;
/// they are filled by move only after the construction session is released.
pub(in crate::j2c::encode) fn try_preencoded_owned_skeleton(
    image: &PreencodedHtj2k97Image,
    tracker: &mut ConstructionTracker<'_, '_>,
) -> NativeEncodePipelineResult<Vec<Vec<PreparedResolutionPacket>>> {
    let mut components = tracker.try_vec::<Vec<PreparedResolutionPacket>>(
        image.components.len(),
        "owned preencoded 9/7 prepared component owners",
    )?;
    for (component_idx, component) in image.components.iter().enumerate() {
        let component_idx = u16::try_from(component_idx).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("component index exceeds u16")
        })?;
        let mut packets = tracker.try_vec::<PreparedResolutionPacket>(
            component.resolutions.len(),
            "owned preencoded 9/7 prepared resolutions",
        )?;
        for (resolution_idx, resolution) in component.resolutions.iter().enumerate() {
            let mut subbands = tracker.try_vec::<PreparedEncodeSubband>(
                resolution.subbands.len(),
                "owned preencoded 9/7 prepared subbands",
            )?;
            for subband in &resolution.subbands {
                let mut shapes = tracker.try_vec::<PreparedEncodeCodeBlock>(
                    subband.code_blocks.len(),
                    "owned preencoded 9/7 code-block shapes",
                )?;
                for block in &subband.code_blocks {
                    shapes.push(empty_code_block_shape(block));
                }
                let encoded = tracker.try_vec::<crate::EncodedHtJ2kCodeBlock>(
                    subband.code_blocks.len(),
                    "owned preencoded 9/7 payload owners",
                )?;
                subbands.push(prepared_subband_from_shapes(
                    subband,
                    shapes,
                    Some(encoded),
                )?);
            }
            packets.push(PreparedResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx).map_err(|_| {
                    NativeEncodePipelineError::arithmetic_overflow("resolution index exceeds u32")
                })?,
                precinct: 0,
                subbands,
            });
        }
        components.push(packets);
    }
    Ok(components)
}

pub(in crate::j2c::encode) fn move_preencoded_payloads_into_skeleton(
    image: PreencodedHtj2k97Image,
    prepared_components: &mut [Vec<PreparedResolutionPacket>],
) -> Result<(), &'static str> {
    if image.components.len() != prepared_components.len() {
        return Err("preencoded component count changed during preparation");
    }
    for (source_component, target_packets) in image.components.into_iter().zip(prepared_components)
    {
        if source_component.resolutions.len() != target_packets.len() {
            return Err("preencoded resolution count changed during preparation");
        }
        for (source_resolution, target_packet) in
            source_component.resolutions.into_iter().zip(target_packets)
        {
            if source_resolution.subbands.len() != target_packet.subbands.len() {
                return Err("preencoded subband count changed during preparation");
            }
            for (source_subband, target_subband) in source_resolution
                .subbands
                .into_iter()
                .zip(&mut target_packet.subbands)
            {
                if source_subband.code_blocks.len() != target_subband.code_blocks.len() {
                    return Err("preencoded code-block count changed during preparation");
                }
                let target_payloads = target_subband
                    .preencoded_ht_code_blocks
                    .as_mut()
                    .ok_or("preencoded payload destination missing")?;
                if !target_payloads.is_empty() {
                    return Err("preencoded payload destination was not empty");
                }
                if target_payloads.capacity() < source_subband.code_blocks.len() {
                    return Err("preencoded payload destination capacity changed");
                }
                for source_block in source_subband.code_blocks {
                    target_payloads.push(source_block.encoded);
                }
            }
        }
    }
    Ok(())
}

fn empty_code_block_shape(block: &PreencodedHtj2k97CodeBlock) -> PreparedEncodeCodeBlock {
    PreparedEncodeCodeBlock {
        coefficients: PreparedCodeBlockCoefficients::Empty,
        width: block.width,
        height: block.height,
    }
}

fn prepared_subband_from_shapes(
    subband: &PreencodedHtj2k97Subband,
    code_blocks: Vec<PreparedEncodeCodeBlock>,
    encoded: Option<Vec<crate::EncodedHtJ2kCodeBlock>>,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    prepared_subband_metadata(
        subband.num_cbs_x,
        subband.num_cbs_y,
        subband.total_bitplanes,
        subband.sub_band_type,
        code_blocks,
        encoded,
    )
}

fn prepared_subband_metadata(
    num_cbs_x: u32,
    num_cbs_y: u32,
    total_bitplanes: u8,
    sub_band_type: crate::J2kSubBandType,
    code_blocks: Vec<PreparedEncodeCodeBlock>,
    encoded: Option<Vec<crate::EncodedHtJ2kCodeBlock>>,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let code_block_width = code_blocks
        .iter()
        .map(|block| block.width)
        .max()
        .unwrap_or(0);
    let code_block_height = code_blocks
        .iter()
        .map(|block| block.height)
        .max()
        .unwrap_or(0);
    let width = precomputed_subband_width(num_cbs_x, code_blocks.iter().map(|block| block.width))
        .map_err(NativeEncodePipelineError::arithmetic_overflow)?;
    let height = precomputed_subband_height(
        num_cbs_x,
        num_cbs_y,
        code_blocks.iter().map(|block| block.height),
    )
    .map_err(NativeEncodePipelineError::arithmetic_overflow)?;
    Ok(PreparedEncodeSubband {
        code_block_width,
        code_block_height,
        width,
        height,
        code_blocks,
        preencoded_ht_code_blocks: encoded,
        num_cbs_x,
        num_cbs_y,
        sub_band_type: internal_sub_band_type(sub_band_type),
        total_bitplanes,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 1,
    })
}

pub(in crate::j2c::encode) fn precomputed_subband_width(
    width_in_blocks: u32,
    widths: impl Iterator<Item = u32>,
) -> Result<u32, &'static str> {
    if width_in_blocks == 0 {
        return Ok(0);
    }
    widths
        .take(width_in_blocks as usize)
        .try_fold(0_u32, |width, block_width| {
            width
                .checked_add(block_width)
                .ok_or("precomputed subband width overflow")
        })
}

pub(in crate::j2c::encode) fn precomputed_subband_height(
    width_in_blocks: u32,
    height_in_blocks: u32,
    heights: impl Iterator<Item = u32>,
) -> Result<u32, &'static str> {
    if width_in_blocks == 0 || height_in_blocks == 0 {
        return Ok(0);
    }
    heights
        .step_by(width_in_blocks as usize)
        .take(height_in_blocks as usize)
        .try_fold(0_u32, |height, block_height| {
            height
                .checked_add(block_height)
                .ok_or("precomputed subband height overflow")
        })
}

pub(in crate::j2c::encode) fn compact_payload_slice<'a>(
    payload: &'a [u8],
    range: &Range<usize>,
) -> Result<&'a [u8], &'static str> {
    if range.start > range.end || range.end > payload.len() {
        return Err("HTJ2K payload range out of bounds");
    }
    Ok(&payload[range.start..range.end])
}

#[cfg(test)]
mod test_support;
#[cfg(test)]
pub(in crate::j2c::encode) use self::test_support::prepared_subband_from_preencoded_owned_for_test;
