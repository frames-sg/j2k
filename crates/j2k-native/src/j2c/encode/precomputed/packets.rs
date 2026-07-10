// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    internal_sub_band_type, raw_pixel_bytes_per_sample, vec, BlockCodingMode,
    PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Subband, PreparedCompactCodeBlock,
    PreparedCompactResolutionPacket, PreparedCompactSubband, PreparedEncodeCodeBlock,
    PreparedEncodeSubband, PreparedResolutionPacket, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Subband, Range, Vec,
};

pub(in crate::j2c::encode) fn prepared_resolution_packets_from_prequantized_component(
    component_idx: usize,
    component: &PrequantizedHtj2k97Component,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .iter()
                    .map(prepared_subband_from_prequantized)
                    .collect(),
            })
        })
        .collect()
}

pub(in crate::j2c::encode) fn prepared_subband_from_prequantized(
    subband: &PrequantizedHtj2k97Subband,
) -> PreparedEncodeSubband {
    PreparedEncodeSubband {
        code_blocks: subband
            .code_blocks
            .iter()
            .map(|block| PreparedEncodeCodeBlock {
                coefficients: block.coefficients.iter().copied().map(i64::from).collect(),
                width: block.width,
                height: block.height,
            })
            .collect(),
        preencoded_ht_code_blocks: None,
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
        code_block_width: subband
            .code_blocks
            .iter()
            .map(|block| block.width)
            .max()
            .unwrap_or(0),
        code_block_height: subband
            .code_blocks
            .iter()
            .map(|block| block.height)
            .max()
            .unwrap_or(0),
        width: precomputed_subband_width(
            subband.num_cbs_x,
            subband.code_blocks.iter().map(|block| block.width),
        ),
        height: precomputed_subband_height(
            subband.num_cbs_x,
            subband.num_cbs_y,
            subband.code_blocks.iter().map(|block| block.height),
        ),
        sub_band_type: internal_sub_band_type(subband.sub_band_type),
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 1,
    }
}

pub(in crate::j2c::encode) fn precomputed_subband_width(
    width_in_blocks: u32,
    widths: impl Iterator<Item = u32>,
) -> u32 {
    if width_in_blocks == 0 {
        return 0;
    }

    widths.take(width_in_blocks as usize).sum()
}

pub(in crate::j2c::encode) fn precomputed_subband_height(
    width_in_blocks: u32,
    height_in_blocks: u32,
    heights: impl Iterator<Item = u32>,
) -> u32 {
    if width_in_blocks == 0 || height_in_blocks == 0 {
        return 0;
    }

    heights
        .step_by(width_in_blocks as usize)
        .take(height_in_blocks as usize)
        .sum()
}

pub(in crate::j2c::encode) fn prepared_resolution_packets_from_preencoded_component(
    component_idx: usize,
    component: &PreencodedHtj2k97Component,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .iter()
                    .map(prepared_subband_from_preencoded)
                    .collect(),
            })
        })
        .collect()
}

pub(in crate::j2c::encode) fn prepared_resolution_packets_from_preencoded_component_owned(
    component_idx: usize,
    component: PreencodedHtj2k97Component,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .into_iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .into_iter()
                    .map(prepared_subband_from_preencoded_owned)
                    .collect(),
            })
        })
        .collect()
}

pub(in crate::j2c::encode) fn prepared_resolution_packets_from_preencoded_compact_component<'a>(
    component_idx: usize,
    component: &'a PreencodedHtj2k97CompactComponent,
    payload: &'a [u8],
) -> Result<Vec<PreparedCompactResolutionPacket<'a>>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedCompactResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .iter()
                    .map(|subband| prepared_subband_from_preencoded_compact(subband, payload))
                    .collect::<Result<Vec<_>, &'static str>>()?,
            })
        })
        .collect()
}

pub(in crate::j2c::encode) fn prepared_subband_from_preencoded(
    subband: &PreencodedHtj2k97Subband,
) -> PreparedEncodeSubband {
    PreparedEncodeSubband {
        code_blocks: subband
            .code_blocks
            .iter()
            .map(|block| PreparedEncodeCodeBlock {
                coefficients: Vec::new(),
                width: block.width,
                height: block.height,
            })
            .collect(),
        preencoded_ht_code_blocks: Some(
            subband
                .code_blocks
                .iter()
                .map(|block| block.encoded.clone())
                .collect(),
        ),
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
        code_block_width: subband
            .code_blocks
            .iter()
            .map(|block| block.width)
            .max()
            .unwrap_or(0),
        code_block_height: subband
            .code_blocks
            .iter()
            .map(|block| block.height)
            .max()
            .unwrap_or(0),
        width: precomputed_subband_width(
            subband.num_cbs_x,
            subband.code_blocks.iter().map(|block| block.width),
        ),
        height: precomputed_subband_height(
            subband.num_cbs_x,
            subband.num_cbs_y,
            subband.code_blocks.iter().map(|block| block.height),
        ),
        sub_band_type: internal_sub_band_type(subband.sub_band_type),
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 1,
    }
}

pub(in crate::j2c::encode) fn prepared_subband_from_preencoded_owned(
    subband: PreencodedHtj2k97Subband,
) -> PreparedEncodeSubband {
    let code_block_width = subband
        .code_blocks
        .iter()
        .map(|block| block.width)
        .max()
        .unwrap_or(0);
    let code_block_height = subband
        .code_blocks
        .iter()
        .map(|block| block.height)
        .max()
        .unwrap_or(0);
    let width = precomputed_subband_width(
        subband.num_cbs_x,
        subband.code_blocks.iter().map(|block| block.width),
    );
    let height = precomputed_subband_height(
        subband.num_cbs_x,
        subband.num_cbs_y,
        subband.code_blocks.iter().map(|block| block.height),
    );
    let code_blocks = subband
        .code_blocks
        .into_iter()
        .map(|block| {
            let PreencodedHtj2k97CodeBlock {
                width,
                height,
                encoded,
            } = block;
            (
                PreparedEncodeCodeBlock {
                    coefficients: Vec::new(),
                    width,
                    height,
                },
                encoded,
            )
        })
        .collect::<Vec<_>>();
    let (code_blocks, preencoded_ht_code_blocks): (Vec<_>, Vec<_>) =
        code_blocks.into_iter().unzip();

    PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks: Some(preencoded_ht_code_blocks),
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
        code_block_width,
        code_block_height,
        width,
        height,
        sub_band_type: internal_sub_band_type(subband.sub_band_type),
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 1,
    }
}

pub(in crate::j2c::encode) fn prepared_subband_from_preencoded_compact<'a>(
    subband: &'a PreencodedHtj2k97CompactSubband,
    payload: &'a [u8],
) -> Result<PreparedCompactSubband<'a>, &'static str> {
    let code_blocks = subband
        .code_blocks
        .iter()
        .map(|block| {
            Ok(PreparedCompactCodeBlock {
                data: compact_payload_slice(payload, &block.payload_range)?,
                cleanup_length: block.cleanup_length,
                refinement_length: block.refinement_length,
                num_coding_passes: block.num_coding_passes,
                num_zero_bitplanes: block.num_zero_bitplanes,
            })
        })
        .collect::<Result<Vec<_>, &'static str>>()?;

    Ok(PreparedCompactSubband {
        code_blocks,
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
    })
}

pub(in crate::j2c::encode) fn compact_payload_slice<'a>(
    payload: &'a [u8],
    range: &Range<usize>,
) -> Result<&'a [u8], &'static str> {
    if range.start > range.end || range.end > payload.len() {
        return Err("HTJ2K payload range out of bounds");
    }
    Ok(&payload[range.clone()])
}

pub(in crate::j2c::encode) fn zero_pixel_buffer(
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
) -> Result<Vec<u8>, &'static str> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)?;
    let len = width as usize;
    let len = len
        .checked_mul(height as usize)
        .and_then(|value| value.checked_mul(usize::from(num_components)))
        .and_then(|value| value.checked_mul(bytes_per_sample))
        .ok_or("pixel buffer dimensions overflow")?;
    Ok(vec![0; len])
}
