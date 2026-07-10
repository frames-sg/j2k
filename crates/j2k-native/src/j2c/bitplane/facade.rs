// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::build::{CodeBlock, SubBandType};
use super::super::codestream::CodeBlockStyle;
use super::super::decode::{DecompositionStorage, TileDecodeContext};
use super::observer::{
    J2kBlockDecodeStats, J2kDecodeObserver, NoJ2kDecodeStats, RecordingJ2kDecodeStats,
};
use super::schedule::{decode_code_block_segments_inner, decode_inner};
use super::state::BitPlaneDecodeContext;
use crate::error::{DecodingError, Result};
use crate::J2kCodeBlockSegment;

/// Decode the layers of the given code block into coefficients.
///
/// The result will be stored in the form of a vector of signs and magnitudes
/// in the bitplane decoder context.
pub(crate) fn decode(
    code_block: &CodeBlock,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
    tile_ctx: &mut TileDecodeContext,
    storage: &DecompositionStorage<'_>,
    strict: bool,
) -> Result<()> {
    tile_ctx.bit_plane_decode_context.reset(
        code_block,
        sub_band_type,
        style,
        total_bitplanes,
        strict,
    )?;
    tile_ctx.bit_plane_decode_buffers.reset();

    decode_inner(
        code_block,
        storage,
        &mut tile_ctx.bit_plane_decode_context,
        &mut tile_ctx.bit_plane_decode_buffers,
    )
    .ok_or(DecodingError::CodeBlockDecodeFailure)?;

    Ok(())
}

pub(crate) fn decode_code_block_segments_validated(
    data: &[u8],
    segments: &[J2kCodeBlockSegment],
    width: u32,
    height: u32,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
    total_bitplanes: u8,
    sub_band_type: SubBandType,
    code_block_style: &CodeBlockStyle,
    strict: bool,
    ctx: &mut BitPlaneDecodeContext,
) -> Result<()> {
    let mut observer = NoJ2kDecodeStats;
    decode_code_block_segments_validated_with_observer(
        data,
        segments,
        width,
        height,
        missing_bit_planes,
        number_of_coding_passes,
        total_bitplanes,
        sub_band_type,
        code_block_style,
        strict,
        ctx,
        &mut observer,
    )
}

pub(crate) fn decode_code_block_segments_validated_profiled(
    data: &[u8],
    segments: &[J2kCodeBlockSegment],
    width: u32,
    height: u32,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
    total_bitplanes: u8,
    sub_band_type: SubBandType,
    code_block_style: &CodeBlockStyle,
    strict: bool,
    ctx: &mut BitPlaneDecodeContext,
    stats: &mut J2kBlockDecodeStats,
    profile_enabled: bool,
) -> Result<()> {
    let mut observer = RecordingJ2kDecodeStats {
        stats,
        profile_enabled,
    };
    decode_code_block_segments_validated_with_observer(
        data,
        segments,
        width,
        height,
        missing_bit_planes,
        number_of_coding_passes,
        total_bitplanes,
        sub_band_type,
        code_block_style,
        strict,
        ctx,
        &mut observer,
    )
}

pub(super) fn decode_code_block_segments_validated_with_observer<O: J2kDecodeObserver>(
    data: &[u8],
    segments: &[J2kCodeBlockSegment],
    width: u32,
    height: u32,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
    total_bitplanes: u8,
    sub_band_type: SubBandType,
    code_block_style: &CodeBlockStyle,
    strict: bool,
    ctx: &mut BitPlaneDecodeContext,
    observer: &mut O,
) -> Result<()> {
    ctx.reset_for_job(
        width,
        height,
        missing_bit_planes,
        number_of_coding_passes,
        sub_band_type,
        code_block_style,
        total_bitplanes,
        strict,
    )?;

    if number_of_coding_passes == 0 || ctx.bitplanes == 0 {
        return Ok(());
    }

    decode_code_block_segments_inner(data, segments, number_of_coding_passes, ctx, observer)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;

    Ok(())
}
