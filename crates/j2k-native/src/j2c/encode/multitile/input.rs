// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tile extraction and ROI clipping for multi-tile encode.

use alloc::vec::Vec;

use super::super::allocation::{checked_add_bytes, host_allocation_failed};
use super::super::{
    raw_pixel_bytes_per_sample, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeSession,
};

mod roi;
pub(super) use roi::roi_regions_for_tile;

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
pub(super) fn extract_interleaved_tile(
    pixels: &[u8],
    image_width: u32,
    x0: u32,
    y0: u32,
    tile_width: u32,
    tile_height: u32,
    num_components: u16,
    bit_depth: u8,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let bytes_per_sample =
        raw_pixel_bytes_per_sample(bit_depth).map_err(NativeEncodePipelineError::invalid_input)?;
    let bytes_per_pixel = usize::from(num_components)
        .checked_mul(bytes_per_sample)
        .ok_or(NativeEncodePipelineError::arithmetic_overflow(
            "multi-tile pixel stride",
        ))?;
    let row_bytes = usize::try_from(tile_width)
        .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("multi-tile width"))?
        .checked_mul(bytes_per_pixel)
        .ok_or(NativeEncodePipelineError::arithmetic_overflow(
            "multi-tile row byte count",
        ))?;
    let out_len = interleaved_tile_output_len(tile_width, tile_height, num_components, bit_depth)?;
    let mut tile = Vec::new();
    tile.try_reserve_exact(out_len)
        .map_err(|_| host_allocation_failed("multi-tile pixel scratch", out_len))?;
    let image_row_bytes = usize::try_from(image_width)
        .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("image width"))?
        .checked_mul(bytes_per_pixel)
        .ok_or(NativeEncodePipelineError::arithmetic_overflow(
            "image row byte count",
        ))?;
    let x_byte_offset = usize::try_from(x0)
        .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("tile x offset"))?
        .checked_mul(bytes_per_pixel)
        .ok_or(NativeEncodePipelineError::arithmetic_overflow(
            "tile x byte offset",
        ))?;

    let y_end =
        y0.checked_add(tile_height)
            .ok_or(NativeEncodePipelineError::arithmetic_overflow(
                "tile y extent",
            ))?;
    for y in y0..y_end {
        let row_start = usize::try_from(y)
            .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("tile y offset"))?
            .checked_mul(image_row_bytes)
            .and_then(|offset| offset.checked_add(x_byte_offset))
            .ok_or(NativeEncodePipelineError::arithmetic_overflow(
                "tile row offset",
            ))?;
        let row_end = row_start.checked_add(row_bytes).ok_or(
            NativeEncodePipelineError::arithmetic_overflow("tile row range"),
        )?;
        tile.extend_from_slice(pixels.get(row_start..row_end).ok_or(
            NativeEncodePipelineError::invalid_input("tile row range outside source pixels"),
        )?);
    }

    Ok(tile)
}

pub(super) fn interleaved_tile_output_len(
    tile_width: u32,
    tile_height: u32,
    num_components: u16,
    bit_depth: u8,
) -> NativeEncodePipelineResult<usize> {
    let bytes_per_sample =
        raw_pixel_bytes_per_sample(bit_depth).map_err(NativeEncodePipelineError::invalid_input)?;
    usize::try_from(tile_width)
        .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("multi-tile width"))?
        .checked_mul(usize::from(num_components))
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .and_then(|row_bytes| row_bytes.checked_mul(tile_height as usize))
        .ok_or(NativeEncodePipelineError::arithmetic_overflow(
            "multi-tile byte count",
        ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the component-plane copy boundary keeps validated tile geometry and its retained owner baseline explicit"
)]
pub(in crate::j2c::encode) fn extract_component_plane_tile_for_session(
    data: &[u8],
    image_width: u32,
    x0: u32,
    y0: u32,
    tile_width: u32,
    tile_height: u32,
    bit_depth: u8,
    retained_base_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let bytes_per_sample =
        raw_pixel_bytes_per_sample(bit_depth).map_err(NativeEncodePipelineError::invalid_input)?;
    let row_bytes = usize::try_from(tile_width)
        .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("multi-tile width"))?
        .checked_mul(bytes_per_sample)
        .ok_or(NativeEncodePipelineError::arithmetic_overflow(
            "typed multi-tile row byte count",
        ))?;
    let out_len = row_bytes
        .checked_mul(
            usize::try_from(tile_height)
                .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("multi-tile height"))?,
        )
        .ok_or(NativeEncodePipelineError::arithmetic_overflow(
            "typed multi-tile byte count",
        ))?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            out_len,
            "typed multi-tile component scratch",
        )?,
        "typed multi-tile component scratch",
    )?;
    let mut tile = Vec::new();
    tile.try_reserve_exact(out_len)
        .map_err(|_| host_allocation_failed("typed multi-tile component scratch", out_len))?;
    let image_row_bytes = usize::try_from(image_width)
        .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("image width"))?
        .checked_mul(bytes_per_sample)
        .ok_or(NativeEncodePipelineError::arithmetic_overflow(
            "image row byte count",
        ))?;
    let x_byte_offset = usize::try_from(x0)
        .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("tile x offset"))?
        .checked_mul(bytes_per_sample)
        .ok_or(NativeEncodePipelineError::arithmetic_overflow(
            "tile x byte offset",
        ))?;
    let y_end =
        y0.checked_add(tile_height)
            .ok_or(NativeEncodePipelineError::arithmetic_overflow(
                "tile y extent",
            ))?;
    for y in y0..y_end {
        let row_start = usize::try_from(y)
            .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("tile y offset"))?
            .checked_mul(image_row_bytes)
            .and_then(|offset| offset.checked_add(x_byte_offset))
            .ok_or(NativeEncodePipelineError::arithmetic_overflow(
                "tile row offset",
            ))?;
        let row_end = row_start.checked_add(row_bytes).ok_or(
            NativeEncodePipelineError::arithmetic_overflow("tile row range"),
        )?;
        tile.extend_from_slice(data.get(row_start..row_end).ok_or(
            NativeEncodePipelineError::invalid_input(
                "component plane tile row range outside source data",
            ),
        )?);
    }
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            tile.capacity(),
            "typed multi-tile component scratch",
        )?,
        "typed multi-tile component scratch",
    )?;
    Ok(tile)
}
