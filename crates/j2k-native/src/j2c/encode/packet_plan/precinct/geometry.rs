// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked JPEG 2000 resolution/subband precinct geometry.

use super::super::super::{
    BlockCodingMode, NativeEncodePipelineError, NativeEncodePipelineResult, PreparedEncodeSubband,
    PreparedResolutionPacket, SubBandType,
};

mod block_mapping;
pub(super) use block_mapping::{precinct_index_for_block, precinct_subband_geometry};

#[derive(Clone, Copy)]
pub(super) struct ResolutionPrecinctGrid {
    first_column: u32,
    first_row: u32,
    pub(super) columns: u32,
    pub(super) rows: u32,
}

impl ResolutionPrecinctGrid {
    fn packet_count(self) -> NativeEncodePipelineResult<usize> {
        let count = u64::from(self.columns)
            .checked_mul(u64::from(self.rows))
            .ok_or_else(|| {
                NativeEncodePipelineError::arithmetic_overflow("precinct packet count overflow")
            })?;
        usize::try_from(count).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("precinct packet count exceeds usize")
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct SubbandPrecinctGrid {
    first_column: u32,
    first_row: u32,
    columns: u32,
    rows: u32,
    horizontal_span: u32,
    vertical_span: u32,
}

#[derive(Clone, Copy)]
pub(super) struct PreparedSubbandShape {
    origin: (u32, u32),
    block_columns: u32,
    block_rows: u32,
    pub(super) code_block_horizontal_span: u32,
    pub(super) code_block_vertical_span: u32,
    horizontal_span: u32,
    vertical_span: u32,
    pub(super) sub_band_type: SubBandType,
    pub(super) total_bitplanes: u8,
    pub(super) block_coding_mode: BlockCodingMode,
    pub(super) ht_target_coding_passes: u8,
}

impl From<&PreparedEncodeSubband> for PreparedSubbandShape {
    fn from(subband: &PreparedEncodeSubband) -> Self {
        Self {
            origin: subband.origin,
            block_columns: subband.num_cbs_x,
            block_rows: subband.num_cbs_y,
            code_block_horizontal_span: subband.code_block_width,
            code_block_vertical_span: subband.code_block_height,
            horizontal_span: subband.width,
            vertical_span: subband.height,
            sub_band_type: subband.sub_band_type,
            total_bitplanes: subband.total_bitplanes,
            block_coding_mode: subband.block_coding_mode,
            ht_target_coding_passes: subband.ht_target_coding_passes,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct PrecinctSubbandGeometry {
    pub(super) origin: (u32, u32),
    pub(super) block_columns: u32,
    pub(super) block_rows: u32,
    pub(super) horizontal_span: u32,
    pub(super) vertical_span: u32,
}

/// Map both reference-grid tile endpoints to component coordinates (T.800 B-12).
pub(super) fn component_bounds(
    tile: j2k_types::J2kRect,
    component_sampling: &[(u8, u8)],
    component: usize,
) -> NativeEncodePipelineResult<j2k_types::J2kRect> {
    let (sx, sy) = if component_sampling.is_empty() {
        (1, 1)
    } else {
        component_sampling.get(component).copied().ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "missing component sampling for precinct partitioning",
            )
        })?
    };
    if sx == 0 || sy == 0 {
        return Err(NativeEncodePipelineError::internal_invariant(
            "component sampling must be nonzero",
        ));
    }
    Ok(j2k_types::J2kRect {
        x0: tile.x0.div_ceil(u32::from(sx)),
        y0: tile.y0.div_ceil(u32::from(sy)),
        x1: tile.x1.div_ceil(u32::from(sx)),
        y1: tile.y1.div_ceil(u32::from(sy)),
    })
}

pub(super) fn component_split_packet_count(
    packets: &[PreparedResolutionPacket],
    bounds: j2k_types::J2kRect,
    num_decomposition_levels: u8,
    precinct_exponents: &[(u8, u8)],
) -> NativeEncodePipelineResult<usize> {
    packets.iter().try_fold(0usize, |count, packet| {
        let (horizontal_exponent, vertical_exponent) =
            precinct_exponents_for_resolution(precinct_exponents, packet.resolution)?;
        let grid = resolution_precinct_grid(
            bounds,
            num_decomposition_levels,
            packet.resolution,
            horizontal_exponent,
            vertical_exponent,
        )?;
        count.checked_add(grid.packet_count()?).ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow(
                "component precinct packet count overflow",
            )
        })
    })
}

pub(super) fn subband_precinct_grid(
    shape: PreparedSubbandShape,
    resolution: u32,
    horizontal_exponent: u8,
    vertical_exponent: u8,
    resolution_grid: ResolutionPrecinctGrid,
) -> NativeEncodePipelineResult<SubbandPrecinctGrid> {
    let subband_horizontal_exponent = if resolution > 0 {
        horizontal_exponent.checked_sub(1).ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow(
                "nonzero resolution precinct exponent underflow",
            )
        })?
    } else {
        horizontal_exponent
    };
    let subband_vertical_exponent = if resolution > 0 {
        vertical_exponent.checked_sub(1).ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow(
                "nonzero resolution precinct exponent underflow",
            )
        })?
    } else {
        vertical_exponent
    };
    let horizontal_span = pow2_u32(u32::from(subband_horizontal_exponent))?;
    let vertical_span = pow2_u32(u32::from(subband_vertical_exponent))?;
    if shape.code_block_horizontal_span != 0
        && shape.code_block_vertical_span != 0
        && (horizontal_span < shape.code_block_horizontal_span
            || vertical_span < shape.code_block_vertical_span)
    {
        return Err(NativeEncodePipelineError::invalid_input(
            "precinct dimensions must not reduce encoder code-block dimensions",
        ));
    }
    Ok(SubbandPrecinctGrid {
        first_column: resolution_grid.first_column,
        first_row: resolution_grid.first_row,
        columns: resolution_grid.columns,
        rows: resolution_grid.rows,
        horizontal_span,
        vertical_span,
    })
}

pub(super) fn precinct_exponents_for_resolution(
    precinct_exponents: &[(u8, u8)],
    resolution: u32,
) -> NativeEncodePipelineResult<(u8, u8)> {
    if precinct_exponents.is_empty() {
        return Ok((15, 15));
    }
    let resolution = usize::try_from(resolution).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("resolution index exceeds usize")
    })?;
    precinct_exponents.get(resolution).copied().ok_or_else(|| {
        NativeEncodePipelineError::internal_invariant("missing precinct exponents for resolution")
    })
}

/// Partition absolute resolution bounds on the zero-anchored grid (T.800 B-14/B-16).
pub(super) fn resolution_precinct_grid(
    bounds: j2k_types::J2kRect,
    num_decomposition_levels: u8,
    resolution: u32,
    horizontal_exponent: u8,
    vertical_exponent: u8,
) -> NativeEncodePipelineResult<ResolutionPrecinctGrid> {
    let shift = u32::from(num_decomposition_levels)
        .checked_sub(resolution)
        .ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("resolution exceeds decomposition count")
        })?;
    let scale = 1_u64
        .checked_shl(shift)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("resolution scale"))?;
    let span_x = u64::from(pow2_u32(u32::from(horizontal_exponent))?);
    let span_y = u64::from(pow2_u32(u32::from(vertical_exponent))?);
    let x0 = u64::from(bounds.x0).div_ceil(scale);
    let y0 = u64::from(bounds.y0).div_ceil(scale);
    let x1 = u64::from(bounds.x1).div_ceil(scale);
    let y1 = u64::from(bounds.y1).div_ceil(scale);
    let checked = |value| {
        u32::try_from(value)
            .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("resolution precinct grid"))
    };
    Ok(ResolutionPrecinctGrid {
        first_column: checked(x0 / span_x)?,
        first_row: checked(y0 / span_y)?,
        columns: checked(if x0 == x1 {
            0
        } else {
            x1.div_ceil(span_x) - x0 / span_x
        })?,
        rows: checked(if y0 == y1 {
            0
        } else {
            y1.div_ceil(span_y) - y0 / span_y
        })?,
    })
}

fn pow2_u32(exponent: u32) -> NativeEncodePipelineResult<u32> {
    1_u32.checked_shl(exponent).ok_or_else(|| {
        NativeEncodePipelineError::arithmetic_overflow("precinct exponent exceeds u32 shift width")
    })
}
