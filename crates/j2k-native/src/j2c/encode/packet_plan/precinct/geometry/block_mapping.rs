// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked mapping between subband code blocks and precinct-local owners.

use super::{PrecinctSubbandGeometry, PreparedSubbandShape, SubbandPrecinctGrid};
use crate::j2c::encode::{NativeEncodePipelineError, NativeEncodePipelineResult};

pub(in crate::j2c::encode::packet_plan::precinct) fn precinct_subband_geometry(
    shape: PreparedSubbandShape,
    grid: SubbandPrecinctGrid,
    precinct_column: u32,
    precinct_row: u32,
) -> NativeEncodePipelineResult<PrecinctSubbandGeometry> {
    if shape.horizontal_span == 0
        || shape.vertical_span == 0
        || shape.code_block_horizontal_span == 0
        || shape.code_block_vertical_span == 0
    {
        return Ok(empty_precinct_geometry());
    }
    let horizontal_precinct_origin = precinct_column
        .checked_mul(grid.horizontal_span)
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("precinct x coordinate overflow")
        })?;
    let vertical_precinct_origin =
        precinct_row
            .checked_mul(grid.vertical_span)
            .ok_or_else(|| {
                NativeEncodePipelineError::arithmetic_overflow("precinct y coordinate overflow")
            })?;
    let horizontal_start = horizontal_precinct_origin.min(shape.horizontal_span);
    let vertical_start = vertical_precinct_origin.min(shape.vertical_span);
    let horizontal_end = horizontal_precinct_origin
        .checked_add(grid.horizontal_span)
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("precinct x extent overflow")
        })?
        .min(shape.horizontal_span);
    let vertical_end = vertical_precinct_origin
        .checked_add(grid.vertical_span)
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("precinct y extent overflow")
        })?
        .min(shape.vertical_span);
    if horizontal_start >= horizontal_end || vertical_start >= vertical_end {
        return Ok(empty_precinct_geometry());
    }

    let horizontal_block_start = (horizontal_start / shape.code_block_horizontal_span)
        .checked_mul(shape.code_block_horizontal_span)
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("precinct code-block x start overflow")
        })?;
    let vertical_block_start = (vertical_start / shape.code_block_vertical_span)
        .checked_mul(shape.code_block_vertical_span)
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("precinct code-block y start overflow")
        })?;
    let horizontal_block_end = horizontal_end
        .div_ceil(shape.code_block_horizontal_span)
        .checked_mul(shape.code_block_horizontal_span)
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("precinct code-block x end overflow")
        })?;
    let vertical_block_end = vertical_end
        .div_ceil(shape.code_block_vertical_span)
        .checked_mul(shape.code_block_vertical_span)
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("precinct code-block y end overflow")
        })?;
    Ok(PrecinctSubbandGeometry {
        block_columns: horizontal_block_end
            .checked_sub(horizontal_block_start)
            .ok_or_else(|| {
                NativeEncodePipelineError::arithmetic_overflow(
                    "precinct code-block x range underflow",
                )
            })?
            / shape.code_block_horizontal_span,
        block_rows: vertical_block_end
            .checked_sub(vertical_block_start)
            .ok_or_else(|| {
                NativeEncodePipelineError::arithmetic_overflow(
                    "precinct code-block y range underflow",
                )
            })?
            / shape.code_block_vertical_span,
        horizontal_span: horizontal_end - horizontal_start,
        vertical_span: vertical_end - vertical_start,
    })
}

pub(in crate::j2c::encode::packet_plan::precinct) fn precinct_index_for_block(
    block_index: usize,
    shape: PreparedSubbandShape,
    grid: SubbandPrecinctGrid,
) -> NativeEncodePipelineResult<usize> {
    if shape.block_columns == 0 {
        return Err(NativeEncodePipelineError::internal_invariant(
            "precinct code-block grid width is zero",
        ));
    }
    let block_index = u32::try_from(block_index).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("code-block index exceeds u32")
    })?;
    let block_column = block_index % shape.block_columns;
    let block_row = block_index / shape.block_columns;
    if block_row >= shape.block_rows {
        return Err(NativeEncodePipelineError::internal_invariant(
            "precinct code-block index out of range",
        ));
    }
    let precinct_column = block_column
        .checked_mul(shape.code_block_horizontal_span)
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("code-block x coordinate overflow")
        })?
        / grid.horizontal_span;
    let precinct_row = block_row
        .checked_mul(shape.code_block_vertical_span)
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("code-block y coordinate overflow")
        })?
        / grid.vertical_span;
    if precinct_column >= grid.columns || precinct_row >= grid.rows {
        return Err(NativeEncodePipelineError::internal_invariant(
            "precinct code-block destination out of range",
        ));
    }
    let packet_index = u64::from(precinct_row)
        .checked_mul(u64::from(grid.columns))
        .and_then(|value| value.checked_add(u64::from(precinct_column)))
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow(
                "precinct code-block destination overflow",
            )
        })?;
    usize::try_from(packet_index).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow(
            "precinct code-block destination exceeds usize",
        )
    })
}

const fn empty_precinct_geometry() -> PrecinctSubbandGeometry {
    PrecinctSubbandGeometry {
        block_columns: 0,
        block_rows: 0,
        horizontal_span: 0,
        vertical_span: 0,
    }
}
