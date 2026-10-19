// SPDX-License-Identifier: MIT OR Apache-2.0

//! Intersections of absolute subband, precinct, and code-block grids (T.800 B.6/B.7).

use super::{PrecinctSubbandGeometry, PreparedSubbandShape, SubbandPrecinctGrid};
use crate::j2c::encode::{NativeEncodePipelineError, NativeEncodePipelineResult};

pub(in crate::j2c::encode::packet_plan::precinct) fn precinct_subband_geometry(
    shape: PreparedSubbandShape,
    grid: SubbandPrecinctGrid,
    precinct_column: u32,
    precinct_row: u32,
) -> NativeEncodePipelineResult<PrecinctSubbandGeometry> {
    let intersect_axis = |origin: u32,
                          extent: u32,
                          cell: u32,
                          first: u32,
                          span: u32,
                          block: u32|
     -> NativeEncodePipelineResult<(u32, u32, u32)> {
        let cell_start = (u64::from(first) + u64::from(cell)) * u64::from(span);
        let start = cell_start.max(u64::from(origin));
        let end = (cell_start + u64::from(span)).min(u64::from(origin) + u64::from(extent));
        if start >= end || block == 0 {
            return Ok((0, 0, 0));
        }
        let count = end.div_ceil(u64::from(block)) - start / u64::from(block);
        let narrow = |value| {
            u32::try_from(value).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("precinct subband intersection")
            })
        };
        Ok((narrow(start)?, narrow(end - start)?, narrow(count)?))
    };
    let (x, width, columns) = intersect_axis(
        shape.origin.0,
        shape.horizontal_span,
        precinct_column,
        grid.first_column,
        grid.horizontal_span,
        shape.code_block_horizontal_span,
    )?;
    let (y, height, rows) = intersect_axis(
        shape.origin.1,
        shape.vertical_span,
        precinct_row,
        grid.first_row,
        grid.vertical_span,
        shape.code_block_vertical_span,
    )?;
    Ok(PrecinctSubbandGeometry {
        origin: (x, y),
        horizontal_span: width,
        vertical_span: height,
        block_columns: if height == 0 { 0 } else { columns },
        block_rows: if width == 0 { 0 } else { rows },
    })
}

pub(in crate::j2c::encode::packet_plan::precinct) fn precinct_index_for_block(
    block_index: usize,
    shape: PreparedSubbandShape,
    grid: SubbandPrecinctGrid,
) -> NativeEncodePipelineResult<usize> {
    if shape.block_columns == 0
        || shape.code_block_horizontal_span == 0
        || shape.code_block_vertical_span == 0
    {
        return Err(NativeEncodePipelineError::internal_invariant(
            "empty subband code-block grid",
        ));
    }
    let block_index = u64::try_from(block_index)
        .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("code-block index"))?;
    let column = block_index % u64::from(shape.block_columns);
    let row = block_index / u64::from(shape.block_columns);
    if row >= u64::from(shape.block_rows) {
        return Err(NativeEncodePipelineError::internal_invariant(
            "code-block index exceeds subband",
        ));
    }
    let x = (u64::from(shape.origin.0 / shape.code_block_horizontal_span) + column)
        * u64::from(shape.code_block_horizontal_span);
    let y = (u64::from(shape.origin.1 / shape.code_block_vertical_span) + row)
        * u64::from(shape.code_block_vertical_span);
    let column = (x / u64::from(grid.horizontal_span)).checked_sub(u64::from(grid.first_column));
    let row = (y / u64::from(grid.vertical_span)).checked_sub(u64::from(grid.first_row));
    let (Some(column), Some(row)) = (column, row) else {
        return Err(NativeEncodePipelineError::internal_invariant(
            "code-block precedes precinct grid",
        ));
    };
    if column >= u64::from(grid.columns) || row >= u64::from(grid.rows) {
        return Err(NativeEncodePipelineError::internal_invariant(
            "code-block exceeds precinct grid",
        ));
    }
    usize::try_from(row * u64::from(grid.columns) + column)
        .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("precinct index exceeds usize"))
}
