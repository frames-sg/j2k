// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked grid and per-tile geometry for standard multi-tile encode.

use super::super::super::{NativeEncodePipelineError, NativeEncodePipelineResult};
use super::super::MultiTileEncodeRequest;

pub(in crate::j2c::encode::multitile) struct TileGrid {
    columns: u32,
    rows: u32,
    tile_count: usize,
}

impl TileGrid {
    pub(in crate::j2c::encode::multitile) fn try_new(
        request: &MultiTileEncodeRequest<'_, '_>,
    ) -> NativeEncodePipelineResult<Self> {
        if request.tile_width == 0 || request.tile_height == 0 {
            return Err(NativeEncodePipelineError::invalid_input(
                "multi-tile dimensions must be non-zero",
            ));
        }
        let columns = request.width.div_ceil(request.tile_width);
        let rows = request.height.div_ceil(request.tile_height);
        let tile_count =
            columns
                .checked_mul(rows)
                .ok_or(NativeEncodePipelineError::arithmetic_overflow(
                    "multi-tile tile count",
                ))?;
        if tile_count > u32::from(u16::MAX) + 1 {
            return Err(NativeEncodePipelineError::unsupported(
                "multi-tile encode supports at most 65536 tiles",
            ));
        }
        let tile_count = usize::try_from(tile_count)
            .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("multi-tile tile count"))?;
        Ok(Self {
            columns,
            rows,
            tile_count,
        })
    }

    pub(in crate::j2c::encode::multitile) const fn columns(&self) -> u32 {
        self.columns
    }

    pub(in crate::j2c::encode::multitile) const fn rows(&self) -> u32 {
        self.rows
    }

    pub(in crate::j2c::encode::multitile) const fn tile_count(&self) -> usize {
        self.tile_count
    }
}

#[derive(Clone, Copy)]
pub(in crate::j2c::encode::multitile) struct TilePosition {
    pub(super) index: u16,
    pub(super) origin_x: u32,
    pub(super) origin_y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl TilePosition {
    pub(in crate::j2c::encode::multitile) fn try_new(
        request: &MultiTileEncodeRequest<'_, '_>,
        grid: &TileGrid,
        row: u32,
        column: u32,
    ) -> NativeEncodePipelineResult<Self> {
        let index = row
            .checked_mul(grid.columns)
            .and_then(|base| base.checked_add(column))
            .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("tile index"))?;
        let index = u16::try_from(index)
            .map_err(|_| NativeEncodePipelineError::internal_invariant("tile index exceeds u16"))?;
        let origin_x = column
            .checked_mul(request.tile_width)
            .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("tile x offset"))?;
        let origin_y = row
            .checked_mul(request.tile_height)
            .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("tile y offset"))?;
        let remaining_width = request.width.checked_sub(origin_x).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("tile x offset exceeds image width")
        })?;
        let remaining_height = request.height.checked_sub(origin_y).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("tile y offset exceeds image height")
        })?;
        Ok(Self {
            index,
            origin_x,
            origin_y,
            width: remaining_width.min(request.tile_width),
            height: remaining_height.min(request.tile_height),
        })
    }
}
