// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{Downscale, PixelFormat, Rect};

const VIEWPORT_TILE_EDGE: u32 = 96;
const VIEWPORT_TILE_COLS: u32 = 6;
const VIEWPORT_TILE_ROWS: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// One source-to-destination region in a composed viewport.
pub struct ViewportTile {
    /// Source region in the JPEG image before downscaling.
    pub source_roi: Rect,
    /// Destination rectangle in the viewport after downscaling.
    pub dest: Rect,
}

#[derive(Debug, PartialEq, Eq)]
/// Move-only planned viewport decode made of one or more source tiles.
pub struct ViewportWorkload {
    /// Downscale factor applied to every source tile.
    pub scale: Downscale,
    /// Output viewport dimensions in pixels.
    pub viewport_dims: (u32, u32),
    /// Tiles to decode and place into the viewport.
    pub tiles: Vec<ViewportTile>,
}

#[derive(Clone, Copy)]
pub(super) struct CpuViewportComposeRequest<'a> {
    pub(super) fmt: PixelFormat,
    pub(super) scale: Downscale,
    pub(super) viewport_dims: (u32, u32),
    pub(super) tiles: &'a [ViewportTile],
    pub(super) tile_metadata_capacity: usize,
    pub(super) external_live_bytes: usize,
}

/// Compute the bounding source rectangle covering all tiles in a workload.
pub fn viewport_source_bounds(workload: &ViewportWorkload) -> Rect {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    for tile in &workload.tiles {
        min_x = min_x.min(tile.source_roi.x);
        min_y = min_y.min(tile.source_roi.y);
        max_x = max_x.max(tile.source_roi.x.saturating_add(tile.source_roi.w));
        max_y = max_y.max(tile.source_roi.y.saturating_add(tile.source_roi.h));
    }

    Rect {
        x: min_x,
        y: min_y,
        w: max_x.saturating_sub(min_x),
        h: max_y.saturating_sub(min_y),
    }
}

/// Return whether the workload covers a contiguous viewport without overlaps.
pub fn is_contiguous_viewport_workload(workload: &ViewportWorkload) -> bool {
    if workload.tiles.is_empty() {
        return false;
    }

    let source = viewport_source_bounds(workload);
    let scaled_source = source.scaled_covering(workload.scale);
    if (scaled_source.w, scaled_source.h) != workload.viewport_dims {
        return false;
    }

    let viewport_area = u64::from(workload.viewport_dims.0) * u64::from(workload.viewport_dims.1);
    let mut area_sum = 0u64;

    for tile in &workload.tiles {
        let scaled_tile = tile.source_roi.scaled_covering(workload.scale);
        let expected = Rect {
            x: scaled_tile.x.saturating_sub(scaled_source.x),
            y: scaled_tile.y.saturating_sub(scaled_source.y),
            w: scaled_tile.w,
            h: scaled_tile.h,
        };
        if tile.dest != expected {
            return false;
        }
        if tile.dest.x.saturating_add(tile.dest.w) > workload.viewport_dims.0
            || tile.dest.y.saturating_add(tile.dest.h) > workload.viewport_dims.1
        {
            return false;
        }

        area_sum = area_sum.saturating_add(u64::from(tile.dest.w) * u64::from(tile.dest.h));
    }

    for (idx, tile) in workload.tiles.iter().enumerate() {
        let tile_right = tile.dest.x.saturating_add(tile.dest.w);
        let tile_bottom = tile.dest.y.saturating_add(tile.dest.h);
        for other in &workload.tiles[idx + 1..] {
            let other_right = other.dest.x.saturating_add(other.dest.w);
            let other_bottom = other.dest.y.saturating_add(other.dest.h);
            let separated = tile_right <= other.dest.x
                || other_right <= tile.dest.x
                || tile_bottom <= other.dest.y
                || other_bottom <= tile.dest.y;
            if !separated {
                return false;
            }
        }
    }

    area_sum == viewport_area
}

/// Suggest a fixed-size centered viewport workload for an image.
pub fn suggest_viewport_workload(dimensions: (u32, u32)) -> Option<ViewportWorkload> {
    let scales = [
        Downscale::Eighth,
        Downscale::Quarter,
        Downscale::Half,
        Downscale::None,
    ];
    let viewport_dims = (
        VIEWPORT_TILE_EDGE * VIEWPORT_TILE_COLS,
        VIEWPORT_TILE_EDGE * VIEWPORT_TILE_ROWS,
    );
    for scale in scales {
        let denom = scale.denominator();
        let Some(x) = viewport_origin(dimensions.0, viewport_dims.0.saturating_mul(denom), denom)
        else {
            continue;
        };
        let Some(y) = viewport_origin(dimensions.1, viewport_dims.1.saturating_mul(denom), denom)
        else {
            continue;
        };
        let source_viewport = Rect {
            x,
            y,
            w: viewport_dims.0.saturating_mul(denom),
            h: viewport_dims.1.saturating_mul(denom),
        };
        let scaled_source = source_viewport.scaled_covering(scale);
        if (scaled_source.w, scaled_source.h) != viewport_dims {
            continue;
        }
        let source_tile = VIEWPORT_TILE_EDGE.saturating_mul(denom);
        let mut tiles = Vec::with_capacity((VIEWPORT_TILE_COLS * VIEWPORT_TILE_ROWS) as usize);
        for row in 0..VIEWPORT_TILE_ROWS {
            for col in 0..VIEWPORT_TILE_COLS {
                tiles.push(ViewportTile {
                    source_roi: Rect {
                        x: source_viewport.x + col * source_tile,
                        y: source_viewport.y + row * source_tile,
                        w: source_tile,
                        h: source_tile,
                    },
                    dest: Rect {
                        x: col * VIEWPORT_TILE_EDGE,
                        y: row * VIEWPORT_TILE_EDGE,
                        w: VIEWPORT_TILE_EDGE,
                        h: VIEWPORT_TILE_EDGE,
                    },
                });
            }
        }

        return Some(ViewportWorkload {
            scale,
            viewport_dims,
            tiles,
        });
    }

    None
}

fn viewport_origin(full_extent: u32, viewport_extent: u32, align: u32) -> Option<u32> {
    if viewport_extent > full_extent || align == 0 {
        return None;
    }

    let centered = (full_extent - viewport_extent) / 2;
    Some(centered - centered % align)
}
