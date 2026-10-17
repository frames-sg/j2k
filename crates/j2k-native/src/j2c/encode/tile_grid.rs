// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-tile encode geometry the tile-at-a-time encoder can represent.
//!
//! Multi-tile encode codes each tile as a standalone image whose origin is
//! zero. A decoder instead places every tile on the image grid: the wavelet
//! low/high split follows the parity of absolute coordinates (Formula B-15),
//! and code-block and precinct partitions are anchored at zero in sub-band
//! and resolution coordinates (B.6, B.7). The two agree only when each
//! component-tile origin is a multiple of `2^levels` and no sub-band or
//! resolution of the tile crosses a code-block or precinct boundary unless it
//! starts on one. Other geometries would produce codestreams that decode to
//! the wrong samples, so they are rejected.

/// Default precinct exponent when no precinct partition is signalled.
const DEFAULT_PRECINCT_EXPONENT: u8 = 15;

pub(super) const MISALIGNED_TILE_GRID: &str =
    "multi-tile encode requires tile origins on the wavelet, code-block, and precinct grids";

/// Whether a component tile spanning `origin..end` on one axis can be coded
/// as a standalone image. `code_block_exponent` is the full code-block size
/// exponent and `precinct_exponent(r)` the precinct exponent for resolution
/// `r` in COD order.
pub(super) fn component_tile_axis_is_aligned(
    origin: u32,
    end: u32,
    levels: u8,
    code_block_exponent: u8,
    precinct_exponent: impl Fn(u8) -> u8,
) -> bool {
    if origin == 0 {
        return true;
    }
    let (origin, end) = (u64::from(origin), u64::from(end));
    let Some(level_grid) = 1_u64.checked_shl(u32::from(levels)) else {
        return false;
    };
    if !origin.is_multiple_of(level_grid) {
        return false;
    }
    let code_block = 1_u64 << code_block_exponent;
    (0..=levels).all(|resolution| {
        let resolution_shift = levels - resolution;
        // Sub-bands of resolution r > 0 come from decomposition level
        // levels - r + 1. The low-pass range is the widest on each axis.
        let band_shift = if resolution == 0 {
            resolution_shift
        } else {
            resolution_shift + 1
        };
        let precinct = 1_u64 << precinct_exponent(resolution).min(DEFAULT_PRECINCT_EXPONENT);
        range_respects_grid(
            origin >> resolution_shift,
            end.div_ceil(1 << resolution_shift),
            precinct,
        ) && range_respects_grid(
            origin >> band_shift,
            end.div_ceil(1 << band_shift),
            code_block,
        )
    })
}

/// True when the partition of `start..end` anchored at `start` matches the
/// partition anchored at zero: either `start` is on the grid, or the range
/// does not cross a grid line.
fn range_respects_grid(start: u64, end: u64, grid: u64) -> bool {
    start.is_multiple_of(grid) || (start / grid + 1) * grid >= end
}

/// Validate every tile origin of a rectangular tile grid on one axis for a
/// component with sampling factor `sampling`.
pub(super) fn tile_axis_is_aligned(
    image_extent: u32,
    tile_extent: u32,
    sampling: u32,
    levels: u8,
    code_block_exponent: u8,
    precinct_exponent: impl Fn(u8) -> u8 + Copy,
) -> bool {
    let component_extent = image_extent.div_ceil(sampling);
    let mut tile_origin = tile_extent;
    while tile_origin < image_extent {
        let tile_end = tile_origin.saturating_add(tile_extent).min(image_extent);
        let origin = tile_origin.div_ceil(sampling);
        let end = tile_end.div_ceil(sampling).min(component_extent);
        if !component_tile_axis_is_aligned(
            origin,
            end,
            levels,
            code_block_exponent,
            precinct_exponent,
        ) {
            return false;
        }
        tile_origin = tile_end;
    }
    true
}

/// Validate a tile grid for components with the given sampling factors.
/// `code_block_exponents` and `precinct_exponents` use the
/// [`super::EncodeOptions`] conventions.
pub(super) fn validate_tile_grid(
    image: (u32, u32),
    tile: (u32, u32),
    sampling: impl IntoIterator<Item = (u32, u32)>,
    levels: u8,
    code_block_exponents: (u8, u8),
    precinct_exponents: &[(u8, u8)],
) -> Result<(), &'static str> {
    let horizontal = |resolution: u8| {
        precinct_exponents
            .get(usize::from(resolution))
            .map_or(DEFAULT_PRECINCT_EXPONENT, |exponents| exponents.0)
    };
    let vertical = |resolution: u8| {
        precinct_exponents
            .get(usize::from(resolution))
            .map_or(DEFAULT_PRECINCT_EXPONENT, |exponents| exponents.1)
    };
    let code_block_width = code_block_exponents.0.saturating_add(2);
    let code_block_height = code_block_exponents.1.saturating_add(2);
    for (x_sampling, y_sampling) in sampling {
        if !tile_axis_is_aligned(
            image.0,
            tile.0,
            x_sampling.max(1),
            levels,
            code_block_width,
            horizontal,
        ) || !tile_axis_is_aligned(
            image.1,
            tile.1,
            y_sampling.max(1),
            levels,
            code_block_height,
            vertical,
        ) {
            return Err(MISALIGNED_TILE_GRID);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{tile_axis_is_aligned, validate_tile_grid};

    const NO_PRECINCTS: fn(u8) -> u8 = |_| 15;

    fn axis(image: u32, tile: u32, levels: u8) -> bool {
        tile_axis_is_aligned(image, tile, 1, levels, 6, NO_PRECINCTS)
    }

    #[test]
    fn accepts_power_of_two_tiles_and_single_tiles() {
        assert!(axis(301, 64, 1));
        assert!(axis(301, 64, 5));
        assert!(axis(16_384, 1024, 5));
        assert!(axis(16_384, 256, 5));
        // A tile at least as large as the image has no interior origin.
        assert!(axis(301, 301, 5));
        assert!(axis(301, 512, 5));
    }

    #[test]
    fn rejects_tile_origins_off_the_wavelet_grid() {
        assert!(!axis(30, 15, 1));
        assert!(!axis(40, 18, 2));
        assert!(axis(40, 20, 2));
        assert!(!axis(6000, 1000, 5));
    }

    #[test]
    fn rejects_sub_bands_crossing_a_code_block_boundary() {
        // 48 px tiles with one level put sub-band 48..72 across 64.
        assert!(!axis(203, 48, 1));
        assert!(!axis(301, 48, 1));
        // Without a wavelet level the tile itself crosses 64 and 128.
        assert!(!axis(301, 96, 0));
        // 48 px tiles whose sub-bands never cross a boundary are fine.
        assert!(axis(96, 48, 1));
    }

    #[test]
    fn rejects_resolutions_crossing_a_precinct_boundary() {
        // 192 px tiles sit on the 64 px code-block grid, but the tile at
        // 192..384 crosses the 128 px precinct boundary at 256.
        assert!(!tile_axis_is_aligned(600, 192, 1, 0, 6, |_| 7));
        assert!(tile_axis_is_aligned(600, 192, 1, 0, 6, NO_PRECINCTS));
        // The default 2^15 precinct still applies to very wide images.
        assert!(!axis(40_000, 20_480, 0));
        assert!(axis(32_768, 20_480, 0));
    }

    #[test]
    fn applies_component_sampling_to_tile_origins() {
        // A 64 px tile grid on a component sampled by 3 starts at 22.
        assert!(!tile_axis_is_aligned(301, 64, 3, 1, 6, NO_PRECINCTS));
        assert!(tile_axis_is_aligned(301, 64, 2, 1, 6, NO_PRECINCTS));
    }

    #[test]
    fn validates_both_axes_for_every_component() {
        assert_eq!(
            validate_tile_grid((301, 203), (64, 64), [(1, 1)], 1, (4, 4), &[]),
            Ok(())
        );
        assert!(validate_tile_grid((301, 203), (64, 48), [(1, 1)], 1, (4, 4), &[]).is_err());
        assert!(validate_tile_grid((301, 203), (48, 64), [(1, 1)], 1, (4, 4), &[]).is_err());
        assert!(
            validate_tile_grid((301, 203), (64, 64), [(1, 1), (3, 1)], 1, (4, 4), &[]).is_err()
        );
    }
}
