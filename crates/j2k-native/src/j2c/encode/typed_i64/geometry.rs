// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sampling-aware high-bit tile geometry.

use super::super::{max_decomposition_levels, EncodeTypedComponentPlane};

#[expect(
    clippy::similar_names,
    reason = "paired axis names follow JPEG 2000 sampling notation"
)]
pub(super) fn min_sampled_tile_component_decomposition_levels(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
) -> Result<u8, &'static str> {
    let num_x_tiles = width.div_ceil(tile_width);
    let num_y_tiles = height.div_ceil(tile_height);
    let mut levels: Option<u8> = None;
    for tile_y in 0..num_y_tiles {
        for tile_x in 0..num_x_tiles {
            let x0 = tile_x * tile_width;
            let y0 = tile_y * tile_height;
            let actual_width = (width - x0).min(tile_width);
            let actual_height = (height - y0).min(tile_height);
            for plane in planes {
                let x_rsiz = u32::from(plane.x_rsiz);
                let y_rsiz = u32::from(plane.y_rsiz);
                let component_image_width = width.div_ceil(x_rsiz);
                let component_image_height = height.div_ceil(y_rsiz);
                let (_, component_tile_width) =
                    sampled_tile_component_axis(x0, actual_width, x_rsiz, component_image_width)?;
                let (_, component_tile_height) =
                    sampled_tile_component_axis(y0, actual_height, y_rsiz, component_image_height)?;
                let component_levels =
                    max_decomposition_levels(component_tile_width, component_tile_height);
                levels = Some(levels.map_or(component_levels, |min| min.min(component_levels)));
            }
        }
    }
    Ok(levels.unwrap_or(0))
}

pub(super) fn sampled_tile_component_axis(
    tile_origin: u32,
    tile_extent: u32,
    sampling: u32,
    component_extent: u32,
) -> Result<(u32, u32), &'static str> {
    let tile_end = tile_origin
        .checked_add(tile_extent)
        .ok_or("tile component bounds overflow")?;
    let start = tile_origin.div_ceil(sampling).min(component_extent);
    let end = tile_end.div_ceil(sampling).min(component_extent);
    Ok((start, end.saturating_sub(start)))
}
