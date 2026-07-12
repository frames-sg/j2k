// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tile-local clipping for caller-provided ROI regions.

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{checked_element_bytes, host_allocation_failed};
use crate::j2c::encode::{EncodeRoiRegion, NativeEncodePipelineError, NativeEncodePipelineResult};

#[expect(
    clippy::similar_names,
    reason = "paired axis and ROI names follow JPEG 2000 specification notation"
)]
pub(in crate::j2c::encode::multitile) fn roi_regions_for_tile(
    roi_regions: &[EncodeRoiRegion],
    tile_x: u32,
    tile_y: u32,
    tile_width: u32,
    tile_height: u32,
) -> NativeEncodePipelineResult<Vec<EncodeRoiRegion>> {
    let tile_x1 =
        tile_x
            .checked_add(tile_width)
            .ok_or(NativeEncodePipelineError::arithmetic_overflow(
                "tile ROI x bound",
            ))?;
    let tile_y1 =
        tile_y
            .checked_add(tile_height)
            .ok_or(NativeEncodePipelineError::arithmetic_overflow(
                "tile ROI y bound",
            ))?;
    let requested_bytes =
        checked_element_bytes::<EncodeRoiRegion>(roi_regions.len(), "multi-tile ROI scratch")?;
    let mut clipped = Vec::new();
    clipped
        .try_reserve_exact(roi_regions.len())
        .map_err(|_| host_allocation_failed("multi-tile ROI scratch", requested_bytes))?;

    for region in roi_regions {
        let region_x1 =
            region
                .x
                .checked_add(region.width)
                .ok_or(NativeEncodePipelineError::invalid_input(
                    "ROI region x bound overflows",
                ))?;
        let region_y1 =
            region
                .y
                .checked_add(region.height)
                .ok_or(NativeEncodePipelineError::invalid_input(
                    "ROI region y bound overflows",
                ))?;
        let x0 = region.x.max(tile_x);
        let y0 = region.y.max(tile_y);
        let x1 = region_x1.min(tile_x1);
        let y1 = region_y1.min(tile_y1);
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        clipped.push(EncodeRoiRegion {
            component: region.component,
            x: x0 - tile_x,
            y: y0 - tile_y,
            width: x1 - x0,
            height: y1 - y0,
            shift: region.shift,
        });
    }

    Ok(clipped)
}
