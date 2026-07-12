// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sampling-aware typed tile extraction and borrowed plane views.

use alloc::vec::Vec;

use super::super::super::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use super::super::super::multitile::extract_component_plane_tile_for_session;
use super::super::super::{
    EncodeTypedComponentPlane, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeSession,
};
use super::super::geometry::sampled_tile_component_axis;
use super::TypedI64MultiTileRequest;

pub(super) struct TilePlaneData {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

pub(super) struct TilePlaneViews<'a> {
    pub(super) planes: Vec<EncodeTypedComponentPlane<'a>>,
    pub(super) component_dimensions: Vec<(u32, u32)>,
}

#[expect(
    clippy::similar_names,
    reason = "paired axis and sampling names follow JPEG 2000 notation"
)]
pub(super) fn try_extract_tile_planes(
    request: &TypedI64MultiTileRequest<'_, '_>,
    x0: u32,
    y0: u32,
    actual_width: u32,
    actual_height: u32,
    retained_base_bytes: usize,
) -> NativeEncodePipelineResult<Vec<TilePlaneData>> {
    let requested_outer = checked_element_bytes::<TilePlaneData>(
        request.planes.len(),
        "typed i64 tile plane owners",
    )?;
    request.session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            requested_outer,
            "typed i64 tile plane owners",
        )?,
        "typed i64 tile plane owners",
    )?;
    let mut tile_planes = Vec::new();
    tile_planes
        .try_reserve_exact(request.planes.len())
        .map_err(|_| host_allocation_failed("typed i64 tile plane owners", requested_outer))?;
    for plane in request.planes {
        let x_rsiz = u32::from(plane.x_rsiz);
        let y_rsiz = u32::from(plane.y_rsiz);
        let component_image_width = request.width.div_ceil(x_rsiz);
        let component_image_height = request.height.div_ceil(y_rsiz);
        let (component_x0, width) =
            sampled_tile_component_axis(x0, actual_width, x_rsiz, component_image_width)
                .map_err(NativeEncodePipelineError::arithmetic_overflow)?;
        let (component_y0, height) =
            sampled_tile_component_axis(y0, actual_height, y_rsiz, component_image_height)
                .map_err(NativeEncodePipelineError::arithmetic_overflow)?;
        if width == 0 || height == 0 {
            return Err(NativeEncodePipelineError::internal_invariant(
                "sampled tile component dimensions must be non-zero",
            ));
        }
        let prior_bytes = tile_plane_data_retained_bytes(&tile_planes)?;
        let data = extract_component_plane_tile_for_session(
            plane.data,
            component_image_width,
            component_x0,
            component_y0,
            width,
            height,
            plane.bit_depth,
            checked_add_bytes(
                retained_base_bytes,
                prior_bytes,
                "typed i64 tile plane scratch",
            )?,
            request.session,
        )?;
        tile_planes.push(TilePlaneData {
            data,
            width,
            height,
        });
    }
    request.session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            tile_plane_data_retained_bytes(&tile_planes)?,
            "typed i64 tile plane scratch",
        )?,
        "typed i64 tile plane scratch",
    )?;
    Ok(tile_planes)
}

pub(super) fn try_tile_plane_views<'a>(
    source_planes: &[EncodeTypedComponentPlane<'_>],
    tile_data: &'a [TilePlaneData],
    retained_base_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<TilePlaneViews<'a>> {
    if source_planes.len() != tile_data.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "typed tile plane count mismatch",
        ));
    }
    let requested_planes = checked_element_bytes::<EncodeTypedComponentPlane<'_>>(
        source_planes.len(),
        "typed i64 tile plane views",
    )?;
    let requested_dimensions = checked_element_bytes::<(u32, u32)>(
        source_planes.len(),
        "typed i64 tile component dimensions",
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_add_bytes(
                requested_planes,
                requested_dimensions,
                "typed i64 tile plane views",
            )?,
            "typed i64 tile plane views",
        )?,
        "typed i64 tile plane views",
    )?;
    let mut planes = Vec::new();
    planes
        .try_reserve_exact(source_planes.len())
        .map_err(|_| host_allocation_failed("typed i64 tile plane views", requested_planes))?;
    let mut dimensions = Vec::new();
    dimensions
        .try_reserve_exact(source_planes.len())
        .map_err(|_| {
            host_allocation_failed("typed i64 tile component dimensions", requested_dimensions)
        })?;
    for (source, tile) in source_planes.iter().zip(tile_data) {
        planes.push(EncodeTypedComponentPlane {
            data: &tile.data,
            x_rsiz: source.x_rsiz,
            y_rsiz: source.y_rsiz,
            bit_depth: source.bit_depth,
            signed: source.signed,
        });
        dimensions.push((tile.width, tile.height));
    }
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_add_bytes(
                checked_element_bytes::<EncodeTypedComponentPlane<'_>>(
                    planes.capacity(),
                    "typed i64 tile plane views",
                )?,
                checked_element_bytes::<(u32, u32)>(
                    dimensions.capacity(),
                    "typed i64 tile component dimensions",
                )?,
                "typed i64 tile plane views",
            )?,
            "typed i64 tile plane views",
        )?,
        "typed i64 tile plane views",
    )?;
    Ok(TilePlaneViews {
        planes,
        component_dimensions: dimensions,
    })
}

pub(super) fn tile_plane_data_retained_bytes(
    planes: &Vec<TilePlaneData>,
) -> NativeEncodePipelineResult<usize> {
    let mut bytes =
        checked_element_bytes::<TilePlaneData>(planes.capacity(), "typed i64 tile plane owners")?;
    for plane in planes {
        bytes = checked_add_bytes(bytes, plane.data.capacity(), "typed i64 tile plane data")?;
    }
    Ok(bytes)
}
