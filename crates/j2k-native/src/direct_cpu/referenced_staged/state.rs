// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{bail, DecodingError, Result};
use crate::{J2kDirectGrayscalePlan, J2kWaveletTransform};

use super::super::{resize_and_zero, J2kDirectCpuScratch, StagedDirectRoute, StagedDirectState};
use super::{finish::finish_tile_components, prepare::prepare_entropy_bands};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum EntropyRoute {
    Classic,
    Htj2k,
}

pub(super) fn begin_staged_image(
    scratch: &mut J2kDirectCpuScratch,
    route: StagedDirectRoute,
    tile_count: usize,
    output_rect: crate::J2kRect,
    component_count: usize,
) -> Result<()> {
    scratch.staged_state = None;
    if tile_count == 0
        || !matches!(component_count, 1 | 3 | 4)
        || scratch.component_band_sets.len() < component_count
        || scratch.component_planes.len() < component_count
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let dimensions = (output_rect.width(), output_rect.height());
    let plane_len = usize::try_from(dimensions.0)
        .ok()
        .and_then(|width| {
            usize::try_from(dimensions.1)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    for component_index in 0..component_count {
        scratch.component_band_sets[component_index].reset();
        let plane = &mut scratch.component_planes[component_index];
        plane.width = dimensions.0;
        plane.height = dimensions.1;
        resize_and_zero(&mut plane.samples, plane_len)?;
    }
    scratch.staged_state = Some(StagedDirectState {
        route,
        next_tile: 0,
        active_tile: None,
        tile_count,
    });
    Ok(())
}

pub(super) fn prepare_staged_tile(
    scratch: &mut J2kDirectCpuScratch,
    route: StagedDirectRoute,
    tile_index: usize,
    components: &[J2kDirectGrayscalePlan],
    entropy_route: EntropyRoute,
) -> Result<()> {
    let Some(state) = scratch.staged_state else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };
    if state.route != route
        || state.next_tile != tile_index
        || state.active_tile.is_some()
        || tile_index >= state.tile_count
    {
        scratch.staged_state = None;
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    if let Err(error) = prepare_entropy_bands(components, entropy_route, scratch) {
        scratch.staged_state = None;
        return Err(error);
    }
    if let Some(state) = scratch.staged_state.as_mut() {
        state.active_tile = Some(tile_index);
    }
    Ok(())
}

pub(super) fn finish_staged_tile(
    scratch: &mut J2kDirectCpuScratch,
    route: StagedDirectRoute,
    tile_index: usize,
    components: &[J2kDirectGrayscalePlan],
    color_transform: Option<([u8; 3], bool, J2kWaveletTransform)>,
    destination: crate::J2kRect,
    signed: bool,
) -> Result<()> {
    validate_active_tile(scratch, route, tile_index)?;
    if let Err(error) =
        finish_tile_components(components, color_transform, destination, signed, scratch)
    {
        scratch.staged_state = None;
        return Err(error);
    }
    for bands in &mut scratch.component_band_sets[..components.len()] {
        bands.reset();
    }
    let Some(state) = scratch.staged_state.as_mut() else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };
    state.active_tile = None;
    state.next_tile = state
        .next_tile
        .checked_add(1)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    Ok(())
}

pub(super) fn finish_staged_image(
    scratch: &mut J2kDirectCpuScratch,
    route: StagedDirectRoute,
    tile_count: usize,
) -> Result<()> {
    let Some(state) = scratch.staged_state else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };
    if state.route != route
        || state.tile_count != tile_count
        || state.next_tile != tile_count
        || state.active_tile.is_some()
    {
        scratch.staged_state = None;
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    scratch.staged_state = None;
    Ok(())
}

pub(super) fn validate_active_tile(
    scratch: &J2kDirectCpuScratch,
    route: StagedDirectRoute,
    tile_index: usize,
) -> Result<()> {
    if scratch.staged_state.is_some_and(|state| {
        state.route == route
            && state.next_tile == tile_index
            && state.active_tile == Some(tile_index)
            && tile_index < state.tile_count
    }) {
        Ok(())
    } else {
        Err(DecodingError::CodeBlockDecodeFailure.into())
    }
}
