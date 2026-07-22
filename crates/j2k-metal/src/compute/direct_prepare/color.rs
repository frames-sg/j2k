// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained classic and HTJ2K RGB/RGBA plan preparation.

use super::{
    append_referenced_classic_component_steps, append_referenced_htj2k_component_steps,
    finish_referenced_component_plan, validate_payload_record_span,
    validate_referenced_component_metadata, Error, J2kReferencedClassicPlan,
    J2kReferencedHtj2kPlan, NativeGrayscalePlan, PreparedDirectColorPlan,
    PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep, ReferencedClassicPayloadCursor,
};
use std::sync::Arc;

#[cfg(target_os = "macos")]
pub(crate) fn prepare_referenced_classic_color_plan(
    referenced: &J2kReferencedClassicPlan,
    input: &[u8],
    signed: bool,
) -> Result<PreparedDirectColorPlan, Error> {
    let prepared = prepare_referenced_classic_color_tiles(
        referenced,
        input,
        3,
        "J2K MetalDirect referenced classic RGB plan",
    )?;
    Ok(PreparedDirectColorPlan {
        dimensions: prepared.dimensions,
        bit_depths: [
            prepared.bit_depths[0],
            prepared.bit_depths[1],
            prepared.bit_depths[2],
        ],
        alpha_bit_depth: None,
        signed,
        mct: prepared.mct,
        transform: prepared.transform,
        component_plans: prepared.component_plans,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_referenced_classic_rgba_plan(
    referenced: &J2kReferencedClassicPlan,
    input: &[u8],
    signed: bool,
) -> Result<PreparedDirectColorPlan, Error> {
    let prepared = prepare_referenced_classic_color_tiles(
        referenced,
        input,
        4,
        "J2K MetalDirect referenced classic RGBA plan",
    )?;
    Ok(PreparedDirectColorPlan {
        dimensions: prepared.dimensions,
        bit_depths: [
            prepared.bit_depths[0],
            prepared.bit_depths[1],
            prepared.bit_depths[2],
        ],
        alpha_bit_depth: Some(prepared.bit_depths[3]),
        signed,
        mct: prepared.mct,
        transform: prepared.transform,
        component_plans: prepared.component_plans,
    })
}

#[cfg(target_os = "macos")]
fn prepare_referenced_classic_color_tiles(
    referenced: &J2kReferencedClassicPlan,
    input: &[u8],
    expected_component_count: usize,
    context: &'static str,
) -> Result<PreparedReferencedColorTiles, Error> {
    let first = referenced
        .tiles()
        .first()
        .and_then(|tile| referenced_color_geometry(tile, expected_component_count))
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "J2K Metal color prepared path received incompatible classic J2K tile geometry",
        })?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(context);
    let mut component_steps = budget.try_vec(expected_component_count, context)?;
    for component_index in 0..expected_component_count {
        let step_count = crate::batch_allocation::checked_count_sum(
            referenced.tiles().iter().map(|tile| {
                referenced_color_geometry(tile, expected_component_count)
                    .and_then(|geometry| geometry.component_plans.get(component_index))
                    .map_or(0, |component| component.steps.len())
            }),
            context,
        )?;
        component_steps.push(budget.try_vec(step_count, context)?);
    }
    let mut payloads = ReferencedClassicPayloadCursor::new(input, referenced);
    for tile in referenced.tiles() {
        let geometry =
            referenced_color_geometry(tile, expected_component_count)
                .ok_or(Error::UnsupportedMetalRequest {
                reason:
                    "J2K Metal color prepared path received incompatible classic J2K tile geometry",
            })?;
        validate_referenced_color_metadata(first, geometry)?;
        let expected_end = validate_payload_record_span(
            tile.payload_records(),
            payloads.next_payload,
            referenced.payloads().len(),
            "classic color tile",
        )?;
        for (component, steps) in geometry.component_plans.iter().zip(&mut component_steps) {
            append_referenced_classic_component_steps(component, &mut payloads, steps, context)?;
        }
        if payloads.next_payload != expected_end {
            return Err(Error::MetalStateInvariant {
                state: "classic color tile payload traversal",
                reason: "tile geometry job count does not match its payload-record span",
            });
        }
    }
    payloads.ensure_exhausted()?;
    finish_referenced_color_tiles(first, component_steps, context)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct ReferencedColorGeometry<'a> {
    dimensions: (u32, u32),
    bit_depths: [u8; 4],
    mct: bool,
    transform: j2k_native::J2kWaveletTransform,
    component_plans: &'a [NativeGrayscalePlan],
}

#[cfg(target_os = "macos")]
struct PreparedReferencedColorTiles {
    dimensions: (u32, u32),
    bit_depths: [u8; 4],
    mct: bool,
    transform: j2k_native::J2kWaveletTransform,
    component_plans: Vec<PreparedDirectGrayscalePlan>,
}

#[cfg(target_os = "macos")]
fn referenced_color_geometry(
    tile: &j2k_native::J2kReferencedTilePlan,
    expected_component_count: usize,
) -> Option<ReferencedColorGeometry<'_>> {
    match expected_component_count {
        3 => tile
            .color_geometry()
            .map(|geometry| ReferencedColorGeometry {
                dimensions: geometry.dimensions,
                bit_depths: [
                    geometry.bit_depths[0],
                    geometry.bit_depths[1],
                    geometry.bit_depths[2],
                    0,
                ],
                mct: geometry.mct,
                transform: geometry.transform,
                component_plans: &geometry.component_plans,
            }),
        4 => tile
            .rgba_geometry()
            .map(|geometry| ReferencedColorGeometry {
                dimensions: geometry.dimensions,
                bit_depths: geometry.bit_depths,
                mct: geometry.mct,
                transform: geometry.transform,
                component_plans: &geometry.component_plans,
            }),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn validate_referenced_color_metadata(
    first: ReferencedColorGeometry<'_>,
    tile: ReferencedColorGeometry<'_>,
) -> Result<(), Error> {
    if tile.dimensions != first.dimensions
        || tile.bit_depths != first.bit_depths
        || tile.mct != first.mct
        || tile.transform != first.transform
        || tile.component_plans.len() != first.component_plans.len()
    {
        return Err(Error::MetalStateInvariant {
            state: "referenced multi-tile color metadata",
            reason: "tile color dimensions, precision, transform, or component count changed",
        });
    }
    for (first_component, tile_component) in first.component_plans.iter().zip(tile.component_plans)
    {
        validate_referenced_component_metadata(first_component, tile_component)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn finish_referenced_color_tiles(
    first: ReferencedColorGeometry<'_>,
    component_steps: Vec<Vec<PreparedDirectGrayscaleStep>>,
    context: &'static str,
) -> Result<PreparedReferencedColorTiles, Error> {
    if component_steps.len() != first.component_plans.len() {
        return Err(Error::MetalStateInvariant {
            state: "referenced multi-tile color preparation",
            reason: "prepared component step count does not match color metadata",
        });
    }
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(context);
    let mut component_plans = budget.try_vec(component_steps.len(), context)?;
    for (component, steps) in first.component_plans.iter().zip(component_steps) {
        component_plans.push(finish_referenced_component_plan(
            component.dimensions,
            component.bit_depth,
            steps,
            context,
        )?);
    }
    Ok(PreparedReferencedColorTiles {
        dimensions: first.dimensions,
        bit_depths: first.bit_depths,
        mct: first.mct,
        transform: first.transform,
        component_plans,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_referenced_htj2k_color_plan(
    referenced: &J2kReferencedHtj2kPlan,
    input: &Arc<[u8]>,
    signed: bool,
) -> Result<PreparedDirectColorPlan, Error> {
    let prepared = prepare_referenced_htj2k_color_tiles(
        referenced,
        input,
        3,
        "J2K MetalDirect referenced HTJ2K RGB plan",
    )?;
    Ok(PreparedDirectColorPlan {
        dimensions: prepared.dimensions,
        bit_depths: [
            prepared.bit_depths[0],
            prepared.bit_depths[1],
            prepared.bit_depths[2],
        ],
        alpha_bit_depth: None,
        signed,
        mct: prepared.mct,
        transform: prepared.transform,
        component_plans: prepared.component_plans,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_referenced_htj2k_rgba_plan(
    referenced: &J2kReferencedHtj2kPlan,
    input: &Arc<[u8]>,
    signed: bool,
) -> Result<PreparedDirectColorPlan, Error> {
    let prepared = prepare_referenced_htj2k_color_tiles(
        referenced,
        input,
        4,
        "J2K MetalDirect referenced HTJ2K RGBA plan",
    )?;
    Ok(PreparedDirectColorPlan {
        dimensions: prepared.dimensions,
        bit_depths: [
            prepared.bit_depths[0],
            prepared.bit_depths[1],
            prepared.bit_depths[2],
        ],
        alpha_bit_depth: Some(prepared.bit_depths[3]),
        signed,
        mct: prepared.mct,
        transform: prepared.transform,
        component_plans: prepared.component_plans,
    })
}

#[cfg(target_os = "macos")]
fn prepare_referenced_htj2k_color_tiles(
    referenced: &J2kReferencedHtj2kPlan,
    input: &Arc<[u8]>,
    expected_component_count: usize,
    context: &'static str,
) -> Result<PreparedReferencedColorTiles, Error> {
    let first = referenced
        .tiles()
        .first()
        .and_then(|tile| referenced_color_geometry(tile, expected_component_count))
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "J2K Metal color prepared path received incompatible HTJ2K tile geometry",
        })?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(context);
    let mut component_steps = budget.try_vec(expected_component_count, context)?;
    for component_index in 0..expected_component_count {
        let step_count = crate::batch_allocation::checked_count_sum(
            referenced.tiles().iter().map(|tile| {
                referenced_color_geometry(tile, expected_component_count)
                    .and_then(|geometry| geometry.component_plans.get(component_index))
                    .map_or(0, |component| component.steps.len())
            }),
            context,
        )?;
        component_steps.push(budget.try_vec(step_count, context)?);
    }
    let mut payload_cursor = 0usize;
    for tile in referenced.tiles() {
        let geometry = referenced_color_geometry(tile, expected_component_count).ok_or(
            Error::UnsupportedMetalRequest {
                reason: "J2K Metal color prepared path received incompatible HTJ2K tile geometry",
            },
        )?;
        validate_referenced_color_metadata(first, geometry)?;
        let expected_end = validate_payload_record_span(
            tile.payload_records(),
            payload_cursor,
            referenced.payloads().len(),
            "HTJ2K color tile",
        )?;
        for (component, steps) in geometry.component_plans.iter().zip(&mut component_steps) {
            append_referenced_htj2k_component_steps(
                component,
                input,
                referenced.payloads(),
                &mut payload_cursor,
                steps,
            )?;
        }
        if payload_cursor != expected_end {
            return Err(Error::MetalStateInvariant {
                state: "HTJ2K color tile payload traversal",
                reason: "tile geometry job count does not match its payload-record span",
            });
        }
    }
    if payload_cursor != referenced.payloads().len() {
        return Err(Error::MetalKernel {
            message: format!("{context} has unused payload ranges"),
        });
    }
    finish_referenced_color_tiles(first, component_steps, context)
}
