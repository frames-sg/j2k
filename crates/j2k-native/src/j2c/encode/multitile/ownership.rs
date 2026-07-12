// SPDX-License-Identifier: MIT OR Apache-2.0

//! Capacity accounting for the owners retained across nested tile encodes.

use alloc::vec::Vec;

use super::super::allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed};
use super::super::tile_parts::{encoded_tile_parts_retained_bytes, EncodedTilePart};
use super::super::{EncodeOptions, NativeEncodeSession};
use crate::EncodeResult;

pub(in crate::j2c::encode) fn reserve_tile_parts(
    tile_count: usize,
    planning_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> super::super::NativeEncodePipelineResult<Vec<EncodedTilePart>> {
    let requested_bytes =
        checked_element_bytes::<EncodedTilePart>(tile_count, "multi-tile part owner reservation")?;
    session.checked_phase(
        checked_add_bytes(
            planning_bytes,
            requested_bytes,
            "multi-tile part owner reservation",
        )?,
        "multi-tile part owner reservation",
    )?;
    let mut parts = Vec::new();
    parts.try_reserve_exact(tile_count).map_err(|_| {
        host_allocation_failed("multi-tile part owner reservation", requested_bytes)
    })?;
    let actual_bytes = checked_element_bytes::<EncodedTilePart>(
        parts.capacity(),
        "multi-tile part owner reservation",
    )?;
    session.checked_phase(
        checked_add_bytes(
            planning_bytes,
            actual_bytes,
            "multi-tile part owner reservation",
        )?,
        "multi-tile part owner reservation",
    )?;
    Ok(parts)
}

pub(in crate::j2c::encode) fn append_encoded_tile_parts(
    tile_bodies: &mut Vec<EncodedTilePart>,
    mut new_parts: Vec<EncodedTilePart>,
    planning_bytes: usize,
    scratch_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> super::super::NativeEncodePipelineResult<()> {
    let accumulated_bytes = encoded_tile_parts_retained_bytes(tile_bodies, tile_bodies.capacity())?;
    let new_part_bytes = encoded_tile_parts_retained_bytes(&new_parts, new_parts.capacity())?;
    let future_len = tile_bodies.len().checked_add(new_parts.len()).ok_or(
        crate::EncodeError::ArithmeticOverflow {
            what: "multi-tile accumulated part count",
        },
    )?;
    let needs_growth = tile_bodies.capacity().saturating_sub(tile_bodies.len()) < new_parts.len();
    let requested_outer_bytes = if needs_growth {
        checked_element_bytes::<EncodedTilePart>(future_len, "multi-tile accumulated part owners")?
    } else {
        0
    };
    session.checked_phase(
        checked_add_bytes(
            checked_add_bytes(
                planning_bytes,
                checked_add_bytes(
                    accumulated_bytes,
                    new_part_bytes,
                    "multi-tile accumulated payloads",
                )?,
                "multi-tile retained owners",
            )?,
            checked_add_bytes(
                scratch_bytes,
                requested_outer_bytes,
                "multi-tile accumulation growth",
            )?,
            "multi-tile accumulation peak",
        )?,
        "multi-tile accumulation peak",
    )?;
    tile_bodies
        .try_reserve_exact(new_parts.len())
        .map_err(|_| {
            host_allocation_failed("multi-tile accumulated part owners", requested_outer_bytes)
        })?;
    session.checked_phase(
        checked_add_bytes(
            checked_add_bytes(
                planning_bytes,
                encoded_tile_parts_retained_bytes(tile_bodies, tile_bodies.capacity())?,
                "multi-tile retained owners",
            )?,
            checked_add_bytes(new_part_bytes, scratch_bytes, "multi-tile append owners")?,
            "multi-tile append peak",
        )?,
        "multi-tile append peak",
    )?;
    tile_bodies.append(&mut new_parts);
    drop(new_parts);
    session.checked_phase(
        checked_add_bytes(
            planning_bytes,
            encoded_tile_parts_retained_bytes(tile_bodies, tile_bodies.capacity())?,
            "retained multi-tile payloads",
        )?,
        "retained multi-tile payloads",
    )?;
    Ok(())
}

pub(in crate::j2c::encode) fn quantization_retained_bytes(
    quant_params: &Vec<(u16, u16)>,
) -> EncodeResult<usize> {
    checked_element_bytes::<(u16, u16)>(quant_params.capacity(), "multi-tile quantization")
}

pub(in crate::j2c::encode) fn encode_options_retained_bytes(
    options: &EncodeOptions,
) -> EncodeResult<usize> {
    let mut bytes = checked_element_bytes::<u64>(
        options.quality_layer_byte_targets.capacity(),
        "multi-tile quality targets",
    )?;
    if let Some(component_sampling) = &options.component_sampling {
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<(u8, u8)>(
                component_sampling.capacity(),
                "multi-tile component sampling",
            )?,
            "multi-tile component sampling",
        )?;
    }
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<u8>(
            options.roi_component_shifts.capacity(),
            "multi-tile ROI shifts",
        )?,
        "multi-tile ROI shifts",
    )?;
    checked_add_bytes(
        bytes,
        checked_element_bytes::<(u8, u8)>(
            options.precinct_exponents.capacity(),
            "multi-tile precinct exponents",
        )?,
        "multi-tile precinct exponents",
    )
}

#[cfg(test)]
#[path = "ownership/tests.rs"]
mod tests;
