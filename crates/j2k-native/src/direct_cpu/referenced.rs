// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parse-free CPU execution for referenced HTJ2K plans.

mod component;
mod output;
mod payload;

pub(super) use output::{decoded_color_components, decoded_components, decoded_plane};
pub use output::{J2kDirectDecodedComponents, J2kDirectDecodedPlane};
pub(super) use payload::payload_slice;

use crate::error::{bail, DecodingError, Result};
use crate::{HtCodeBlockPayloadRanges, J2kReferencedHtj2kPlan};

use super::allocation::prepare_referenced_direct_scratch;
use super::J2kDirectCpuScratch;
use component::{execute_color_components_referenced, execute_component_plan_referenced};
use payload::{validate_payload_ranges, ReferencedPayloadCursor};

/// Execute retained per-tile Gray/RGB/RGBA HTJ2K geometry without reparsing packets.
///
/// Compressed cleanup/refinement ranges are validated against `encoded_input`
/// and combined in one retained scratch buffer. Reconstructed component owners
/// remain in `scratch` and are borrowed by the returned view.
#[doc(hidden)]
pub fn execute_referenced_htj2k_plan<'scratch>(
    plan: &J2kReferencedHtj2kPlan,
    encoded_input: &[u8],
    signed: bool,
    scratch: &'scratch mut J2kDirectCpuScratch,
) -> Result<J2kDirectDecodedComponents<'scratch>> {
    execute_referenced_htj2k_plan_with_payloads(
        plan,
        encoded_input,
        plan.payloads(),
        signed,
        scratch,
    )
}

/// Execute retained HTJ2K geometry from caller-flattened payload ranges.
///
/// `payloads` must remain in the plan's component/step/job traversal order,
/// but their ranges may point anywhere inside the shared `payload_arena`.
#[doc(hidden)]
pub fn execute_referenced_htj2k_plan_from_payloads<'scratch>(
    plan: &J2kReferencedHtj2kPlan,
    payload_arena: &[u8],
    payloads: &[HtCodeBlockPayloadRanges],
    signed: bool,
    scratch: &'scratch mut J2kDirectCpuScratch,
) -> Result<J2kDirectDecodedComponents<'scratch>> {
    execute_referenced_htj2k_plan_with_payloads(plan, payload_arena, payloads, signed, scratch)
}

fn execute_referenced_htj2k_plan_with_payloads<'scratch>(
    plan: &J2kReferencedHtj2kPlan,
    encoded_input: &[u8],
    payload_ranges: &[HtCodeBlockPayloadRanges],
    signed: bool,
    scratch: &'scratch mut J2kDirectCpuScratch,
) -> Result<J2kDirectDecodedComponents<'scratch>> {
    if payload_ranges.len() != plan.payloads().len() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    validate_payload_ranges(encoded_input, payload_ranges)?;
    prepare_referenced_direct_scratch(plan, scratch)?;
    {
        let J2kDirectCpuScratch {
            component_band_sets,
            component_planes,
            compressed_payload,
            classic_workspace: _,
            ht_workspace,
            staged_state: _,
        } = scratch;
        let mut payloads =
            ReferencedPayloadCursor::new(encoded_input, payload_ranges, compressed_payload);
        let mut output_initialized = [false; 4];

        for tile in plan.tiles() {
            if let Some(geometry) = tile.grayscale_geometry() {
                let bands = component_band_sets
                    .first_mut()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                let output = component_planes
                    .first_mut()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                execute_component_plan_referenced(
                    geometry,
                    bands,
                    output,
                    &mut payloads,
                    ht_workspace,
                    &mut output_initialized[0],
                )?;
            } else if let Some(geometry) = tile.color_geometry() {
                execute_color_components_referenced(
                    &geometry.component_plans,
                    3,
                    geometry.bit_depths,
                    geometry.mct,
                    geometry.transform,
                    signed,
                    component_band_sets,
                    component_planes,
                    &mut payloads,
                    ht_workspace,
                    &mut output_initialized,
                    tile.destination_rect(),
                )?;
            } else if let Some(geometry) = tile.rgba_geometry() {
                execute_color_components_referenced(
                    &geometry.component_plans,
                    4,
                    [
                        geometry.bit_depths[0],
                        geometry.bit_depths[1],
                        geometry.bit_depths[2],
                    ],
                    geometry.mct,
                    geometry.transform,
                    signed,
                    component_band_sets,
                    component_planes,
                    &mut payloads,
                    ht_workspace,
                    &mut output_initialized,
                    tile.destination_rect(),
                )?;
            } else {
                bail!(DecodingError::CodeBlockDecodeFailure);
            }
        }
        payloads.ensure_exhausted()?;
    }

    decoded_components(plan, scratch)
}
