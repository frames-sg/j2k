// SPDX-License-Identifier: MIT OR Apache-2.0

//! Construction of the retained standard single-tile execution plan.

use crate::j2c::encode::allocation::checked_element_bytes;
use crate::j2c::encode::{
    ht_target_coding_passes_for_options, max_decomposition_levels,
    reversible_guard_bits_for_marker_limit, BlockCodingMode, CodeBlockGeometry,
    EncodeComponentSampleInfo, EncodeOptions, EncodeRoiRegion, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession,
};

use super::construction::PlanConstruction;
use super::{SingleTilePlan, ValidatedSingleTileInput};

mod owners;
mod params;

use owners::{try_build_plan_owners, PlanOwners};
use params::build_encode_params;

struct BuildRequest<'a> {
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &'a EncodeOptions,
    block_coding_mode: BlockCodingMode,
    roi_regions: &'a [EncodeRoiRegion],
    component_sample_info: &'a [EncodeComponentSampleInfo],
}

#[derive(Clone, Copy)]
struct PlanGeometry {
    use_mct: bool,
    num_levels: u8,
    guard_bits: u8,
    guard_delta: u8,
    cb_width: u32,
    cb_height: u32,
    ht_target_coding_passes: u8,
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
pub(in crate::j2c::encode::single_tile) fn build_single_tile_plan(
    validated: ValidatedSingleTileInput,
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    roi_regions: &[EncodeRoiRegion],
    component_sample_info: &[EncodeComponentSampleInfo],
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<SingleTilePlan> {
    let request = BuildRequest {
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        block_coding_mode,
        roi_regions,
        component_sample_info,
    };
    let ValidatedSingleTileInput {
        num_pixels,
        component_sampling,
        high_bit_exact,
        code_block_geometry,
    } = validated;
    let component_sampling_bytes = checked_element_bytes::<(u8, u8)>(
        component_sampling.capacity(),
        "single-tile component sampling",
    )?;
    let mut construction = PlanConstruction::new(session, component_sampling_bytes);
    let geometry = resolve_geometry(&request, high_bit_exact, code_block_geometry)?;
    let owners = try_build_plan_owners(&request, geometry, component_sampling, &mut construction)?;
    let PlanOwners {
        step_sizes,
        quant_params,
        component_step_sizes,
        roi_plans,
        roi_component_shifts,
        params: param_owners,
    } = owners;
    let params = build_encode_params(&request, geometry, param_owners);
    let plan = SingleTilePlan {
        num_pixels,
        high_bit_exact,
        use_mct: geometry.use_mct,
        num_levels: geometry.num_levels,
        guard_bits: geometry.guard_bits,
        step_sizes,
        quant_params,
        component_step_sizes,
        roi_plans,
        roi_component_shifts,
        cb_width: geometry.cb_width,
        cb_height: geometry.cb_height,
        ht_target_coding_passes: geometry.ht_target_coding_passes,
        params,
    };
    let retained_bytes = super::super::ownership::single_tile_plan_retained_bytes(&plan)?;
    session.checked_phase(retained_bytes, "retained single-tile plan")?;
    Ok(plan)
}

fn resolve_geometry(
    request: &BuildRequest<'_>,
    high_bit_exact: bool,
    code_block_geometry: CodeBlockGeometry,
) -> NativeEncodePipelineResult<PlanGeometry> {
    let use_mct = request.options.use_mct && matches!(request.num_components, 3 | 4);
    let num_levels = request
        .options
        .num_decomposition_levels
        .min(max_decomposition_levels(request.width, request.height));
    let requested_guard_bits = requested_guard_bits(request.options, use_mct);
    let guard_bits = if high_bit_exact && request.options.reversible {
        reversible_guard_bits_for_marker_limit(request.bit_depth, num_levels, requested_guard_bits)
            .map_err(NativeEncodePipelineError::unsupported)?
    } else {
        requested_guard_bits
    };
    Ok(PlanGeometry {
        use_mct,
        num_levels,
        guard_bits,
        guard_delta: guard_bits.saturating_sub(requested_guard_bits),
        cb_width: code_block_geometry.width,
        cb_height: code_block_geometry.height,
        ht_target_coding_passes: ht_target_coding_passes_for_options(
            request.options,
            request.block_coding_mode,
        ),
    })
}

fn requested_guard_bits(options: &EncodeOptions, use_mct: bool) -> u8 {
    if options.reversible {
        if use_mct {
            options.guard_bits.max(2)
        } else {
            options.guard_bits
        }
    } else {
        options.guard_bits.max(2)
    }
}
