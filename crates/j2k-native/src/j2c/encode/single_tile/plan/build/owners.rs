// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible construction of every allocation retained by a standard tile plan.

use crate::j2c::encode::{
    adjust_component_step_sizes_for_guard_delta, adjust_reversible_step_sizes_for_guard_delta,
    max_total_bitplanes_for_components, validate_precinct_exponents_for_options,
    ComponentRoiEncodePlan, EncodeComponentSampleInfo, NativeEncodePipelineError,
    NativeEncodePipelineResult, QuantStepSize, Vec,
};

use super::super::construction::{try_roi_plans, PlanConstruction};
use super::{BuildRequest, PlanGeometry};

pub(super) struct EncodeParamOwners {
    pub(super) component_sample_info: Vec<EncodeComponentSampleInfo>,
    pub(super) component_quantization_step_sizes: Vec<Vec<(u16, u16)>>,
    pub(super) component_sampling: Vec<(u8, u8)>,
    pub(super) roi_component_shifts: Vec<u8>,
    pub(super) precinct_exponents: Vec<(u8, u8)>,
}

pub(super) struct PlanOwners {
    pub(super) step_sizes: Vec<QuantStepSize>,
    pub(super) quant_params: Vec<(u16, u16)>,
    pub(super) component_step_sizes: Vec<Vec<QuantStepSize>>,
    pub(super) roi_plans: Vec<ComponentRoiEncodePlan>,
    pub(super) roi_component_shifts: Vec<u8>,
    pub(super) params: EncodeParamOwners,
}

pub(super) fn try_build_plan_owners(
    request: &BuildRequest<'_>,
    geometry: PlanGeometry,
    component_sampling: Vec<(u8, u8)>,
    construction: &mut PlanConstruction<'_, '_>,
) -> NativeEncodePipelineResult<PlanOwners> {
    let mut step_sizes = construction.try_step_sizes(
        request.bit_depth,
        geometry.num_levels,
        request.options.reversible,
        geometry.guard_bits,
        request.options,
    )?;
    if request.options.reversible && geometry.guard_delta != 0 {
        adjust_reversible_step_sizes_for_guard_delta(&mut step_sizes, geometry.guard_delta)
            .map_err(NativeEncodePipelineError::unsupported)?;
    }
    let quant_params =
        construction.try_quantization(&step_sizes, "single-tile default quantization")?;
    let mut component_step_sizes = construction.try_component_step_sizes(
        request.component_sample_info,
        geometry.num_levels,
        request.options.reversible,
        geometry.guard_bits,
        request.options,
    )?;
    if request.options.reversible && geometry.guard_delta != 0 {
        adjust_component_step_sizes_for_guard_delta(
            &mut component_step_sizes,
            geometry.guard_delta,
        )
        .map_err(NativeEncodePipelineError::unsupported)?;
    }
    let component_quantization_step_sizes =
        construction.try_component_quantization(&component_step_sizes)?;
    validate_precinct_exponents_for_options(request.options, geometry.num_levels)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let precinct_exponents = construction.try_copy_slice(
        &request.options.precinct_exponents,
        "single-tile precinct exponents",
    )?;
    let max_base_bitplanes =
        max_total_bitplanes_for_components(&step_sizes, &component_step_sizes, geometry.guard_bits)
            .map_err(NativeEncodePipelineError::internal_invariant)?;
    let roi_plans = try_roi_plans(
        construction,
        request.options,
        request.roi_regions,
        request.num_components,
        request.width,
        request.height,
        &component_sampling,
        max_base_bitplanes,
        request.block_coding_mode,
    )?;
    let roi_component_shifts =
        construction.try_map_slice(&roi_plans, "single-tile ROI shifts", |plan| plan.shift)?;
    let marker_roi_component_shifts =
        construction.try_copy_slice(&roi_component_shifts, "single-tile marker ROI shifts")?;
    let component_sample_info = construction.try_copy_slice(
        request.component_sample_info,
        "single-tile component metadata",
    )?;

    Ok(PlanOwners {
        step_sizes,
        quant_params,
        component_step_sizes,
        roi_plans,
        roi_component_shifts,
        params: EncodeParamOwners {
            component_sample_info,
            component_quantization_step_sizes,
            component_sampling,
            roi_component_shifts: marker_roi_component_shifts,
            precinct_exponents,
        },
    })
}
