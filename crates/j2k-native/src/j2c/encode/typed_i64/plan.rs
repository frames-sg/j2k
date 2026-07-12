// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible high-bit plan construction and consume-only phase transitions.

use alloc::vec::Vec;

use super::super::allocation::checked_add_bytes;
use super::super::{
    adjust_component_step_sizes_for_guard_delta, adjust_reversible_step_sizes_for_guard_delta,
    block_coding_mode, ht_target_coding_passes_for_options, reversible_guard_bits_for_marker_limit,
    validate_code_block_geometry, BlockCodingMode, EncodeComponentSampleInfo, EncodeOptions,
    EncodeParams, EncodeTypedComponentPlane, I64SubbandEncodeSettings, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession, QuantStepSize,
};

mod accounting;
mod construction;
mod options;
mod transition;
pub(super) use construction::try_precinct_exponents;
use construction::{
    try_component_quantization, try_component_sample_info, try_component_sampling,
    try_component_step_sizes, try_quantization, try_step_sizes, PlanConstruction,
};
pub(super) use options::try_high_bit_options;

pub(super) struct TypedI64HighBitPlan {
    num_levels: u8,
    max_bit_depth: u8,
    guard_bits: u8,
    quant_params: Vec<(u16, u16)>,
    component_step_sizes: Vec<Vec<QuantStepSize>>,
    component_sample_info: Vec<EncodeComponentSampleInfo>,
    component_quantization_step_sizes: Vec<Vec<(u16, u16)>>,
    component_sampling: Vec<(u8, u8)>,
    block_coding_mode: BlockCodingMode,
    cb_width: u32,
    cb_height: u32,
}

pub(super) struct TypedI64ExecutionPlan {
    pub(super) params: EncodeParams,
    pub(super) quant_params: Vec<(u16, u16)>,
    pub(super) component_step_sizes: Vec<Vec<QuantStepSize>>,
    cb_width: u32,
    cb_height: u32,
}

pub(super) struct TypedI64ExecutionRequest<'a, 'input> {
    pub(super) dimensions: (u32, u32),
    pub(super) tile_dimensions: (u32, u32),
    pub(super) num_components: u16,
    pub(super) options: &'a EncodeOptions,
    pub(super) precinct_exponents: Vec<(u8, u8)>,
    pub(super) retained_base_bytes: usize,
    pub(super) session: &'a NativeEncodeSession<'input>,
}

impl TypedI64HighBitPlan {
    pub(super) fn try_new(
        planes: &[EncodeTypedComponentPlane<'_>],
        options: &EncodeOptions,
        num_levels: u8,
        retained_base_bytes: usize,
        session: &NativeEncodeSession<'_>,
    ) -> NativeEncodePipelineResult<Self> {
        let code_block_geometry = validate_code_block_geometry(options)
            .map_err(NativeEncodePipelineError::invalid_input)?;
        let max_bit_depth = planes
            .iter()
            .map(|plane| plane.bit_depth)
            .max()
            .ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant(
                    "validated typed component set is unexpectedly empty",
                )
            })?;
        let guard_bits =
            reversible_guard_bits_for_marker_limit(max_bit_depth, num_levels, options.guard_bits)
                .map_err(NativeEncodePipelineError::unsupported)?;
        let guard_delta = guard_bits.saturating_sub(options.guard_bits);
        let mut construction = PlanConstruction::new(session, retained_base_bytes);
        let mut step_sizes = try_step_sizes(
            max_bit_depth,
            num_levels,
            guard_bits,
            options,
            &mut construction,
        )?;
        let component_sample_info = try_component_sample_info(planes, &mut construction)?;
        let mut component_step_sizes = try_component_step_sizes(
            &component_sample_info,
            num_levels,
            guard_bits,
            options,
            &mut construction,
        )?;
        if guard_delta != 0 {
            adjust_reversible_step_sizes_for_guard_delta(&mut step_sizes, guard_delta)
                .map_err(NativeEncodePipelineError::unsupported)?;
            adjust_component_step_sizes_for_guard_delta(&mut component_step_sizes, guard_delta)
                .map_err(NativeEncodePipelineError::unsupported)?;
        }
        if step_sizes.iter().any(|step| step.exponent > 31)
            || component_step_sizes
                .iter()
                .flatten()
                .any(|step| step.exponent > 31)
        {
            return Err(NativeEncodePipelineError::unsupported(
                "25-38 bit typed component-plane encode exceeds the current no-quantization guard/exponent signaling limit",
            ));
        }
        let quant_params = try_quantization(&step_sizes, &mut construction)?;
        let component_quantization_step_sizes =
            try_component_quantization(&component_step_sizes, &mut construction)?;
        let component_sampling = try_component_sampling(planes, &mut construction)?;
        let plan = Self {
            num_levels,
            max_bit_depth,
            guard_bits,
            quant_params,
            component_step_sizes,
            component_sample_info,
            component_quantization_step_sizes,
            component_sampling,
            block_coding_mode: block_coding_mode(options),
            cb_width: code_block_geometry.width,
            cb_height: code_block_geometry.height,
        };
        let retained_bytes = plan.retained_bytes()?;
        session.checked_phase(
            checked_add_bytes(
                retained_base_bytes,
                retained_bytes,
                "typed i64 retained plan",
            )?,
            "typed i64 retained plan",
        )?;
        drop(step_sizes);
        Ok(plan)
    }

    pub(super) fn component_sampling(&self) -> &[(u8, u8)] {
        &self.component_sampling
    }
}

impl TypedI64ExecutionPlan {
    pub(super) fn subband_settings(
        &self,
        options: &EncodeOptions,
    ) -> I64SubbandEncodeSettings<'static> {
        subband_settings(
            self.params.guard_bits,
            self.params.block_coding_mode,
            self.cb_width,
            self.cb_height,
            options,
        )
    }

    pub(super) fn into_final_parts(self) -> (EncodeParams, Vec<(u16, u16)>) {
        let Self {
            params,
            quant_params,
            component_step_sizes: _,
            cb_width: _,
            cb_height: _,
        } = self;
        (params, quant_params)
    }
}

fn subband_settings(
    guard_bits: u8,
    block_coding_mode: BlockCodingMode,
    cb_width: u32,
    cb_height: u32,
    options: &EncodeOptions,
) -> I64SubbandEncodeSettings<'static> {
    I64SubbandEncodeSettings {
        guard_bits,
        cb_width,
        cb_height,
        roi_shift: 0,
        roi_regions: &[],
        roi_scale: 1,
        block_coding_mode,
        ht_target_coding_passes: ht_target_coding_passes_for_options(options, block_coding_mode),
    }
}

#[cfg(test)]
#[path = "plan/tests.rs"]
mod tests;
