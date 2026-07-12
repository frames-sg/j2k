// SPDX-License-Identifier: MIT OR Apache-2.0

//! Consume-only transition from a validated high-bit plan to execution state.

use alloc::vec::Vec;

use super::{TypedI64ExecutionPlan, TypedI64ExecutionRequest, TypedI64HighBitPlan};
use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::{BlockCodingMode, EncodeParams, NativeEncodePipelineResult};

impl TypedI64HighBitPlan {
    pub(in crate::j2c::encode::typed_i64) fn try_into_execution(
        self,
        request: TypedI64ExecutionRequest<'_, '_>,
    ) -> NativeEncodePipelineResult<TypedI64ExecutionPlan> {
        let TypedI64ExecutionRequest {
            dimensions,
            tile_dimensions,
            num_components,
            options,
            precinct_exponents,
            retained_base_bytes,
            session,
        } = request;
        let plan_bytes = self.retained_bytes()?;
        let precinct_bytes = checked_element_bytes::<(u8, u8)>(
            precinct_exponents.capacity(),
            "typed i64 precinct exponents",
        )?;
        let roi_bytes =
            checked_element_bytes::<u8>(usize::from(num_components), "typed i64 zero ROI shifts")?;
        session.checked_phase(
            checked_add_bytes(
                retained_base_bytes,
                checked_add_bytes(
                    plan_bytes,
                    checked_add_bytes(precinct_bytes, roi_bytes, "typed i64 marker plan")?,
                    "typed i64 marker plan",
                )?,
                "typed i64 marker plan",
            )?,
            "typed i64 marker plan",
        )?;
        let mut roi_component_shifts = Vec::new();
        roi_component_shifts
            .try_reserve_exact(usize::from(num_components))
            .map_err(|_| host_allocation_failed("typed i64 zero ROI shifts", roi_bytes))?;
        roi_component_shifts.resize(usize::from(num_components), 0);
        let Self {
            num_levels,
            max_bit_depth,
            guard_bits,
            quant_params,
            component_step_sizes,
            component_sample_info,
            component_quantization_step_sizes,
            component_sampling,
            block_coding_mode,
            cb_width,
            cb_height,
        } = self;
        let params = EncodeParams {
            width: dimensions.0,
            height: dimensions.1,
            tile_width: tile_dimensions.0,
            tile_height: tile_dimensions.1,
            num_components,
            bit_depth: max_bit_depth,
            signed: component_sample_info.iter().all(|info| info.signed),
            component_sample_info,
            component_quantization_step_sizes,
            num_decomposition_levels: num_levels,
            reversible: true,
            code_block_width_exp: options.code_block_width_exp,
            code_block_height_exp: options.code_block_height_exp,
            num_layers: options.num_layers,
            use_mct: false,
            guard_bits,
            block_coding_mode,
            progression_order: options.progression_order,
            write_tlm: options.write_tlm,
            write_plt: options.write_plt,
            write_plm: options.write_plm,
            write_ppm: options.write_ppm,
            write_ppt: options.write_ppt,
            write_sop: options.write_sop,
            write_eph: options.write_eph,
            terminate_coding_passes: block_coding_mode == BlockCodingMode::Classic
                && options.num_layers > 1,
            component_sampling,
            roi_component_shifts,
            precinct_exponents,
        };
        let execution = TypedI64ExecutionPlan {
            params,
            quant_params,
            component_step_sizes,
            cb_width,
            cb_height,
        };
        session.checked_phase(
            checked_add_bytes(
                retained_base_bytes,
                execution.retained_bytes()?,
                "typed i64 execution plan",
            )?,
            "typed i64 execution plan",
        )?;
        Ok(execution)
    }
}
