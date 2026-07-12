// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed planning transitions for multi-tile encode.

use alloc::vec::Vec;

use super::super::allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed};
use super::super::single_tile::ownership::encode_params_retained_bytes;
use super::super::tile_parts::{encoded_tile_parts_retained_bytes, EncodedTilePart};
use super::super::{
    adjust_component_step_sizes_for_guard_delta, adjust_reversible_step_sizes_for_guard_delta,
    max_decomposition_levels, max_total_bitplanes_for_components, quantize,
    reversible_guard_bits_for_marker_limit, validate_precinct_exponents_for_options,
    validate_roi_encode_options_nonallocating, BlockCodingMode, EncodeComponentSampleInfo,
    EncodeOptions, EncodeParams, EncodeRoiRegion, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession, QuantStepSize, MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};
use super::ownership::encode_options_retained_bytes;

mod accounting;
use accounting::{
    final_plan_requested_bytes, requested_options_clone_bytes, requested_step_graph_bytes,
    step_graph_retained_bytes,
};
mod copy;
pub(in crate::j2c::encode) use copy::{
    try_clone_options, try_clone_options_with_component_sampling, try_copy_slice,
};
use copy::{try_component_sampling, try_roi_shifts};

pub(super) struct MultiTileLoopPlan {
    num_levels: u8,
    use_mct: bool,
    guard_bits: u8,
    child_options: EncodeOptions,
    retained_bytes: usize,
}

pub(super) struct MultiTileFinalPlan {
    pub(super) params: EncodeParams,
    pub(super) quant_params: Vec<(u16, u16)>,
}

struct FinalPlanOwners {
    params: EncodeParams,
    quant_params: Vec<(u16, u16)>,
    step_sizes: Vec<QuantStepSize>,
    component_step_sizes: Vec<Vec<QuantStepSize>>,
}

pub(super) struct LoopPlanRequest<'a, 'input> {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) tile_width: u32,
    pub(super) tile_height: u32,
    pub(super) num_components: u16,
    pub(super) bit_depth: u8,
    pub(super) options: &'a EncodeOptions,
    pub(super) roi_regions: &'a [EncodeRoiRegion],
    pub(super) component_sample_info: &'a [EncodeComponentSampleInfo],
    pub(super) block_coding_mode: BlockCodingMode,
    pub(super) session: &'a NativeEncodeSession<'input>,
}

pub(super) struct FinalPlanRequest<'a, 'input> {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) tile_width: u32,
    pub(super) tile_height: u32,
    pub(super) num_components: u16,
    pub(super) bit_depth: u8,
    pub(super) signed: bool,
    pub(super) options: &'a EncodeOptions,
    pub(super) roi_regions: &'a [EncodeRoiRegion],
    pub(super) component_sample_info: &'a [EncodeComponentSampleInfo],
    pub(super) block_coding_mode: BlockCodingMode,
    pub(super) tile_bodies: &'a Vec<EncodedTilePart>,
    pub(super) session: &'a NativeEncodeSession<'input>,
}

pub(super) fn build_loop_plan(
    request: &LoopPlanRequest<'_, '_>,
) -> NativeEncodePipelineResult<MultiTileLoopPlan> {
    let min_tile_width = if request.width.is_multiple_of(request.tile_width) {
        request.tile_width
    } else {
        request.width % request.tile_width
    };
    let min_tile_height = if request.height.is_multiple_of(request.tile_height) {
        request.tile_height
    } else {
        request.height % request.tile_height
    };
    let num_levels = request
        .options
        .num_decomposition_levels
        .min(max_decomposition_levels(min_tile_width, min_tile_height));
    validate_precinct_exponents_for_options(request.options, num_levels)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let use_mct = request.options.use_mct && matches!(request.num_components, 3 | 4);
    let requested_guard_bits = requested_guard_bits(request.options, use_mct);
    let high_bit_exact = request.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH;
    let guard_bits = if high_bit_exact && request.options.reversible {
        reversible_guard_bits_for_marker_limit(request.bit_depth, num_levels, requested_guard_bits)
            .map_err(NativeEncodePipelineError::unsupported)?
    } else {
        requested_guard_bits
    };
    let guard_delta = guard_bits.saturating_sub(requested_guard_bits);
    let requested_step_bytes =
        requested_step_graph_bytes(num_levels, request.component_sample_info.len())?;
    request
        .session
        .checked_phase(requested_step_bytes, "multi-tile validation step graph")?;
    let (mut step_sizes, mut component_step_sizes) = build_step_graph(
        request.bit_depth,
        num_levels,
        guard_bits,
        request.options,
        request.component_sample_info,
    )?;
    if request.options.reversible && guard_delta != 0 {
        adjust_reversible_step_sizes_for_guard_delta(&mut step_sizes, guard_delta)
            .map_err(NativeEncodePipelineError::unsupported)?;
        adjust_component_step_sizes_for_guard_delta(&mut component_step_sizes, guard_delta)
            .map_err(NativeEncodePipelineError::unsupported)?;
    }
    request.session.checked_phase(
        step_graph_retained_bytes(&step_sizes, &component_step_sizes)?,
        "multi-tile validation step graph",
    )?;
    validate_roi_encode_options_nonallocating(
        request.options,
        request.roi_regions,
        request.num_components,
        request.width,
        request.height,
        max_total_bitplanes_for_components(&step_sizes, &component_step_sizes, guard_bits)
            .map_err(NativeEncodePipelineError::internal_invariant)?,
        request.block_coding_mode,
    )?;
    drop(step_sizes);
    drop(component_step_sizes);

    let requested_options_bytes = requested_options_clone_bytes(request.options)?;
    request
        .session
        .checked_phase(requested_options_bytes, "multi-tile child options")?;
    let mut child_options = try_clone_options(request.options)?;
    child_options.use_ht_block_coding =
        request.block_coding_mode == BlockCodingMode::HighThroughput;
    child_options.num_decomposition_levels = num_levels;
    child_options.tile_size = None;
    child_options.write_tlm = false;
    child_options.write_plt = request.options.write_plt
        || request.options.write_plm
        || request.options.write_ppm
        || request.options.write_ppt
        || request.options.tile_part_packet_limit.is_some();
    child_options.write_plm = false;
    child_options.write_ppm = request.options.write_ppm || request.options.write_ppt;
    child_options.write_ppt = false;
    child_options.tile_part_packet_limit = None;
    let retained_bytes = encode_options_retained_bytes(&child_options)?;
    request
        .session
        .checked_phase(retained_bytes, "multi-tile child options")?;
    Ok(MultiTileLoopPlan {
        num_levels,
        use_mct,
        guard_bits,
        child_options,
        retained_bytes,
    })
}

impl MultiTileLoopPlan {
    pub(super) const fn child_options(&self) -> &EncodeOptions {
        &self.child_options
    }

    pub(super) const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(super) fn into_final_plan(
        self,
        request: &FinalPlanRequest<'_, '_>,
    ) -> NativeEncodePipelineResult<MultiTileFinalPlan> {
        let Self {
            num_levels,
            use_mct,
            guard_bits,
            child_options,
            retained_bytes: _,
        } = self;
        drop(child_options);
        let tile_bytes =
            encoded_tile_parts_retained_bytes(request.tile_bodies, request.tile_bodies.capacity())?;
        let requested_bytes = final_plan_requested_bytes(
            num_levels,
            request.num_components,
            request.component_sample_info.len(),
            request.options.precinct_exponents.len(),
        )?;
        request.session.checked_phase(
            checked_add_bytes(
                tile_bytes,
                requested_bytes,
                "multi-tile final plan construction",
            )?,
            "multi-tile final plan construction",
        )?;

        let FinalPlanOwners {
            params,
            quant_params,
            step_sizes,
            component_step_sizes,
        } = build_final_plan_owners(request, num_levels, use_mct, guard_bits)?;
        let live_construction_bytes = checked_add_bytes(
            tile_bytes,
            checked_add_bytes(
                step_graph_retained_bytes(&step_sizes, &component_step_sizes)?,
                checked_add_bytes(
                    encode_params_retained_bytes(&params)?,
                    checked_element_bytes::<(u16, u16)>(
                        quant_params.capacity(),
                        "multi-tile quantization",
                    )?,
                    "multi-tile final marker owners",
                )?,
                "multi-tile final plan construction",
            )?,
            "multi-tile final plan construction",
        )?;
        request.session.checked_phase(
            live_construction_bytes,
            "multi-tile final plan construction",
        )?;
        drop(step_sizes);
        drop(component_step_sizes);
        Ok(MultiTileFinalPlan {
            params,
            quant_params,
        })
    }
}

fn build_final_plan_owners(
    request: &FinalPlanRequest<'_, '_>,
    num_levels: u8,
    use_mct: bool,
    guard_bits: u8,
) -> NativeEncodePipelineResult<FinalPlanOwners> {
    let guard_delta = guard_bits.saturating_sub(requested_guard_bits(request.options, use_mct));
    let (mut step_sizes, mut component_step_sizes) = build_step_graph(
        request.bit_depth,
        num_levels,
        guard_bits,
        request.options,
        request.component_sample_info,
    )?;
    if request.options.reversible && guard_delta != 0 {
        adjust_reversible_step_sizes_for_guard_delta(&mut step_sizes, guard_delta)
            .map_err(NativeEncodePipelineError::unsupported)?;
        adjust_component_step_sizes_for_guard_delta(&mut component_step_sizes, guard_delta)
            .map_err(NativeEncodePipelineError::unsupported)?;
    }
    let quant_params = try_quantization(&step_sizes, "multi-tile quantization")?;
    let params = EncodeParams {
        width: request.width,
        height: request.height,
        tile_width: request.tile_width,
        tile_height: request.tile_height,
        num_components: request.num_components,
        bit_depth: request.bit_depth,
        signed: request.signed,
        component_sample_info: try_copy_slice(
            request.component_sample_info,
            "multi-tile component metadata",
        )?,
        component_quantization_step_sizes: try_component_quantization(&component_step_sizes)?,
        num_decomposition_levels: num_levels,
        reversible: request.options.reversible,
        code_block_width_exp: request.options.code_block_width_exp,
        code_block_height_exp: request.options.code_block_height_exp,
        num_layers: request.options.num_layers,
        use_mct,
        guard_bits,
        block_coding_mode: request.block_coding_mode,
        progression_order: request.options.progression_order,
        write_tlm: request.options.write_tlm,
        write_plt: request.options.write_plt,
        write_plm: request.options.write_plm,
        write_ppm: request.options.write_ppm,
        write_ppt: request.options.write_ppt,
        write_sop: request.options.write_sop,
        write_eph: request.options.write_eph,
        terminate_coding_passes: request.block_coding_mode == BlockCodingMode::Classic
            && request.options.num_layers > 1,
        component_sampling: try_component_sampling(request.options, request.num_components)?,
        roi_component_shifts: try_roi_shifts(
            request.options,
            request.roi_regions,
            request.num_components,
        )?,
        precinct_exponents: try_copy_slice(
            &request.options.precinct_exponents,
            "multi-tile precinct exponents",
        )?,
    };
    Ok(FinalPlanOwners {
        params,
        quant_params,
        step_sizes,
        component_step_sizes,
    })
}

fn requested_guard_bits(options: &EncodeOptions, use_mct: bool) -> u8 {
    if options.reversible && !use_mct {
        options.guard_bits
    } else {
        options.guard_bits.max(2)
    }
}

fn build_step_graph(
    bit_depth: u8,
    num_levels: u8,
    guard_bits: u8,
    options: &EncodeOptions,
    component_sample_info: &[EncodeComponentSampleInfo],
) -> NativeEncodePipelineResult<(Vec<QuantStepSize>, Vec<Vec<QuantStepSize>>)> {
    let step_sizes = try_step_sizes(bit_depth, num_levels, guard_bits, options)?;
    let outer_bytes = checked_element_bytes::<Vec<QuantStepSize>>(
        component_sample_info.len(),
        "multi-tile component step owners",
    )?;
    let mut component_steps = Vec::new();
    component_steps
        .try_reserve_exact(component_sample_info.len())
        .map_err(|_| host_allocation_failed("multi-tile component step owners", outer_bytes))?;
    for info in component_sample_info {
        component_steps.push(try_step_sizes(
            info.bit_depth,
            num_levels,
            guard_bits,
            options,
        )?);
    }
    Ok((step_sizes, component_steps))
}

fn try_step_sizes(
    bit_depth: u8,
    num_levels: u8,
    guard_bits: u8,
    options: &EncodeOptions,
) -> NativeEncodePipelineResult<Vec<QuantStepSize>> {
    let count = usize::from(num_levels) * 3 + 1;
    let bytes = checked_element_bytes::<QuantStepSize>(count, "multi-tile step sizes")?;
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(count)
        .map_err(|_| host_allocation_failed("multi-tile step sizes", bytes))?;
    quantize::append_step_sizes_with_irreversible_profile(
        &mut steps,
        bit_depth,
        num_levels,
        options.reversible,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    Ok(steps)
}

fn try_quantization(
    steps: &[QuantStepSize],
    what: &'static str,
) -> NativeEncodePipelineResult<Vec<(u16, u16)>> {
    let bytes = checked_element_bytes::<(u16, u16)>(steps.len(), what)?;
    let mut quantization = Vec::new();
    quantization
        .try_reserve_exact(steps.len())
        .map_err(|_| host_allocation_failed(what, bytes))?;
    quantization.extend(steps.iter().map(|step| (step.exponent, step.mantissa)));
    Ok(quantization)
}

fn try_component_quantization(
    component_steps: &[Vec<QuantStepSize>],
) -> NativeEncodePipelineResult<Vec<Vec<(u16, u16)>>> {
    let outer_bytes = checked_element_bytes::<Vec<(u16, u16)>>(
        component_steps.len(),
        "multi-tile component quantization owners",
    )?;
    let mut quantization = Vec::new();
    quantization
        .try_reserve_exact(component_steps.len())
        .map_err(|_| {
            host_allocation_failed("multi-tile component quantization owners", outer_bytes)
        })?;
    for steps in component_steps {
        quantization.push(try_quantization(
            steps,
            "multi-tile component quantization",
        )?);
    }
    Ok(quantization)
}

#[cfg(test)]
#[path = "plan/tests.rs"]
mod tests;
