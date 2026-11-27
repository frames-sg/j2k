// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-component DWT-band preparation under the retained transform budget.

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::{
    profile, roi_subband_scale, EncodeComponentSampleInfo, EncodeOptions,
    J2kEncodeStageAccelerator, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeSession, PreparedCodeBlockCoefficients, PreparedEncodeSubband,
    PreparedResolutionPacket, SubBandType, MAX_REVERSIBLE_NO_QUANT_EXPONENT,
};
use crate::EncodeError;

use super::super::coefficient_source::{DwtBandView, DwtComponentSource};
use super::super::ownership::prepared_packet_tree_retained_bytes;
use super::super::plan::SingleTilePlan;
use super::dwt_band::{prepare_dwt_band_for_session, DwtBandEncodeRequest, DwtBandEncodeSettings};

mod ownership;

use ownership::{subband_prepare_retained_bytes, try_own_packet_subbands};

#[derive(Clone, Copy)]
pub(super) struct ComponentPacketRequest<'a, 'input> {
    pub(super) num_components: u16,
    pub(super) bit_depth: u8,
    pub(super) options: &'a EncodeOptions,
    pub(super) component_sample_info: &'a [EncodeComponentSampleInfo],
    pub(super) retained_base_bytes: usize,
    pub(super) profile_enabled: bool,
    pub(super) session: &'a NativeEncodeSession<'input>,
}

struct BandSettingsInput<'a> {
    origin: (u32, u32),
    step_size: crate::j2c::quantize::QuantStepSize,
    bit_depth: u8,
    sub_band_type: SubBandType,
    roi_shift: u8,
    roi_regions: &'a [crate::j2c::encode::ComponentRoiEncodeRegion],
    roi_scale: u32,
    retained_base_bytes: usize,
}

#[derive(Clone, Copy)]
struct SubbandSlot {
    component_idx: usize,
    step_idx: usize,
    resolution: usize,
    subband_pos: usize,
}

/// Prepare every component's packets, dropping each source once its code
/// blocks are prepared: they own their samples, so a source taken by value
/// frees its coefficients there. Returns the bytes those drops freed.
pub(super) fn prepare_component_packets<S: DwtComponentSource>(
    request: &ComponentPacketRequest<'_, '_>,
    plan: &mut SingleTilePlan,
    decompositions: impl ExactSizeIterator<Item = S>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<(Vec<Vec<PreparedResolutionPacket>>, u128, usize)> {
    if decompositions.len() != usize::from(request.num_components) {
        return Err(NativeEncodePipelineError::internal_invariant(
            "DWT source count does not match component count",
        ));
    }
    let component_owner_bytes = checked_element_bytes::<Vec<PreparedResolutionPacket>>(
        usize::from(request.num_components),
        "prepared component packet owners",
    )?;
    request.session.checked_phase(
        checked_add_bytes(
            request.retained_base_bytes,
            component_owner_bytes,
            "prepared component packet owners",
        )?,
        "prepared component packet owners",
    )?;
    let mut component_packets = Vec::new();
    component_packets
        .try_reserve_exact(usize::from(request.num_components))
        .map_err(|_| {
            host_allocation_failed("prepared component packet owners", component_owner_bytes)
        })?;
    let stage_start = profile::profile_now(request.profile_enabled);
    let mut released = 0;
    for (component_idx, decomposition) in decompositions.enumerate() {
        let retained_base_bytes = request
            .retained_base_bytes
            .checked_sub(released)
            .ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant(
                    "released DWT sources exceed the retained baseline",
                )
            })?;
        prepare_one_component(
            component_idx,
            &decomposition,
            &ComponentPacketRequest {
                retained_base_bytes,
                ..*request
            },
            plan,
            &mut component_packets,
            accelerator,
        )?;
        released = checked_add_bytes(
            released,
            decomposition.owned_bytes()?,
            "released DWT sources",
        )?;
    }
    Ok((
        component_packets,
        profile::elapsed_us(stage_start),
        released,
    ))
}

fn prepare_one_component<S: DwtComponentSource>(
    component_idx: usize,
    decomposition: &S,
    request: &ComponentPacketRequest<'_, '_>,
    plan: &mut SingleTilePlan,
    completed: &mut Vec<Vec<PreparedResolutionPacket>>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    if decomposition.level_count() != usize::from(plan.num_levels) {
        return Err(NativeEncodePipelineError::internal_invariant(
            "DWT source level count does not match encode plan",
        ));
    }
    let component = u16::try_from(component_idx).map_err(|_| {
        NativeEncodePipelineError::internal_invariant("component index exceeds u16")
    })?;
    let component_bit_depth = request
        .component_sample_info
        .get(component_idx)
        .map_or(request.bit_depth, |info| info.bit_depth);
    let roi_shift = plan
        .roi_component_shifts
        .get(component_idx)
        .copied()
        .unwrap_or(0);
    if plan.roi_plans.get(component_idx).is_none() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "ROI plan count does not match component count",
        ));
    }
    let packet_count = usize::from(plan.num_levels) + 1;
    let packet_owner_bytes = checked_element_bytes::<PreparedResolutionPacket>(
        packet_count,
        "prepared resolution packet owners",
    )?;
    request.session.checked_phase(
        checked_add_bytes(
            checked_add_bytes(
                request.retained_base_bytes,
                prepared_packet_tree_retained_bytes(completed, completed.capacity())?,
                "prepared packet tree",
            )?,
            packet_owner_bytes,
            "prepared resolution packet owners",
        )?,
        "prepared resolution packet owners",
    )?;
    let mut packets = Vec::new();
    packets.try_reserve_exact(packet_count).map_err(|_| {
        host_allocation_failed("prepared resolution packet owners", packet_owner_bytes)
    })?;

    prepare_ll_packet(
        component,
        component_idx,
        component_bit_depth,
        roi_shift,
        decomposition,
        request,
        plan,
        completed,
        &mut packets,
        accelerator,
    )?;
    prepare_detail_packets(
        component,
        component_idx,
        component_bit_depth,
        roi_shift,
        decomposition,
        request,
        plan,
        completed,
        &mut packets,
        accelerator,
    )?;

    completed.push(packets);
    request.session.checked_phase(
        checked_add_bytes(
            request.retained_base_bytes,
            prepared_packet_tree_retained_bytes(completed, completed.capacity())?,
            "prepared packet tree",
        )?,
        "prepared packet tree",
    )?;
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the LL request keeps component coding metadata and retained owners explicit"
)]
fn prepare_ll_packet<S: DwtComponentSource>(
    component: u16,
    component_idx: usize,
    bit_depth: u8,
    roi_shift: u8,
    decomposition: &S,
    request: &ComponentPacketRequest<'_, '_>,
    plan: &mut SingleTilePlan,
    completed: &mut Vec<Vec<PreparedResolutionPacket>>,
    packets: &mut Vec<PreparedResolutionPacket>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    let ll = decomposition.ll()?;
    let origin = super::super::super::tile_grid::subband_origin(
        plan.component_origin(component_idx),
        plan.num_levels,
        SubBandType::LowLow,
    )?;
    let roi_scale = roi_subband_scale(plan.num_levels, None)
        .map_err(NativeEncodePipelineError::internal_invariant)?;
    let retained_base_bytes =
        subband_prepare_retained_bytes(request.retained_base_bytes, completed, packets, &[])?;
    let slot = SubbandSlot {
        component_idx,
        step_idx: 0,
        resolution: 0,
        subband_pos: 0,
    };
    let ll_subband = prepare_band_with_reversible_exponent(
        ll,
        slot,
        origin,
        bit_depth,
        SubBandType::LowLow,
        roi_shift,
        roi_scale,
        retained_base_bytes,
        request,
        plan,
        completed,
        accelerator,
    )?;
    let subbands = try_own_packet_subbands(
        [ll_subband],
        request.retained_base_bytes,
        completed,
        packets,
        request.session,
    )?;
    packets.push(PreparedResolutionPacket {
        component,
        resolution: 0,
        precinct: 0,
        subbands,
    });
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "detail-band preparation keeps component coding metadata and retained owners explicit"
)]
fn prepare_detail_packets<S: DwtComponentSource>(
    component: u16,
    component_idx: usize,
    bit_depth: u8,
    roi_shift: u8,
    decomposition: &S,
    request: &ComponentPacketRequest<'_, '_>,
    plan: &mut SingleTilePlan,
    completed: &mut Vec<Vec<PreparedResolutionPacket>>,
    packets: &mut Vec<PreparedResolutionPacket>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    for level_idx in 0..decomposition.level_count() {
        let level = decomposition.level(level_idx)?.ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "DWT source level missing after validated count",
            )
        })?;
        let depth = u8::try_from(level_idx)
            .ok()
            .and_then(|index| plan.num_levels.checked_sub(index))
            .ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant(
                    "DWT source level exceeds the planned decomposition depth",
                )
            })?;
        let step_base = 1 + level_idx * 3;
        let roi_scale = roi_subband_scale(plan.num_levels, Some(level_idx))
            .map_err(NativeEncodePipelineError::internal_invariant)?;
        let resolution = level_idx + 1;
        let comp_origin = plan.component_origin(component_idx);

        let hl = prepare_detail_band(
            level.hl,
            SubbandSlot {
                component_idx,
                step_idx: step_base,
                resolution,
                subband_pos: 0,
            },
            comp_origin,
            depth,
            bit_depth,
            SubBandType::HighLow,
            roi_shift,
            roi_scale,
            &[],
            request,
            plan,
            completed,
            packets,
            accelerator,
        )?;
        let lh = prepare_detail_band(
            level.lh,
            SubbandSlot {
                component_idx,
                step_idx: step_base + 1,
                resolution,
                subband_pos: 1,
            },
            comp_origin,
            depth,
            bit_depth,
            SubBandType::LowHigh,
            roi_shift,
            roi_scale,
            &[&hl],
            request,
            plan,
            completed,
            packets,
            accelerator,
        )?;
        let hh = prepare_detail_band(
            level.hh,
            SubbandSlot {
                component_idx,
                step_idx: step_base + 2,
                resolution,
                subband_pos: 2,
            },
            comp_origin,
            depth,
            bit_depth,
            SubBandType::HighHigh,
            roi_shift,
            roi_scale,
            &[&hl, &lh],
            request,
            plan,
            completed,
            packets,
            accelerator,
        )?;
        let subbands = try_own_packet_subbands(
            [hl, lh, hh],
            request.retained_base_bytes,
            completed,
            packets,
            request.session,
        )?;
        packets.push(PreparedResolutionPacket {
            component,
            resolution: u32::try_from(resolution).map_err(|_| {
                NativeEncodePipelineError::internal_invariant("resolution index exceeds u32")
            })?,
            precinct: 0,
            subbands,
        });
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "detail-band helper keeps geometry, Vec capacity accounting, and mutable plan explicit"
)]
fn prepare_detail_band(
    band: DwtBandView<'_>,
    slot: SubbandSlot,
    comp_origin: (u32, u32),
    depth: u8,
    bit_depth: u8,
    sub_band_type: SubBandType,
    roi_shift: u8,
    roi_scale: u32,
    prior_bands: &[&PreparedEncodeSubband],
    request: &ComponentPacketRequest<'_, '_>,
    plan: &mut SingleTilePlan,
    completed: &mut Vec<Vec<PreparedResolutionPacket>>,
    packets: &Vec<PreparedResolutionPacket>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let origin = super::super::super::tile_grid::subband_origin(comp_origin, depth, sub_band_type)?;
    let retained_base_bytes = subband_prepare_retained_bytes(
        request.retained_base_bytes,
        completed,
        packets,
        prior_bands,
    )?;
    prepare_band_with_reversible_exponent(
        band,
        slot,
        origin,
        bit_depth,
        sub_band_type,
        roi_shift,
        roi_scale,
        retained_base_bytes,
        request,
        plan,
        completed,
        accelerator,
    )
}

/// Prepare one DWT subband, dynamically bumping its reversible 5/3 quantization
/// exponent when actual lifting coefficients exceed the nominal dynamic-range
/// bitplane count (e.g., high-contrast edges or overlays after multi-level 5/3).
#[expect(
    clippy::too_many_arguments,
    reason = "reversible exponent reconciliation sits directly at the subband preparation boundary"
)]
fn prepare_band_with_reversible_exponent(
    band: DwtBandView<'_>,
    slot: SubbandSlot,
    origin: (u32, u32),
    bit_depth: u8,
    sub_band_type: SubBandType,
    roi_shift: u8,
    roi_scale: u32,
    retained_base_bytes: usize,
    request: &ComponentPacketRequest<'_, '_>,
    plan: &mut SingleTilePlan,
    completed: &mut [Vec<PreparedResolutionPacket>],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let step_size = current_step_size(plan, slot);
    let attempt = {
        let initial_input = BandSettingsInput {
            origin,
            step_size,
            bit_depth,
            sub_band_type,
            roi_shift,
            roi_regions: &plan.roi_plans[slot.component_idx].regions,
            roi_scale,
            retained_base_bytes,
        };
        prepare_dwt_band_for_session(
            DwtBandEncodeRequest {
                band,
                settings: band_settings(&initial_input, request, plan),
            },
            accelerator,
        )
    };
    let can_bump = request.options.reversible && roi_shift == 0;
    let mut subband = match attempt {
        Ok(subband) => subband,
        Err(error) if can_bump && is_ht_bitplane_overflow(&error) => {
            let max_abs = dwt_band_max_abs_i32(band);
            let required_bits = crate::math::bit_width_u32(max_abs);
            let current_bits = plan
                .guard_bits
                .saturating_add(u8::try_from(step_size.exponent).unwrap_or(u8::MAX))
                .saturating_sub(1);
            let delta = required_bits.saturating_sub(current_bits);
            if delta == 0 {
                return Err(error);
            }
            bump_reversible_subband_exponent(plan, slot, delta, completed)?;
            let bumped_step = current_step_size(plan, slot);
            let retry_input = BandSettingsInput {
                origin,
                step_size: bumped_step,
                bit_depth,
                sub_band_type,
                roi_shift,
                roi_regions: &plan.roi_plans[slot.component_idx].regions,
                roi_scale,
                retained_base_bytes,
            };
            prepare_dwt_band_for_session(
                DwtBandEncodeRequest {
                    band,
                    settings: band_settings(&retry_input, request, plan),
                },
                accelerator,
            )?
        }
        Err(error) => return Err(error),
    };
    if can_bump && subband.preencoded_ht_code_blocks.is_none() {
        let max_abs = prepared_subband_max_abs(&subband);
        let required_bits = crate::math::bit_width_u64(max_abs);
        if required_bits > subband.total_bitplanes {
            let delta = required_bits - subband.total_bitplanes;
            bump_reversible_subband_exponent(plan, slot, delta, completed)?;
            subband.total_bitplanes = required_bits;
        }
    }
    Ok(subband)
}

fn current_step_size(
    plan: &SingleTilePlan,
    slot: SubbandSlot,
) -> crate::j2c::quantize::QuantStepSize {
    plan.component_step_sizes
        .get(slot.component_idx)
        .map_or(plan.step_sizes.as_slice(), Vec::as_slice)[slot.step_idx]
}

fn is_ht_bitplane_overflow(error: &NativeEncodePipelineError) -> bool {
    matches!(
        error,
        NativeEncodePipelineError::InvalidInput(
            "HTJ2K block magnitude exceeds configured bitplane count"
        ) | NativeEncodePipelineError::Typed(EncodeError::InvalidInput {
            what: "HTJ2K block magnitude exceeds configured bitplane count",
        })
    )
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "reversible 5/3 f32 coefficients fit in i32"
)]
fn dwt_band_max_abs_i32(band: DwtBandView<'_>) -> u32 {
    let mut maximum = 0_u32;
    match band {
        DwtBandView::Contiguous { coefficients, .. } => {
            for &value in coefficients {
                maximum = maximum.max((value as i32).unsigned_abs());
            }
        }
        DwtBandView::Packed(view) => {
            for y in 0..view.height() {
                if let Some(row) = view.row(y) {
                    for &value in row {
                        maximum = maximum.max((value as i32).unsigned_abs());
                    }
                }
            }
        }
    }
    maximum
}

fn prepared_subband_max_abs(subband: &PreparedEncodeSubband) -> u64 {
    let mut maximum = 0_u64;
    for block in &subband.code_blocks {
        match &block.coefficients {
            PreparedCodeBlockCoefficients::I32(values) => {
                for &v in values {
                    maximum = maximum.max(u64::from(v.unsigned_abs()));
                }
            }
            PreparedCodeBlockCoefficients::I64(values) => {
                for &v in values {
                    maximum = maximum.max(v.unsigned_abs());
                }
            }
            PreparedCodeBlockCoefficients::Empty => {}
        }
    }
    maximum
}

fn bump_reversible_subband_exponent(
    plan: &mut SingleTilePlan,
    slot: SubbandSlot,
    delta: u8,
    completed: &mut [Vec<PreparedResolutionPacket>],
) -> NativeEncodePipelineResult<()> {
    let delta_u16 = u16::from(delta);
    if let Some(comp_steps) = plan.component_step_sizes.get_mut(slot.component_idx) {
        let next = comp_steps[slot.step_idx]
            .exponent
            .checked_add(delta_u16)
            .filter(|&exp| exp <= MAX_REVERSIBLE_NO_QUANT_EXPONENT)
            .ok_or_else(|| {
                NativeEncodePipelineError::invalid_input(
                    "reversible subband dynamic range exceeds marker exponent limit",
                )
            })?;
        comp_steps[slot.step_idx].exponent = next;
        if let Some(params_steps) = plan
            .params
            .component_quantization_step_sizes
            .get_mut(slot.component_idx)
        {
            params_steps[slot.step_idx].0 = next;
        }
        return Ok(());
    }
    let next = plan.step_sizes[slot.step_idx]
        .exponent
        .checked_add(delta_u16)
        .filter(|&exp| exp <= MAX_REVERSIBLE_NO_QUANT_EXPONENT)
        .ok_or_else(|| {
            NativeEncodePipelineError::invalid_input(
                "reversible subband dynamic range exceeds marker exponent limit",
            )
        })?;
    plan.step_sizes[slot.step_idx].exponent = next;
    plan.quant_params[slot.step_idx].0 = next;
    for prior_component in completed.iter_mut().take(slot.component_idx) {
        if let Some(packet) = prior_component.get_mut(slot.resolution) {
            if let Some(subband) = packet.subbands.get_mut(slot.subband_pos) {
                subband.total_bitplanes =
                    subband.total_bitplanes.checked_add(delta).ok_or_else(|| {
                        NativeEncodePipelineError::arithmetic_overflow(
                            "bumped reversible subband bitplane count",
                        )
                    })?;
                if let Some(blocks) = subband.preencoded_ht_code_blocks.as_mut() {
                    for block in blocks {
                        block.num_zero_bitplanes =
                            block.num_zero_bitplanes.checked_add(delta).ok_or_else(|| {
                                NativeEncodePipelineError::arithmetic_overflow(
                                    "bumped reversible HT zero-bitplane count",
                                )
                            })?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn band_settings<'a, 'input>(
    input: &'a BandSettingsInput<'a>,
    request: &'a ComponentPacketRequest<'a, 'input>,
    plan: &'a SingleTilePlan,
) -> DwtBandEncodeSettings<'a, 'input> {
    DwtBandEncodeSettings {
        origin: input.origin,
        step_size: &input.step_size,
        bit_depth: input.bit_depth,
        guard_bits: plan.guard_bits,
        reversible: request.options.reversible,
        block_coding_mode: plan.params.block_coding_mode,
        cb_width: plan.cb_width,
        cb_height: plan.cb_height,
        sub_band_type: input.sub_band_type,
        roi_shift: input.roi_shift,
        roi_regions: input.roi_regions,
        roi_scale: input.roi_scale,
        ht_target_coding_passes: plan.ht_target_coding_passes,
        single_layer: plan.params.num_layers == 1,
        session: request.session,
        retained_base_bytes: input.retained_base_bytes,
    }
}
