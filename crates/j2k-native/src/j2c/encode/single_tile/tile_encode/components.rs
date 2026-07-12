// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-component DWT-band preparation under the retained transform budget.

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::{
    profile, roi_subband_scale, EncodeComponentSampleInfo, EncodeOptions,
    J2kEncodeStageAccelerator, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeSession, PreparedResolutionPacket, SubBandType,
};

use super::super::coefficient_source::DwtComponentSource;
use super::super::ownership::prepared_packet_tree_retained_bytes;
use super::super::plan::SingleTilePlan;
use super::dwt_band::{prepare_dwt_band_for_session, DwtBandEncodeRequest, DwtBandEncodeSettings};

mod ownership;

use ownership::{subband_prepare_retained_bytes, try_own_packet_subbands};

pub(super) struct ComponentPacketRequest<'a, 'input, S> {
    pub(super) num_components: u16,
    pub(super) bit_depth: u8,
    pub(super) options: &'a EncodeOptions,
    pub(super) component_sample_info: &'a [EncodeComponentSampleInfo],
    pub(super) plan: &'a SingleTilePlan,
    pub(super) decompositions: &'a [S],
    pub(super) retained_base_bytes: usize,
    pub(super) profile_enabled: bool,
    pub(super) session: &'a NativeEncodeSession<'input>,
}

struct BandSettingsInput<'a> {
    step_size: &'a crate::j2c::quantize::QuantStepSize,
    bit_depth: u8,
    sub_band_type: SubBandType,
    roi_shift: u8,
    roi_regions: &'a [crate::j2c::encode::ComponentRoiEncodeRegion],
    roi_scale: u32,
    retained_base_bytes: usize,
}

pub(super) fn prepare_component_packets<S: DwtComponentSource>(
    request: &ComponentPacketRequest<'_, '_, S>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<(Vec<Vec<PreparedResolutionPacket>>, u128)> {
    if request.decompositions.len() != usize::from(request.num_components) {
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
    for (component_idx, decomposition) in request.decompositions.iter().enumerate() {
        prepare_one_component(
            component_idx,
            decomposition,
            request,
            &mut component_packets,
            accelerator,
        )?;
    }
    Ok((component_packets, profile::elapsed_us(stage_start)))
}

fn prepare_one_component<S: DwtComponentSource>(
    component_idx: usize,
    decomposition: &S,
    request: &ComponentPacketRequest<'_, '_, S>,
    completed: &mut Vec<Vec<PreparedResolutionPacket>>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    if decomposition.level_count() != usize::from(request.plan.num_levels) {
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
    let component_steps = request
        .plan
        .component_step_sizes
        .get(component_idx)
        .map_or(request.plan.step_sizes.as_slice(), Vec::as_slice);
    let roi_shift = request
        .plan
        .roi_component_shifts
        .get(component_idx)
        .copied()
        .unwrap_or(0);
    let roi_plan = request.plan.roi_plans.get(component_idx).ok_or_else(|| {
        NativeEncodePipelineError::internal_invariant(
            "ROI plan count does not match component count",
        )
    })?;
    let packet_count = usize::from(request.plan.num_levels) + 1;
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
        component_bit_depth,
        component_steps,
        roi_shift,
        &roi_plan.regions,
        decomposition,
        request,
        completed,
        &mut packets,
        accelerator,
    )?;
    prepare_detail_packets(
        component,
        component_bit_depth,
        component_steps,
        roi_shift,
        &roi_plan.regions,
        decomposition,
        request,
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
    bit_depth: u8,
    steps: &[crate::j2c::quantize::QuantStepSize],
    roi_shift: u8,
    roi_regions: &[crate::j2c::encode::ComponentRoiEncodeRegion],
    decomposition: &S,
    request: &ComponentPacketRequest<'_, '_, S>,
    completed: &Vec<Vec<PreparedResolutionPacket>>,
    packets: &mut Vec<PreparedResolutionPacket>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    let ll = decomposition.ll()?;
    let ll_subband = prepare_dwt_band_for_session(
        DwtBandEncodeRequest {
            band: ll,
            settings: band_settings(
                &BandSettingsInput {
                    step_size: &steps[0],
                    bit_depth,
                    sub_band_type: SubBandType::LowLow,
                    roi_shift,
                    roi_regions,
                    roi_scale: roi_subband_scale(request.plan.num_levels, None)
                        .map_err(NativeEncodePipelineError::internal_invariant)?,
                    retained_base_bytes: subband_prepare_retained_bytes(
                        request.retained_base_bytes,
                        completed,
                        packets,
                        &[],
                    )?,
                },
                request,
            ),
        },
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
    bit_depth: u8,
    steps: &[crate::j2c::quantize::QuantStepSize],
    roi_shift: u8,
    roi_regions: &[crate::j2c::encode::ComponentRoiEncodeRegion],
    decomposition: &S,
    request: &ComponentPacketRequest<'_, '_, S>,
    completed: &Vec<Vec<PreparedResolutionPacket>>,
    packets: &mut Vec<PreparedResolutionPacket>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    for level_idx in 0..decomposition.level_count() {
        let level = decomposition.level(level_idx)?.ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "DWT source level missing after validated count",
            )
        })?;
        let step_base = 1 + level_idx * 3;
        let roi_scale = roi_subband_scale(request.plan.num_levels, Some(level_idx))
            .map_err(NativeEncodePipelineError::internal_invariant)?;
        let hl = prepare_dwt_band_for_session(
            DwtBandEncodeRequest {
                band: level.hl,
                settings: band_settings(
                    &BandSettingsInput {
                        step_size: &steps[step_base],
                        bit_depth,
                        sub_band_type: SubBandType::HighLow,
                        roi_shift,
                        roi_regions,
                        roi_scale,
                        retained_base_bytes: subband_prepare_retained_bytes(
                            request.retained_base_bytes,
                            completed,
                            packets,
                            &[],
                        )?,
                    },
                    request,
                ),
            },
            accelerator,
        )?;
        let lh = prepare_dwt_band_for_session(
            DwtBandEncodeRequest {
                band: level.lh,
                settings: band_settings(
                    &BandSettingsInput {
                        step_size: &steps[step_base + 1],
                        bit_depth,
                        sub_band_type: SubBandType::LowHigh,
                        roi_shift,
                        roi_regions,
                        roi_scale,
                        retained_base_bytes: subband_prepare_retained_bytes(
                            request.retained_base_bytes,
                            completed,
                            packets,
                            &[&hl],
                        )?,
                    },
                    request,
                ),
            },
            accelerator,
        )?;
        let hh = prepare_dwt_band_for_session(
            DwtBandEncodeRequest {
                band: level.hh,
                settings: band_settings(
                    &BandSettingsInput {
                        step_size: &steps[step_base + 2],
                        bit_depth,
                        sub_band_type: SubBandType::HighHigh,
                        roi_shift,
                        roi_regions,
                        roi_scale,
                        retained_base_bytes: subband_prepare_retained_bytes(
                            request.retained_base_bytes,
                            completed,
                            packets,
                            &[&hl, &lh],
                        )?,
                    },
                    request,
                ),
            },
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
            resolution: u32::try_from(level_idx + 1).map_err(|_| {
                NativeEncodePipelineError::internal_invariant("resolution index exceeds u32")
            })?,
            precinct: 0,
            subbands,
        });
    }
    Ok(())
}

fn band_settings<'a, 'input, S>(
    input: &BandSettingsInput<'a>,
    request: &'a ComponentPacketRequest<'a, 'input, S>,
) -> DwtBandEncodeSettings<'a, 'input> {
    DwtBandEncodeSettings {
        step_size: input.step_size,
        bit_depth: input.bit_depth,
        guard_bits: request.plan.guard_bits,
        reversible: request.options.reversible,
        block_coding_mode: request.plan.params.block_coding_mode,
        cb_width: request.plan.cb_width,
        cb_height: request.plan.cb_height,
        sub_band_type: input.sub_band_type,
        roi_shift: input.roi_shift,
        roi_regions: input.roi_regions,
        roi_scale: input.roi_scale,
        ht_target_coding_passes: request.plan.ht_target_coding_passes,
        session: request.session,
        retained_base_bytes: input.retained_base_bytes,
    }
}
