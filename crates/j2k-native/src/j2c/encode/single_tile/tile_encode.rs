// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::*;
use super::plan::SingleTilePlan;

pub(super) struct EncodedTilePackets {
    pub(super) packetized_tile: packet_encode::PacketizedTileData,
    pub(super) subband_prepare_us: u128,
    pub(super) block_encode_us: u128,
    pub(super) packetize_us: u128,
}

pub(super) fn encode_tile_packets(
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    options: &EncodeOptions,
    component_sample_info: &[EncodeComponentSampleInfo],
    plan: &SingleTilePlan,
    decompositions: &[DwtDecomposition],
    profile_enabled: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<EncodedTilePackets, &'static str> {
    let mut component_resolution_packets = Vec::with_capacity(num_components as usize);
    let stage_start = profile::profile_now(profile_enabled);
    for (component_idx, decomposition) in decompositions
        .iter()
        .take(num_components as usize)
        .enumerate()
    {
        let component = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
        let component_bit_depth = component_sample_info
            .get(component_idx)
            .map_or(bit_depth, |info| info.bit_depth);
        let component_steps = plan
            .component_step_sizes
            .get(component_idx)
            .map_or(plan.step_sizes.as_slice(), Vec::as_slice);
        let roi_shift = plan
            .roi_component_shifts
            .get(component_idx)
            .copied()
            .unwrap_or(0);
        let roi_plan = plan
            .roi_plans
            .get(component_idx)
            .ok_or("ROI plan count does not match component count")?;
        let mut packets = Vec::with_capacity(plan.num_levels as usize + 1);

        let ll_roi_scale = roi_subband_scale(plan.num_levels, None)?;
        let ll_subband = prepare_subband(
            &decomposition.ll,
            decomposition.ll_width,
            decomposition.ll_height,
            &component_steps[0],
            component_bit_depth,
            plan.guard_bits,
            options.reversible,
            plan.params.block_coding_mode,
            plan.cb_width,
            plan.cb_height,
            SubBandType::LowLow,
            roi_shift,
            &roi_plan.regions,
            ll_roi_scale,
            plan.ht_target_coding_passes,
            accelerator,
        )?;
        packets.push(PreparedResolutionPacket {
            component,
            resolution: 0,
            precinct: 0,
            subbands: vec![ll_subband],
        });

        for (level_idx, level) in decomposition.levels.iter().enumerate() {
            let step_base = 1 + level_idx * 3;
            let level_roi_scale = roi_subband_scale(plan.num_levels, Some(level_idx))?;
            let hl_subband = prepare_subband(
                &level.hl,
                level.high_width,
                level.low_height,
                &component_steps[step_base],
                component_bit_depth,
                plan.guard_bits,
                options.reversible,
                plan.params.block_coding_mode,
                plan.cb_width,
                plan.cb_height,
                SubBandType::HighLow,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                plan.ht_target_coding_passes,
                accelerator,
            )?;
            let lh_subband = prepare_subband(
                &level.lh,
                level.low_width,
                level.high_height,
                &component_steps[step_base + 1],
                component_bit_depth,
                plan.guard_bits,
                options.reversible,
                plan.params.block_coding_mode,
                plan.cb_width,
                plan.cb_height,
                SubBandType::LowHigh,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                plan.ht_target_coding_passes,
                accelerator,
            )?;
            let hh_subband = prepare_subband(
                &level.hh,
                level.high_width,
                level.high_height,
                &component_steps[step_base + 2],
                component_bit_depth,
                plan.guard_bits,
                options.reversible,
                plan.params.block_coding_mode,
                plan.cb_width,
                plan.cb_height,
                SubBandType::HighHigh,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                plan.ht_target_coding_passes,
                accelerator,
            )?;

            packets.push(PreparedResolutionPacket {
                component,
                resolution: u32::try_from(level_idx + 1)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: vec![hl_subband, lh_subband, hh_subband],
            });
        }

        component_resolution_packets.push(packets);
    }
    let subband_prepare_us = profile::elapsed_us(stage_start);

    let component_resolution_packets = split_component_resolution_packets_by_precinct(
        component_resolution_packets,
        width,
        height,
        plan.num_levels,
        &plan.params.precinct_exponents,
    )?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, options)?;
    let stage_start = profile::profile_now(profile_enabled);
    let (resolution_packets, packet_descriptors, allow_packetization_accelerator) =
        if options.num_layers > 1 {
            let (resolution_packets, packet_descriptors) =
                encode_prepared_resolution_packets_layered(
                    prepared_resolution_packets,
                    options.num_layers,
                    options.progression_order,
                    &options.quality_layer_byte_targets,
                    accelerator,
                )?;
            (resolution_packets, packet_descriptors, false)
        } else {
            let packet_descriptors = packet_descriptors_for_order(
                &prepared_resolution_packets,
                1,
                options.progression_order,
            )?;
            let resolution_packets =
                encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
            (resolution_packets, packet_descriptors, true)
        };
    let block_encode_us = profile::elapsed_us(stage_start);

    let stage_start = profile::profile_now(profile_enabled);
    let mut resolution_packets = resolution_packets;
    let packetized_tile = packetize_resolution_packets_with_options(
        &mut resolution_packets,
        &packet_descriptors,
        options.num_layers,
        num_components,
        options.progression_order,
        packet_encode::PacketMarkerOptions {
            write_sop: plan.params.write_sop,
            write_eph: plan.params.write_eph,
            separate_packet_headers: plan.params.write_ppm || plan.params.write_ppt,
        },
        allow_packetization_accelerator,
        packetization_requires_scalar(&plan.params, options.tile_part_packet_limit),
        accelerator,
    )?;
    let packetize_us = profile::elapsed_us(stage_start);

    Ok(EncodedTilePackets {
        packetized_tile,
        subband_prepare_us,
        block_encode_us,
        packetize_us,
    })
}
