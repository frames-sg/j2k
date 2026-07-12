// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::{
    encode_prepared_resolution_packets_for_session,
    encode_prepared_resolution_packets_layered_for_session,
    ordered_prepared_resolution_packets_for_session, packet_descriptors_for_order_for_session,
    packet_encode, packetization_requires_scalar,
    packetize_resolution_packets_with_options_for_session, profile,
    split_component_resolution_packets_by_precinct_for_session, EncodeComponentSampleInfo,
    EncodeOptions, J2kEncodeStageAccelerator, J2kPacketizationPacketDescriptor,
    NativeEncodePipelineResult, NativeEncodeSession,
};
use super::coefficient_source::DwtComponentSource;
use super::ownership::{
    prepared_packet_tree_retained_bytes, prepared_packets_retained_bytes,
    single_tile_plan_retained_bytes,
};
use super::plan::SingleTilePlan;

mod components;
mod dwt_band;
use components::{prepare_component_packets, ComponentPacketRequest};

pub(super) struct EncodedTilePackets {
    pub(super) packetized_tile: packet_encode::PacketizedTileData,
    pub(super) subband_prepare_us: u128,
    pub(super) block_encode_us: u128,
    pub(super) packetize_us: u128,
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
pub(super) fn encode_tile_packets<S: DwtComponentSource>(
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    options: &EncodeOptions,
    component_sample_info: &[EncodeComponentSampleInfo],
    plan: &SingleTilePlan,
    decompositions: &[S],
    source_retained_bytes: usize,
    profile_enabled: bool,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<EncodedTilePackets> {
    let retained_base_bytes = checked_add_bytes(
        single_tile_plan_retained_bytes(plan)?,
        source_retained_bytes,
        "retained transform and single-tile plan owners",
    )?;
    let (component_resolution_packets, subband_prepare_us) = prepare_component_packets(
        &ComponentPacketRequest {
            num_components,
            bit_depth,
            options,
            component_sample_info,
            plan,
            decompositions,
            retained_base_bytes,
            profile_enabled,
            session,
        },
        accelerator,
    )?;

    let component_resolution_packets = split_component_resolution_packets_by_precinct_for_session(
        component_resolution_packets,
        width,
        height,
        plan.num_levels,
        &plan.params.precinct_exponents,
        session,
        retained_base_bytes,
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            prepared_packet_tree_retained_bytes(
                &component_resolution_packets,
                component_resolution_packets.capacity(),
            )?,
            "precinct-split prepared packet tree",
        )?,
        "precinct-split prepared packet tree",
    )?;
    let prepared_resolution_packets = ordered_prepared_resolution_packets_for_session(
        component_resolution_packets,
        options,
        session,
        retained_base_bytes,
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            prepared_packets_retained_bytes(
                &prepared_resolution_packets,
                prepared_resolution_packets.capacity(),
            )?,
            "ordered prepared packet tree",
        )?,
        "ordered prepared packet tree",
    )?;
    let stage_start = profile::profile_now(profile_enabled);
    let (resolution_packets, packet_descriptors, allow_packetization_accelerator) =
        if options.num_layers > 1 {
            let (resolution_packets, packet_descriptors) =
                encode_prepared_resolution_packets_layered_for_session(
                    prepared_resolution_packets,
                    options.num_layers,
                    options.progression_order,
                    &options.quality_layer_byte_targets,
                    session,
                    retained_base_bytes,
                    accelerator,
                )?;
            (resolution_packets, packet_descriptors, false)
        } else {
            let packet_descriptors = packet_descriptors_for_order_for_session(
                &prepared_resolution_packets,
                prepared_resolution_packets.capacity(),
                1,
                options.progression_order,
                session,
                retained_base_bytes,
            )?;
            let descriptor_bytes = checked_element_bytes::<J2kPacketizationPacketDescriptor>(
                packet_descriptors.capacity(),
                "packet descriptor owners",
            )?;
            let resolution_packets = encode_prepared_resolution_packets_for_session(
                prepared_resolution_packets,
                session,
                checked_add_bytes(
                    retained_base_bytes,
                    descriptor_bytes,
                    "Tier-1 packet descriptor baseline",
                )?,
                accelerator,
            )?;
            (resolution_packets, packet_descriptors, true)
        };
    let block_encode_us = profile::elapsed_us(stage_start);

    let stage_start = profile::profile_now(profile_enabled);
    let packet_phase_owners = (plan, decompositions);
    let packet_session = session.checked_child_session(
        &packet_phase_owners,
        retained_base_bytes,
        "retained transform and single-tile plan owners",
    )?;
    let packetized_tile = packetize_resolution_packets_with_options_for_session(
        &resolution_packets,
        resolution_packets.capacity(),
        &packet_descriptors,
        packet_descriptors.capacity(),
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
        &packet_session,
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
