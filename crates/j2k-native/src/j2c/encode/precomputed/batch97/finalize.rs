// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-image Tier-2 and final writer handoff under one aggregate batch cap.

use super::super::orchestrator::{plan_metadata_bytes, Prepared97PacketPlan};
use super::super::{
    packet_encode, packetization_requires_scalar,
    packetize_resolution_packets_with_options_for_session,
    write_single_tile_packetized_codestream_for_session, J2kEncodeStageAccelerator,
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession, ResolutionPacket,
    Vec,
};
use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::EncodeResult;

pub(super) fn packetize_and_finalize_batch(
    plans: Vec<Prepared97PacketPlan>,
    groups: Vec<Vec<ResolutionPacket>>,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<Vec<u8>>> {
    if plans.len() != groups.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "encoded image count mismatch",
        ));
    }
    let plan_outer_capacity = plans.capacity();
    let group_outer_capacity = groups.capacity();
    let output_requested =
        checked_element_bytes::<Vec<u8>>(plans.len(), "precomputed 9/7 codestream owners")?;
    session.checked_phase(
        checked_add_bytes(
            batch_plan_metadata_bytes(&plans, plan_outer_capacity)?,
            checked_add_bytes(
                batch_group_bytes(&groups, group_outer_capacity)?,
                output_requested,
                "precomputed 9/7 batch output owners",
            )?,
            "precomputed 9/7 batch output owners",
        )?,
        "precomputed 9/7 batch output owners",
    )?;
    let mut codestreams = Vec::new();
    codestreams.try_reserve_exact(plans.len()).map_err(|_| {
        host_allocation_failed("precomputed 9/7 codestream owners", output_requested)
    })?;
    let output_outer_bytes = checked_element_bytes::<Vec<u8>>(
        codestreams.capacity(),
        "precomputed 9/7 codestream owners",
    )?;
    let mut plans = plans.into_iter();
    let mut groups = groups.into_iter();
    while let Some(plan) = plans.next() {
        let resolution_packets = groups.next().ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("encoded image count mismatch")
        })?;
        let remaining = BatchTailOwners {
            plans: plans.as_slice(),
            plan_outer_capacity,
            groups: groups.as_slice(),
            group_outer_capacity,
            codestreams: &codestreams,
            output_outer_bytes,
        };
        let codestream =
            finalize_batch_plan(plan, resolution_packets, &remaining, session, accelerator)?;
        codestreams.push(codestream);
    }
    if groups.next().is_some() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "encoded image count mismatch",
        ));
    }
    Ok(codestreams)
}

struct BatchTailOwners<'a> {
    plans: &'a [Prepared97PacketPlan],
    plan_outer_capacity: usize,
    groups: &'a [Vec<ResolutionPacket>],
    group_outer_capacity: usize,
    codestreams: &'a [Vec<u8>],
    output_outer_bytes: usize,
}

fn finalize_batch_plan(
    plan: Prepared97PacketPlan,
    resolution_packets: Vec<ResolutionPacket>,
    remaining: &BatchTailOwners<'_>,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let Prepared97PacketPlan {
        params,
        quant_params,
        packet_descriptors,
        prepared_packets,
        tile_part_packet_limit,
    } = plan;
    if !prepared_packets.is_empty() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "batch finalization retained unexpected prepared packets",
        ));
    }
    let current_metadata = plan_metadata_bytes(&params, &quant_params, 0)?;
    let other_live = batch_iteration_live_bytes(remaining, current_metadata)?;
    let packetized = {
        let packet_owners = (
            remaining.plans,
            remaining.groups,
            remaining.codestreams,
            &params,
            &quant_params,
        );
        let packet_session = session.checked_child_session(
            &packet_owners,
            other_live,
            "precomputed 9/7 aggregate packetization owners",
        )?;
        packetize_resolution_packets_with_options_for_session(
            &resolution_packets,
            resolution_packets.capacity(),
            &packet_descriptors,
            packet_descriptors.capacity(),
            1,
            params.num_components,
            params.progression_order,
            packet_encode::PacketMarkerOptions {
                write_sop: params.write_sop,
                write_eph: params.write_eph,
                separate_packet_headers: params.write_ppm || params.write_ppt,
            },
            true,
            packetization_requires_scalar(&params, tile_part_packet_limit),
            &packet_session,
            accelerator,
        )?
    };
    drop(resolution_packets);
    drop(packet_descriptors);
    let final_owners = (
        remaining.plans,
        remaining.groups,
        remaining.codestreams,
        &params,
        &quant_params,
    );
    let final_session = session.checked_child_session(
        &final_owners,
        other_live,
        "precomputed 9/7 aggregate finalization owners",
    )?;
    write_single_tile_packetized_codestream_for_session(
        &params,
        &packetized,
        &quant_params,
        tile_part_packet_limit,
        0,
        &final_session,
    )
}

fn batch_iteration_live_bytes(
    remaining: &BatchTailOwners<'_>,
    current_metadata: usize,
) -> EncodeResult<usize> {
    let mut bytes = checked_element_bytes::<Prepared97PacketPlan>(
        remaining.plan_outer_capacity,
        "precomputed 9/7 retained plan owners",
    )?;
    bytes = checked_add_bytes(
        bytes,
        nested_plan_metadata_bytes(remaining.plans)?,
        "precomputed 9/7 remaining plan metadata",
    )?;
    bytes = checked_add_bytes(
        bytes,
        batch_group_bytes(remaining.groups, remaining.group_outer_capacity)?,
        "precomputed 9/7 remaining encoded packets",
    )?;
    bytes = checked_add_bytes(
        bytes,
        remaining.output_outer_bytes,
        "precomputed 9/7 output owners",
    )?;
    for codestream in remaining.codestreams {
        bytes = checked_add_bytes(
            bytes,
            codestream.capacity(),
            "precomputed 9/7 prior codestream payloads",
        )?;
    }
    checked_add_bytes(
        bytes,
        current_metadata,
        "precomputed 9/7 current plan metadata",
    )
}

pub(super) fn batch_plan_metadata_bytes(
    plans: &[Prepared97PacketPlan],
    outer_capacity: usize,
) -> EncodeResult<usize> {
    checked_add_bytes(
        checked_element_bytes::<Prepared97PacketPlan>(
            outer_capacity,
            "precomputed 9/7 batch plan owners",
        )?,
        nested_plan_metadata_bytes(plans)?,
        "precomputed 9/7 batch plan metadata",
    )
}

fn nested_plan_metadata_bytes(plans: &[Prepared97PacketPlan]) -> EncodeResult<usize> {
    plans.iter().try_fold(0usize, |bytes, plan| {
        checked_add_bytes(
            bytes,
            plan.metadata_retained_bytes()?,
            "batch plan metadata",
        )
    })
}

fn batch_group_bytes(
    groups: &[Vec<ResolutionPacket>],
    outer_capacity: usize,
) -> EncodeResult<usize> {
    let mut bytes = checked_element_bytes::<Vec<ResolutionPacket>>(
        outer_capacity,
        "precomputed 9/7 encoded group owners",
    )?;
    for group in groups {
        bytes = checked_add_bytes(
            bytes,
            packet_encode::owned_packet_retained_bytes_for_public_descriptors(
                group,
                group.capacity(),
                0,
                0,
            )?,
            "precomputed 9/7 encoded packet groups",
        )?;
    }
    Ok(bytes)
}
