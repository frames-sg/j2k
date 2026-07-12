// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible batch preparation, flattening, shared Tier-1, and regrouping.

use super::super::allocation::add_capacity;
use super::super::orchestrator::Prepared97PacketPlan;
use super::super::{
    encode_prepared_resolution_packets_for_session, packet_encode,
    prepare_precomputed_htj2k97_image_for_batch, EncodeOptions, J2kEncodeStageAccelerator,
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
    PrecomputedHtj2k97Image, PreparedResolutionPacket, ResolutionPacket, Vec,
};
use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};

use super::finalize::{batch_plan_metadata_bytes, packetize_and_finalize_batch};

pub(super) fn prepare_batch_plans(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<Prepared97PacketPlan>> {
    let requested = checked_element_bytes::<Prepared97PacketPlan>(
        images.len(),
        "precomputed 9/7 batch plan owners",
    )?;
    session.checked_phase(requested, "precomputed 9/7 batch plan owners")?;
    let mut plans: Vec<Prepared97PacketPlan> = Vec::new();
    plans
        .try_reserve_exact(images.len())
        .map_err(|_| host_allocation_failed("precomputed 9/7 batch plan owners", requested))?;
    let outer_bytes = checked_element_bytes::<Prepared97PacketPlan>(
        plans.capacity(),
        "precomputed 9/7 batch plan owners",
    )?;
    for image in images {
        let prior_bytes = checked_add_bytes(
            outer_bytes,
            plans.iter().try_fold(0usize, |bytes, plan| {
                checked_add_bytes(bytes, plan.retained_bytes()?, "precomputed 9/7 batch plans")
            })?,
            "precomputed 9/7 batch plans",
        )?;
        plans.push(prepare_precomputed_htj2k97_image_for_batch(
            image,
            options,
            session,
            prior_bytes,
        )?);
    }
    Ok(plans)
}

pub(super) fn encode_prepared_batch(
    mut plans: Vec<Prepared97PacketPlan>,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<Vec<u8>>> {
    let packet_count = plans.iter().try_fold(0usize, |count, plan| {
        count
            .checked_add(plan.packet_count())
            .ok_or(crate::EncodeError::ArithmeticOverflow {
                what: "precomputed 9/7 batch packet count",
            })
    })?;
    let plan_outer_capacity = plans.capacity();
    let prepared_bytes = plans.iter().try_fold(0usize, |bytes, plan| {
        checked_add_bytes(
            bytes,
            plan.retained_bytes()?,
            "precomputed 9/7 prepared plans",
        )
    })?;
    let full_plan_bytes = add_capacity::<Prepared97PacketPlan>(
        prepared_bytes,
        plan_outer_capacity,
        "precomputed 9/7 batch plan owners",
    )?;
    let mut packet_counts = try_usize_vec(plans.len(), session, full_plan_bytes)?;
    for plan in &plans {
        packet_counts.push(plan.packet_count());
    }
    let packet_count_bytes =
        checked_element_bytes::<usize>(packet_counts.capacity(), "precomputed 9/7 packet counts")?;
    let requested_packets = checked_element_bytes::<PreparedResolutionPacket>(
        packet_count,
        "precomputed 9/7 flattened packet owners",
    )?;
    session.checked_phase(
        checked_add_bytes(
            full_plan_bytes,
            checked_add_bytes(
                packet_count_bytes,
                requested_packets,
                "precomputed 9/7 flattened packet overlap",
            )?,
            "precomputed 9/7 flattened packet overlap",
        )?,
        "precomputed 9/7 flattened packet overlap",
    )?;
    let mut all_packets = Vec::new();
    all_packets.try_reserve_exact(packet_count).map_err(|_| {
        host_allocation_failed("precomputed 9/7 flattened packet owners", requested_packets)
    })?;
    let actual_packet_owner_bytes = checked_element_bytes::<PreparedResolutionPacket>(
        all_packets.capacity(),
        "precomputed 9/7 flattened packet owners",
    )?;
    session.checked_phase(
        checked_add_bytes(
            full_plan_bytes,
            checked_add_bytes(
                packet_count_bytes,
                actual_packet_owner_bytes,
                "precomputed 9/7 flattened packet overlap",
            )?,
            "precomputed 9/7 flattened packet overlap",
        )?,
        "precomputed 9/7 flattened packet overlap",
    )?;
    for plan in &mut plans {
        let mut packets = core::mem::take(&mut plan.prepared_packets);
        all_packets.append(&mut packets);
    }
    let descriptor_and_plan_bytes = batch_plan_metadata_bytes(&plans, plan_outer_capacity)?;
    let encoded = encode_prepared_resolution_packets_for_session(
        all_packets,
        session,
        checked_add_bytes(
            descriptor_and_plan_bytes,
            packet_count_bytes,
            "precomputed 9/7 Tier-1 batch metadata",
        )?,
        accelerator,
    )?;
    let groups = split_encoded_packets(
        encoded,
        &packet_counts,
        session,
        checked_add_bytes(
            descriptor_and_plan_bytes,
            packet_count_bytes,
            "precomputed 9/7 packet regrouping metadata",
        )?,
    )?;
    drop(packet_counts);
    packetize_and_finalize_batch(plans, groups, session, accelerator)
}

fn split_encoded_packets(
    encoded: Vec<ResolutionPacket>,
    packet_counts: &[usize],
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
) -> NativeEncodePipelineResult<Vec<Vec<ResolutionPacket>>> {
    let encoded_bytes = packet_encode::owned_packet_retained_bytes_for_public_descriptors(
        &encoded,
        encoded.capacity(),
        0,
        0,
    )?;
    let outer_requested = checked_element_bytes::<Vec<ResolutionPacket>>(
        packet_counts.len(),
        "precomputed 9/7 encoded group owners",
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_add_bytes(encoded_bytes, outer_requested, "encoded packet regrouping")?,
            "encoded packet regrouping",
        )?,
        "encoded packet regrouping",
    )?;
    let mut groups = Vec::new();
    groups.try_reserve_exact(packet_counts.len()).map_err(|_| {
        host_allocation_failed("precomputed 9/7 encoded group owners", outer_requested)
    })?;
    let group_outer_bytes = checked_element_bytes::<Vec<ResolutionPacket>>(
        groups.capacity(),
        "precomputed 9/7 encoded group owners",
    )?;
    let mut encoded = encoded.into_iter();
    let mut destination_outer_bytes = group_outer_bytes;
    for &packet_count in packet_counts {
        let requested = checked_element_bytes::<ResolutionPacket>(
            packet_count,
            "precomputed 9/7 encoded packet group",
        )?;
        session.checked_phase(
            checked_add_bytes(
                retained_base_bytes,
                checked_add_bytes(
                    encoded_bytes,
                    checked_add_bytes(destination_outer_bytes, requested, "encoded regrouping")?,
                    "encoded regrouping",
                )?,
                "encoded regrouping",
            )?,
            "encoded regrouping",
        )?;
        let mut group = Vec::new();
        group.try_reserve_exact(packet_count).map_err(|_| {
            host_allocation_failed("precomputed 9/7 encoded packet group", requested)
        })?;
        let next_destination_outer_bytes = checked_add_bytes(
            destination_outer_bytes,
            checked_element_bytes::<ResolutionPacket>(
                group.capacity(),
                "precomputed 9/7 encoded packet group",
            )?,
            "encoded regrouping",
        )?;
        session.checked_phase(
            checked_add_bytes(
                retained_base_bytes,
                checked_add_bytes(
                    encoded_bytes,
                    next_destination_outer_bytes,
                    "encoded regrouping",
                )?,
                "encoded regrouping",
            )?,
            "encoded regrouping",
        )?;
        destination_outer_bytes = next_destination_outer_bytes;
        for _ in 0..packet_count {
            group.push(encoded.next().ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant("encoded packet count mismatch")
            })?);
        }
        groups.push(group);
    }
    if encoded.next().is_some() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "encoded packet count mismatch",
        ));
    }
    Ok(groups)
}

fn try_usize_vec(
    count: usize,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
) -> NativeEncodePipelineResult<Vec<usize>> {
    let requested = checked_element_bytes::<usize>(count, "precomputed 9/7 packet counts")?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            requested,
            "precomputed 9/7 packet counts",
        )?,
        "precomputed 9/7 packet counts",
    )?;
    let mut counts = Vec::new();
    counts
        .try_reserve_exact(count)
        .map_err(|_| host_allocation_failed("precomputed 9/7 packet counts", requested))?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_element_bytes::<usize>(counts.capacity(), "precomputed 9/7 packet counts")?,
            "precomputed 9/7 packet counts",
        )?,
        "precomputed 9/7 packet counts",
    )?;
    Ok(counts)
}
