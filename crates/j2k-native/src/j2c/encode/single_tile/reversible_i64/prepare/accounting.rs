// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible owner allocation and retained-capacity checks for exact-i64 packets.

use alloc::vec::Vec;

use super::{component_planes_retained_bytes, PacketTreeContext};
use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::tier1_allocation::prepared_packet_tree_ownership;
use crate::j2c::encode::{NativeEncodePipelineResult, PreparedResolutionPacket};

pub(super) fn try_packet_tree_owners(
    components: &Vec<Vec<i64>>,
    context: PacketTreeContext<'_, '_>,
) -> NativeEncodePipelineResult<Vec<Vec<PreparedResolutionPacket>>> {
    let component_count = components.len();
    let requested_outer = checked_element_bytes::<Vec<PreparedResolutionPacket>>(
        component_count,
        "exact i64 prepared component owners",
    )?;
    let component_bytes = component_planes_retained_bytes(components)?;
    context.session.checked_phase(
        checked_add_bytes(
            context.retained_plan_bytes,
            checked_add_bytes(
                component_bytes,
                requested_outer,
                "exact i64 prepared component owners",
            )?,
            "exact i64 preparation",
        )?,
        "exact i64 preparation",
    )?;
    let mut component_packets = Vec::new();
    component_packets
        .try_reserve_exact(component_count)
        .map_err(|_| {
            host_allocation_failed("exact i64 prepared component owners", requested_outer)
        })?;
    context.session.checked_phase(
        checked_add_bytes(
            context.retained_plan_bytes,
            checked_add_bytes(
                component_planes_retained_bytes(components)?,
                prepared_packet_tree_ownership(&component_packets, component_packets.capacity())?
                    .total()?,
                "exact i64 prepared component owners",
            )?,
            "exact i64 preparation",
        )?,
        "exact i64 preparation",
    )?;
    Ok(component_packets)
}

pub(super) fn check_packet_tree(
    components: &Vec<Vec<i64>>,
    component_packets: &Vec<Vec<PreparedResolutionPacket>>,
    context: PacketTreeContext<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    context.session.checked_phase(
        checked_add_bytes(
            context.retained_plan_bytes,
            checked_add_bytes(
                component_planes_retained_bytes(components)?,
                prepared_packet_tree_ownership(component_packets, component_packets.capacity())?
                    .total()?,
                "exact i64 prepared packet tree",
            )?,
            "exact i64 prepared packet tree",
        )?,
        "exact i64 prepared packet tree",
    )?;
    Ok(())
}
