// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw exact-i64 component to prepared-packet transition.

use alloc::vec::Vec;

use super::super::super::allocation::checked_add_bytes;
use super::super::super::tier1_allocation::{
    prepared_packet_tree_ownership, prepared_packets_ownership,
};
use super::super::super::typed_i64::{prepare_i64_component_packets, I64ComponentPrepareRequest};
use super::super::super::{
    I64SubbandEncodeSettings, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeSession, PreparedResolutionPacket,
};
use super::super::plan::SingleTilePlan;
use super::input::component_planes_retained_bytes;

mod accounting;
use accounting::{check_packet_tree, try_packet_tree_owners};

pub(super) fn prepare_i64_component_packet_tree(
    components: &mut Vec<Vec<i64>>,
    plan: &SingleTilePlan,
    retained_plan_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<Vec<PreparedResolutionPacket>>> {
    let context = PacketTreeContext {
        plan,
        retained_plan_bytes,
        session,
    };
    let mut component_packets = try_packet_tree_owners(components, context)?;
    for component_index in 0..components.len() {
        let packets =
            prepare_i64_component(components, &component_packets, component_index, context)?;
        component_packets.push(packets);
    }
    check_packet_tree(components, &component_packets, context)?;
    Ok(component_packets)
}

#[derive(Clone, Copy)]
struct PacketTreeContext<'a, 'input> {
    plan: &'a SingleTilePlan,
    retained_plan_bytes: usize,
    session: &'a NativeEncodeSession<'input>,
}

fn prepare_i64_component(
    components: &mut Vec<Vec<i64>>,
    component_packets: &Vec<Vec<PreparedResolutionPacket>>,
    component_index: usize,
    context: PacketTreeContext<'_, '_>,
) -> NativeEncodePipelineResult<Vec<PreparedResolutionPacket>> {
    let samples = core::mem::take(components.get_mut(component_index).ok_or_else(|| {
        NativeEncodePipelineError::internal_invariant("exact i64 component index is out of range")
    })?);
    let component_base = checked_add_bytes(
        context.retained_plan_bytes,
        checked_add_bytes(
            component_planes_retained_bytes(components)?,
            prepared_packet_tree_ownership(component_packets, component_packets.capacity())?
                .total()?,
            "exact i64 retained preparation owners",
        )?,
        "exact i64 retained preparation owners",
    )?;
    let roi_plan = context.plan.roi_plans.get(component_index).ok_or_else(|| {
        NativeEncodePipelineError::internal_invariant(
            "ROI plan count does not match component count",
        )
    })?;
    let component_steps = context
        .plan
        .component_step_sizes
        .get(component_index)
        .map_or(context.plan.step_sizes.as_slice(), Vec::as_slice);
    let packets = prepare_i64_component_packets(
        samples,
        I64ComponentPrepareRequest {
            component: u16::try_from(component_index).map_err(|_| {
                NativeEncodePipelineError::internal_invariant("component index exceeds u16")
            })?,
            width: context.plan.params.width,
            height: context.plan.params.height,
            num_levels: context.plan.num_levels,
            step_sizes: component_steps,
            subband_settings: I64SubbandEncodeSettings {
                guard_bits: context.plan.guard_bits,
                cb_width: context.plan.cb_width,
                cb_height: context.plan.cb_height,
                roi_shift: context
                    .plan
                    .roi_component_shifts
                    .get(component_index)
                    .copied()
                    .unwrap_or(0),
                roi_regions: &roi_plan.regions,
                roi_scale: 1,
                block_coding_mode: context.plan.params.block_coding_mode,
                ht_target_coding_passes: context.plan.ht_target_coding_passes,
            },
            retained_base_bytes: component_base,
            session: context.session,
        },
    )?;
    let incoming_bytes = prepared_packets_ownership(&packets, packets.capacity())?.total()?;
    context.session.checked_phase(
        checked_add_bytes(
            component_base,
            incoming_bytes,
            "exact i64 prepared component",
        )?,
        "exact i64 prepared component",
    )?;
    Ok(packets)
}
