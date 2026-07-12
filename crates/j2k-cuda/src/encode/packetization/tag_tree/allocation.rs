// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::allocation::HostPhaseBudget;

use super::super::types::{
    packetization_plan_allocation_error, CudaHtj2kPacketizationPlanError, PacketizationPlanResult,
    CUDA_PACKETIZATION_TAG_TREE_ALLOCATION,
};

pub(super) fn try_tag_tree_vec_with_capacity<T>(
    host_budget: &mut HostPhaseBudget,
    capacity: usize,
) -> PacketizationPlanResult<Vec<T>> {
    host_budget
        .try_vec_with_capacity(capacity)
        .map_err(packetization_plan_allocation_error)
}

pub(super) fn try_tag_tree_vec_filled<T: Clone>(
    host_budget: &mut HostPhaseBudget,
    len: usize,
    value: T,
) -> PacketizationPlanResult<Vec<T>> {
    host_budget
        .try_vec_filled(len, value)
        .map_err(packetization_plan_allocation_error)
}

pub(super) fn checked_tag_tree_retained_bytes(
    u32_level_capacities: [usize; 2],
    usize_level_capacity: usize,
    node_capacities: [usize; 3],
    cap: usize,
) -> PacketizationPlanResult<usize> {
    let mut total = 0usize;
    for bytes in u32_level_capacities
        .into_iter()
        .chain(node_capacities)
        .map(|capacity| capacity.checked_mul(core::mem::size_of::<u32>()))
        .chain(core::iter::once(
            usize_level_capacity.checked_mul(core::mem::size_of::<usize>()),
        ))
    {
        total = total
            .checked_add(
                bytes.ok_or(CudaHtj2kPacketizationPlanError::ArithmeticOverflow(
                    "CUDA HTJ2K packetization tag-tree byte count overflow",
                ))?,
            )
            .ok_or(CudaHtj2kPacketizationPlanError::ArithmeticOverflow(
                "CUDA HTJ2K packetization tag-tree byte count overflow",
            ))?;
        if total > cap {
            return Err(CudaHtj2kPacketizationPlanError::MemoryCapExceeded {
                what: CUDA_PACKETIZATION_TAG_TREE_ALLOCATION,
                requested: total,
                cap,
            });
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests;
