// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::allocation::HostPhaseBudget;
use crate::encode::stage_error::{arithmetic_overflow, internal_invariant, CudaStageResult};

use super::host_budget::account_encoded_resolution_owners;
use super::htj2k_allocation_error;
use super::types::CudaEncodedHtj2kResolution;

pub(super) fn cuda_order_component_resolution_packets(
    component_packets: Vec<Vec<CudaEncodedHtj2kResolution>>,
    num_components: u16,
) -> CudaStageResult<Vec<CudaEncodedHtj2kResolution>> {
    if component_packets.len() != usize::from(num_components) {
        return Err(internal_invariant(
            "CUDA HTJ2K tile component packet count mismatch",
        ));
    }
    let resolution_count = component_packets.first().map_or(0usize, Vec::len);
    let mut host_budget = HostPhaseBudget::new("j2k CUDA HTJ2K resolution ordering");
    host_budget
        .account_vec(&component_packets)
        .map_err(htj2k_allocation_error)?;
    for component in &component_packets {
        account_encoded_resolution_owners(&mut host_budget, component, component.capacity())?;
    }
    let mut component_iters = host_budget
        .try_vec_with_capacity(component_packets.len())
        .map_err(htj2k_allocation_error)?;
    component_iters.extend(component_packets.into_iter().map(Vec::into_iter));
    let packet_count = resolution_count
        .checked_mul(component_iters.len())
        .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K tile resolution packet count"))?;
    let mut resolution_packets = host_budget
        .try_vec_with_capacity(packet_count)
        .map_err(htj2k_allocation_error)?;

    for _resolution in 0..resolution_count {
        for component in &mut component_iters {
            resolution_packets.push(component.next().ok_or_else(|| {
                internal_invariant("CUDA HTJ2K tile component resolution count mismatch")
            })?);
        }
    }
    if component_iters
        .iter_mut()
        .any(|component| component.next().is_some())
    {
        return Err(internal_invariant(
            "CUDA HTJ2K tile component resolution count mismatch",
        ));
    }

    Ok(resolution_packets)
}
