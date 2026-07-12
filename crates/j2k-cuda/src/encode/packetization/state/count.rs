// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{CudaHtj2kPacketizationPlanError, PacketizationPlanResult};

pub(in crate::encode::packetization) fn cuda_packetization_state_count(
    descriptors: &[j2k::J2kPacketizationPacketDescriptor],
) -> PacketizationPlanResult<usize> {
    let descriptor_count = descriptors.len();
    let mut max_state = None;
    for descriptor in descriptors {
        let state_index = usize::try_from(descriptor.state_index).map_err(|_| {
            CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packet descriptor state index exceeds usize",
            )
        })?;
        if state_index >= descriptor_count {
            return Err(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization descriptor state index exceeds descriptor count",
            ));
        }
        max_state = Some(max_state.map_or(state_index, |current: usize| current.max(state_index)));
    }
    checked_cuda_packetization_state_count(max_state)
}

pub(in crate::encode::packetization) fn checked_cuda_packetization_state_count(
    max_state: Option<usize>,
) -> PacketizationPlanResult<usize> {
    max_state.map_or(Ok(0), |max_state| {
        max_state
            .checked_add(1)
            .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization descriptor state count overflow",
            ))
    })
}
