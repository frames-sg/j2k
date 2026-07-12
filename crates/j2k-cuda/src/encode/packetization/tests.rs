// SPDX-License-Identifier: MIT OR Apache-2.0

use super::flatten::flatten_cuda_htj2k_packetization_job_classified;
use super::state::checked_cuda_packetization_state_count;
use super::types::{packetization_plan_allocation_error, CudaHtj2kPacketizationPlanError};
use j2k::{
    J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor, J2kPacketizationProgressionOrder,
    J2kPacketizationResolution,
};

mod ht_segment;

#[test]
fn sparse_descriptor_state_index_is_rejected_before_state_allocation() {
    let descriptor = J2kPacketizationPacketDescriptor {
        packet_index: 0,
        state_index: u32::MAX,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    };
    let resolution = J2kPacketizationResolution {
        subbands: Vec::new(),
    };
    let error = flatten_cuda_htj2k_packetization_job_classified(J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 1,
        num_components: 1,
        code_block_count: 0,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[descriptor],
        resolutions: &[resolution],
    })
    .expect_err("sparse state indexes must fail before host allocation");

    assert_eq!(
        error,
        CudaHtj2kPacketizationPlanError::Invalid(
            "CUDA HTJ2K packetization descriptor state index exceeds descriptor count"
        )
    );
}

#[test]
fn descriptor_state_count_addition_is_checked() {
    assert_eq!(
        checked_cuda_packetization_state_count(Some(usize::MAX)).unwrap_err(),
        CudaHtj2kPacketizationPlanError::Invalid(
            "CUDA HTJ2K packetization descriptor state count overflow"
        )
    );
}

#[test]
fn packetization_plan_allocation_failure_keeps_its_typed_category() {
    let source = crate::Error::HostAllocationFailed {
        bytes: 4096,
        what: "test packetization plan",
    };

    assert_eq!(
        packetization_plan_allocation_error(source),
        CudaHtj2kPacketizationPlanError::HostAllocation {
            bytes: 4096,
            what: "test packetization plan"
        }
    );
}
