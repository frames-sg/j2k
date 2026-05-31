// SPDX-License-Identifier: Apache-2.0

use signinum_j2k_cuda::{CudaEncodeStageAccelerator, CudaEncodeStageTimings};
use signinum_j2k_native::{
    J2kEncodeStageAccelerator, J2kPacketizationEncodeJob, J2kPacketizationProgressionOrder,
};

#[test]
fn cuda_encode_stage_timings_are_publicly_readable_and_resettable() {
    let mut accelerator = CudaEncodeStageAccelerator::with_profile_collection(true);

    assert_eq!(
        accelerator.collected_stage_timings(),
        CudaEncodeStageTimings::default()
    );

    accelerator.reset_collected_stage_timings();
    assert_eq!(
        accelerator.collected_stage_timings(),
        CudaEncodeStageTimings::default()
    );
}

#[test]
fn cuda_encode_stage_can_prefer_cpu_packetization() {
    let mut accelerator = CudaEncodeStageAccelerator::default().prefer_cpu_packetization(true);
    let job = J2kPacketizationEncodeJob {
        resolution_count: 0,
        num_layers: 1,
        num_components: 1,
        code_block_count: 0,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[],
        resolutions: &[],
    };

    assert!(accelerator.encode_packetization(job).unwrap().is_none());
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 0);
}
