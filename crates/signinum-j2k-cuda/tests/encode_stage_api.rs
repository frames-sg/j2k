// SPDX-License-Identifier: Apache-2.0

use signinum_j2k_cuda::{CudaEncodeStageAccelerator, CudaEncodeStageTimings};

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
