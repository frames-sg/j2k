// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    driver::CuDevicePtr, error::CudaError, execution::elapsed_event_us_ceil,
    CudaJ2kIdwtBatchStageProfile,
};

use super::super::CudaJ2kIdwtBatchKernelMode;

impl crate::J2kCudaEngine<'_> {
    pub(in crate::j2k_decode) fn profile_j2k_idwt_batch_mode_ptr(
        &self,
        mode: CudaJ2kIdwtBatchKernelMode,
        jobs_ptr: CuDevicePtr,
        max_width: usize,
        max_height: usize,
        job_count: usize,
        final_stage: bool,
    ) -> Result<CudaJ2kIdwtBatchStageProfile, CudaError> {
        let start = self.create_event()?;
        start.record_default_stream()?;
        self.launch_j2k_idwt_batch_interleave_horizontal_ptr(
            mode, jobs_ptr, max_width, max_height, job_count, false,
        )?;
        let horizontal_end = self.create_event()?;
        horizontal_end.record_default_stream()?;
        self.launch_j2k_idwt_batch_vertical_ptr(
            mode, jobs_ptr, max_width, max_height, job_count, false,
        )?;
        let end = self.create_event()?;
        end.record_default_stream()?;
        end.synchronize()?;
        Ok(CudaJ2kIdwtBatchStageProfile {
            final_stage,
            elapsed_us: elapsed_event_us_ceil(&start, &end)?,
            interleave_horizontal_us: elapsed_event_us_ceil(&start, &horizontal_end)?,
            vertical_us: elapsed_event_us_ceil(&horizontal_end, &end)?,
        })
    }
}
