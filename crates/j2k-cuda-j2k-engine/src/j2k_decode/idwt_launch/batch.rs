// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{driver::CuDevicePtr, error::CudaError};

use super::super::CudaJ2kIdwtBatchKernelMode;

impl crate::J2kCudaEngine<'_> {
    pub(in crate::j2k_decode) fn launch_j2k_idwt_batch_interleave_horizontal_ptr(
        &self,
        mode: CudaJ2kIdwtBatchKernelMode,
        jobs_ptr: CuDevicePtr,
        max_width: usize,
        max_height: usize,
        job_count: usize,
        synchronize_each_launch: bool,
    ) -> Result<(), CudaError> {
        match mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                .launch_j2k_idwt_interleave_horizontal_53_multi_ptr(
                    jobs_ptr,
                    max_width,
                    max_height,
                    job_count,
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
                    jobs_ptr,
                    max_width,
                    max_height,
                    job_count,
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self
                .launch_j2k_idwt_interleave_horizontal_multi_ptr(
                    jobs_ptr,
                    max_height,
                    job_count,
                    synchronize_each_launch,
                ),
        }
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_batch_vertical_ptr(
        &self,
        mode: CudaJ2kIdwtBatchKernelMode,
        jobs_ptr: CuDevicePtr,
        max_width: usize,
        max_height: usize,
        job_count: usize,
        synchronize_each_launch: bool,
    ) -> Result<(), CudaError> {
        match mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                .launch_j2k_idwt_vertical_53_multi_ptr(
                    jobs_ptr,
                    max_width,
                    max_height,
                    job_count,
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_vertical_97_multi_ptr(
                    jobs_ptr,
                    max_width,
                    max_height,
                    job_count,
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self.launch_j2k_idwt_vertical_multi_ptr(
                jobs_ptr,
                max_width,
                job_count,
                synchronize_each_launch,
            ),
        }
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_batch_mode_ptr(
        &self,
        mode: CudaJ2kIdwtBatchKernelMode,
        jobs_ptr: CuDevicePtr,
        max_width: usize,
        max_height: usize,
        job_count: usize,
        synchronize_each_launch: bool,
    ) -> Result<usize, CudaError> {
        self.launch_j2k_idwt_batch_interleave_horizontal_ptr(
            mode,
            jobs_ptr,
            max_width,
            max_height,
            job_count,
            synchronize_each_launch,
        )?;
        self.launch_j2k_idwt_batch_vertical_ptr(
            mode,
            jobs_ptr,
            max_width,
            max_height,
            job_count,
            synchronize_each_launch,
        )?;
        Ok(2)
    }
}
