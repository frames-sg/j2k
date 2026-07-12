// SPDX-License-Identifier: MIT OR Apache-2.0

use std::os::raw::c_uint;

use crate::{
    context::CudaContext,
    error::CudaError,
    execution::{cuda_kernel_param, CudaLaunchMode},
    kernels::{
        htj2k_codeblock_launch_geometry, htj2k_codeblock_sample_launch_geometry, CudaKernel,
        CudaLaunchGeometry,
    },
    memory::CudaDeviceBuffer,
};

use super::types::{Htj2kDecodeCodeblocksLaunch, Htj2kDecodeCodeblocksMultiLaunch};

impl CudaContext {
    #[expect(
        clippy::similar_names,
        reason = "per-job block and byte offsets intentionally share domain terminology"
    )]
    pub(super) fn launch_htj2k_decode_codeblocks(
        &self,
        launch: Htj2kDecodeCodeblocksLaunch<'_>,
    ) -> Result<(), CudaError> {
        let Htj2kDecodeCodeblocksLaunch {
            payload,
            coefficients,
            jobs,
            tables,
            statuses,
            job_count,
            mode,
        } = launch;
        let mut payload_ptr = payload.device_ptr();
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = tables.vlc_table0.device_ptr();
        let mut vlc_table1_ptr = tables.vlc_table1.device_ptr();
        let mut uvlc_table0_ptr = tables.uvlc_table0.device_ptr();
        let mut uvlc_table1_ptr = tables.uvlc_table1.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count = c_uint::try_from(job_count)
            .map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = cuda_kernel_params!(
            payload_ptr,
            coefficients_ptr,
            jobs_ptr,
            vlc_table0_ptr,
            vlc_table1_ptr,
            uvlc_table0_ptr,
            uvlc_table1_ptr,
            statuses_ptr,
            job_count
        );
        let geometry = htj2k_codeblock_launch_geometry(job_count as usize).ok_or(
            CudaError::LengthTooLarge {
                len: job_count as usize,
            },
        )?;

        self.launch_htj2k_decode_kernel(
            CudaKernel::Htj2kDecodeCodeblocks,
            geometry,
            &mut params,
            mode,
        )
    }

    #[expect(
        clippy::similar_names,
        reason = "per-job block and byte offsets intentionally share domain terminology"
    )]
    pub(super) fn launch_htj2k_decode_codeblocks_multi(
        &self,
        launch: Htj2kDecodeCodeblocksMultiLaunch<'_>,
    ) -> Result<(), CudaError> {
        let Htj2kDecodeCodeblocksMultiLaunch {
            kernel,
            payload,
            jobs,
            tables,
            statuses,
            job_count,
            mode,
        } = launch;
        let mut payload_ptr = payload.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = tables.vlc_table0.device_ptr();
        let mut vlc_table1_ptr = tables.vlc_table1.device_ptr();
        let mut uvlc_table0_ptr = tables.uvlc_table0.device_ptr();
        let mut uvlc_table1_ptr = tables.uvlc_table1.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count = c_uint::try_from(job_count)
            .map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = cuda_kernel_params!(
            payload_ptr,
            jobs_ptr,
            vlc_table0_ptr,
            vlc_table1_ptr,
            uvlc_table0_ptr,
            uvlc_table1_ptr,
            statuses_ptr,
            job_count
        );
        let geometry = htj2k_codeblock_launch_geometry(job_count as usize).ok_or(
            CudaError::LengthTooLarge {
                len: job_count as usize,
            },
        )?;

        self.launch_htj2k_decode_kernel(kernel, geometry, &mut params, mode)
    }

    fn launch_htj2k_decode_kernel<const N: usize>(
        &self,
        kernel: CudaKernel,
        geometry: CudaLaunchGeometry,
        params: &mut [*mut std::ffi::c_void; N],
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let function = self.htj2k_decode_kernel_function(kernel)?;
        match mode {
            CudaLaunchMode::Sync => self.launch_kernel(function, geometry, params),
            CudaLaunchMode::Async => self.launch_kernel_async(function, geometry, params),
        }
    }

    fn htj2k_decode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner.cuda_oxide_htj2k_decode_kernel_function(kernel)
    }

    pub(super) fn launch_j2k_dequantize_htj2k_codeblocks(
        &self,
        coefficients: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(coefficients_ptr, jobs_ptr);
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;

        self.launch_j2k_dequantize_kernel(
            CudaKernel::J2kDequantizeHtj2kCodeblocks,
            geometry,
            &mut params,
            mode,
        )
    }

    pub(super) fn launch_j2k_dequantize_htj2k_codeblocks_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;

        self.launch_j2k_dequantize_kernel(
            CudaKernel::J2kDequantizeHtj2kCodeblocksMulti,
            geometry,
            &mut params,
            mode,
        )
    }

    pub(crate) fn launch_j2k_dequantize_htj2k_cleanup_jobs_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;

        self.launch_j2k_dequantize_kernel(
            CudaKernel::J2kDequantizeHtj2kCleanupJobsMulti,
            geometry,
            &mut params,
            mode,
        )
    }

    fn launch_j2k_dequantize_kernel<const N: usize>(
        &self,
        kernel: CudaKernel,
        geometry: CudaLaunchGeometry,
        params: &mut [*mut std::ffi::c_void; N],
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let function = self.j2k_dequantize_kernel_function(kernel)?;
        match mode {
            CudaLaunchMode::Sync => self.launch_kernel(function, geometry, params),
            CudaLaunchMode::Async => self.launch_kernel_async(function, geometry, params),
        }
    }

    fn j2k_dequantize_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner.cuda_oxide_j2k_dequantize_kernel_function(kernel)
    }
}
