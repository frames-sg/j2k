// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::CudaContext,
    driver::CuFunction,
    error::CudaError,
    execution::cuda_kernel_param,
    kernels::{
        htj2k_codeblock_sample_launch_geometry, htj2k_encode_codeblock_launch_geometry, CudaKernel,
    },
    memory::CudaDeviceBuffer,
};

use super::types::{CudaHtj2kEncodeCodeblocksLaunch, CudaHtj2kEncodeMultiInputLaunch};

impl CudaContext {
    pub(super) fn launch_htj2k_encode_codeblocks(
        &self,
        request: &CudaHtj2kEncodeCodeblocksLaunch<'_>,
    ) -> Result<(), CudaError> {
        let function = self.htj2k_encode_kernel_function(CudaKernel::Htj2kEncodeCodeblocks)?;
        let mut coefficients_ptr = request.coefficients.device_ptr();
        let mut output_ptr = request.output.device_ptr();
        let mut jobs_ptr = request.jobs.device_ptr();
        let mut vlc_table0_ptr = request.tables.vlc_table0.device_ptr();
        let mut vlc_table1_ptr = request.tables.vlc_table1.device_ptr();
        let mut uvlc_table_ptr = request.tables.uvlc_table.device_ptr();
        let mut statuses_ptr = request.statuses.device_ptr();
        let mut job_count_u64 =
            u64::try_from(request.job_count).map_err(|_| CudaError::LengthTooLarge {
                len: request.job_count,
            })?;
        let mut params = cuda_kernel_params!(
            coefficients_ptr,
            output_ptr,
            jobs_ptr,
            vlc_table0_ptr,
            vlc_table1_ptr,
            uvlc_table_ptr,
            statuses_ptr,
            job_count_u64
        );
        let geometry = htj2k_encode_codeblock_launch_geometry(request.job_count).ok_or(
            CudaError::LengthTooLarge {
                len: request.job_count,
            },
        )?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    pub(super) fn launch_htj2k_encode_codeblocks_multi_input(
        &self,
        request: &CudaHtj2kEncodeMultiInputLaunch<'_>,
    ) -> Result<(), CudaError> {
        self.launch_htj2k_encode_multi_input_kernel(
            CudaKernel::Htj2kEncodeCodeblocksMultiInput,
            request,
        )
    }

    pub(super) fn launch_htj2k_encode_codeblocks_multi_input_cleanup(
        &self,
        request: &CudaHtj2kEncodeMultiInputLaunch<'_>,
    ) -> Result<(), CudaError> {
        self.launch_htj2k_encode_multi_input_kernel(
            CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup,
            request,
        )
    }

    pub(super) fn launch_htj2k_encode_codeblocks_multi_input_cleanup_64(
        &self,
        request: &CudaHtj2kEncodeMultiInputLaunch<'_>,
    ) -> Result<(), CudaError> {
        self.launch_htj2k_encode_multi_input_kernel(
            CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup64,
            request,
        )
    }

    fn launch_htj2k_encode_multi_input_kernel(
        &self,
        kernel: CudaKernel,
        request: &CudaHtj2kEncodeMultiInputLaunch<'_>,
    ) -> Result<(), CudaError> {
        let function = self.htj2k_encode_kernel_function(kernel)?;
        let mut output_ptr = request.output.device_ptr();
        let mut jobs_ptr = request.jobs.device_ptr();
        let mut vlc_table0_ptr = request.tables.vlc_table0.device_ptr();
        let mut vlc_table1_ptr = request.tables.vlc_table1.device_ptr();
        let mut uvlc_table_ptr = request.tables.uvlc_table.device_ptr();
        let mut statuses_ptr = request.statuses.device_ptr();
        let mut job_count_u64 =
            u64::try_from(request.job_count).map_err(|_| CudaError::LengthTooLarge {
                len: request.job_count,
            })?;
        let mut params = cuda_kernel_params!(
            output_ptr,
            jobs_ptr,
            vlc_table0_ptr,
            vlc_table1_ptr,
            uvlc_table_ptr,
            statuses_ptr,
            job_count_u64
        );
        let geometry = htj2k_encode_codeblock_launch_geometry(request.job_count).ok_or(
            CudaError::LengthTooLarge {
                len: request.job_count,
            },
        )?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    pub(crate) fn launch_htj2k_compact_codeblocks(
        &self,
        scratch: &CudaDeviceBuffer,
        compact: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self.htj2k_encode_kernel_function(CudaKernel::Htj2kCompactCodeblocks)?;
        let mut scratch_ptr = scratch.device_ptr();
        let mut compact_ptr = compact.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut job_count_u64 =
            u64::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = cuda_kernel_params!(scratch_ptr, compact_ptr, jobs_ptr, job_count_u64);
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    fn htj2k_encode_kernel_function(&self, kernel: CudaKernel) -> Result<CuFunction, CudaError> {
        if kernel.is_htj2k_encode_codeblock_stage() {
            self.inner.cuda_oxide_htj2k_encode_kernel_function(kernel)
        } else {
            self.inner.cuda_oxide_j2k_encode_kernel_function(kernel)
        }
    }
}
