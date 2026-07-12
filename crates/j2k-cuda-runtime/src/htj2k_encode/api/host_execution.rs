// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{context::CudaContext, error::CudaError, memory::CudaDeviceBuffer};

use super::super::{
    planning::htj2k_encode_kernel_jobs_with_live_host_bytes,
    types::{
        CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeKernelJob, CudaHtj2kEncodeResources,
        CudaHtj2kEncodedCodeBlocks,
    },
};

impl CudaContext {
    pub(super) fn encode_htj2k_codeblocks_device_with_resources(
        &self,
        coefficient_buffer: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        caller_live_host_bytes: usize,
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let kernel_jobs = htj2k_encode_kernel_jobs_with_live_host_bytes(
            jobs,
            coefficient_count,
            caller_live_host_bytes,
        )?;
        self.encode_htj2k_kernel_jobs_device_with_resources(
            coefficient_buffer,
            &kernel_jobs,
            kernel_jobs.capacity(),
            caller_live_host_bytes,
            resources,
        )
    }

    fn encode_htj2k_kernel_jobs_device_with_resources(
        &self,
        coefficient_buffer: &CudaDeviceBuffer,
        kernel_jobs: &[CudaHtj2kEncodeKernelJob],
        kernel_jobs_capacity: usize,
        caller_live_host_bytes: usize,
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let pool = self.buffer_pool();
        self.encode_htj2k_kernel_jobs_device_with_resources_and_pool(
            coefficient_buffer,
            kernel_jobs,
            kernel_jobs_capacity,
            caller_live_host_bytes,
            resources,
            &pool,
        )
    }
}
