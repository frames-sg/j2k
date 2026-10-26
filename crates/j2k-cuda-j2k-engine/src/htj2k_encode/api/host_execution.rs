// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    error::CudaError,
    memory::{CudaBufferPool, CudaDeviceBuffer},
};

use super::super::{
    planning::{
        htj2k_encode_kernel_jobs_with_live_host_bytes,
        htj2k_encode_region_kernel_jobs_with_live_host_bytes,
    },
    types::{
        empty_htj2k_encoded_code_blocks, CudaHtj2kEncodeCodeBlockJob,
        CudaHtj2kEncodeCodeBlockRegionJob, CudaHtj2kEncodeKernelJob, CudaHtj2kEncodeResources,
        CudaHtj2kEncodeStageTimings, CudaHtj2kEncodedCodeBlocks,
    },
};

impl crate::J2kCudaEngine<'_> {
    pub(super) fn encode_htj2k_region_chunks(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let mut budget =
            HostPhaseBudget::with_live_bytes("CUDA HTJ2K chunked region encode", live_host_bytes)?;
        let mut encoded = empty_htj2k_encoded_code_blocks();
        encoded.code_blocks = budget.try_vec_with_capacity(jobs.len())?;
        self.prepare_operation()?;
        for chunk in jobs.chunks(super::MAX_REGION_ENCODE_JOBS) {
            let kernel_jobs = htj2k_encode_region_kernel_jobs_with_live_host_bytes(
                chunk,
                coefficient_count,
                budget.live_bytes(),
            )?;
            let part = self.encode_htj2k_kernel_jobs_device_with_resources_and_pool(
                coefficients,
                &kernel_jobs,
                kernel_jobs.capacity(),
                budget.live_bytes(),
                resources,
                pool,
            )?;
            // Completion has synchronized the readback and returned its device scratch
            // to the pool. Only the encoded payloads survive into the next chunk.
            for block in &part.code_blocks {
                budget.account_vec(&block.data)?;
            }
            encoded.execution.kernel_dispatches = encoded
                .execution
                .kernel_dispatches
                .saturating_add(part.execution.kernel_dispatches);
            encoded.execution.copy_kernel_dispatches = encoded
                .execution
                .copy_kernel_dispatches
                .saturating_add(part.execution.copy_kernel_dispatches);
            let previous = encoded.stage_timings;
            let next = part.stage_timings;
            encoded.stage_timings = CudaHtj2kEncodeStageTimings::from_parts(
                previous.ht_kernel_us.saturating_add(next.ht_kernel_us),
                previous
                    .ht_status_readback_us
                    .saturating_add(next.ht_status_readback_us),
                previous.ht_compact_us.saturating_add(next.ht_compact_us),
                previous
                    .ht_output_readback_us
                    .saturating_add(next.ht_output_readback_us),
            );
            encoded.code_blocks.extend(part.code_blocks);
        }
        Ok(encoded)
    }

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
        let pool = self.context.buffer_pool();
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
