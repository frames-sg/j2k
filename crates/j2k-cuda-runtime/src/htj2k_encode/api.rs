// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::{CudaContext, CudaHtj2kCompactEncodedCodeBlocks},
    error::CudaError,
    execution::CudaExecutionStats,
    memory::{CudaBufferPool, CudaDeviceBuffer},
};

use super::{
    context_validation::validate_htj2k_encode_context,
    planning::{
        htj2k_encode_kernel_jobs_with_live_host_bytes,
        htj2k_encode_multi_input_kernel_jobs_with_live_host_bytes,
        htj2k_encode_region_kernel_jobs_with_live_host_bytes,
    },
    types::{
        empty_htj2k_encoded_code_blocks, validate_resident_coefficient_capacity,
        CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob,
        CudaHtj2kEncodeResidentTarget, CudaHtj2kEncodeResources, CudaHtj2kEncodeStageTimings,
        CudaHtj2kEncodeTables, CudaHtj2kEncodedCodeBlocks,
    },
};

mod host_execution;
mod resources;

impl CudaContext {
    /// Encode multiple HTJ2K cleanup-pass code blocks with one CUDA dispatch.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblocks(
        &self,
        coefficients: &[i32],
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let resources = self.upload_htj2k_encode_resources(tables)?;
        self.encode_htj2k_codeblocks_with_resources(coefficients, jobs, &resources)
    }

    /// Encode multiple HTJ2K cleanup-pass code blocks with pre-uploaded lookup tables.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblocks_with_resources(
        &self,
        coefficients: &[i32],
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        self.encode_htj2k_codeblocks_with_resources_and_live_host_bytes(
            coefficients,
            jobs,
            resources,
            0,
        )
    }

    /// Encode host code blocks while accounting caller-live host owners.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblocks_with_resources_and_live_host_bytes(
        &self,
        coefficients: &[i32],
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        resources: &CudaHtj2kEncodeResources,
        live_host_bytes: usize,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        validate_htj2k_encode_context(
            self,
            std::iter::empty::<&CudaDeviceBuffer>(),
            Some(resources),
            None,
        )?;
        if jobs.is_empty() {
            return Ok(CudaHtj2kEncodedCodeBlocks {
                code_blocks: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }

        self.inner.set_current()?;
        let coefficient_buffer = self.upload_i32_pinned(coefficients)?;
        self.encode_htj2k_codeblocks_device_with_resources(
            &coefficient_buffer,
            coefficients.len(),
            jobs,
            live_host_bytes,
            resources,
        )
    }

    /// Encode multiple HTJ2K cleanup-pass code blocks from resident quantized coefficients.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblocks_resident(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        validate_htj2k_encode_context(self, [coefficients], None, None)?;
        let resources = self.upload_htj2k_encode_resources(tables)?;
        let pool = self.buffer_pool();
        self.encode_htj2k_codeblocks_resident_with_resources_and_pool(
            coefficients,
            coefficient_count,
            jobs,
            &resources,
            &pool,
        )
    }

    /// Encode multiple cleanup-pass code blocks from resident coefficients with
    /// lookup table reuse and caller-owned transient buffer reuse.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblocks_resident_with_resources_and_pool(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        validate_htj2k_encode_context(self, [coefficients], Some(resources), Some(pool))?;
        if jobs.is_empty() {
            return Ok(empty_htj2k_encoded_code_blocks());
        }
        validate_resident_coefficient_capacity(coefficients, coefficient_count)?;
        let kernel_jobs =
            htj2k_encode_kernel_jobs_with_live_host_bytes(jobs, coefficient_count, 0)?;
        self.inner.set_current()?;
        self.encode_htj2k_kernel_jobs_device_with_resources_and_pool(
            coefficients,
            &kernel_jobs,
            kernel_jobs.capacity(),
            0,
            resources,
            pool,
        )
    }

    /// Encode multiple cleanup-pass code-block batches from independent
    /// resident coefficient buffers with one CUDA dispatch.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(
        &self,
        targets: &[CudaHtj2kEncodeResidentTarget<'_>],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        self.encode_htj2k_codeblocks_multi_resident_with_resources_and_pool_and_live_host_bytes(
            targets, resources, pool, 0,
        )
    }

    /// Encode resident batches while accounting caller-live host metadata.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblocks_multi_resident_with_resources_and_pool_and_live_host_bytes(
        &self,
        targets: &[CudaHtj2kEncodeResidentTarget<'_>],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        self.encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool_and_live_host_bytes(
            targets,
            resources,
            pool,
            live_host_bytes,
        )?
        .into_owned_code_blocks_with_live_host_bytes(live_host_bytes)
    }

    /// Encode multiple cleanup-pass code-block batches from independent resident
    /// coefficient buffers with one CUDA dispatch, returning one compact payload
    /// plus per-block ranges.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
        &self,
        targets: &[CudaHtj2kEncodeResidentTarget<'_>],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kCompactEncodedCodeBlocks, CudaError> {
        self.encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool_and_live_host_bytes(
            targets,
            resources,
            pool,
            0,
        )
    }

    /// Encode compact resident batches while accounting caller-live metadata.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool_and_live_host_bytes(
        &self,
        targets: &[CudaHtj2kEncodeResidentTarget<'_>],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<CudaHtj2kCompactEncodedCodeBlocks, CudaError> {
        validate_htj2k_encode_context(
            self,
            targets.iter().map(|target| target.coefficients),
            Some(resources),
            Some(pool),
        )?;
        let kernel_jobs =
            htj2k_encode_multi_input_kernel_jobs_with_live_host_bytes(targets, live_host_bytes)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaHtj2kCompactEncodedCodeBlocks {
                payload: Vec::new(),
                code_blocks: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }
        self.inner.set_current()?;
        self.encode_htj2k_multi_input_kernel_jobs_device_compact_with_resources_and_pool(
            &kernel_jobs,
            kernel_jobs.capacity(),
            live_host_bytes,
            resources,
            pool,
        )
    }

    /// Encode cleanup-pass code blocks from strided resident coefficient regions.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblock_regions_resident(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        validate_htj2k_encode_context(self, [coefficients], None, None)?;
        let resources = self.upload_htj2k_encode_resources(tables)?;
        let pool = self.buffer_pool();
        self.encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
            coefficients,
            coefficient_count,
            jobs,
            &resources,
            &pool,
        )
    }

    /// Encode strided resident code-block regions with pre-uploaded lookup
    /// tables and caller-owned transient buffer reuse.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        self.encode_htj2k_codeblock_regions_resident_with_resources_and_pool_and_live_host_bytes(
            coefficients,
            coefficient_count,
            jobs,
            resources,
            pool,
            0,
        )
    }

    /// Encode resident regions while accounting caller-live host metadata.
    #[doc(hidden)]
    pub fn encode_htj2k_codeblock_regions_resident_with_resources_and_pool_and_live_host_bytes(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        validate_htj2k_encode_context(self, [coefficients], Some(resources), Some(pool))?;
        if jobs.is_empty() {
            return Ok(empty_htj2k_encoded_code_blocks());
        }
        validate_resident_coefficient_capacity(coefficients, coefficient_count)?;
        let kernel_jobs = htj2k_encode_region_kernel_jobs_with_live_host_bytes(
            jobs,
            coefficient_count,
            live_host_bytes,
        )?;
        self.inner.set_current()?;
        self.encode_htj2k_kernel_jobs_device_with_resources_and_pool(
            coefficients,
            &kernel_jobs,
            kernel_jobs.capacity(),
            live_host_bytes,
            resources,
            pool,
        )
    }
}
