// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    bytes::{htj2k_cleanup_multi_jobs_as_bytes, htj2k_statuses_byte_len},
    context::CudaContext,
    error::CudaError,
    execution::{CudaExecutionStats, CudaLaunchMode},
    memory::{pooled_device_buffer, CudaBufferPool},
};

use super::super::{
    context_validation::validate_cleanup_context,
    planning::{
        htj2k_cleanup_multi_kernel_jobs_with_live_host_bytes, htj2k_decode_multi_kernel_for_jobs,
    },
    queued::CudaQueuedHtj2kCleanup,
    status_group::CudaQueuedHtj2kCleanupGroup,
    types::{
        htj2k_decode_kernel_tables, CudaHtj2kCleanupTarget, CudaHtj2kDecodeResources,
        Htj2kDecodeCodeblocksMultiLaunch,
    },
};

impl CudaContext {
    /// Enqueue HTJ2K cleanup passes for multiple output buffers with one CUDA
    /// dispatch. The returned value must be kept live until `finish` validates
    /// the kernel statuses after the default stream has completed.
    ///
    /// # Safety
    ///
    /// Every target coefficient buffer must remain allocated and must not be
    /// mutated or reused until the returned cleanup is finished or dropped.
    /// Target allocations and each target's job write regions must be
    /// pairwise disjoint; both conditions are validated before launch.
    /// The decode payload and table resources must remain live for the same
    /// duration. The resources, targets, and pool must belong to this context
    /// (validated at runtime), and all pool clones must remain confined to this
    /// context's default stream until that completion point.
    #[doc(hidden)]
    pub unsafe fn decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedHtj2kCleanup, CudaError> {
        // SAFETY: this wrapper preserves the caller's target and pool lifetime
        // requirements and contributes no additional caller-live host owners.
        unsafe {
            self.decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool_and_live_host_bytes(
                resources,
                targets,
                pool,
                0,
            )
        }
    }

    /// Enqueue HTJ2K cleanup while accounting caller-live host metadata.
    ///
    /// # Safety
    ///
    /// The target buffers, resources, and pool must satisfy the same lifetime,
    /// aliasing, context, and stream-confinement requirements as
    /// [`Self::decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool`].
    #[doc(hidden)]
    pub unsafe fn decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool_and_live_host_bytes(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<CudaQueuedHtj2kCleanup, CudaError> {
        // SAFETY: this forwards the caller's complete target/resource lifetime
        // contract and requests an owned per-launch status allocation.
        unsafe {
            self.enqueue_htj2k_cleanup_multi_impl(resources, targets, pool, live_host_bytes, None)
        }
    }

    /// Enqueue one HTJ2K cleanup launch into a group-owned status arena.
    ///
    /// # Safety
    ///
    /// In addition to the normal queued-cleanup requirements, `status_group`
    /// must outlive the returned cleanup and finish it before exposing output.
    #[doc(hidden)]
    pub unsafe fn decode_htj2k_codeblocks_cleanup_multi_enqueue_into_status_group(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
        status_group: &CudaQueuedHtj2kCleanupGroup,
        status_offset: usize,
    ) -> Result<CudaQueuedHtj2kCleanup, CudaError> {
        // SAFETY: this forwards the caller's target/resource/status-group
        // lifetime and aliasing requirements unchanged.
        unsafe {
            self.enqueue_htj2k_cleanup_multi_impl(
                resources,
                targets,
                pool,
                live_host_bytes,
                Some((status_group, status_offset)),
            )
        }
    }

    unsafe fn enqueue_htj2k_cleanup_multi_impl(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
        status_group: Option<(&CudaQueuedHtj2kCleanupGroup, usize)>,
    ) -> Result<CudaQueuedHtj2kCleanup, CudaError> {
        validate_cleanup_context(self, resources, targets, pool)?;
        let kernel_jobs = htj2k_cleanup_multi_kernel_jobs_with_live_host_bytes(
            targets,
            resources.payload_len,
            live_host_bytes,
        )?;
        if kernel_jobs.is_empty() {
            return Ok(CudaQueuedHtj2kCleanup {
                context: self.clone(),
                resources: Vec::new(),
                status_buffer: None,
                status_count: 0,
                status_offset: status_group.map_or(0, |(_, offset)| offset),
                uses_external_status_group: status_group.is_some(),
                kernel_name: "j2k_htj2k_decode_codeblocks_multi",
                execution: CudaExecutionStats::default(),
                pool_reuse_guard: None,
                finish_host_live_bytes: 0,
            });
        }
        self.inner.set_current()?;
        let (decode_kernel, decode_kernel_name) = htj2k_decode_multi_kernel_for_jobs(&kernel_jobs);
        let tables = htj2k_decode_kernel_tables(resources)?;

        let mut host_budget = HostPhaseBudget::with_live_bytes(
            "CUDA queued HTJ2K cleanup metadata",
            live_host_bytes,
        )?;
        host_budget.account_vec(&kernel_jobs)?;
        let mut queued_resources = host_budget.try_vec_with_capacity(1)?;
        let jobs_buffer = pool.upload(htj2k_cleanup_multi_jobs_as_bytes(&kernel_jobs))?;
        queued_resources.push(jobs_buffer);
        let mut finish_budget = HostPhaseBudget::with_live_bytes(
            "CUDA queued HTJ2K cleanup retained metadata",
            live_host_bytes,
        )?;
        finish_budget.account_vec(&queued_resources)?;
        let status_buffer = status_group
            .is_none()
            .then(|| pool.take(htj2k_statuses_byte_len(kernel_jobs.len())?))
            .transpose()?;
        let payload_buffer = resources.payload.buffer()?;
        let jobs_device_buffer = pooled_device_buffer(&queued_resources[0])?;
        let (status_device_buffer, status_byte_offset, status_offset) = match status_group {
            Some((group, offset)) => {
                let (buffer, byte_offset) = group.status_destination(offset, kernel_jobs.len())?;
                (buffer, byte_offset, offset)
            }
            None => (
                pooled_device_buffer(status_buffer.as_ref().ok_or(
                    CudaError::InternalInvariant {
                        what: "owned HTJ2K status allocation disappeared before launch",
                    },
                )?)?,
                0,
                0,
            ),
        };
        let pool_reuse_guard = pool.defer_reuse()?;
        let launch_result =
            self.launch_htj2k_decode_codeblocks_multi(Htj2kDecodeCodeblocksMultiLaunch {
                kernel: decode_kernel,
                payload: payload_buffer,
                jobs: jobs_device_buffer,
                tables,
                statuses: status_device_buffer,
                status_byte_offset,
                job_count: kernel_jobs.len(),
                mode: CudaLaunchMode::Async,
            });
        if let Err(error) = launch_result {
            return pool_reuse_guard.synchronize_then_error(error);
        }

        Ok(CudaQueuedHtj2kCleanup {
            context: self.clone(),
            resources: queued_resources,
            status_buffer,
            status_count: kernel_jobs.len(),
            status_offset,
            uses_external_status_group: status_group.is_some(),
            kernel_name: decode_kernel_name,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
            pool_reuse_guard: Some(pool_reuse_guard),
            finish_host_live_bytes: finish_budget.live_bytes(),
        })
    }
}
