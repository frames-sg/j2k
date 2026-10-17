// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device-resident immutable arenas for an exact ordered prepared HT group.

pub(super) mod cache;

use crate::metal_types::Buffer;
use j2k_core::HtGpuJobChunkLimits;

use super::{
    execution::validate_pass_homogeneous_jobs, HtBatchInput, J2kHtCleanupBatchJob,
    PackedMetalHtChunkJobs,
};
use crate::engine::Error;

pub(in crate::engine) struct PreparedMetalHtExecution {
    chunks: Vec<PreparedMetalHtChunk>,
    job_count: usize,
}

pub(in crate::engine) struct PreparedMetalHtChunk {
    pub(in crate::engine) bucket: j2k_core::HtGpuJobPassBucket,
    jobs: Vec<J2kHtCleanupBatchJob>,
    pub(in crate::engine) source_indices: Vec<usize>,
    pub(in crate::engine) coded_buffer: Buffer,
    pub(in crate::engine) jobs_buffer: Buffer,
}

// SAFETY: A prepared execution is immutable after construction. Its host
// arenas are read-only, and its retained coded/job Metal buffers are bound only
// for shader reads while each submission owns separate writable outputs.
unsafe impl Send for PreparedMetalHtExecution {}
// SAFETY: Concurrent readers cannot mutate the host arenas or retained input
// buffers through this type; all GPU writes target per-submission storage.
unsafe impl Sync for PreparedMetalHtExecution {}

impl PreparedMetalHtExecution {
    fn prepare(
        inputs: crate::engine::HtInputAllocator<'_>,
        batches: &[HtBatchInput<'_>],
        limits: HtGpuJobChunkLimits,
    ) -> Result<Self, Error> {
        let plan = super::plan_metal_ht_chunks(batches, limits)?;
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(plan.chunk_count())
            .map_err(|source| Error::PreparedPlanCacheAllocation {
                context: "J2K Metal prepared HT execution chunks",
                source,
            })?;
        for chunk_index in 0..plan.chunk_count() {
            // Payload bytes are written once, straight into the shared buffer
            // the kernels read; no host copy is packed or retained.
            let (packed, coded_buffer) = plan.pack_chunk_into_buffer(chunk_index, inputs)?;
            validate_pass_homogeneous_jobs(packed.bucket, &packed.jobs)?;
            #[cfg(test)]
            crate::engine::test_counters::record_ht_immutable_payload_upload();
            #[cfg(test)]
            crate::engine::test_counters::record_ht_immutable_job_upload();
            let jobs_buffer = inputs.copied_slice(&packed.jobs)?;
            chunks.push(PreparedMetalHtChunk::new(packed, coded_buffer, jobs_buffer));
        }
        Ok(Self {
            chunks,
            job_count: plan.job_count(),
        })
    }

    pub(in crate::engine) fn chunks(&self) -> &[PreparedMetalHtChunk] {
        &self.chunks
    }

    pub(in crate::engine) const fn job_count(&self) -> usize {
        self.job_count
    }
}

impl PreparedMetalHtChunk {
    fn new(packed: PackedMetalHtChunkJobs, coded_buffer: Buffer, jobs_buffer: Buffer) -> Self {
        Self {
            bucket: packed.bucket,
            jobs: packed.jobs,
            source_indices: packed.source_indices,
            coded_buffer,
            jobs_buffer,
        }
    }

    pub(in crate::engine) fn job_count(&self) -> usize {
        self.jobs.len()
    }
}
