// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device-resident immutable arenas for an exact ordered prepared HT group.

pub(super) mod cache;

use j2k_core::HtGpuJobChunkLimits;
use metal::{Buffer, Device};

use super::{
    execution::validate_pass_homogeneous_chunk, HtBatchInput, J2kHtCleanupBatchJob,
    PackedMetalHtChunk,
};
use crate::compute::{copied_slice_buffer, Error};

pub(in crate::compute) struct PreparedMetalHtExecution {
    chunks: Vec<PreparedMetalHtChunk>,
    job_count: usize,
}

pub(in crate::compute) struct PreparedMetalHtChunk {
    pub(in crate::compute) bucket: j2k_core::HtGpuJobPassBucket,
    coded_data: Vec<u8>,
    jobs: Vec<J2kHtCleanupBatchJob>,
    pub(in crate::compute) source_indices: Vec<usize>,
    pub(in crate::compute) coded_buffer: Buffer,
    pub(in crate::compute) jobs_buffer: Buffer,
}

impl PreparedMetalHtExecution {
    fn prepare(
        device: &Device,
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
            let packed = plan.pack_chunk(chunk_index)?;
            validate_pass_homogeneous_chunk(&packed)?;
            #[cfg(test)]
            crate::compute::test_counters::record_ht_immutable_payload_upload();
            let coded_buffer = copied_slice_buffer(device, &packed.coded_data)?;
            #[cfg(test)]
            crate::compute::test_counters::record_ht_immutable_job_upload();
            let jobs_buffer = copied_slice_buffer(device, &packed.jobs)?;
            chunks.push(PreparedMetalHtChunk::new(packed, coded_buffer, jobs_buffer));
        }
        Ok(Self {
            chunks,
            job_count: plan.job_count(),
        })
    }

    pub(in crate::compute) fn chunks(&self) -> &[PreparedMetalHtChunk] {
        &self.chunks
    }

    pub(in crate::compute) const fn job_count(&self) -> usize {
        self.job_count
    }
}

impl PreparedMetalHtChunk {
    fn new(packed: PackedMetalHtChunk, coded_buffer: Buffer, jobs_buffer: Buffer) -> Self {
        Self {
            bucket: packed.bucket,
            coded_data: packed.coded_data,
            jobs: packed.jobs,
            source_indices: packed.source_indices,
            coded_buffer,
            jobs_buffer,
        }
    }

    pub(in crate::compute) fn job_count(&self) -> usize {
        self.jobs.len()
    }
}
