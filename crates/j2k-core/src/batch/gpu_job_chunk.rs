// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded, backend-neutral HTJ2K GPU job chunk planning.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use super::BatchInfrastructureError;

mod planner;
pub use self::planner::plan_ht_gpu_job_chunks;

/// HTJ2K coding-pass family used to keep cleanup-heavy jobs on a fused path.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HtGpuJobPassBucket {
    /// One cleanup pass and no refinement payload.
    CleanupOnly,
    /// Cleanup followed by one `SigProp` pass.
    SigProp,
    /// Cleanup, `SigProp`, and `MagRef`, including any future pass count above three.
    MagRef,
}

/// One flattened HTJ2K job submitted to the shared chunk planner.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HtGpuJobChunkRequest {
    source_index: usize,
    coding_passes: u8,
    payload_bytes: usize,
    descriptor_bytes: usize,
}

impl HtGpuJobChunkRequest {
    /// Describe one job without borrowing backend-specific payload or descriptor types.
    #[doc(hidden)]
    #[must_use]
    pub const fn new(
        source_index: usize,
        coding_passes: u8,
        payload_bytes: usize,
        descriptor_bytes: usize,
    ) -> Self {
        Self {
            source_index,
            coding_passes,
            payload_bytes,
            descriptor_bytes,
        }
    }

    /// Caller input index that owns this job.
    #[doc(hidden)]
    #[must_use]
    pub const fn source_index(self) -> usize {
        self.source_index
    }

    /// Number of HT coding passes represented by the job.
    #[doc(hidden)]
    #[must_use]
    pub const fn coding_passes(self) -> u8 {
        self.coding_passes
    }

    /// Compressed cleanup and refinement payload bytes.
    #[doc(hidden)]
    #[must_use]
    pub const fn payload_bytes(self) -> usize {
        self.payload_bytes
    }

    /// Backend descriptor-arena bytes required by the job.
    #[doc(hidden)]
    #[must_use]
    pub const fn descriptor_bytes(self) -> usize {
        self.descriptor_bytes
    }
}

/// Per-chunk limits enforced independently for every pass bucket.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HtGpuJobChunkLimits {
    jobs: NonZeroUsize,
    payload_bytes: usize,
    descriptor_bytes: usize,
}

impl HtGpuJobChunkLimits {
    /// Construct explicit job, compressed-payload, and descriptor-arena limits.
    #[doc(hidden)]
    #[must_use]
    pub const fn new(
        max_jobs: NonZeroUsize,
        max_payload_bytes: usize,
        max_descriptor_bytes: usize,
    ) -> Self {
        Self {
            jobs: max_jobs,
            payload_bytes: max_payload_bytes,
            descriptor_bytes: max_descriptor_bytes,
        }
    }

    /// Maximum jobs in one chunk.
    #[doc(hidden)]
    #[must_use]
    pub const fn max_jobs(self) -> NonZeroUsize {
        self.jobs
    }

    /// Maximum compressed payload bytes in one chunk.
    #[doc(hidden)]
    #[must_use]
    pub const fn max_payload_bytes(self) -> usize {
        self.payload_bytes
    }

    /// Maximum backend descriptor bytes in one chunk.
    #[doc(hidden)]
    #[must_use]
    pub const fn max_descriptor_bytes(self) -> usize {
        self.descriptor_bytes
    }
}

/// Limit dimension exceeded by one job before any chunk is constructed.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HtGpuJobChunkLimit {
    /// Compressed cleanup and refinement payload bytes.
    PayloadBytes,
    /// Backend descriptor-arena bytes.
    DescriptorBytes,
}

/// Failure returned by bounded HTJ2K GPU chunk planning.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum HtGpuJobChunkPlanError {
    /// A submitted job does not contain the mandatory cleanup pass.
    #[error(
        "HTJ2K source {source_index} job {original_job_index} has invalid coding-pass count {coding_passes}"
    )]
    InvalidCodingPassCount {
        /// Caller input index that owns the invalid job.
        source_index: usize,
        /// Position of the invalid job in the submitted flattened job slice.
        original_job_index: usize,
        /// Invalid coding-pass count.
        coding_passes: u8,
    },
    /// One job cannot fit an otherwise valid empty chunk.
    #[error(
        "HTJ2K source {source_index} job {original_job_index} requires {requested} {limit:?}, cap {cap}"
    )]
    SingleJobTooLarge {
        /// Caller input index that owns the oversized job.
        source_index: usize,
        /// Position of the oversized job in the submitted flattened job slice.
        original_job_index: usize,
        /// Limit dimension exceeded by the job.
        limit: HtGpuJobChunkLimit,
        /// Bytes required by the single job.
        requested: usize,
        /// Configured per-chunk byte cap.
        cap: usize,
    },
    /// Host planning metadata could not be represented or allocated safely.
    #[error(transparent)]
    BatchInfrastructure(#[from] BatchInfrastructureError),
}

/// Stable identity retained for one job after pass-family bucketing.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HtGpuJobChunkEntry {
    original_job_index: usize,
    source_index: usize,
}

impl HtGpuJobChunkEntry {
    pub(super) const fn new(original_job_index: usize, source_index: usize) -> Self {
        Self {
            original_job_index,
            source_index,
        }
    }

    /// Position in the caller's original flattened job slice.
    #[doc(hidden)]
    #[must_use]
    pub const fn original_job_index(self) -> usize {
        self.original_job_index
    }

    /// Caller input index that owns this job.
    #[doc(hidden)]
    #[must_use]
    pub const fn source_index(self) -> usize {
        self.source_index
    }
}

/// One bounded, pass-homogeneous range in [`HtGpuJobChunkPlan::entries`].
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HtGpuJobChunk {
    bucket: HtGpuJobPassBucket,
    entry_start: usize,
    entry_end: usize,
    payload_bytes: usize,
    descriptor_bytes: usize,
}

impl HtGpuJobChunk {
    pub(super) const fn new(
        bucket: HtGpuJobPassBucket,
        entry_start: usize,
        entry_end: usize,
        payload_bytes: usize,
        descriptor_bytes: usize,
    ) -> Self {
        Self {
            bucket,
            entry_start,
            entry_end,
            payload_bytes,
            descriptor_bytes,
        }
    }

    /// Cleanup-only, SigProp, or MagRef pass family for every entry.
    #[doc(hidden)]
    #[must_use]
    pub const fn bucket(self) -> HtGpuJobPassBucket {
        self.bucket
    }

    /// Number of jobs in the chunk.
    #[doc(hidden)]
    #[must_use]
    pub const fn job_count(self) -> usize {
        self.entry_end - self.entry_start
    }

    /// Aggregate compressed payload bytes in the chunk.
    #[doc(hidden)]
    #[must_use]
    pub const fn payload_bytes(self) -> usize {
        self.payload_bytes
    }

    /// Aggregate backend descriptor bytes in the chunk.
    #[doc(hidden)]
    #[must_use]
    pub const fn descriptor_bytes(self) -> usize {
        self.descriptor_bytes
    }
}

/// Complete bounded plan with flat stable-identity storage shared by all chunks.
#[doc(hidden)]
#[derive(Debug, Default, PartialEq, Eq)]
pub struct HtGpuJobChunkPlan {
    chunks: Vec<HtGpuJobChunk>,
    entries: Vec<HtGpuJobChunkEntry>,
}

impl HtGpuJobChunkPlan {
    pub(super) const fn new(chunks: Vec<HtGpuJobChunk>, entries: Vec<HtGpuJobChunkEntry>) -> Self {
        Self { chunks, entries }
    }

    /// Chunks ordered cleanup-only, SigProp, then MagRef; stable within each bucket.
    #[doc(hidden)]
    #[must_use]
    pub fn chunks(&self) -> &[HtGpuJobChunk] {
        &self.chunks
    }

    /// Flat stable-identity entries referenced by the chunk descriptors.
    #[doc(hidden)]
    #[must_use]
    pub fn entries(&self) -> &[HtGpuJobChunkEntry] {
        &self.entries
    }

    /// Entries belonging to one chunk index, or `None` when the index is invalid.
    #[doc(hidden)]
    #[must_use]
    pub fn chunk_entries(&self, chunk_index: usize) -> Option<&[HtGpuJobChunkEntry]> {
        let chunk = self.chunks.get(chunk_index)?;
        self.entries.get(chunk.entry_start..chunk.entry_end)
    }
}

#[cfg(test)]
mod tests;
