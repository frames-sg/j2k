// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Error, J2kClassicCleanupBatchJob, J2kClassicSegment};

pub(super) struct DistinctClassicMetadata {
    pub(super) coded_data: Vec<u8>,
    pub(super) jobs: Vec<J2kClassicCleanupBatchJob>,
    pub(super) segments: Vec<J2kClassicSegment>,
    pub(super) source_indices: Vec<usize>,
}

pub(super) fn allocate_distinct_classic_metadata(
    coded_len: usize,
    job_count: usize,
    segment_count: usize,
    mut budget: crate::batch_allocation::BatchMetadataBudget,
) -> Result<DistinctClassicMetadata, Error> {
    let requests = [
        crate::batch_allocation::BatchMetadataRequest::of::<u8>(coded_len),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kClassicCleanupBatchJob>(job_count),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kClassicSegment>(segment_count),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(job_count),
    ];
    budget.preflight(&requests)?;
    Ok(DistinctClassicMetadata {
        coded_data: budget.try_vec(
            coded_len,
            "classic J2K MetalDirect distinct color coded payload",
        )?,
        jobs: budget.try_vec(job_count, "classic J2K MetalDirect distinct color jobs")?,
        segments: budget.try_vec(
            segment_count,
            "classic J2K MetalDirect distinct color segments",
        )?,
        source_indices: budget.try_vec(
            job_count,
            "classic J2K MetalDirect distinct color status sources",
        )?,
    })
}
