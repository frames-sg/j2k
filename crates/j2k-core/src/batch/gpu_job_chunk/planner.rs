// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{
    HtGpuJobChunk, HtGpuJobChunkEntry, HtGpuJobChunkLimit, HtGpuJobChunkLimits, HtGpuJobChunkPlan,
    HtGpuJobChunkPlanError, HtGpuJobChunkRequest, HtGpuJobPassBucket,
};
use crate::{
    try_batch_reserve_for_push, BatchAllocationBudget, BatchAllocationRequest,
    BatchInfrastructureError,
};

const BUCKET_ORDER: [HtGpuJobPassBucket; 3] = [
    HtGpuJobPassBucket::CleanupOnly,
    HtGpuJobPassBucket::SigProp,
    HtGpuJobPassBucket::MagRef,
];

/// Build bounded pass-homogeneous chunks while retaining caller job identity.
#[doc(hidden)]
pub fn plan_ht_gpu_job_chunks(
    jobs: &[HtGpuJobChunkRequest],
    limits: HtGpuJobChunkLimits,
) -> Result<HtGpuJobChunkPlan, HtGpuJobChunkPlanError> {
    preflight_jobs(jobs, limits)?;
    if jobs.is_empty() {
        return Ok(HtGpuJobChunkPlan::default());
    }

    let mut budget = BatchAllocationBudget::new("HTJ2K GPU chunk planning metadata");
    budget.preflight(&[
        BatchAllocationRequest::of::<HtGpuJobChunkEntry>(jobs.len()),
        BatchAllocationRequest::of::<HtGpuJobChunk>(jobs.len()),
    ])?;
    let mut entries = budget.try_vec(jobs.len(), "HTJ2K GPU chunk job identities")?;
    let mut chunks = budget.try_vec(jobs.len(), "HTJ2K GPU chunk descriptors")?;

    for bucket in BUCKET_ORDER {
        plan_bucket(jobs, limits, bucket, &mut entries, &mut chunks)?;
    }
    Ok(HtGpuJobChunkPlan::new(chunks, entries))
}

fn preflight_jobs(
    jobs: &[HtGpuJobChunkRequest],
    limits: HtGpuJobChunkLimits,
) -> Result<(), HtGpuJobChunkPlanError> {
    for (original_job_index, job) in jobs.iter().copied().enumerate() {
        bucket_for(job, original_job_index)?;
        ensure_single_job_fits(
            job,
            original_job_index,
            HtGpuJobChunkLimit::PayloadBytes,
            job.payload_bytes(),
            limits.max_payload_bytes(),
        )?;
        ensure_single_job_fits(
            job,
            original_job_index,
            HtGpuJobChunkLimit::DescriptorBytes,
            job.descriptor_bytes(),
            limits.max_descriptor_bytes(),
        )?;
    }
    Ok(())
}

fn ensure_single_job_fits(
    job: HtGpuJobChunkRequest,
    original_job_index: usize,
    limit: HtGpuJobChunkLimit,
    requested: usize,
    cap: usize,
) -> Result<(), HtGpuJobChunkPlanError> {
    if requested > cap {
        return Err(HtGpuJobChunkPlanError::SingleJobTooLarge {
            source_index: job.source_index(),
            original_job_index,
            limit,
            requested,
            cap,
        });
    }
    Ok(())
}

fn plan_bucket(
    jobs: &[HtGpuJobChunkRequest],
    limits: HtGpuJobChunkLimits,
    bucket: HtGpuJobPassBucket,
    entries: &mut Vec<HtGpuJobChunkEntry>,
    chunks: &mut Vec<HtGpuJobChunk>,
) -> Result<(), HtGpuJobChunkPlanError> {
    let mut entry_start = entries.len();
    let mut payload_bytes = 0usize;
    let mut descriptor_bytes = 0usize;

    for (original_job_index, job) in jobs.iter().copied().enumerate() {
        if bucket_for(job, original_job_index)? != bucket {
            continue;
        }
        let job_count = entries.len() - entry_start;
        if job_count == limits.max_jobs().get()
            || exceeds(
                payload_bytes,
                job.payload_bytes(),
                limits.max_payload_bytes(),
            )
            || exceeds(
                descriptor_bytes,
                job.descriptor_bytes(),
                limits.max_descriptor_bytes(),
            )
        {
            finish_chunk(
                chunks,
                bucket,
                entry_start,
                entries.len(),
                payload_bytes,
                descriptor_bytes,
            )?;
            entry_start = entries.len();
            payload_bytes = 0;
            descriptor_bytes = 0;
        }

        try_batch_reserve_for_push(entries, "HTJ2K GPU chunk job identities")?;
        entries.push(HtGpuJobChunkEntry::new(
            original_job_index,
            job.source_index(),
        ));
        payload_bytes = payload_bytes.checked_add(job.payload_bytes()).ok_or(
            BatchInfrastructureError::AllocationTooLarge {
                what: "HTJ2K GPU chunk payload bytes",
                requested: usize::MAX,
                cap: limits.max_payload_bytes(),
            },
        )?;
        descriptor_bytes = descriptor_bytes.checked_add(job.descriptor_bytes()).ok_or(
            BatchInfrastructureError::AllocationTooLarge {
                what: "HTJ2K GPU chunk descriptor bytes",
                requested: usize::MAX,
                cap: limits.max_descriptor_bytes(),
            },
        )?;
    }

    finish_chunk(
        chunks,
        bucket,
        entry_start,
        entries.len(),
        payload_bytes,
        descriptor_bytes,
    )
}

fn finish_chunk(
    chunks: &mut Vec<HtGpuJobChunk>,
    bucket: HtGpuJobPassBucket,
    entry_start: usize,
    entry_end: usize,
    payload_bytes: usize,
    descriptor_bytes: usize,
) -> Result<(), HtGpuJobChunkPlanError> {
    if entry_start == entry_end {
        return Ok(());
    }
    try_batch_reserve_for_push(chunks, "HTJ2K GPU chunk descriptors")?;
    chunks.push(HtGpuJobChunk::new(
        bucket,
        entry_start,
        entry_end,
        payload_bytes,
        descriptor_bytes,
    ));
    Ok(())
}

fn bucket_for(
    job: HtGpuJobChunkRequest,
    original_job_index: usize,
) -> Result<HtGpuJobPassBucket, HtGpuJobChunkPlanError> {
    match job.coding_passes() {
        0 => Err(HtGpuJobChunkPlanError::InvalidCodingPassCount {
            source_index: job.source_index(),
            original_job_index,
            coding_passes: 0,
        }),
        1 => Ok(HtGpuJobPassBucket::CleanupOnly),
        2 => Ok(HtGpuJobPassBucket::SigProp),
        _ => Ok(HtGpuJobPassBucket::MagRef),
    }
}

fn exceeds(current: usize, additional: usize, cap: usize) -> bool {
    current
        .checked_add(additional)
        .is_none_or(|requested| requested > cap)
}
