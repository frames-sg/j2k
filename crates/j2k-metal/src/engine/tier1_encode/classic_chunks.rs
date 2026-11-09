// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-staged classic Tier-1 batches split into runs whose Metal buffers
//! each fit one allocation.
//!
//! A batch's output buffer reserves the conservative per-block capacity, so a
//! whole 2048x2048 RGB image asks for about 1 GB in one buffer, past the
//! allocation limit. Contiguous runs of code blocks are encoded one after
//! another and their blocks returned in job order; a batch that fits runs as
//! one dispatch, as before.

use std::mem::size_of;
use std::ops::Range;

use crate::metal_types::prelude::*;

use super::super::abi::{J2kClassicEncodeBatchJob, J2kClassicEncodeStatus, J2kClassicSegment};
use super::super::{
    classic_encode_output_capacity, classic_encode_segment_capacity, classic_style_flags,
    with_runtime, EncodedJ2kCodeBlock, Error, J2kTier1CodeBlockEncodeJob, MetalRuntime,
};
use super::encode_classic_tier1_chunk;

/// Batch buffers a classic Tier-1 dispatch allocates per job: coefficients,
/// job descriptors, statuses, output bytes and pass segments. The host copies
/// of the coefficients and job descriptors, charged together to one host
/// budget, stay under half the output bytes, so the output limit covers them.
const CLASSIC_TIER1_BUFFERS: usize = 5;

type ClassicTier1JobBytes = [usize; CLASSIC_TIER1_BUFFERS];

/// Encodes `jobs` in as few runs as keep every batch buffer within the
/// largest allocation Metal and this crate allow, returning blocks in job
/// order. A single job too large on its own runs alone and fails at its
/// allocation with the usual typed `BufferAllocationTooLarge` error, as the
/// resident classic chunk plan and the single-block encode do. A code block
/// holds at most 4096 samples, so that takes a limit under 8 MiB.
pub(crate) fn encode_classic_tier1_code_blocks(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    let limit = with_runtime(|runtime| Ok(classic_tier1_buffer_limit(runtime)))?;
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("classic J2K Metal Tier-1 encode runs");
    let mut job_bytes = budget.try_vec(jobs.len(), "classic J2K Metal Tier-1 job buffer bytes")?;
    for job in jobs {
        job_bytes.push(classic_tier1_job_bytes(job)?);
    }
    let mut runs = budget.try_vec(jobs.len(), "classic J2K Metal Tier-1 encode runs")?;
    classic_tier1_runs(&job_bytes, limit, &mut runs);
    if runs.len() <= 1 {
        return encode_classic_tier1_chunk(jobs);
    }

    let mut encoded = budget.try_vec(jobs.len(), "classic J2K Metal Tier-1 encoded blocks")?;
    for run in runs {
        #[cfg(test)]
        crate::engine::test_counters::record_classic_tier1_run();
        encoded.extend(encode_classic_tier1_chunk(&jobs[run])?);
    }
    Ok(encoded)
}

/// The largest single buffer a classic Tier-1 batch may allocate: the
/// device's `maxBufferLength`, capped by the host allocation ceiling that
/// every `j2k-metal-support` buffer allocation enforces.
fn classic_tier1_buffer_limit(runtime: &MetalRuntime) -> usize {
    #[cfg(test)]
    if let Some(limit) = crate::engine::test_counters::classic_tier1_buffer_limit_for_test() {
        return limit;
    }
    runtime
        .device
        .maxBufferLength()
        .min(j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES)
}

/// Bytes `job` adds to each batch buffer, sized as the batch sizes them.
fn classic_tier1_job_bytes(
    job: &J2kTier1CodeBlockEncodeJob<'_>,
) -> Result<ClassicTier1JobBytes, Error> {
    let overflow = || Error::MetalKernel {
        message: "classic J2K Metal encode buffer size overflow".to_string(),
    };
    let samples = usize::try_from(job.width)
        .ok()
        .zip(usize::try_from(job.height).ok())
        .and_then(|(width, height)| width.checked_mul(height))
        .ok_or_else(overflow)?;
    let segments =
        classic_encode_segment_capacity(classic_style_flags(job.style), job.total_bitplanes)
            .checked_mul(size_of::<J2kClassicSegment>())
            .ok_or_else(overflow)?;
    Ok([
        samples.checked_mul(size_of::<i32>()).ok_or_else(overflow)?,
        size_of::<J2kClassicEncodeBatchJob>(),
        size_of::<J2kClassicEncodeStatus>(),
        classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?,
        segments,
    ])
}

/// Splits jobs into contiguous runs, in order, so that each run's summed
/// bytes stay within `limit` in every buffer. A job that alone exceeds
/// `limit` forms a run of its own.
fn classic_tier1_runs(
    job_bytes: &[ClassicTier1JobBytes],
    limit: usize,
    runs: &mut Vec<Range<usize>>,
) {
    let mut start = 0;
    let mut totals = [0usize; CLASSIC_TIER1_BUFFERS];
    for (index, bytes) in job_bytes.iter().enumerate() {
        let fits = totals
            .iter()
            .zip(bytes)
            .all(|(total, add)| total.checked_add(*add).is_some_and(|sum| sum <= limit));
        if !fits && index > start {
            runs.push(start..index);
            start = index;
            totals = [0; CLASSIC_TIER1_BUFFERS];
        }
        for (total, add) in totals.iter_mut().zip(bytes) {
            *total = total.saturating_add(*add);
        }
    }
    if start < job_bytes.len() {
        runs.push(start..job_bytes.len());
    }
}

#[cfg(test)]
mod tests {
    use super::{classic_tier1_runs, ClassicTier1JobBytes};

    /// Each run as its (start, end) job indices.
    fn runs(job_bytes: &[ClassicTier1JobBytes], limit: usize) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        classic_tier1_runs(job_bytes, limit, &mut runs);
        runs.into_iter().map(|run| (run.start, run.end)).collect()
    }

    #[test]
    fn a_batch_within_the_limit_is_one_run() {
        let jobs = [[4, 1, 1, 10, 1]; 3];
        assert_eq!(runs(&jobs, 30), [(0, 3)]);
    }

    #[test]
    fn runs_split_where_any_buffer_would_pass_the_limit() {
        // At 10, output bytes decide the first split and coefficient bytes the
        // second; at 11, segment bytes decide the second.
        let jobs = [
            [1, 1, 1, 6, 1],
            [1, 1, 1, 4, 1],
            [1, 1, 1, 1, 1],
            [9, 1, 1, 1, 2],
            [2, 1, 1, 1, 10],
        ];
        assert_eq!(runs(&jobs, 10), [(0, 2), (2, 4), (4, 5)]);
        assert_eq!(runs(&jobs, 11), [(0, 3), (3, 4), (4, 5)]);
    }

    #[test]
    fn a_run_may_fill_a_buffer_exactly() {
        let jobs = [[1, 1, 1, 5, 1]; 4];
        assert_eq!(runs(&jobs, 10), [(0, 2), (2, 4)]);
    }

    #[test]
    fn an_oversized_job_runs_alone_and_keeps_order() {
        let jobs = [[1, 1, 1, 2, 1], [1, 1, 1, 50, 1], [1, 1, 1, 2, 1]];
        assert_eq!(runs(&jobs, 10), [(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn no_jobs_make_no_runs() {
        assert_eq!(runs(&[], 10), Vec::<(usize, usize)>::new());
    }
}
