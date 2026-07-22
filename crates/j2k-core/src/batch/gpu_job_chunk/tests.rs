// SPDX-License-Identifier: MIT OR Apache-2.0

use core::num::NonZeroUsize;

use alloc::vec::Vec;

use super::{
    plan_ht_gpu_job_chunks, HtGpuJobChunkLimit, HtGpuJobChunkLimits, HtGpuJobChunkPlanError,
    HtGpuJobChunkRequest, HtGpuJobPassBucket,
};

fn limits(jobs: usize, payload: usize, descriptors: usize) -> HtGpuJobChunkLimits {
    HtGpuJobChunkLimits::new(
        NonZeroUsize::new(jobs).expect("non-zero test job limit"),
        payload,
        descriptors,
    )
}

fn job(
    source_index: usize,
    coding_passes: u8,
    payload_bytes: usize,
    descriptor_bytes: usize,
) -> HtGpuJobChunkRequest {
    HtGpuJobChunkRequest::new(source_index, coding_passes, payload_bytes, descriptor_bytes)
}

fn identities(plan: &super::HtGpuJobChunkPlan, chunk_index: usize) -> Vec<(usize, usize)> {
    plan.chunk_entries(chunk_index)
        .expect("planned chunk entries")
        .iter()
        .map(|entry| (entry.original_job_index(), entry.source_index()))
        .collect()
}

#[test]
fn interleaved_jobs_are_bucketed_in_stable_original_order() {
    let jobs = [
        job(7, 3, 2, 1),
        job(2, 1, 2, 1),
        job(8, 2, 2, 1),
        job(2, 1, 2, 1),
        job(7, 4, 2, 1),
    ];
    let plan = plan_ht_gpu_job_chunks(&jobs, limits(8, 32, 32)).expect("chunk plan");

    assert_eq!(
        plan.chunks()
            .iter()
            .map(|chunk| chunk.bucket())
            .collect::<Vec<_>>(),
        [
            HtGpuJobPassBucket::CleanupOnly,
            HtGpuJobPassBucket::SigProp,
            HtGpuJobPassBucket::MagRef,
        ]
    );
    assert_eq!(identities(&plan, 0), [(1, 2), (3, 2)]);
    assert_eq!(identities(&plan, 1), [(2, 8)]);
    assert_eq!(identities(&plan, 2), [(0, 7), (4, 7)]);
}

#[test]
fn exact_job_payload_and_descriptor_boundaries_share_one_chunk() {
    let jobs = [job(0, 1, 2, 1), job(1, 1, 3, 2)];
    let plan = plan_ht_gpu_job_chunks(&jobs, limits(2, 5, 3)).expect("exact boundaries");

    assert_eq!(plan.chunks().len(), 1);
    let chunk = &plan.chunks()[0];
    assert_eq!(chunk.job_count(), 2);
    assert_eq!(chunk.payload_bytes(), 5);
    assert_eq!(chunk.descriptor_bytes(), 3);
    assert_eq!(identities(&plan, 0), [(0, 0), (1, 1)]);
}

#[test]
fn tiny_caps_split_without_losing_job_or_source_order() {
    let jobs = [
        job(4, 2, 1, 1),
        job(3, 2, 1, 1),
        job(4, 2, 1, 1),
        job(3, 2, 1, 1),
    ];
    let plan = plan_ht_gpu_job_chunks(&jobs, limits(2, 2, 2)).expect("tiny chunk caps");

    assert_eq!(plan.chunks().len(), 2);
    assert!(plan
        .chunks()
        .iter()
        .all(|chunk| chunk.bucket() == HtGpuJobPassBucket::SigProp));
    assert_eq!(identities(&plan, 0), [(0, 4), (1, 3)]);
    assert_eq!(identities(&plan, 1), [(2, 4), (3, 3)]);
}

#[test]
fn payload_and_descriptor_caps_each_force_a_boundary() {
    let payload_jobs = [job(0, 1, 2, 1), job(1, 1, 2, 1)];
    let payload_plan =
        plan_ht_gpu_job_chunks(&payload_jobs, limits(8, 3, 8)).expect("payload split");
    assert_eq!(payload_plan.chunks().len(), 2);

    let descriptor_jobs = [job(0, 3, 1, 2), job(1, 3, 1, 2)];
    let descriptor_plan =
        plan_ht_gpu_job_chunks(&descriptor_jobs, limits(8, 8, 3)).expect("descriptor split");
    assert_eq!(descriptor_plan.chunks().len(), 2);
}

#[test]
fn single_payload_and_descriptor_oversize_errors_are_source_indexed() {
    let jobs = [job(3, 1, 1, 1), job(44, 1, 6, 2)];
    assert_eq!(
        plan_ht_gpu_job_chunks(&jobs, limits(4, 5, 5)).expect_err("payload too large"),
        HtGpuJobChunkPlanError::SingleJobTooLarge {
            source_index: 44,
            original_job_index: 1,
            limit: HtGpuJobChunkLimit::PayloadBytes,
            requested: 6,
            cap: 5,
        }
    );

    let jobs = [job(9, 3, 1, 7)];
    assert_eq!(
        plan_ht_gpu_job_chunks(&jobs, limits(4, 5, 6)).expect_err("descriptor too large"),
        HtGpuJobChunkPlanError::SingleJobTooLarge {
            source_index: 9,
            original_job_index: 0,
            limit: HtGpuJobChunkLimit::DescriptorBytes,
            requested: 7,
            cap: 6,
        }
    );
}

#[test]
fn zero_pass_job_is_rejected_with_original_identity() {
    let jobs = [job(12, 0, 0, 0)];
    assert_eq!(
        plan_ht_gpu_job_chunks(&jobs, limits(1, 0, 0)).expect_err("zero pass job"),
        HtGpuJobChunkPlanError::InvalidCodingPassCount {
            source_index: 12,
            original_job_index: 0,
            coding_passes: 0,
        }
    );
}

#[test]
fn empty_input_produces_an_allocation_free_empty_plan() {
    let plan = plan_ht_gpu_job_chunks(&[], limits(1, 0, 0)).expect("empty plan");
    assert!(plan.chunks().is_empty());
    assert!(plan.entries().is_empty());
}
