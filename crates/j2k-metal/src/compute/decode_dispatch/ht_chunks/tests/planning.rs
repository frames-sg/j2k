// SPDX-License-Identifier: MIT OR Apache-2.0

use core::{mem::size_of, num::NonZeroUsize};
use std::sync::Arc;

use j2k_core::{BatchInfrastructureError, HtGpuJobChunkLimits, HtGpuJobPassBucket};

use super::super::execution::validate_pass_homogeneous_chunk;
use super::super::{
    allocate_packed_metal_ht_chunk, metal_ht_pipeline_kind_for_bucket, plan_metal_ht_chunks, Error,
    HtBatchInput, HtPayloadSource, J2kHtCleanupBatchJob, MetalHtPipelineKind, PackedMetalHtChunk,
};
use crate::batch_allocation::BatchMetadataBudget;
use crate::compute::PreparedHtExecutionOwner;

mod fixtures;
mod referenced;

fn limits(jobs: usize, payload: usize) -> HtGpuJobChunkLimits {
    HtGpuJobChunkLimits::new(
        NonZeroUsize::new(jobs).expect("test job cap is nonzero"),
        payload,
        jobs * size_of::<J2kHtCleanupBatchJob>(),
    )
}

fn job(coded_offset: u32, coded_len: u32, passes: u32, output_offset: u32) -> J2kHtCleanupBatchJob {
    J2kHtCleanupBatchJob {
        coded_offset,
        width: 1,
        height: 1,
        coded_len,
        cleanup_length: coded_len,
        refinement_length: 0,
        missing_msbs: 0,
        num_bitplanes: 1,
        roi_shift: 0,
        number_of_coding_passes: passes,
        output_stride: 1,
        output_offset,
        dequantization_step: 1.0,
        stripe_causal: 0,
    }
}

#[test]
fn pass_buckets_select_dedicated_metal_ht_pipelines() {
    assert_eq!(
        metal_ht_pipeline_kind_for_bucket(HtGpuJobPassBucket::CleanupOnly),
        MetalHtPipelineKind::CleanupOnly
    );
    assert_eq!(
        metal_ht_pipeline_kind_for_bucket(HtGpuJobPassBucket::SigProp),
        MetalHtPipelineKind::SigProp
    );
    assert_eq!(
        metal_ht_pipeline_kind_for_bucket(HtGpuJobPassBucket::MagRef),
        MetalHtPipelineKind::MagRef
    );
}

#[test]
fn magref_specialization_rejects_more_than_three_coding_passes() {
    let chunk = PackedMetalHtChunk {
        bucket: HtGpuJobPassBucket::MagRef,
        coded_data: vec![0],
        jobs: vec![job(0, 1, 4, 0)],
        source_indices: vec![0],
    };

    assert!(matches!(
        validate_pass_homogeneous_chunk(&chunk),
        Err(Error::UnsupportedMetalRequest {
            reason: "HTJ2K Metal decoding supports at most three coding passes per code block"
        })
    ));
}

#[test]
fn tiny_caps_split_jobs_in_pass_order_and_keep_source_mapping() {
    let first_payload = [10, 11, 12, 13];
    let second_payload = [20, 21, 22, 23];
    let first_jobs = [job(0, 1, 3, 0), job(1, 1, 1, 1)];
    let second_jobs = [job(0, 1, 2, 0), job(1, 1, 1, 1)];
    let first_owner = Arc::new(PreparedHtExecutionOwner);
    let second_owner = Arc::new(PreparedHtExecutionOwner);
    let batches = [
        HtBatchInput {
            source_index: 7,
            payload: HtPayloadSource::Contiguous(&first_payload),
            jobs: &first_jobs,
            output_base: 0,
            execution_owner: &first_owner,
        },
        HtBatchInput {
            source_index: 11,
            payload: HtPayloadSource::Contiguous(&second_payload),
            jobs: &second_jobs,
            output_base: 8,
            execution_owner: &second_owner,
        },
    ];

    let plan = plan_metal_ht_chunks(&batches, limits(1, 1)).expect("tiny chunk plan");
    assert_eq!(plan.job_count(), 4);
    assert_eq!(plan.chunk_count(), 4);

    let packed = [0, 1, 2, 3].map(|index| plan.pack_chunk(index).expect("packed chunk"));
    assert_eq!(
        [
            packed[0].bucket,
            packed[1].bucket,
            packed[2].bucket,
            packed[3].bucket,
        ],
        [
            HtGpuJobPassBucket::CleanupOnly,
            HtGpuJobPassBucket::CleanupOnly,
            HtGpuJobPassBucket::SigProp,
            HtGpuJobPassBucket::MagRef,
        ]
    );
    assert!(packed
        .iter()
        .flat_map(|chunk| chunk.source_indices.iter().copied())
        .eq([7, 11, 11, 7]));
    assert!(packed
        .iter()
        .flat_map(|chunk| chunk.coded_data.iter().copied())
        .eq([11, 21, 20, 10]));
    assert!(packed
        .iter()
        .flat_map(|chunk| chunk.jobs.iter().map(|job| job.output_offset))
        .eq([1, 9, 8, 0]));
}

#[test]
fn packed_chunk_rebases_payload_offsets_without_exceeding_caps() {
    let payload = [1, 2, 3, 4, 5, 6];
    let jobs = [job(1, 2, 1, 0), job(4, 2, 1, 1)];
    let owner = Arc::new(PreparedHtExecutionOwner);
    let batches = [HtBatchInput {
        source_index: 3,
        payload: HtPayloadSource::Contiguous(&payload),
        jobs: &jobs,
        output_base: 16,
        execution_owner: &owner,
    }];

    let plan = plan_metal_ht_chunks(&batches, limits(2, 4)).expect("bounded chunk plan");
    let packed = plan.pack_chunk(0).expect("packed chunk");

    assert_eq!(packed.coded_data, [2, 3, 5, 6]);
    assert_eq!(packed.jobs.len(), 2);
    assert_eq!(packed.jobs[0].coded_offset, 0);
    assert_eq!(packed.jobs[1].coded_offset, 2);
    assert_eq!(packed.jobs[0].output_offset, 16);
    assert_eq!(packed.jobs[1].output_offset, 17);
    assert_eq!(packed.source_indices, [3, 3]);
    assert!(packed.coded_data.len() <= 4);
    assert!(
        packed.jobs.len() * size_of::<J2kHtCleanupBatchJob>()
            <= 2 * size_of::<J2kHtCleanupBatchJob>()
    );
}

#[test]
fn packed_ht_chunk_metadata_honors_exact_cap_and_one_byte_over() {
    let payload_bytes = 7;
    let job_count = 3;
    let exact_cap =
        payload_bytes + job_count * (size_of::<J2kHtCleanupBatchJob>() + size_of::<usize>());
    let owners = allocate_packed_metal_ht_chunk(
        payload_bytes,
        job_count,
        BatchMetadataBudget::with_cap("HTJ2K Metal packed chunk metadata", exact_cap),
    )
    .expect("exact packed HT chunk metadata cap");
    assert_eq!(owners.coded_data.capacity(), payload_bytes);
    assert_eq!(owners.jobs.capacity(), job_count);
    assert_eq!(owners.source_indices.capacity(), job_count);

    assert!(matches!(
        allocate_packed_metal_ht_chunk(
            payload_bytes,
            job_count,
            BatchMetadataBudget::with_cap("HTJ2K Metal packed chunk metadata", exact_cap - 1,),
        ),
        Err(Error::BatchInfrastructure(
            BatchInfrastructureError::AllocationTooLarge { requested, cap, .. }
        )) if requested == exact_cap && cap == exact_cap - 1
    ));
}
