// SPDX-License-Identifier: MIT OR Apache-2.0

use core::{mem::size_of, num::NonZeroUsize};
use std::sync::Arc;

use j2k_core::HtGpuJobChunkLimits;
use j2k_native::{HtCodeBlockPayloadRanges, J2kCodestreamRange};

use super::super::super::{
    plan_metal_ht_chunks, Error, HtBatchInput, HtPayloadSource, J2kHtCleanupBatchJob,
};
use crate::compute::PreparedHtExecutionOwner;

fn job(
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    output_offset: u32,
) -> J2kHtCleanupBatchJob {
    J2kHtCleanupBatchJob {
        coded_offset: 999,
        width: 1,
        height: 1,
        coded_len,
        cleanup_length,
        refinement_length,
        missing_msbs: 0,
        num_bitplanes: 1,
        roi_shift: 0,
        number_of_coding_passes: if refinement_length == 0 { 1 } else { 2 },
        output_stride: 1,
        output_offset,
        dequantization_step: 1.0,
        stripe_causal: 0,
    }
}

fn limits(jobs: usize, payload: usize) -> HtGpuJobChunkLimits {
    HtGpuJobChunkLimits::new(
        NonZeroUsize::new(jobs).expect("nonzero test job cap"),
        payload,
        jobs * size_of::<J2kHtCleanupBatchJob>(),
    )
}

#[test]
fn referenced_payload_ranges_are_validated_before_chunk_packing() {
    let bytes = Arc::<[u8]>::from([10, 11, 12]);
    let ranges = [HtCodeBlockPayloadRanges {
        cleanup: J2kCodestreamRange {
            offset: 2,
            length: 2,
        },
        refinement: None,
    }];
    let jobs = [job(2, 2, 0, 0)];
    let owner = Arc::new(PreparedHtExecutionOwner);
    let input = [HtBatchInput {
        source_index: 7,
        payload: HtPayloadSource::Referenced {
            input: &bytes,
            ranges: &ranges,
        },
        jobs: &jobs,
        output_base: 0,
        execution_owner: &owner,
    }];

    let Err(error) = plan_metal_ht_chunks(&input, limits(1, 2)) else {
        panic!("out-of-bounds referenced payload must fail before packing")
    };
    assert!(matches!(error, Error::MetalKernel { .. }));
    assert!(error.to_string().contains("exceeds retained input"));
}

#[test]
fn referenced_payload_pack_preserves_job_order_and_rebases_offsets() {
    let bytes = Arc::<[u8]>::from([0, 10, 0, 11, 0, 20, 21, 0, 22]);
    let ranges = [
        HtCodeBlockPayloadRanges {
            cleanup: J2kCodestreamRange {
                offset: 1,
                length: 1,
            },
            refinement: Some(J2kCodestreamRange {
                offset: 3,
                length: 1,
            }),
        },
        HtCodeBlockPayloadRanges {
            cleanup: J2kCodestreamRange {
                offset: 5,
                length: 2,
            },
            refinement: Some(J2kCodestreamRange {
                offset: 8,
                length: 1,
            }),
        },
    ];
    let jobs = [job(2, 1, 1, 2), job(3, 2, 1, 4)];
    let owner = Arc::new(PreparedHtExecutionOwner);
    let input = [HtBatchInput {
        source_index: 3,
        payload: HtPayloadSource::Referenced {
            input: &bytes,
            ranges: &ranges,
        },
        jobs: &jobs,
        output_base: 16,
        execution_owner: &owner,
    }];

    let plan = plan_metal_ht_chunks(&input, limits(2, 5)).expect("referenced chunk plan");
    let packed = plan.pack_chunk(0).expect("referenced packed chunk");

    assert_eq!(packed.coded_data, [10, 11, 20, 21, 22]);
    assert_eq!(packed.jobs[0].coded_offset, 0);
    assert_eq!(packed.jobs[1].coded_offset, 2);
    assert_eq!(packed.jobs[0].output_offset, 18);
    assert_eq!(packed.jobs[1].output_offset, 20);
    assert_eq!(packed.source_indices, [3, 3]);
}
