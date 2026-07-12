// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

use j2k_core::BatchInfrastructureError;

use super::{
    allocate_distinct_classic_metadata, Error, J2kClassicCleanupBatchJob, J2kClassicSegment,
};
use crate::batch_allocation::BatchMetadataBudget;

#[test]
fn distinct_classic_metadata_honors_exact_cap_and_one_byte_over() {
    let coded_len = 11;
    let job_count = 3;
    let segment_count = 5;
    let exact_cap = coded_len
        + job_count * size_of::<J2kClassicCleanupBatchJob>()
        + segment_count * size_of::<J2kClassicSegment>();
    let owners = allocate_distinct_classic_metadata(
        coded_len,
        job_count,
        segment_count,
        BatchMetadataBudget::with_cap(
            "classic J2K MetalDirect distinct color submission",
            exact_cap,
        ),
    )
    .expect("exact distinct classic metadata cap");
    assert_eq!(owners.coded_data.capacity(), coded_len);
    assert_eq!(owners.jobs.capacity(), job_count);
    assert_eq!(owners.segments.capacity(), segment_count);

    assert!(matches!(
        allocate_distinct_classic_metadata(
            coded_len,
            job_count,
            segment_count,
            BatchMetadataBudget::with_cap(
                "classic J2K MetalDirect distinct color submission",
                exact_cap - 1,
            ),
        ),
        Err(Error::BatchInfrastructure(
            BatchInfrastructureError::AllocationTooLarge {
                requested,
                cap,
                ..
            }
        )) if requested == exact_cap && cap == exact_cap - 1
    ));
}
