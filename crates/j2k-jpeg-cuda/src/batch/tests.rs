// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CudaJpegBatch;
use crate::{CudaSession, Error};

#[test]
fn batch_preserves_order_and_owned_iteration() {
    let session = CudaSession::default();
    let mut batch = CudaJpegBatch::try_with_capacity(&session, 3, "test batch").unwrap();
    batch.try_push(2).unwrap();
    batch.try_push(4).unwrap();
    batch.try_push(6).unwrap();

    assert_eq!(batch.as_slice(), [2, 4, 6]);
    assert_eq!(batch.into_iter().collect::<Vec<_>>(), [2, 4, 6]);
}

#[test]
fn batch_rejects_growth_past_its_leased_capacity_without_mutation() {
    let session = CudaSession::default();
    let mut batch = CudaJpegBatch::try_with_capacity(&session, 1, "test batch").unwrap();
    batch.try_push(7_u64).unwrap();

    let error = batch
        .try_push(9)
        .expect_err("fixed-capacity batch growth must fail");

    assert!(matches!(
        error,
        Error::BatchCapacityExceeded {
            capacity: 1,
            what: "test batch",
        }
    ));
    assert_eq!(batch.as_slice(), [7]);
    assert_eq!(batch.items.capacity(), 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn batch_capacity_is_leased_until_the_batch_drops() {
    let session = CudaSession::default();
    let batch = CudaJpegBatch::<u64>::try_with_capacity(&session, 3, "test batch").unwrap();
    let expected = j2k_core::host_capacity_bytes::<u64>(batch.items.capacity());

    assert_eq!(
        session
            .owned_cuda_host_memory_diagnostics()
            .unwrap()
            .active_owner_bytes,
        expected
    );
    drop(batch);
    assert_eq!(
        session
            .owned_cuda_host_memory_diagnostics()
            .unwrap()
            .active_owner_bytes,
        0
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn owning_iterator_keeps_capacity_leased_until_the_iterator_drops() {
    let session = CudaSession::default();
    let mut batch = CudaJpegBatch::<u64>::try_with_capacity(&session, 2, "test batch").unwrap();
    batch.try_push(3).unwrap();
    batch.try_push(5).unwrap();
    let expected = j2k_core::host_capacity_bytes::<u64>(batch.items.capacity());

    let mut iter = batch.into_iter();
    assert_eq!(iter.next(), Some(3));
    assert_eq!(
        session
            .owned_cuda_host_memory_diagnostics()
            .unwrap()
            .active_owner_bytes,
        expected
    );

    drop(iter);
    assert_eq!(
        session
            .owned_cuda_host_memory_diagnostics()
            .unwrap()
            .active_owner_bytes,
        0
    );
}
