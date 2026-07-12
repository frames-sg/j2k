// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::error::JpegError;
use crate::internal::checkpoint::{
    checked_actual_checkpoint_live_bytes, checked_checkpoint_reservation_peak,
    checked_checkpoint_workspace_bytes, host_allocation_error,
    reconcile_actual_checkpoint_capacity, reserve_checkpoint_capacity, try_checkpoint_vec,
    CpuCheckpointCache, DeviceCheckpoint,
};

#[test]
fn checkpoint_and_terminated_reader_live_byte_boundary_is_exact() {
    let checkpoint_count = 3;
    let terminated_copy_bytes = 17;
    let requested =
        checkpoint_count * core::mem::size_of::<DeviceCheckpoint>() + terminated_copy_bytes;
    assert_eq!(
        checked_checkpoint_workspace_bytes(checkpoint_count, terminated_copy_bytes, requested),
        Ok(requested)
    );
    assert_eq!(
        checked_checkpoint_workspace_bytes(checkpoint_count, terminated_copy_bytes, requested - 1),
        Err(JpegError::MemoryCapExceeded {
            requested,
            cap: requested - 1,
        })
    );
    assert_eq!(
        checked_checkpoint_workspace_bytes(usize::MAX, 1, usize::MAX),
        Err(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: usize::MAX,
        })
    );
}

#[test]
fn allocator_returned_checkpoint_capacity_is_postchecked() {
    let checkpoint_bytes = core::mem::size_of::<DeviceCheckpoint>();
    assert_eq!(
        checked_actual_checkpoint_live_bytes::<DeviceCheckpoint>(
            checkpoint_bytes,
            2,
            checkpoint_bytes * 2,
        ),
        Err(JpegError::MemoryCapExceeded {
            requested: checkpoint_bytes * 3,
            cap: checkpoint_bytes * 2,
        })
    );
}

#[test]
fn retained_checkpoint_cache_reduces_the_next_workspace_boundary() {
    let checkpoint_bytes = core::mem::size_of::<DeviceCheckpoint>();
    let mut cache = CpuCheckpointCache::default();
    reserve_checkpoint_capacity(&mut cache.checkpoints, 3, 0, 3 * checkpoint_bytes)
        .expect("cache reservation fits its boundary");
    let retained = cache
        .retained_allocation_bytes()
        .expect("retained checkpoint capacity fits usize");
    assert_eq!(retained, cache.checkpoints.capacity() * checkpoint_bytes);

    let terminated_copy_bytes = 5;
    let new_workspace_bytes = checkpoint_bytes + terminated_copy_bytes;
    let aggregate_cap = retained + new_workspace_bytes;
    assert_eq!(
        checked_checkpoint_workspace_bytes(1, terminated_copy_bytes, aggregate_cap - retained),
        Ok(new_workspace_bytes)
    );
    assert_eq!(
        checked_checkpoint_workspace_bytes(1, terminated_copy_bytes, aggregate_cap - retained - 1),
        Err(JpegError::MemoryCapExceeded {
            requested: new_workspace_bytes,
            cap: new_workspace_bytes - 1,
        })
    );
}

#[test]
fn checkpoint_cache_growth_counts_decoder_baseline_old_and_replacement_exactly() {
    let checkpoint_bytes = core::mem::size_of::<DeviceCheckpoint>();
    let baseline = 17;
    let retained_cache = 2 * checkpoint_bytes;
    let replacement_cache = 5 * checkpoint_bytes;
    let exact = baseline + retained_cache + replacement_cache;

    assert_eq!(
        checked_checkpoint_reservation_peak(baseline, retained_cache, replacement_cache, exact,),
        Ok(exact)
    );
    assert_eq!(
        checked_checkpoint_reservation_peak(baseline, retained_cache, replacement_cache, exact - 1,),
        Err(JpegError::MemoryCapExceeded {
            requested: exact,
            cap: exact - 1,
        })
    );
}

#[test]
fn prepopulated_checkpoint_growth_counts_external_output_and_scratch_exactly() {
    let checkpoint_bytes = core::mem::size_of::<DeviceCheckpoint>();
    let decoder_retained = 17;
    let output_capacity = 19;
    let scratch_capacity = 23;
    let phase_baseline = decoder_retained + output_capacity + scratch_capacity;
    let old_capacity = 2;
    let replacement_capacity = 5;
    let exact =
        phase_baseline + old_capacity * checkpoint_bytes + replacement_capacity * checkpoint_bytes;

    let mut exact_cache = Vec::<DeviceCheckpoint>::with_capacity(old_capacity);
    reserve_checkpoint_capacity(
        &mut exact_cache,
        replacement_capacity,
        phase_baseline,
        exact,
    )
    .expect("external owners plus cache replacement fit the exact boundary");
    assert!(exact_cache.capacity() >= replacement_capacity);

    let mut rejected_cache = Vec::<DeviceCheckpoint>::with_capacity(old_capacity);
    let retained_capacity = rejected_cache.capacity();
    assert_eq!(
        reserve_checkpoint_capacity(
            &mut rejected_cache,
            replacement_capacity,
            phase_baseline,
            exact - 1,
        ),
        Err(JpegError::MemoryCapExceeded {
            requested: exact,
            cap: exact - 1,
        })
    );
    assert_eq!(rejected_cache.capacity(), retained_capacity);
}

#[test]
fn failed_actual_cache_postcheck_releases_the_retained_allocation() {
    let mut checkpoints = Vec::<DeviceCheckpoint>::with_capacity(2);
    let actual_bytes = checkpoints.capacity() * core::mem::size_of::<DeviceCheckpoint>();
    let baseline = 1;
    let error = reconcile_actual_checkpoint_capacity(&mut checkpoints, baseline, 0, actual_bytes)
        .expect_err("allocator overcapacity must fail its aggregate postcheck");

    assert_eq!(
        error,
        JpegError::MemoryCapExceeded {
            requested: baseline + actual_bytes,
            cap: actual_bytes,
        }
    );
    assert_eq!(checkpoints.capacity(), 0);
}

#[test]
fn checkpoint_reserve_failure_keeps_its_typed_category() {
    assert_eq!(
        host_allocation_error(8192),
        JpegError::HostAllocationFailed { bytes: 8192 }
    );
}

#[test]
fn checkpoint_reservation_fails_before_exceeding_the_host_cap() {
    let checkpoint_bytes = core::mem::size_of::<DeviceCheckpoint>();
    let error = try_checkpoint_vec(2, checkpoint_bytes)
        .expect_err("two checkpoints must exceed a one-checkpoint cap");
    assert_eq!(
        error,
        JpegError::MemoryCapExceeded {
            requested: checkpoint_bytes * 2,
            cap: checkpoint_bytes,
        }
    );

    let overflow = try_checkpoint_vec(usize::MAX, usize::MAX)
        .expect_err("checkpoint byte calculation must reject overflow");
    assert_eq!(
        overflow,
        JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: usize::MAX,
        }
    );

    let mut cached = Vec::new();
    let error = reserve_checkpoint_capacity(&mut cached, 2, 0, checkpoint_bytes)
        .expect_err("lazy checkpoint growth must use the same host cap");
    assert_eq!(
        error,
        JpegError::MemoryCapExceeded {
            requested: checkpoint_bytes * 2,
            cap: checkpoint_bytes,
        }
    );
    assert!(cached.is_empty());
}
