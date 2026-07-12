// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex};

use super::{
    CudaPinnedUploadStagingPoolLimits, PinnedUploadStagingAdmission, PinnedUploadStagingPool,
};
use crate::{context::PinnedUploadStaging, CudaError};

fn staging(id: usize, len: usize) -> PinnedUploadStaging {
    PinnedUploadStaging {
        ptr: id as *mut u8,
        len,
    }
}

fn admit(pool: &mut PinnedUploadStagingPool, id: usize, len: usize) {
    assert_eq!(
        pool.admission(len).expect("classify staging admission"),
        PinnedUploadStagingAdmission::Admit
    );
    pool.begin_new_active_checkout(len)
        .expect("begin staging checkout");
    pool.try_admit_active(staging(id, len))
        .unwrap_or_else(|(error, _)| panic!("admit staging: {error}"));
}

#[test]
fn exact_byte_and_count_limits_admit_and_one_over_evicts() {
    let mut pool = PinnedUploadStagingPool::with_limits(CudaPinnedUploadStagingPoolLimits {
        max_cached_bytes: 100,
        max_cached_buffers: 2,
    });
    admit(&mut pool, 1, 40);
    admit(&mut pool, 2, 60);
    let diagnostics = pool.diagnostics().expect("inspect diagnostics");
    assert_eq!(diagnostics.cached_buffers, 2);
    assert_eq!(diagnostics.cached_bytes, 100);
    assert_eq!(
        pool.admission(1).expect("classify byte overflow"),
        PinnedUploadStagingAdmission::Evict
    );

    let mut count_pool = PinnedUploadStagingPool::with_limits(CudaPinnedUploadStagingPoolLimits {
        max_cached_bytes: 100,
        max_cached_buffers: 1,
    });
    admit(&mut count_pool, 3, 1);
    assert_eq!(
        count_pool.admission(1).expect("classify count overflow"),
        PinnedUploadStagingAdmission::Evict
    );
}

#[test]
fn retained_cache_plus_current_request_honors_exact_host_cap() {
    let mut pool = PinnedUploadStagingPool::with_limits(CudaPinnedUploadStagingPoolLimits {
        max_cached_bytes: 100,
        max_cached_buffers: 2,
    });
    admit(&mut pool, 1, 16);
    assert!(pool
        .cached_plus_request_fits_host_cap(48, 64)
        .expect("check exact aggregate"));
    assert!(!pool
        .cached_plus_request_fits_host_cap(49, 64)
        .expect("check one-over aggregate"));

    let evicted = pool
        .evict_largest_oldest()
        .expect("evict retained staging")
        .expect("retained staging exists");
    assert_eq!(evicted.len, 16);
    assert!(pool
        .cached_plus_request_fits_host_cap(49, 64)
        .expect("request fits after eviction"));
}

#[test]
fn oversized_candidate_is_rejected_without_displacing_reusable_staging() {
    let mut pool = PinnedUploadStagingPool::with_limits(CudaPinnedUploadStagingPoolLimits {
        max_cached_bytes: 64,
        max_cached_buffers: 2,
    });
    admit(&mut pool, 1, 32);
    assert_eq!(
        pool.admission(65).expect("classify oversized staging"),
        PinnedUploadStagingAdmission::Reject
    );
    pool.note_rejection();
    let diagnostics = pool.diagnostics().expect("inspect diagnostics");
    assert_eq!(diagnostics.cached_buffers, 1);
    assert_eq!(diagnostics.cached_bytes, 32);
    assert_eq!(diagnostics.rejected_buffers, 1);
}

#[test]
fn uncertain_release_quarantine_is_separate_from_bounded_reuse() {
    let mut pool = PinnedUploadStagingPool::with_limits(CudaPinnedUploadStagingPoolLimits {
        max_cached_bytes: 4,
        max_cached_buffers: 1,
    });
    pool.try_retain_after_uncertain_release(staging(1, 10))
        .unwrap_or_else(|(error, _)| panic!("quarantine uncertain staging: {error}"));
    let diagnostics = pool.diagnostics().expect("inspect diagnostics");
    assert_eq!(diagnostics.cached_buffers, 0);
    assert_eq!(diagnostics.cached_bytes, 0);
    assert_eq!(diagnostics.uncertain_buffers, 1);
    assert_eq!(diagnostics.uncertain_bytes, 10);
    assert_eq!(diagnostics.retained_bytes, 10);
    assert_eq!(diagnostics.peak_retained_bytes, 10);
    assert!(matches!(
        pool.admission(1),
        Err(CudaError::StatePoisoned { .. })
    ));
    assert!(matches!(
        pool.take_best_fit(1),
        Err(CudaError::StatePoisoned { .. })
    ));
}

#[test]
fn best_fit_take_and_largest_oldest_eviction_are_deterministic() {
    let limits = CudaPinnedUploadStagingPoolLimits {
        max_cached_bytes: 512,
        max_cached_buffers: 8,
    };
    let mut pool = PinnedUploadStagingPool::with_limits(limits);
    admit(&mut pool, 1, 64);
    admit(&mut pool, 2, 32);
    admit(&mut pool, 3, 64);
    admit(&mut pool, 4, 48);

    let best = pool
        .take_best_fit(40)
        .expect("take best fit")
        .expect("best fit exists");
    assert_eq!(best.ptr as usize, 4);
    let oldest_equal_fit = pool
        .take_best_fit(64)
        .expect("take equal fit")
        .expect("equal fit exists");
    assert_eq!(oldest_equal_fit.ptr as usize, 1);

    let evicted = pool
        .evict_largest_oldest()
        .expect("evict largest staging")
        .expect("eviction victim exists");
    assert_eq!(evicted.ptr as usize, 3);
    assert_eq!(
        pool.diagnostics()
            .expect("inspect diagnostics")
            .evicted_buffers,
        1
    );
}

fn churn_distinct_sizes() -> Vec<usize> {
    let mut pool = PinnedUploadStagingPool::with_limits(CudaPinnedUploadStagingPoolLimits {
        max_cached_bytes: 256,
        max_cached_buffers: 5,
    });
    for id in 1..=4_096usize {
        let len = 1 + ((id * 37) % 128);
        loop {
            match pool.admission(len).expect("classify churn admission") {
                PinnedUploadStagingAdmission::Admit => {
                    pool.begin_new_active_checkout(len)
                        .expect("begin churn checkout");
                    pool.try_admit_active(staging(id, len))
                        .unwrap_or_else(|(error, _)| panic!("admit churn staging: {error}"));
                    break;
                }
                PinnedUploadStagingAdmission::Evict => {
                    pool.evict_largest_oldest()
                        .expect("evict churn staging")
                        .expect("churn eviction victim");
                }
                PinnedUploadStagingAdmission::Reject => panic!("churn candidate must fit alone"),
            }
        }
        let diagnostics = pool.diagnostics().expect("inspect diagnostics");
        assert!(diagnostics.cached_buffers <= diagnostics.limits.max_cached_buffers);
        assert!(diagnostics.cached_bytes <= diagnostics.limits.max_cached_bytes);
    }

    let mut retained = Vec::new();
    while let Some(buffer) = pool.take_best_fit(0).expect("drain by best fit") {
        retained.push(buffer.len);
    }
    retained
}

#[test]
fn long_distinct_size_churn_has_stable_bounded_retention() {
    assert_eq!(churn_distinct_sizes(), churn_distinct_sizes());
}

#[test]
fn byte_accounting_overflow_is_typed() {
    let mut pool = PinnedUploadStagingPool::with_limits(CudaPinnedUploadStagingPoolLimits {
        max_cached_bytes: usize::MAX,
        max_cached_buffers: usize::MAX,
    });
    pool.cached_bytes = usize::MAX;
    let error = pool.admission(1).expect_err("byte overflow must fail");
    assert!(matches!(error, CudaError::InternalInvariant { .. }));
}

#[test]
fn arc_aliases_observe_one_diagnostics_ledger() {
    let owner = Arc::new(Mutex::new(PinnedUploadStagingPool::with_limits(
        CudaPinnedUploadStagingPoolLimits {
            max_cached_bytes: 64,
            max_cached_buffers: 2,
        },
    )));
    let alias = Arc::clone(&owner);
    admit(&mut owner.lock().expect("lock owner"), 1, 32);
    assert!(Arc::ptr_eq(&owner, &alias));
    assert_eq!(
        alias
            .lock()
            .expect("lock alias")
            .diagnostics()
            .expect("inspect alias diagnostics")
            .cached_bytes,
        32
    );
}
