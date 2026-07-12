// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Mutex;

use super::{
    retain_pinned_upload_staging_after_abandoned_checkout,
    retain_pinned_upload_staging_after_lock_poison,
    retain_pinned_upload_staging_after_release_failure, PinnedUploadStagingPool,
};
use crate::{context::PinnedUploadStaging, CudaError};

#[test]
fn poisoned_pool_retains_returned_raw_allocation_wrapper() {
    let pool = Mutex::new(PinnedUploadStagingPool::new());
    pool.lock()
        .expect("lock staging pool")
        .begin_new_active_checkout(17)
        .expect("begin active checkout");
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = pool.lock().expect("lock staging pool");
        panic!("poison staging pool");
    }));
    assert!(panic_result.is_err());

    let Err(poisoned) = pool.lock() else {
        panic!("staging pool must be poisoned");
    };
    let error = retain_pinned_upload_staging_after_lock_poison(
        poisoned,
        PinnedUploadStaging {
            ptr: std::ptr::null_mut(),
            len: 17,
        },
    );
    assert!(matches!(error, CudaError::StatePoisoned { .. }));

    let retained = match pool.lock() {
        Ok(_) => panic!("staging pool must remain poisoned"),
        Err(poisoned) => poisoned.into_inner(),
    };
    let diagnostics = retained.diagnostics().expect("inspect retained staging");
    assert_eq!(diagnostics.cached_buffers, 0);
    assert_eq!(diagnostics.cached_bytes, 0);
    assert_eq!(diagnostics.uncertain_buffers, 1);
    assert_eq!(diagnostics.uncertain_bytes, 17);
    assert_eq!(diagnostics.retained_bytes, 17);
}

#[test]
fn failed_release_retains_wrapper_in_clean_or_poisoned_pool() {
    let clean_pool = Mutex::new(PinnedUploadStagingPool::new());
    retain_pinned_upload_staging_after_release_failure(
        clean_pool.lock(),
        PinnedUploadStaging {
            ptr: std::ptr::null_mut(),
            len: 23,
        },
    )
    .expect("retain in clean pool");
    assert_eq!(
        clean_pool
            .lock()
            .expect("lock clean pool")
            .diagnostics()
            .expect("inspect clean pool")
            .uncertain_bytes,
        23
    );

    let poisoned_pool = Mutex::new(PinnedUploadStagingPool::new());
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = poisoned_pool.lock().expect("lock staging pool");
        panic!("poison staging pool");
    }));
    assert!(panic_result.is_err());
    retain_pinned_upload_staging_after_release_failure(
        poisoned_pool.lock(),
        PinnedUploadStaging {
            ptr: std::ptr::null_mut(),
            len: 29,
        },
    )
    .expect("retain in poisoned pool");
    let Err(poisoned) = poisoned_pool.lock() else {
        panic!("staging pool must remain poisoned");
    };
    assert_eq!(
        poisoned
            .into_inner()
            .diagnostics()
            .expect("inspect poisoned pool")
            .uncertain_bytes,
        29
    );
}

#[test]
fn abandoned_or_unwound_checkout_uses_prepared_quarantine_and_fails_closed() {
    let pool = Mutex::new(PinnedUploadStagingPool::new());
    {
        let mut pool = pool.lock().expect("lock staging pool");
        pool.prepare_unwind_quarantine_slots()
            .expect("reserve unwind quarantine");
        pool.begin_new_active_checkout(31)
            .expect("begin active checkout");
    }
    retain_pinned_upload_staging_after_abandoned_checkout(
        pool.lock(),
        PinnedUploadStaging {
            ptr: std::ptr::null_mut(),
            len: 31,
        },
    );

    let mut pool = pool.lock().expect("lock staging pool");
    let diagnostics = pool.diagnostics().expect("inspect unwind quarantine");
    assert_eq!(diagnostics.uncertain_buffers, 1);
    assert_eq!(diagnostics.uncertain_bytes, 31);
    assert!(matches!(
        pool.take_best_fit(1),
        Err(CudaError::StatePoisoned { .. })
    ));
}
