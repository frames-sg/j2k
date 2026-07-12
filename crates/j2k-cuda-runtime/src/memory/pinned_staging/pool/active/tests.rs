// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    CudaPinnedUploadStagingPoolLimits, PinnedUploadStagingAdmission, PinnedUploadStagingPool,
};
use crate::context::PinnedUploadStaging;

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
fn best_fit_actual_length_drives_transaction_exact_and_one_over() {
    let mut pool = PinnedUploadStagingPool::with_limits(CudaPinnedUploadStagingPoolLimits {
        max_cached_bytes: 64,
        max_cached_buffers: 2,
    });
    admit(&mut pool, 1, 16);
    admit(&mut pool, 2, 48);
    let checkout = pool
        .take_best_fit(40)
        .expect("take best fit")
        .expect("best fit exists");
    assert_eq!(checkout.len, 48);
    let transaction_bytes = pool
        .diagnostics()
        .expect("account checkout transaction")
        .retained_bytes;
    assert_eq!(transaction_bytes, 64);
    assert!(transaction_bytes <= 64);
    assert!(transaction_bytes > 63);
}

#[test]
fn active_checkout_is_visible_in_current_and_peak_diagnostics() {
    let mut pool = PinnedUploadStagingPool::with_limits(CudaPinnedUploadStagingPoolLimits {
        max_cached_bytes: 64,
        max_cached_buffers: 2,
    });
    admit(&mut pool, 1, 48);
    let checkout = pool
        .take_best_fit(40)
        .expect("take active staging")
        .expect("active staging exists");
    assert_eq!(checkout.len, 48);
    let diagnostics = pool.diagnostics().expect("inspect active checkout");
    assert_eq!(diagnostics.cached_bytes, 0);
    assert_eq!(diagnostics.active_buffers, 1);
    assert_eq!(diagnostics.active_bytes, 48);
    assert_eq!(diagnostics.retained_bytes, 48);
    assert_eq!(diagnostics.peak_active_bytes, 48);
    assert_eq!(diagnostics.peak_retained_bytes, 48);
}

#[test]
fn confirmed_new_checkout_updates_actual_retained_high_water() {
    let mut pool = PinnedUploadStagingPool::with_limits(CudaPinnedUploadStagingPoolLimits {
        max_cached_bytes: 64,
        max_cached_buffers: 2,
    });
    pool.begin_new_active_checkout(512)
        .expect("reserve new checkout accounting");
    pool.confirm_new_active_checkout()
        .expect("confirm new checkout allocation");
    let diagnostics = pool.diagnostics().expect("inspect new checkout");
    assert_eq!(diagnostics.active_bytes, 512);
    assert_eq!(diagnostics.retained_bytes, 512);
    assert_eq!(diagnostics.peak_active_bytes, 512);
    assert_eq!(diagnostics.peak_retained_bytes, 512);
}

#[test]
fn unwind_quarantine_reservation_scales_with_live_checkouts() {
    let mut pool = PinnedUploadStagingPool::new();
    for active in 0..3 {
        pool.prepare_unwind_quarantine_slots()
            .expect("reserve quarantine slots");
        assert!(pool.uncertain.capacity() >= active + 2);
        pool.begin_new_active_checkout(1)
            .expect("begin active checkout");
    }
}
