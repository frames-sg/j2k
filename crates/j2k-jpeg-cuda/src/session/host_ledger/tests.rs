// SPDX-License-Identifier: MIT OR Apache-2.0

use super::SharedHostLedger;
use crate::Error;
use std::sync::atomic::Ordering;

#[test]
fn exact_reservation_releases_and_one_over_never_mutates() {
    let ledger = SharedHostLedger::new();
    let lease = ledger.reserve(7, 5, 12, "test ledger").unwrap();
    assert_eq!(ledger.active_bytes().unwrap(), 5);
    drop(lease);
    assert_eq!(ledger.active_bytes().unwrap(), 0);
    assert_eq!(ledger.diagnostics().unwrap().peak_active_bytes, 5);
    assert_eq!(ledger.diagnostics().unwrap().peak_combined_bytes, 12);
    assert!(matches!(
        ledger.reserve(7, 5, 11, "test ledger"),
        Err(Error::HostAllocationTooLarge {
            requested: 12,
            cap: 11,
            ..
        })
    ));
    assert_eq!(ledger.active_bytes().unwrap(), 0);
    assert_eq!(ledger.diagnostics().unwrap().peak_active_bytes, 5);
    assert_eq!(ledger.diagnostics().unwrap().peak_combined_bytes, 12);
}

#[test]
fn impossible_release_poison_is_fail_closed_without_panicking() {
    let ledger = SharedHostLedger::new();
    let lease = ledger.reserve(0, 1, 1, "test ledger").unwrap();
    ledger.active_bytes.store(0, Ordering::Release);
    drop(lease);
    assert!(matches!(
        ledger.active_bytes(),
        Err(Error::InFlightHostLedgerPoisoned)
    ));
    assert!(matches!(
        ledger.reserve(0, 0, usize::MAX, "test ledger"),
        Err(Error::InFlightHostLedgerPoisoned)
    ));
}
