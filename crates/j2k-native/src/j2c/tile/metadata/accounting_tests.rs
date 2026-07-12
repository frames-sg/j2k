// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact boundary, allocator reconciliation, and transaction rollback tests.

use super::*;
use crate::error::{DecodeError, DecodingError};

#[test]
fn transient_peak_accepts_exact_cap_and_rejects_one_over() {
    validate_transient_peak(4, 4, 8).expect("exact live peak");
    assert_eq!(
        validate_transient_peak(4, 5, 8),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
    assert_eq!(
        TileMetadataBudget::with_cap(9, 8).expect_err("baseline one over"),
        DecodeError::Validation(ValidationError::ImageTooLarge)
    );
}

#[test]
fn allocator_overcapacity_reconciles_final_owner_before_failing_peak() {
    let live_before = 6;
    let old_bytes = 2;
    let planned_bytes = 4;
    let actual_bytes = 5;
    validate_transient_peak(live_before, planned_bytes, 10).expect("planned exact peak");

    let live_after = checked_replacement_bytes(live_before, old_bytes, actual_bytes)
        .expect("actual owner reconciliation");
    assert_eq!(live_after, 9);
    assert_eq!(
        validate_transient_peak(live_before, actual_bytes, 10),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
}

#[test]
fn failed_reserve_keeps_existing_capacity_and_ledger_in_sync() {
    let mut budget = TileMetadataBudget::with_cap(3, 32).expect("test budget");
    let mut values = Vec::<u8>::new();
    let error = budget
        .try_reserve_accounted_with(&mut values, 4, |_values, _target_len| {
            Err(DecodingError::HostAllocationFailed.into())
        })
        .expect_err("simulated allocator failure");

    assert_eq!(
        error,
        DecodeError::Decoding(DecodingError::HostAllocationFailed)
    );
    assert_eq!(values.capacity(), 0);
    assert_eq!(budget.retained_bytes(), 3);
}

#[test]
fn replacement_transfers_new_claim_and_releases_old_capacity() {
    let mut budget = TileMetadataBudget::with_cap(0, 1_024).expect("test budget");
    let mut destination = Vec::new();
    budget
        .try_reserve_retained(&mut destination, 2)
        .expect("old owner");
    destination.extend_from_slice(&[1_u8, 2]);
    let old_capacity = destination.capacity();

    let replacement_capacity;
    {
        let mut transaction = budget.transaction();
        let mut replacement = Vec::new();
        transaction
            .try_reserve_temporary(&mut replacement, 5)
            .expect("replacement owner");
        replacement.extend_from_slice(&[3_u8, 4, 5, 6, 7]);
        replacement_capacity = replacement.capacity();
        transaction
            .replace_owner::<u8, _>(&mut destination, replacement, Vec::capacity)
            .expect("claim transfer");
    }

    assert_ne!(old_capacity, 0);
    assert_eq!(destination, [3, 4, 5, 6, 7]);
    assert_eq!(budget.retained_bytes(), replacement_capacity);
}
