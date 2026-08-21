// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

use j2k_core::BatchInfrastructureError;

use super::packed_cpu_decode_coefficients_in;
use crate::{batch_allocation::BatchMetadataBudget, Error};

const PHASE: &str = "J2K MetalDirect hybrid packed coefficients";

#[test]
fn packed_coefficients_honor_exact_cap_and_charge_actual_capacity() {
    let count = 4;
    let exact_cap = count * size_of::<f32>();
    let mut budget = BatchMetadataBudget::with_cap(PHASE, exact_cap);

    let coefficients =
        packed_cpu_decode_coefficients_in(&mut budget, 1, count).expect("exact coefficient cap");

    assert_eq!(coefficients, vec![0.0; count]);
    assert_eq!(
        budget.live_bytes(),
        coefficients.capacity() * size_of::<f32>()
    );
}

#[test]
fn packed_coefficients_reject_one_byte_under_exact_cap() {
    let count = 4;
    let exact_cap = count * size_of::<f32>();
    let mut budget = BatchMetadataBudget::with_cap(PHASE, exact_cap - 1);

    let error = packed_cpu_decode_coefficients_in(&mut budget, 1, count)
        .expect_err("one byte below coefficient cap");

    assert!(matches!(
        error,
        Error::BatchInfrastructure(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested,
            cap,
        }) if what == PHASE && requested == exact_cap && cap == exact_cap - 1
    ));
    assert_eq!(budget.live_bytes(), 0);
}

#[test]
fn failed_second_coefficient_allocation_preserves_first_owner_and_budget() {
    let first_len = 4;
    let first_bytes = first_len * size_of::<f32>();
    let cap = first_bytes * 2;
    let mut budget = BatchMetadataBudget::with_cap(PHASE, cap);
    let first = packed_cpu_decode_coefficients_in(&mut budget, 1, first_len)
        .expect("first coefficient allocation");
    let charged_after_first = budget.live_bytes();

    let error = packed_cpu_decode_coefficients_in(&mut budget, 1, first_len + 1)
        .expect_err("aggregate coefficient cap");

    assert!(matches!(
        error,
        Error::BatchInfrastructure(BatchInfrastructureError::AllocationTooLarge {
            what: PHASE,
            requested,
            cap: error_cap,
        }) if requested == first_bytes + (first_len + 1) * size_of::<f32>() && error_cap == cap
    ));
    assert_eq!(first, vec![0.0; first_len]);
    assert_eq!(budget.live_bytes(), charged_after_first);
}

#[test]
fn packed_coefficient_count_overflow_is_typed_and_does_not_charge_budget() {
    let mut budget = BatchMetadataBudget::with_cap(PHASE, usize::MAX);

    let error = packed_cpu_decode_coefficients_in(&mut budget, usize::MAX, 2)
        .expect_err("coefficient count overflow");

    assert!(matches!(
        error,
        Error::BatchInfrastructure(BatchInfrastructureError::AllocationTooLarge {
            what: PHASE,
            requested: usize::MAX,
            ..
        })
    ));
    assert_eq!(budget.live_bytes(), 0);
}

#[test]
fn two_individually_valid_buckets_share_one_aggregate_budget() {
    let bucket_len = 4;
    let bucket_bytes = bucket_len * size_of::<f32>();
    let cap = bucket_bytes * 2 - 1;
    let mut budget = BatchMetadataBudget::with_cap(PHASE, cap);
    let first = packed_cpu_decode_coefficients_in(&mut budget, 1, bucket_len)
        .expect("first bucket is individually below cap");

    let error = packed_cpu_decode_coefficients_in(&mut budget, 1, bucket_len)
        .expect_err("two live buckets exceed aggregate cap");

    assert!(matches!(
        error,
        Error::BatchInfrastructure(BatchInfrastructureError::AllocationTooLarge {
            what: PHASE,
            requested,
            cap: error_cap,
        }) if requested == bucket_bytes * 2 && error_cap == cap
    ));
    assert_eq!(first.len(), bucket_len);
    assert_eq!(budget.live_bytes(), first.capacity() * size_of::<f32>());
}
