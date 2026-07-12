// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{BatchResultSlots, DUPLICATE_RESULT, MISSING_RESULT, OUT_OF_RANGE_RESULT};
use crate::JpegToHtj2kError;

fn assert_internal_invariant(error: &JpegToHtj2kError, expected: &'static str) {
    assert!(matches!(
        error,
        JpegToHtj2kError::InternalInvariant { what } if *what == expected
    ));
}

#[test]
fn missing_worker_result_is_an_internal_invariant() {
    let mut results = BatchResultSlots::try_new(2).expect("allocate two result slots");
    results.insert(0, 10).expect("first result is valid");

    assert_internal_invariant(
        &results
            .into_results()
            .expect_err("second result is missing"),
        MISSING_RESULT,
    );
}

#[test]
fn duplicate_worker_result_is_an_internal_invariant() {
    let mut results = BatchResultSlots::try_new(1).expect("allocate one result slot");
    results.insert(0, 10).expect("first result is valid");

    assert_internal_invariant(
        &results.insert(0, 11).expect_err("result is duplicated"),
        DUPLICATE_RESULT,
    );
}

#[test]
fn out_of_range_worker_result_is_an_internal_invariant() {
    let mut results = BatchResultSlots::try_new(1).expect("allocate one result slot");

    assert_internal_invariant(
        &results.insert(1, 10).expect_err("index exceeds slot count"),
        OUT_OF_RANGE_RESULT,
    );
}

#[test]
fn complete_worker_results_preserve_input_order() {
    let mut results = BatchResultSlots::try_new(2).expect("allocate two result slots");
    results.insert(1, 20).expect("second result is valid");
    results.insert(0, 10).expect("first result is valid");

    assert_eq!(
        results.into_results().expect("all results are present"),
        vec![10, 20]
    );
}
