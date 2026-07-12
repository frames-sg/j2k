// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn capped_bytes_reject_growth_past_the_budget_without_allocating() {
    let mut bytes = CappedBytes::new(2);
    bytes.extend_from_slice(&[1, 2]).expect("within cap");
    let error = bytes.push(3).expect_err("third byte exceeds cap");
    assert!(matches!(
        error,
        JpegEncodeError::MemoryCapExceeded { requested: 3, .. }
    ));
}

#[test]
fn impossible_initial_capacity_is_a_typed_allocation_failure() {
    let error = CappedBytes::try_with_capacity(usize::MAX, usize::MAX)
        .expect_err("impossible vector capacity");
    assert!(matches!(
        error,
        JpegEncodeError::HostAllocationFailed { bytes: usize::MAX }
    ));
}

#[test]
fn geometric_growth_counts_retained_and_replacement_storage() {
    let mut bytes = CappedBytes::new(12);
    bytes
        .extend_from_slice(&[0; 8])
        .expect("initial allocation fits");

    assert!(matches!(
        bytes.push(9),
        Err(JpegEncodeError::MemoryCapExceeded { requested, cap: 12 })
            if requested > 12
    ));
}

#[test]
fn allocator_reported_capacity_is_checked_after_reservation() {
    assert!(matches!(
        ensure_capacity_within_limit(65, 64),
        Err(JpegEncodeError::MemoryCapExceeded {
            requested: 65,
            cap: 64,
        })
    ));
}

#[test]
fn encoded_output_module_stays_focused() {
    const SOURCE: &str = include_str!("../encoded_output.rs");
    assert!(
        SOURCE.lines().count() <= 160,
        "encoded output storage should be split before it exceeds 160 lines"
    );
}
