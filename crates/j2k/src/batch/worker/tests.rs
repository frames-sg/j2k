// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn result_slot_mismatch_is_typed() {
    assert!(matches!(
        ensure_disjoint_result_slots(2, 1),
        Err(BatchInfrastructureError::MissingResult { index: 1 })
    ));
    assert!(matches!(
        ensure_disjoint_result_slots(1, 2),
        Err(BatchInfrastructureError::ResultIndexOutOfBounds {
            index: 1,
            job_count: 1,
        })
    ));
}
