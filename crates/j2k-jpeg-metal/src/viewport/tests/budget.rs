// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

use j2k_core::BatchInfrastructureError;

use super::*;

#[test]
fn cpu_viewport_live_budget_honors_exact_cap_and_one_byte_over() {
    let tile_capacity = 3;
    let viewport_len = 17;
    let tile_scratch_len = 11;
    let exact_cap = tile_capacity * size_of::<ViewportTile>() + viewport_len + tile_scratch_len;

    cpu_viewport_allocation_budget_with_cap(
        tile_capacity,
        viewport_len,
        tile_scratch_len,
        exact_cap,
    )
    .expect("exact viewport live-set cap");
    let Err(one_over) = cpu_viewport_allocation_budget_with_cap(
        tile_capacity,
        viewport_len,
        tile_scratch_len,
        exact_cap - 1,
    ) else {
        panic!("viewport live set must reject one byte over cap");
    };
    assert_eq!(
        one_over,
        BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG Metal CPU viewport live allocation",
            requested: exact_cap,
            cap: exact_cap - 1,
        }
    );
}

#[test]
fn cpu_viewport_live_budget_reports_count_overflow() {
    let Err(overflow) = cpu_viewport_allocation_budget_with_cap(usize::MAX, 1, 1, usize::MAX)
    else {
        panic!("viewport tile metadata must reject count overflow");
    };
    assert_eq!(
        overflow,
        BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG Metal CPU viewport live allocation",
            requested: usize::MAX,
            cap: usize::MAX,
        }
    );
}
