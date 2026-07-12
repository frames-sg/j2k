// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::*;
use crate::j2c::encode::allocation::EncodeAllocationLedger;
use crate::{
    EncodeError, J2kForwardDwt53Level, J2kForwardDwt53Output, PrecomputedHtj2k53Component,
    PrecomputedHtj2k53Image,
};

fn vector_with_capacity<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .expect("small coefficient capacity test allocation");
    values
}

fn retained_coefficient_image() -> Reversible53CoefficientImage {
    let mut ll = vector_with_capacity(5);
    ll.extend([0.0_f32; 4]);
    let mut hl = vector_with_capacity(3);
    hl.extend([1.0_f32; 2]);
    let mut lh = vector_with_capacity(4);
    lh.extend([2.0_f32; 2]);
    let mut hh = vector_with_capacity(6);
    hh.extend([3.0_f32; 1]);
    let mut levels = vector_with_capacity(2);
    levels.push(J2kForwardDwt53Level {
        hl,
        lh,
        hh,
        width: 3,
        height: 3,
        low_width: 2,
        low_height: 2,
        high_width: 1,
        high_height: 1,
    });
    let mut components = vector_with_capacity(2);
    components.push(PrecomputedHtj2k53Component {
        x_rsiz: 1,
        y_rsiz: 1,
        dwt: J2kForwardDwt53Output {
            ll,
            ll_width: 2,
            ll_height: 2,
            levels,
        },
    });
    Reversible53CoefficientImage {
        image: PrecomputedHtj2k53Image {
            width: 3,
            height: 3,
            bit_depth: 8,
            signed: false,
            components,
        },
        use_mct: false,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        guard_bits: 2,
    }
}

#[test]
fn coefficient_tree_baseline_accepts_exact_cap_and_rejects_one_byte_over() {
    let image = retained_coefficient_image();
    let checked = image
        .checked_retained_capacity_bytes()
        .expect("checked coefficient capacity");

    let exact = EncodeAllocationLedger::with_test_cap(checked, checked)
        .expect("exact coefficient baseline cap");
    assert_eq!(exact.live_bytes(), checked);
    let error = EncodeAllocationLedger::with_test_cap(checked, checked - 1)
        .expect_err("one byte over coefficient baseline cap");
    assert_eq!(
        error,
        EncodeError::AllocationTooLarge {
            what: "retained native encode inputs",
            requested: checked,
            cap: checked - 1,
        }
    );
}
