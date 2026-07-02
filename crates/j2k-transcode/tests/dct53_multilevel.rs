// SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(clippy::large_types_passed_by_value, dead_code, unreachable_pub)]
#[path = "../src/dct53_2d.rs"]
mod dct53_2d;
#[allow(dead_code, unreachable_pub, unused_imports)]
#[path = "../src/dct_grid.rs"]
mod dct_grid;

pub use dct_grid::DctGridError;

#[allow(clippy::large_types_passed_by_value, dead_code, unreachable_pub)]
#[path = "support/dct53_multilevel.rs"]
mod dct53_multilevel;

use dct53_multilevel::{
    dct8x8_to_dwt53_multilevel_float_linear, idct8x8_then_dwt53_multilevel_float,
};

#[test]
fn dct8x8_multilevel_uses_direct_level_one_and_conventional_ll_recursion() {
    let mut block = [[0.0; 8]; 8];
    block[0][0] = 512.0;
    block[0][1] = -31.0;
    block[1][0] = 27.0;
    block[2][3] = 9.0;
    block[7][7] = -6.0;

    let direct = dct8x8_to_dwt53_multilevel_float_linear(block, 2).expect("valid levels");
    let reference = idct8x8_then_dwt53_multilevel_float(block, 2).expect("valid levels");

    assert_eq!(direct.levels.len(), 2);
    assert_eq!(direct.final_ll_width, 2);
    assert_eq!(direct.final_ll_height, 2);
    assert!(direct.max_abs_diff(&reference) <= 1.0e-9);
}

#[test]
fn dct8x8_multilevel_rejects_zero_levels() {
    let err = dct8x8_to_dwt53_multilevel_float_linear([[0.0; 8]; 8], 0).unwrap_err();

    assert_eq!(err.requested_levels(), 0);
}
