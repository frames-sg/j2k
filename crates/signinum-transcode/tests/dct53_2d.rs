// SPDX-License-Identifier: Apache-2.0

use signinum_transcode::dct53_2d::{dct8x8_to_dwt53_float_linear, idct8x8_then_dwt53_float};

#[test]
fn dct8x8_to_single_level_2d_53_matches_reference() {
    let mut block = [[0.0; 8]; 8];
    block[0][0] = 512.0;
    block[0][1] = -31.0;
    block[1][0] = 27.0;
    block[2][3] = 9.0;

    let direct = dct8x8_to_dwt53_float_linear(block);
    let reference = idct8x8_then_dwt53_float(block);

    assert_eq!(direct.low_width, 4);
    assert_eq!(direct.low_height, 4);
    assert_eq!(direct.high_width, 4);
    assert_eq!(direct.high_height, 4);
    assert!(direct.max_abs_diff(&reference) <= 1.0e-9);
}

#[test]
fn dct8x8_to_2d_53_matches_reference_for_structured_cases() {
    for block in [
        dc_only_block(),
        high_frequency_block(),
        checkerboard_like_block(),
        gradient_like_block(),
    ] {
        let direct = dct8x8_to_dwt53_float_linear(block);
        let reference = idct8x8_then_dwt53_float(block);

        assert!(
            direct.max_abs_diff(&reference) <= 1.0e-9,
            "max diff {}",
            direct.max_abs_diff(&reference)
        );
    }
}

fn dc_only_block() -> [[f64; 8]; 8] {
    let mut block = [[0.0; 8]; 8];
    block[0][0] = 384.0;
    block
}

fn high_frequency_block() -> [[f64; 8]; 8] {
    let mut block = [[0.0; 8]; 8];
    block[7][7] = 64.0;
    block[6][7] = -31.0;
    block[7][6] = 29.0;
    block
}

fn checkerboard_like_block() -> [[f64; 8]; 8] {
    let mut block = [[0.0; 8]; 8];
    for (y, row) in block.iter_mut().enumerate() {
        for (x, coeff) in row.iter_mut().enumerate() {
            if (x + y) % 2 == 0 {
                *coeff = 8.0;
            } else {
                *coeff = -8.0;
            }
        }
    }
    block
}

fn gradient_like_block() -> [[f64; 8]; 8] {
    let mut block = [[0.0; 8]; 8];
    block[0][0] = 256.0;
    block[0][1] = -48.0;
    block[1][0] = 36.0;
    block[0][2] = 12.0;
    block[2][0] = -9.0;
    block
}
