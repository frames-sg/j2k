// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support/dwt_diff.rs"]
mod dwt_diff;

use dwt_diff::max_abs_diff_53;
use j2k_transcode::{
    dct8x8_blocks_then_dwt53_float, dct8x8_blocks_to_dwt53_float_linear, Dwt53TwoDimensional,
};

#[test]
fn dct8x8_to_single_level_2d_53_matches_reference() {
    let mut block = [[0.0; 8]; 8];
    block[0][0] = 512.0;
    block[0][1] = -31.0;
    block[1][0] = 27.0;
    block[2][3] = 9.0;

    let direct = direct_single_block(&block);
    let reference = reference_single_block(&block);

    assert_eq!(direct.low_width, 4);
    assert_eq!(direct.low_height, 4);
    assert_eq!(direct.high_width, 4);
    assert_eq!(direct.high_height, 4);
    assert!(max_abs_diff_53(&direct, &reference) <= 1.0e-9);
}

#[test]
fn dct8x8_to_2d_53_matches_reference_for_structured_cases() {
    for block in [
        dc_only_block(),
        high_frequency_block(),
        checkerboard_like_block(),
        gradient_like_block(),
    ] {
        let direct = direct_single_block(&block);
        let reference = reference_single_block(&block);

        assert!(
            max_abs_diff_53(&direct, &reference) <= 1.0e-9,
            "max diff {}",
            max_abs_diff_53(&direct, &reference)
        );
    }
}

#[test]
fn dct8x8_grid_to_2d_53_crosses_block_boundaries() {
    let blocks = synthetic_grid_blocks(2, 2);

    let direct =
        dct8x8_blocks_to_dwt53_float_linear(&blocks, 2, 2, 13, 11).expect("valid DCT grid");
    let reference = dct8x8_blocks_then_dwt53_float(&blocks, 2, 2, 13, 11).expect("valid DCT grid");

    assert_eq!(direct.low_width, 7);
    assert_eq!(direct.low_height, 6);
    assert_eq!(direct.high_width, 6);
    assert_eq!(direct.high_height, 5);
    assert!(
        max_abs_diff_53(&direct, &reference) <= 1.0e-9,
        "max diff {}",
        max_abs_diff_53(&direct, &reference)
    );
}

#[test]
fn dct8x8_grid_public_wrapper_is_repeatable_for_same_geometry() {
    let blocks = synthetic_grid_blocks(2, 2);
    let direct =
        dct8x8_blocks_to_dwt53_float_linear(&blocks, 2, 2, 13, 11).expect("valid DCT grid");
    let stateless =
        dct8x8_blocks_to_dwt53_float_linear(&blocks, 2, 2, 13, 11).expect("valid DCT grid");

    let repeated =
        dct8x8_blocks_to_dwt53_float_linear(&blocks, 2, 2, 13, 11).expect("valid DCT grid");

    assert!(max_abs_diff_53(&direct, &stateless) <= 1.0e-9);
    assert!(max_abs_diff_53(&repeated, &stateless) <= 1.0e-9);
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

fn direct_single_block(block: &[[f64; 8]; 8]) -> Dwt53TwoDimensional<f64> {
    dct8x8_blocks_to_dwt53_float_linear(&[*block], 1, 1, 8, 8).expect("valid single DCT block")
}

fn reference_single_block(block: &[[f64; 8]; 8]) -> Dwt53TwoDimensional<f64> {
    dct8x8_blocks_then_dwt53_float(&[*block], 1, 1, 8, 8).expect("valid single DCT block")
}

fn synthetic_grid_blocks(block_cols: usize, block_rows: usize) -> Vec<[[f64; 8]; 8]> {
    let mut blocks = Vec::with_capacity(block_cols * block_rows);
    for block_y in 0..block_rows {
        for block_x in 0..block_cols {
            let mut block = [[0.0; 8]; 8];
            block[0][0] = 192.0 + (block_x * 17 + block_y * 23) as f64;
            block[0][1] = -31.0 + block_x as f64;
            block[1][0] = 27.0 - block_y as f64;
            block[2][3] = 9.0;
            block[7][7] = -6.0;
            blocks.push(block);
        }
    }
    blocks
}
