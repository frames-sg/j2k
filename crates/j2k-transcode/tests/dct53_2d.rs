// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_transcode::dct53_2d::{
    dct8x8_blocks_then_dwt53_float, dct8x8_blocks_to_dwt53_float_linear,
    dct8x8_blocks_to_dwt53_float_linear_with_scratch, dct8x8_to_dwt53_float_linear,
    idct8x8_then_dwt53_float, Dct53GridScratch,
};

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
        direct.max_abs_diff(&reference) <= 1.0e-9,
        "max diff {}",
        direct.max_abs_diff(&reference)
    );
}

#[test]
fn dct8x8_grid_scratch_reuses_weight_rows_for_same_geometry() {
    let blocks = synthetic_grid_blocks(2, 2);
    let mut scratch = Dct53GridScratch::default();

    let direct =
        dct8x8_blocks_to_dwt53_float_linear_with_scratch(&blocks, 2, 2, 13, 11, &mut scratch)
            .expect("valid DCT grid");
    let stateless =
        dct8x8_blocks_to_dwt53_float_linear(&blocks, 2, 2, 13, 11).expect("valid DCT grid");
    let capacity_after_first = scratch.weight_row_capacity();

    let repeated =
        dct8x8_blocks_to_dwt53_float_linear_with_scratch(&blocks, 2, 2, 13, 11, &mut scratch)
            .expect("valid DCT grid");

    assert!(capacity_after_first > 0);
    assert_eq!(scratch.weight_row_capacity(), capacity_after_first);
    assert!(direct.max_abs_diff(&stateless) <= 1.0e-9);
    assert!(repeated.max_abs_diff(&stateless) <= 1.0e-9);
}

#[test]
fn dct8x8_grid_scratch_uses_sparse_weight_rows_for_wsi_tile() {
    let dim = 224_usize;
    let block_cols = dim / 8;
    let block_rows = dim / 8;
    let blocks = vec![[[0.0; 8]; 8]; block_cols * block_rows];
    let mut scratch = Dct53GridScratch::default();

    dct8x8_blocks_to_dwt53_float_linear_with_scratch(
        &blocks,
        block_cols,
        block_rows,
        dim,
        dim,
        &mut scratch,
    )
    .expect("valid DCT grid");

    assert!(
        scratch.weight_row_capacity() <= dim * 10,
        "5/3 grid weights should stay sparse at WSI tile sizes, got capacity {}",
        scratch.weight_row_capacity()
    );
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
