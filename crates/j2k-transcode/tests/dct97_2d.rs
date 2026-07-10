// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_transcode::dct8x8_blocks_then_dwt97_float;
use j2k_transcode_test_support::max_abs_diff_97;

#[test]
fn dct8x8_grid_to_2d_97_public_path_matches_reference_for_structured_cases() {
    let blocks = structured_blocks(2, 2);

    for (width, height) in [(8, 8), (13, 11), (16, 16)] {
        let public_path = dct8x8_blocks_then_dwt97_float(&blocks, 2, 2, width, height)
            .expect("public 9/7 IDCT path accepts covered grid");
        let reference = dct8x8_blocks_then_dwt97_float(&blocks, 2, 2, width, height)
            .expect("reference 9/7 IDCT path accepts covered grid");

        assert!(
            max_abs_diff_97(&public_path, &reference) < 1.0e-9,
            "public 9/7 IDCT path diverged for {width}x{height}"
        );
    }
}

#[test]
fn dct8x8_grid_to_2d_97_public_path_matches_reference_after_large_grid() {
    let large_blocks = structured_blocks(32, 32);
    let small_blocks = structured_blocks(2, 2);

    let large = dct8x8_blocks_then_dwt97_float(&large_blocks, 32, 32, 255, 241)
        .expect("public 9/7 IDCT path accepts covered large grid");
    let expected_large = dct8x8_blocks_then_dwt97_float(&large_blocks, 32, 32, 255, 241)
        .expect("reference 9/7 IDCT path accepts covered large grid");

    let small = dct8x8_blocks_then_dwt97_float(&small_blocks, 2, 2, 13, 11)
        .expect("public 9/7 IDCT path accepts covered small grid");
    let expected_small = dct8x8_blocks_then_dwt97_float(&small_blocks, 2, 2, 13, 11)
        .expect("reference 9/7 IDCT path accepts covered small grid");

    assert!(
        max_abs_diff_97(&large, &expected_large) < 1.0e-9,
        "public 9/7 IDCT path diverged for large grid"
    );
    assert!(
        max_abs_diff_97(&small, &expected_small) < 1.0e-9,
        "public 9/7 IDCT path diverged for small grid"
    );
}

#[expect(
    clippy::cast_precision_loss,
    reason = "small deterministic test-grid indices are exactly representable in f64"
)]
fn structured_blocks(block_cols: usize, block_rows: usize) -> Vec<[[f64; 8]; 8]> {
    let mut blocks = Vec::with_capacity(block_cols * block_rows);
    for block_y in 0..block_rows {
        for block_x in 0..block_cols {
            let mut block = [[0.0; 8]; 8];
            block[0][0] = 384.0 + (block_x * 19 + block_y * 23) as f64;
            block[0][1] = -17.0 + block_x as f64;
            block[1][0] = 11.0 - block_y as f64;
            block[2][3] = 7.0;
            block[4][4] = -3.0;
            block[7][7] = 2.0;
            blocks.push(block);
        }
    }
    blocks
}
