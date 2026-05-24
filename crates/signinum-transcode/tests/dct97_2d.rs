// SPDX-License-Identifier: Apache-2.0

use signinum_transcode::dct97_2d::{
    dct8x8_blocks_then_dwt97_float, dct8x8_blocks_to_dwt97_float_linear_rayon_with_scratch,
    dct8x8_blocks_to_dwt97_float_linear_with_scratch, Dct97GridScratch, Dwt97TwoDimensional,
};

#[test]
fn dct8x8_grid_to_2d_97_matches_reference_for_structured_cases() {
    let blocks = structured_blocks(2, 2);
    let mut scratch = Dct97GridScratch::default();

    for (width, height) in [(8, 8), (13, 11), (16, 16)] {
        let direct = dct8x8_blocks_to_dwt97_float_linear_with_scratch(
            &blocks,
            2,
            2,
            width,
            height,
            &mut scratch,
        )
        .expect("direct 9/7 DCT projection accepts covered grid");
        let reference = dct8x8_blocks_then_dwt97_float(&blocks, 2, 2, width, height)
            .expect("reference 9/7 DCT projection accepts covered grid");

        assert!(
            max_abs_diff(&direct, &reference) < 1.0e-9,
            "direct 9/7 DCT projection diverged for {width}x{height}"
        );
    }
}

#[test]
fn dct8x8_grid_to_2d_97_rayon_matches_scalar_for_structured_cases() {
    let blocks = structured_blocks(32, 32);
    let mut scalar_scratch = Dct97GridScratch::default();
    let mut rayon_scratch = Dct97GridScratch::default();

    for (width, height) in [(8, 8), (13, 11), (16, 16), (224, 224), (255, 241)] {
        let scalar = dct8x8_blocks_to_dwt97_float_linear_with_scratch(
            &blocks,
            32,
            32,
            width,
            height,
            &mut scalar_scratch,
        )
        .expect("scalar 9/7 DCT projection accepts covered grid");
        let rayon = dct8x8_blocks_to_dwt97_float_linear_rayon_with_scratch(
            &blocks,
            32,
            32,
            width,
            height,
            &mut rayon_scratch,
        )
        .expect("rayon 9/7 DCT projection accepts covered grid");

        assert!(
            max_abs_diff(&rayon, &scalar) < 1.0e-12,
            "rayon 9/7 DCT projection diverged from scalar for {width}x{height}"
        );
    }
}

fn max_abs_diff(actual: &Dwt97TwoDimensional<f64>, expected: &Dwt97TwoDimensional<f64>) -> f64 {
    assert_eq!(actual.low_width, expected.low_width);
    assert_eq!(actual.low_height, expected.low_height);
    assert_eq!(actual.high_width, expected.high_width);
    assert_eq!(actual.high_height, expected.high_height);

    actual
        .ll
        .iter()
        .zip(expected.ll.iter())
        .chain(actual.hl.iter().zip(expected.hl.iter()))
        .chain(actual.lh.iter().zip(expected.lh.iter()))
        .chain(actual.hh.iter().zip(expected.hh.iter()))
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

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
