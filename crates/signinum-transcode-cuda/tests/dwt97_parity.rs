// SPDX-License-Identifier: Apache-2.0
//
// Tolerance parity: CUDA irreversible 9/7 DCT->wavelet transcode vs the
// signinum-transcode scalar float oracle. Mirrors signinum-transcode-metal's
// dct97.rs (band max-abs-diff <= 2.0e-2; device math is f32).
//
// Compiled only with `cuda-runtime`; asserts only on the CUDA runner.
#![cfg(feature = "cuda-runtime")]

use signinum_test_support::cuda_runtime_required;
use signinum_transcode::accelerator::{DctGridToDwt97Job, DctToWaveletStageAccelerator};
use signinum_transcode::dct97_2d::{dct8x8_blocks_then_dwt97_float, Dwt97TwoDimensional};
use signinum_transcode_cuda::CudaDctToWaveletStageAccelerator;

const TOLERANCE: f64 = 2.0e-2;

/// Deterministic small f64 DCT coefficients.
fn make_blocks(block_cols: usize, block_rows: usize) -> Vec<[[f64; 8]; 8]> {
    let mut blocks = vec![[[0.0f64; 8]; 8]; block_cols * block_rows];
    for (bi, block) in blocks.iter_mut().enumerate() {
        for (fy, row) in block.iter_mut().enumerate() {
            for (fx, coeff) in row.iter_mut().enumerate() {
                *coeff = (((bi * 7 + fy * 8 + fx * 3) % 23) as f64) - 11.0;
            }
        }
    }
    blocks
}

fn band_max_diff(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn max_abs_diff(actual: &Dwt97TwoDimensional<f64>, expected: &Dwt97TwoDimensional<f64>) -> f64 {
    band_max_diff(&actual.ll, &expected.ll)
        .max(band_max_diff(&actual.hl, &expected.hl))
        .max(band_max_diff(&actual.lh, &expected.lh))
        .max(band_max_diff(&actual.hh, &expected.hh))
}

#[test]
fn cuda_dwt97_matches_scalar_oracle_within_tolerance_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let cases = [
        (1usize, 1usize, 8usize, 8usize),
        (2, 2, 16, 16),
        (3, 2, 24, 16),
        (2, 2, 15, 13),
        (2, 3, 16, 23),
    ];

    for (block_cols, block_rows, width, height) in cases {
        let blocks = make_blocks(block_cols, block_rows);
        let job = DctGridToDwt97Job {
            blocks: &blocks,
            block_cols,
            block_rows,
            width,
            height,
        };

        let actual = CudaDctToWaveletStageAccelerator::new_explicit()
            .dct_grid_to_dwt97(job)
            .expect("CUDA 9/7 dispatch should succeed on the runner")
            .expect("CUDA should handle the 9/7 job (explicit mode)");

        let expected =
            dct8x8_blocks_then_dwt97_float(&blocks, block_cols, block_rows, width, height)
                .expect("scalar 9/7 oracle accepts the job");

        let diff = max_abs_diff(&actual, &expected);
        assert!(
            diff <= TOLERANCE,
            "9/7 transcode diverged for {width}x{height} ({block_cols}x{block_rows} blocks): {diff}"
        );
    }
}
