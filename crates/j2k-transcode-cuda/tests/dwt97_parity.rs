// SPDX-License-Identifier: MIT OR Apache-2.0

//
// Exact parity: the CUDA irreversible 9/7 DCT->wavelet transcode must produce
// the canonical binary32 bands of `j2k_codec_math::transcode97` bit for bit,
// the same bands the CPU and Metal backends produce.
//
// Compiled only with `cuda-runtime`; asserts only on the CUDA runner.
#![cfg(feature = "cuda-runtime")]

use j2k_test_support::cuda_runtime_gate;
use j2k_transcode::accelerator::{DctGridToDwt97Job, DctToWaveletStageAccelerator};
use j2k_transcode::Dwt97TwoDimensional;
use j2k_transcode_cuda::CudaDctToWaveletStageAccelerator;
use j2k_transcode_test_support::canonical_dwt97_from_blocks;

/// Deterministic small f64 DCT coefficients.
#[expect(
    clippy::cast_precision_loss,
    reason = "the synthetic coefficient is reduced modulo 23 before exact f64 conversion"
)]
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

fn assert_dwt97_bits_eq(
    actual: &Dwt97TwoDimensional<f64>,
    expected: &Dwt97TwoDimensional<f64>,
    label: &str,
) {
    assert_eq!(
        (
            actual.low_width,
            actual.low_height,
            actual.high_width,
            actual.high_height
        ),
        (
            expected.low_width,
            expected.low_height,
            expected.high_width,
            expected.high_height
        ),
        "{label}: band geometry"
    );
    for (name, actual, expected) in [
        ("ll", &actual.ll, &expected.ll),
        ("hl", &actual.hl, &expected.hl),
        ("lh", &actual.lh, &expected.lh),
        ("hh", &actual.hh, &expected.hh),
    ] {
        assert_eq!(actual.len(), expected.len(), "{label}: {name} length");
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: {name}[{index}] {a} vs {e}"
            );
        }
    }
}

#[test]
fn cuda_dwt97_matches_canonical_bits_when_required() {
    if !cuda_runtime_gate(module_path!()) {
        return;
    }

    let cases = [
        (1usize, 1usize, 8usize, 8usize),
        (2, 2, 16, 16),
        (3, 2, 24, 16),
        (2, 2, 15, 13),
        (2, 3, 16, 23),
        (1, 2, 1, 9),
        (2, 1, 9, 1),
        (1, 1, 1, 1),
        (130, 2, 1033, 13),
        (2, 138, 9, 1100),
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

        let expected = canonical_dwt97_from_blocks(&blocks, block_cols, block_rows, width, height);
        assert_dwt97_bits_eq(
            &actual,
            &expected,
            &format!("{width}x{height} ({block_cols}x{block_rows} blocks)"),
        );
    }
}
