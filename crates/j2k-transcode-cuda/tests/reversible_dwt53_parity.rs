// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Bit-exact parity: CUDA reversible 5/3 DCT->wavelet transcode vs the
//j2k-transcode scalar oracle. Mirrors j2k-transcode-metal's dct53.rs.
//
// Compiled only with the `cuda-runtime` feature; asserts only on the CUDA runner
// (J2K_REQUIRE_CUDA_RUNTIME set), matching the HTJ2K encode parity gate.
#![cfg(feature = "cuda-runtime")]

use j2k_test_support::cuda_runtime_required;
use j2k_transcode::accelerator::{
    DctGridToReversibleDwt53Job, DctToWaveletStageAccelerator, RayonReversibleDwt53Accelerator,
};
use j2k_transcode_cuda::CudaDctToWaveletStageAccelerator;

/// Deterministic small signed "DCT" coefficients (the transcode does an exact
/// integer IDCT, so any integer input exercises the bit-exact path).
fn make_blocks(block_cols: usize, block_rows: usize) -> Vec<[i16; 64]> {
    let mut blocks = vec![[0i16; 64]; block_cols * block_rows];
    for (bi, block) in blocks.iter_mut().enumerate() {
        for (i, coeff) in block.iter_mut().enumerate() {
            *coeff = i16::try_from((bi * 31 + i * 7) % 193).unwrap_or(0) - 96;
        }
    }
    blocks
}

#[test]
fn cuda_reversible_dwt53_matches_scalar_oracle_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    // (block_cols, block_rows, width, height) including non-multiple-of-8 and
    // odd dimensions to exercise the 5/3 boundary cases.
    let cases = [
        (1usize, 1usize, 8usize, 8usize),
        (2, 2, 16, 16),
        (3, 2, 24, 16),
        (2, 2, 15, 13),
        (2, 3, 16, 23),
    ];

    for (block_cols, block_rows, width, height) in cases {
        let blocks = make_blocks(block_cols, block_rows);
        let job = DctGridToReversibleDwt53Job {
            dequantized_blocks: &blocks,
            block_cols,
            block_rows,
            width,
            height,
        };

        let actual = CudaDctToWaveletStageAccelerator::new_explicit()
            .dct_grid_to_reversible_dwt53(job)
            .expect("CUDA reversible 5/3 dispatch should succeed on the runner")
            .expect("CUDA should handle the reversible 5/3 job (explicit mode)");

        let expected = RayonReversibleDwt53Accelerator::default()
            .dct_grid_to_reversible_dwt53(job)
            .expect("scalar reversible 5/3 oracle accepts the job")
            .expect("scalar oracle handles the job");

        assert_eq!(
            actual, expected,
            "reversible 5/3 mismatch for {width}x{height} ({block_cols}x{block_rows} blocks)"
        );
    }
}
