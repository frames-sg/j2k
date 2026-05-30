// SPDX-License-Identifier: Apache-2.0
//
// Fused code-block parity (Gap C): the CUDA staged 9/7 + deadzone quantization
// batch must match the shared signinum-transcode code-block oracle in layout
// exactly and in quantized coefficients within +/-1 (device math is f32 vs the
// oracle's f64 at deadzone boundaries), and must report all five stage timings.
//
// Compiled only with `cuda-runtime`; asserts only on the CUDA runner.
#![cfg(feature = "cuda-runtime")]

use signinum_transcode::accelerator::{
    DctGridToHtj2k97CodeBlockJob, DctToWaveletStageAccelerator, Htj2k97CodeBlockOptions,
    PrequantizedHtj2k97Component,
};
use signinum_transcode::dct97_2d::dct8x8_blocks_then_dwt97_float;
use signinum_transcode::htj2k97_codeblock_oracle::prequantized_component_from_dwt97;
use signinum_transcode_cuda::CudaDctToWaveletStageAccelerator;

fn runtime_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
}

/// Deterministic small f64 DCT coefficients, varied per job by `salt`.
fn make_blocks(block_cols: usize, block_rows: usize, salt: usize) -> Vec<[[f64; 8]; 8]> {
    let mut blocks = vec![[[0.0f64; 8]; 8]; block_cols * block_rows];
    for (bi, block) in blocks.iter_mut().enumerate() {
        for (fy, row) in block.iter_mut().enumerate() {
            for (fx, coeff) in row.iter_mut().enumerate() {
                *coeff = (((bi * 7 + fy * 8 + fx * 3 + salt) % 23) as f64) - 11.0;
            }
        }
    }
    blocks
}

/// Code-block geometry must match exactly: same component/resolution/subband
/// nesting, code-block counts, declared bitplanes, and code-block dimensions.
fn assert_layout_eq(actual: &PrequantizedHtj2k97Component, expected: &PrequantizedHtj2k97Component) {
    assert_eq!(actual.x_rsiz, expected.x_rsiz);
    assert_eq!(actual.y_rsiz, expected.y_rsiz);
    assert_eq!(actual.resolutions.len(), expected.resolutions.len());
    for (actual_res, expected_res) in actual.resolutions.iter().zip(expected.resolutions.iter()) {
        assert_eq!(actual_res.subbands.len(), expected_res.subbands.len());
        for (actual_sub, expected_sub) in actual_res.subbands.iter().zip(expected_res.subbands.iter())
        {
            assert_eq!(actual_sub.sub_band_type, expected_sub.sub_band_type);
            assert_eq!(actual_sub.num_cbs_x, expected_sub.num_cbs_x);
            assert_eq!(actual_sub.num_cbs_y, expected_sub.num_cbs_y);
            assert_eq!(actual_sub.total_bitplanes, expected_sub.total_bitplanes);
            assert_eq!(actual_sub.code_blocks.len(), expected_sub.code_blocks.len());
            for (actual_block, expected_block) in
                actual_sub.code_blocks.iter().zip(expected_sub.code_blocks.iter())
            {
                assert_eq!(actual_block.width, expected_block.width);
                assert_eq!(actual_block.height, expected_block.height);
                assert_eq!(actual_block.coefficients.len(), expected_block.coefficients.len());
            }
        }
    }
}

/// Quantized coefficients must agree within `max_abs_error` (f32 GPU vs f64 oracle).
fn assert_coefficients_close(
    actual: &PrequantizedHtj2k97Component,
    expected: &PrequantizedHtj2k97Component,
    max_abs_error: i32,
) {
    for (actual_res, expected_res) in actual.resolutions.iter().zip(expected.resolutions.iter()) {
        for (actual_sub, expected_sub) in actual_res.subbands.iter().zip(expected_res.subbands.iter())
        {
            for (actual_block, expected_block) in
                actual_sub.code_blocks.iter().zip(expected_sub.code_blocks.iter())
            {
                for (&actual_coeff, &expected_coeff) in actual_block
                    .coefficients
                    .iter()
                    .zip(expected_block.coefficients.iter())
                {
                    assert!(
                        (actual_coeff - expected_coeff).abs() <= max_abs_error,
                        "quantized coefficient diverged: actual {actual_coeff}, expected {expected_coeff}"
                    );
                }
            }
        }
    }
}

#[test]
fn cuda_htj2k97_codeblock_batch_matches_oracle_when_required() {
    if !runtime_required() {
        return;
    }

    // Non-multiple-of-8 dimensions to exercise partial code-blocks and bands.
    let (block_cols, block_rows, width, height) = (4usize, 4usize, 29usize, 31usize);
    let first = make_blocks(block_cols, block_rows, 0);
    let second = make_blocks(block_cols, block_rows, 37);
    let jobs = [
        DctGridToHtj2k97CodeBlockJob {
            blocks: &first,
            block_cols,
            block_rows,
            width,
            height,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        DctGridToHtj2k97CodeBlockJob {
            blocks: &second,
            block_cols,
            block_rows,
            width,
            height,
            x_rsiz: 1,
            y_rsiz: 1,
        },
    ];
    let options = Htj2k97CodeBlockOptions {
        bit_depth: 8,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        irreversible_quantization_scale: 2.5,
    };

    let mut accelerator = CudaDctToWaveletStageAccelerator::new_explicit();
    let actual = accelerator
        .dct_grid_to_htj2k97_codeblock_batch(&jobs, options)
        .expect("CUDA code-block batch dispatch should succeed on the runner")
        .expect("CUDA should handle the code-block batch (explicit mode)");

    assert_eq!(actual.len(), jobs.len());
    for (job, component) in jobs.iter().zip(actual.iter()) {
        let dwt = dct8x8_blocks_then_dwt97_float(
            job.blocks,
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
        )
        .expect("scalar 9/7 oracle accepts the job");
        let expected = prequantized_component_from_dwt97(&dwt, options, job.x_rsiz, job.y_rsiz);
        assert_layout_eq(component, &expected);
        assert_coefficients_close(component, &expected, 1);
    }

    let timings = accelerator
        .last_dwt97_batch_stage_timings()
        .expect("CUDA code-block batch records backend stage timings");
    assert!(timings.pack_upload_us > 0, "pack/upload stage not timed");
    assert!(timings.idct_row_lift_us > 0, "idct+row-lift stage not timed");
    assert!(timings.column_lift_us > 0, "column-lift stage not timed");
    assert!(timings.quantize_codeblock_us > 0, "quantize stage not timed");
    assert!(timings.readback_us > 0, "readback stage not timed");
}
