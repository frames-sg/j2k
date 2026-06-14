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
    IrreversibleQuantizationSubbandScales, PrequantizedHtj2k97Component,
};
use signinum_transcode::dct97_2d::dct8x8_blocks_then_dwt97_float;
use signinum_transcode::htj2k97_codeblock_oracle::prequantized_component_from_dwt97;
use signinum_transcode::{JpegTileBatchInput, JpegToHtj2kOptions, JpegToHtj2kTranscoder};
use signinum_transcode_cuda::CudaDctToWaveletStageAccelerator;

use signinum_test_support::{cuda_runtime_required, jpeg_baseline_420_16x16};

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
fn assert_layout_eq(
    actual: &PrequantizedHtj2k97Component,
    expected: &PrequantizedHtj2k97Component,
) {
    assert_eq!(actual.x_rsiz, expected.x_rsiz);
    assert_eq!(actual.y_rsiz, expected.y_rsiz);
    assert_eq!(actual.resolutions.len(), expected.resolutions.len());
    for (actual_res, expected_res) in actual.resolutions.iter().zip(expected.resolutions.iter()) {
        assert_eq!(actual_res.subbands.len(), expected_res.subbands.len());
        for (actual_sub, expected_sub) in
            actual_res.subbands.iter().zip(expected_res.subbands.iter())
        {
            assert_eq!(actual_sub.sub_band_type, expected_sub.sub_band_type);
            assert_eq!(actual_sub.num_cbs_x, expected_sub.num_cbs_x);
            assert_eq!(actual_sub.num_cbs_y, expected_sub.num_cbs_y);
            assert_eq!(actual_sub.total_bitplanes, expected_sub.total_bitplanes);
            assert_eq!(actual_sub.code_blocks.len(), expected_sub.code_blocks.len());
            for (actual_block, expected_block) in actual_sub
                .code_blocks
                .iter()
                .zip(expected_sub.code_blocks.iter())
            {
                assert_eq!(actual_block.width, expected_block.width);
                assert_eq!(actual_block.height, expected_block.height);
                assert_eq!(
                    actual_block.coefficients.len(),
                    expected_block.coefficients.len()
                );
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
        for (actual_sub, expected_sub) in
            actual_res.subbands.iter().zip(expected_res.subbands.iter())
        {
            for (actual_block, expected_block) in actual_sub
                .code_blocks
                .iter()
                .zip(expected_sub.code_blocks.iter())
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
    if !cuda_runtime_required() {
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
        irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales {
            low_low: 0.9,
            high_low: 1.1,
            low_high: 1.2,
            high_high: 1.5,
        },
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
    assert!(
        timings.idct_row_lift_us > 0,
        "idct+row-lift stage not timed"
    );
    assert!(timings.column_lift_us > 0, "column-lift stage not timed");
    assert!(
        timings.quantize_codeblock_us > 0,
        "quantize stage not timed"
    );
    assert!(timings.readback_us > 0, "readback stage not timed");
}

#[test]
fn cuda_htj2k97_codeblock_batch_rejects_non_uniform_geometry_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    // The fused code-block kernels require uniform geometry; a mixed-geometry
    // batch must surface a typed error in Explicit mode (Auto would fall back to
    // the scalar oracle). There is no single-job GPU code-block entry point, so
    // the whole batch is rejected rather than handled per job.
    let block_cols = 4usize;
    let block_rows = 4usize;
    let first = make_blocks(block_cols, block_rows, 0);
    let second = make_blocks(block_cols, block_rows, 37);
    let jobs = [
        DctGridToHtj2k97CodeBlockJob {
            blocks: &first,
            block_cols,
            block_rows,
            width: 29,
            height: 31,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        DctGridToHtj2k97CodeBlockJob {
            blocks: &second,
            block_cols,
            block_rows,
            width: 24,
            height: 26,
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
        irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales::default(),
    };

    let result = CudaDctToWaveletStageAccelerator::new_explicit()
        .dct_grid_to_htj2k97_codeblock_batch(&jobs, options);
    assert!(
        result.is_err(),
        "explicit CUDA code-block batch must reject non-uniform geometry, got {result:?}"
    );
}

#[test]
fn cuda_resident_htj2k97_batch_matches_host_bounce_codestream_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let jpeg = jpeg_baseline_420_16x16();
    let inputs = vec![
        JpegTileBatchInput {
            bytes: jpeg.as_slice(),
        };
        2
    ];
    let options = JpegToHtj2kOptions::lossy_97();

    let mut host_transcoder = JpegToHtj2kTranscoder::default();
    let mut host_accelerator = CudaDctToWaveletStageAccelerator::new_explicit();
    let host_batch = host_transcoder
        .transcode_batch_with_accelerator(&inputs, &options, &mut host_accelerator)
        .expect("host-bounce CUDA 9/7 batch succeeds");

    let mut resident_transcoder = JpegToHtj2kTranscoder::default();
    let mut resident_accelerator =
        CudaDctToWaveletStageAccelerator::new_explicit_resident_ht_encode();
    let resident_batch = resident_transcoder
        .transcode_batch_with_accelerator(&inputs, &options, &mut resident_accelerator)
        .expect("resident CUDA 9/7 + HT batch succeeds");

    assert_eq!(resident_batch.report.failed_tiles, 0);
    assert_eq!(resident_batch.report.timings.cpu_fallback_jobs, 0);
    assert_eq!(
        resident_batch
            .report
            .timings
            .dwt97_batch_ht_codeblock_dispatches,
        1,
        "resident path should share one CUDA HT code-block encode across compatible batch groups"
    );
    assert_eq!(
        resident_batch.report.timings.dwt97_batch_readback_us, 0,
        "resident path must avoid quantized coefficient readback"
    );

    let host_codestreams = host_batch
        .tiles
        .into_iter()
        .map(|tile| tile.expect("host tile succeeds").codestream)
        .collect::<Vec<_>>();
    let resident_codestreams = resident_batch
        .tiles
        .into_iter()
        .map(|tile| tile.expect("resident tile succeeds").codestream)
        .collect::<Vec<_>>();
    assert_eq!(resident_codestreams, host_codestreams);
}
