// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Batch parity (Gap B): the CUDA same-geometry 9/7 batch must produce the exact
// same bands as the per-job CUDA 9/7 path (identical f32 kernels with per-item
// offsets, --fmad=false), and must report real (non-zero) staged batch timings.
//
// Compiled only with `cuda-runtime`; asserts only on the CUDA runner.
#![cfg(feature = "cuda-runtime")]

use j2k_test_support::cuda_runtime_gate;
use j2k_transcode::accelerator::{DctGridToDwt97Job, DctToWaveletStageAccelerator};
use j2k_transcode_cuda::CudaDctToWaveletStageAccelerator;

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

#[test]
fn cuda_dwt97_batch_matches_per_job_and_reports_stage_timings_when_required() {
    if !cuda_runtime_gate(module_path!()) {
        return;
    }

    // Non-multiple-of-8 dimensions to exercise partial edge blocks/bands.
    let (block_cols, block_rows, width, height) = (4usize, 4usize, 29usize, 31usize);
    let first = make_blocks(block_cols, block_rows, 0);
    let second = make_blocks(block_cols, block_rows, 37);
    let jobs = [
        DctGridToDwt97Job {
            blocks: &first,
            block_cols,
            block_rows,
            width,
            height,
        },
        DctGridToDwt97Job {
            blocks: &second,
            block_cols,
            block_rows,
            width,
            height,
        },
    ];

    let mut accelerator = CudaDctToWaveletStageAccelerator::new_explicit();
    let batch = accelerator
        .dct_grid_to_dwt97_batch(&jobs)
        .expect("CUDA 9/7 batch dispatch should succeed on the runner")
        .expect("CUDA should handle the 9/7 batch (explicit mode)");

    assert_eq!(batch.len(), jobs.len());

    // The batched staged kernels must match the single-job path bit-for-bit.
    for (job, batched) in jobs.iter().zip(batch.iter()) {
        let per_job = CudaDctToWaveletStageAccelerator::new_explicit()
            .dct_grid_to_dwt97(*job)
            .expect("CUDA 9/7 dispatch should succeed on the runner")
            .expect("CUDA should handle the 9/7 job (explicit mode)");
        assert_eq!(
            batched, &per_job,
            "batch item diverged from the per-job 9/7 transcode for {width}x{height}"
        );
    }

    let timings = accelerator
        .last_dwt97_batch_stage_timings()
        .expect("CUDA 9/7 batch records backend stage timings");
    assert!(timings.pack_upload_us > 0, "pack/upload stage not timed");
    assert!(
        timings.idct_row_lift_us > 0,
        "idct+row-lift stage not timed"
    );
    assert!(timings.column_lift_us > 0, "column-lift stage not timed");
    assert_eq!(
        timings.quantize_codeblock_us, 0,
        "band path must not run the quantize stage"
    );
    assert!(timings.readback_us > 0, "readback stage not timed");
}

#[test]
fn cuda_dwt97_batch_non_uniform_geometry_falls_back_to_per_job_when_required() {
    if !cuda_runtime_gate(module_path!()) {
        return;
    }

    // Two jobs share a 4x4 block grid (covers 32x32) but differ in logical
    // dimensions, so the batch cannot use the same-geometry staged path and must
    // fall back to the per-job loop — still bit-exact, just without batch timings.
    let block_cols = 4usize;
    let block_rows = 4usize;
    let first = make_blocks(block_cols, block_rows, 0);
    let second = make_blocks(block_cols, block_rows, 37);
    let jobs = [
        DctGridToDwt97Job {
            blocks: &first,
            block_cols,
            block_rows,
            width: 29,
            height: 31,
        },
        DctGridToDwt97Job {
            blocks: &second,
            block_cols,
            block_rows,
            width: 24,
            height: 26,
        },
    ];

    let mut accelerator = CudaDctToWaveletStageAccelerator::new_explicit();
    let batch = accelerator
        .dct_grid_to_dwt97_batch(&jobs)
        .expect("CUDA 9/7 batch dispatch should succeed on the runner")
        .expect("CUDA should handle the mixed-geometry 9/7 batch via per-job fallback");

    assert_eq!(batch.len(), jobs.len());
    for (job, batched) in jobs.iter().zip(batch.iter()) {
        let per_job = CudaDctToWaveletStageAccelerator::new_explicit()
            .dct_grid_to_dwt97(*job)
            .expect("CUDA 9/7 dispatch should succeed on the runner")
            .expect("CUDA should handle the 9/7 job (explicit mode)");
        assert_eq!(
            batched, &per_job,
            "non-uniform batch item diverged from per-job 9/7 for {}x{}",
            job.width, job.height
        );
    }
}
