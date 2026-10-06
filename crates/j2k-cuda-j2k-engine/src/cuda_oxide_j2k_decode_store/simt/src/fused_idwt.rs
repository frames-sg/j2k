// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    static_mut_refs,
    reason = "CUDA shared-memory statics are accessed through device-scoped references"
)]

//! Final vertical synthesis and RGB packing for full, origin-zero codestream planes.

use crate::{
    abi::CudaJ2kStoreRgb8MctBatchJob,
    color::store_rgb8_mct_values,
    memory::{load_f32, load_job, store_f32},
    sample::floor_f32,
    transform::fused_mul_add_f32,
};
use cuda_device::{SharedArray, kernel, thread};
use cuda_host::cuda_module;

const COLS: u32 = j2k_codec_math::dwt::FUSED_VERTICAL_TILE_COLUMNS;
const ROWS: u32 = j2k_codec_math::dwt::FUSED_VERTICAL_TILE_ROWS;
const THREADS: u32 = j2k_codec_math::dwt::FUSED_VERTICAL_TILE_THREADS;
const HALO: u32 = 4;
const STRIP: u32 = ROWS + 2 * HALO;
const PLANE: u32 = COLS * STRIP;
const SAMPLES: usize = (3 * PLANE) as usize;

#[inline(always)]
fn lift(samples: *mut f32, lane: u32, phase: u32, irreversible: bool) {
    let mut index = lane;
    while index < 3 * PLANE {
        let row = (index % PLANE) / COLS;
        // Four rows of immutable-input halo cover the four lifting steps.
        // Artificial outer strip boundaries cannot affect the stored interior.
        if row > 0 && row + 1 < STRIP && row & 1 == phase & 1 {
            let value = load_f32(samples, index);
            let sum = load_f32(samples, index - COLS) + load_f32(samples, index + COLS);
            let result = if irreversible {
                let coefficient = match phase {
                    0 => j2k_codec_math::dwt::IDWT97_NEG_DELTA_F32,
                    1 => j2k_codec_math::dwt::IDWT97_NEG_GAMMA_F32,
                    2 => j2k_codec_math::dwt::IDWT97_NEG_BETA_F32,
                    _ => j2k_codec_math::dwt::IDWT97_NEG_ALPHA_F32,
                };
                fused_mul_add_f32(sum, coefficient, value)
            } else if phase == 0 {
                value - floor_f32(sum * 0.25 + 0.5)
            } else {
                value + floor_f32(sum * 0.5)
            };
            store_f32(samples, index, result);
        }
        index += THREADS;
    }
    thread::sync_threads();
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn j2k_idwt_vertical_rgb8_mct_batch(jobs: *const CudaJ2kStoreRgb8MctBatchJob) {
        static mut SAMPLES_SHARED: SharedArray<f32, SAMPLES> = SharedArray::UNINIT;
        let item = load_job(unsafe { jobs.add(thread::blockIdx_y() as usize) });
        let job = item.job.store;
        let width = job.copy_width;
        let height = job.copy_height;
        let tiles_x = (width + COLS - 1) / COLS;
        let start_x = (thread::blockIdx_x() % tiles_x) * COLS;
        let start_y = (thread::blockIdx_x() / tiles_x) * ROWS;
        if start_y >= height {
            return;
        }
        let lane = thread::threadIdx_x();
        let samples = unsafe { SAMPLES_SHARED.as_mut_ptr() };
        let irreversible = item.job.irreversible97 != 0;
        let mut index = lane;
        while index < 3 * PLANE {
            let plane = index / PLANE;
            let row = (index % PLANE) / COLS;
            let col = start_x + index % COLS;
            let y = j2k_codec_math::dwt::reflect_index(
                start_y as i64 + row as i64 - HALO as i64,
                height,
            );
            let input = match plane {
                0 => item.plane0_ptr,
                1 => item.plane1_ptr,
                _ => item.plane2_ptr,
            } as usize as *const f32;
            let mut value = if col < width {
                load_f32(input, y * width + col)
            } else {
                0.0
            };
            if irreversible {
                let scale = if row & 1 == 0 {
                    j2k_codec_math::dwt::DWT97_KAPPA_F32
                } else {
                    j2k_codec_math::dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5
                };
                value *= scale;
            }
            store_f32(samples, index, value);
            index += THREADS;
        }
        thread::sync_threads();
        lift(samples, lane, 0, irreversible);
        lift(samples, lane, 1, irreversible);
        if irreversible {
            lift(samples, lane, 2, true);
            lift(samples, lane, 3, true);
        }
        let mut pixel = lane;
        while pixel < ROWS * COLS {
            let row = pixel / COLS;
            let col = pixel % COLS;
            let x = start_x + col;
            let y = start_y + row;
            if x < width && y < height {
                let index = (row + HALO) * COLS + col;
                store_rgb8_mct_values(
                    [
                        load_f32(samples, index),
                        load_f32(samples, PLANE + index),
                        load_f32(samples, 2 * PLANE + index),
                    ],
                    item.output_ptr as usize as *mut u8,
                    item.job,
                    y,
                    x,
                );
            }
            pixel += THREADS;
        }
    }
}
