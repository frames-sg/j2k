// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical binary32 9/7 transcode arithmetic on the device.
//!
//! Every value is computed by the operation sequence of
//! `j2k_codec_math::transcode97`, instantiated with [`PtxArithmetic`], so the
//! bands are bit-identical to the CPU and Metal transcode backends.

use j2k_codec_math::transcode97::{
    idct8_dot, lift97_update, Transcode97Arithmetic, DWT97_HIGH_SCALE_F32, DWT97_LIFT_STEPS_F32,
    DWT97_LOW_SCALE_F32, IDCT8_BASIS_F32,
};

use crate::{
    constants::DWT97_ROW_LIFT_MAX_WIDTH,
    helpers::{load_f32, load_i16, store_f32, PtxArithmetic},
};

/// Basis row `IDCT8_BASIS_F32[sample_idx]` for a sample index in `0..8`.
#[inline(always)]
fn idct8_basis_row(sample_idx: i32) -> [f32; 8] {
    match sample_idx & 7 {
        0 => IDCT8_BASIS_F32[0],
        1 => IDCT8_BASIS_F32[1],
        2 => IDCT8_BASIS_F32[2],
        3 => IDCT8_BASIS_F32[3],
        4 => IDCT8_BASIS_F32[4],
        5 => IDCT8_BASIS_F32[5],
        6 => IDCT8_BASIS_F32[6],
        _ => IDCT8_BASIS_F32[7],
    }
}

/// One canonical IDCT output: the vertical dot products `T[local_y][u]` of
/// basis row `local_y` with each coefficient column, then the horizontal dot
/// product of basis row `local_x` with `T[local_y]`. `coefficient(k)` loads
/// natural-order coefficient `k` as an exact `f32`.
#[inline(always)]
fn idct8x8_canonical_sample(coefficient: impl Fn(u64) -> f32, local_x: i32, local_y: i32) -> f32 {
    let vertical_basis = idct8_basis_row(local_y);
    let column = |u: u64| {
        [
            coefficient(u),
            coefficient(8 + u),
            coefficient(16 + u),
            coefficient(24 + u),
            coefficient(32 + u),
            coefficient(40 + u),
            coefficient(48 + u),
            coefficient(56 + u),
        ]
    };
    let vertical = [
        idct8_dot::<PtxArithmetic>(&vertical_basis, &column(0)),
        idct8_dot::<PtxArithmetic>(&vertical_basis, &column(1)),
        idct8_dot::<PtxArithmetic>(&vertical_basis, &column(2)),
        idct8_dot::<PtxArithmetic>(&vertical_basis, &column(3)),
        idct8_dot::<PtxArithmetic>(&vertical_basis, &column(4)),
        idct8_dot::<PtxArithmetic>(&vertical_basis, &column(5)),
        idct8_dot::<PtxArithmetic>(&vertical_basis, &column(6)),
        idct8_dot::<PtxArithmetic>(&vertical_basis, &column(7)),
    ];
    idct8_dot::<PtxArithmetic>(&idct8_basis_row(local_x), &vertical)
}

#[inline(always)]
pub(crate) fn idct8x8_sample(block: *const f32, local_x: i32, local_y: i32) -> f32 {
    idct8x8_canonical_sample(|index| load_f32(block, index), local_x, local_y)
}

#[inline(always)]
pub(crate) fn idct8x8_sample_i16(block: *const i16, local_x: i32, local_y: i32) -> f32 {
    idct8x8_canonical_sample(|index| f32::from(load_i16(block, index)), local_x, local_y)
}

/// One lifting step over every sample of parity `first` (1 = odd, 0 =
/// even). The missing left neighbour of sample 0 is sample 1, and a missing
/// right neighbour repeats the left one.
#[inline(always)]
fn lift_step(data: *mut f32, n: i32, stride: i32, first: i32, coefficient: f32) {
    let mut i = first;
    while i < n {
        let left = if i > 0 {
            load_f32(data.cast_const(), ((i - 1) * stride) as u64)
        } else {
            load_f32(data.cast_const(), stride as u64)
        };
        let right = if i + 1 < n {
            load_f32(data.cast_const(), ((i + 1) * stride) as u64)
        } else {
            left
        };
        let center = load_f32(data.cast_const(), (i * stride) as u64);
        store_f32(
            data,
            (i * stride) as u64,
            lift97_update::<PtxArithmetic>(center, left, right, coefficient),
        );
        i += 2;
    }
}

/// Canonical in-place forward 9/7 lifting of one strided signal; signals
/// shorter than two samples are left unchanged.
#[inline(always)]
pub(crate) fn forward_lift_97(data: *mut f32, n: i32, stride: i32) {
    if n < 2 {
        return;
    }
    lift_step(data, n, stride, 1, DWT97_LIFT_STEPS_F32[0]);
    lift_step(data, n, stride, 0, DWT97_LIFT_STEPS_F32[1]);
    lift_step(data, n, stride, 1, DWT97_LIFT_STEPS_F32[2]);
    lift_step(data, n, stride, 0, DWT97_LIFT_STEPS_F32[3]);
    scale_parity(data, n, stride, 0, DWT97_LOW_SCALE_F32);
    scale_parity(data, n, stride, 1, DWT97_HIGH_SCALE_F32);
}

#[inline(always)]
fn scale_parity(data: *mut f32, n: i32, stride: i32, first: i32, scale: f32) {
    let mut i = first;
    while i < n {
        let value = load_f32(data.cast_const(), (i * stride) as u64);
        store_f32(data, (i * stride) as u64, PtxArithmetic::mul(value, scale));
        i += 2;
    }
}

/// One cooperative lifting step on a shared-memory row: thread `tid` of
/// `block_step` updates samples `first + 2 * (tid + k * block_step)`.
#[inline(always)]
pub(crate) fn coop_lift_step(
    rows: *mut f32,
    row_lane: i32,
    width: i32,
    tid: i32,
    block_step: i32,
    first: i32,
    coefficient: f32,
) {
    let mut i = tid * 2 + first;
    while i < width {
        let left = if i > 0 {
            load_f32(rows.cast_const(), shared_row_index(row_lane, i - 1))
        } else {
            load_f32(rows.cast_const(), shared_row_index(row_lane, 1))
        };
        let right = if i + 1 < width {
            load_f32(rows.cast_const(), shared_row_index(row_lane, i + 1))
        } else {
            left
        };
        let center = load_f32(rows.cast_const(), shared_row_index(row_lane, i));
        store_f32(
            rows,
            shared_row_index(row_lane, i),
            lift97_update::<PtxArithmetic>(center, left, right, coefficient),
        );
        i += block_step * 2;
    }
}

/// Cooperative low/high scaling of a lifted shared-memory row.
#[inline(always)]
pub(crate) fn coop_scale(rows: *mut f32, row_lane: i32, width: i32, tid: i32, block_step: i32) {
    let mut i = tid;
    while i < width {
        let scale = if i & 1 == 0 {
            DWT97_LOW_SCALE_F32
        } else {
            DWT97_HIGH_SCALE_F32
        };
        let value = load_f32(rows.cast_const(), shared_row_index(row_lane, i));
        store_f32(
            rows,
            shared_row_index(row_lane, i),
            PtxArithmetic::mul(value, scale),
        );
        i += block_step;
    }
}

#[inline(always)]
pub(crate) fn shared_row_index(row_lane: i32, x: i32) -> u64 {
    row_lane as u64 * DWT97_ROW_LIFT_MAX_WIDTH as u64 + x as u64
}
