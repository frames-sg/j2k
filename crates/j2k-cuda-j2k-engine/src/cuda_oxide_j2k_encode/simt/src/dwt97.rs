// SPDX-License-Identifier: MIT OR Apache-2.0

//! T.800 F-11 lifting evaluated at local positions, with absolute parity.

use crate::{
    constants::{
        J2K_FDWT97_ALPHA, J2K_FDWT97_BETA, J2K_FDWT97_DELTA, J2K_FDWT97_GAMMA,
        J2K_FDWT97_INV_KAPPA, J2K_FDWT97_KAPPA,
    },
    helpers::load_f32,
};

#[inline(always)]
fn neighbors(index: u32, length: u32) -> (u32, u32) {
    (
        if index == 0 { 1 } else { index - 1 },
        if index + 1 < length {
            index + 1
        } else {
            index - 1
        },
    )
}

#[inline(always)]
fn high1(src: *const f32, base: u32, stride: u32, length: u32, index: u32) -> f32 {
    let (left, right) = neighbors(index, length);
    load_f32(src, base + index * stride)
        + J2K_FDWT97_ALPHA
            * (load_f32(src, base + left * stride) + load_f32(src, base + right * stride))
}

#[inline(always)]
fn low1(src: *const f32, base: u32, stride: u32, length: u32, index: u32) -> f32 {
    let (left, right) = neighbors(index, length);
    load_f32(src, base + index * stride)
        + J2K_FDWT97_BETA
            * (high1(src, base, stride, length, left) + high1(src, base, stride, length, right))
}

#[inline(always)]
fn high2(src: *const f32, base: u32, stride: u32, length: u32, index: u32) -> f32 {
    let (left, right) = neighbors(index, length);
    high1(src, base, stride, length, index)
        + J2K_FDWT97_GAMMA
            * (low1(src, base, stride, length, left) + low1(src, base, stride, length, right))
}

#[inline(always)]
fn low2(src: *const f32, base: u32, stride: u32, length: u32, index: u32) -> f32 {
    let (left, right) = neighbors(index, length);
    low1(src, base, stride, length, index)
        + J2K_FDWT97_DELTA
            * (high2(src, base, stride, length, left) + high2(src, base, stride, length, right))
}

#[inline(always)]
pub(crate) fn j2k_fdwt97_sample(
    src: *const f32,
    base: u32,
    stride: u32,
    length: u32,
    low_count: u32,
    phase: u32,
    output: u32,
) -> f32 {
    if length == 1 {
        return load_f32(src, base) * if phase == 0 { 1.0 } else { 2.0 };
    }
    if output < low_count {
        low2(src, base, stride, length, output * 2 + phase) * J2K_FDWT97_INV_KAPPA
    } else {
        high2(
            src,
            base,
            stride,
            length,
            (output - low_count) * 2 + 1 - phase,
        ) * J2K_FDWT97_KAPPA
    }
}
