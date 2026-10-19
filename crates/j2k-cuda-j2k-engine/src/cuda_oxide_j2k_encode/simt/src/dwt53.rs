// SPDX-License-Identifier: MIT OR Apache-2.0

//! T.800 F-9/F-10 analysis on an absolute-parity line with symmetric extension.

use crate::helpers::{floor_f32, load_f32};

#[inline(always)]
fn predict(src: *const f32, base: u32, stride: u32, length: u32, index: u32) -> f32 {
    let left = if index == 0 { 1 } else { index - 1 };
    let right = if index + 1 < length {
        index + 1
    } else {
        index - 1
    };
    load_f32(src, base + index * stride)
        - floor_f32(
            (load_f32(src, base + left * stride) + load_f32(src, base + right * stride)) * 0.5,
        )
}

#[inline(always)]
pub(crate) fn j2k_fdwt53_sample(
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
        let index = output * 2 + phase;
        let left = if index == 0 { 1 } else { index - 1 };
        let right = if index + 1 < length {
            index + 1
        } else {
            index - 1
        };
        load_f32(src, base + index * stride)
            + floor_f32(
                (predict(src, base, stride, length, left)
                    + predict(src, base, stride, length, right))
                    * 0.25
                    + 0.5,
            )
    } else {
        predict(
            src,
            base,
            stride,
            length,
            (output - low_count) * 2 + 1 - phase,
        )
    }
}
