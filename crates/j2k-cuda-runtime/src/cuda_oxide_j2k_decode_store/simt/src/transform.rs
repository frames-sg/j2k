// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inverse JPEG 2000 color transform for final-store kernels.

use crate::sample::floor_f32;

#[inline(always)]
pub(crate) fn inverse_mct_sample(
    src0: f32,
    src1: f32,
    src2: f32,
    irreversible97: u32,
) -> (f32, f32, f32) {
    if irreversible97 != 0 {
        (
            src0 + 1.402 * src2,
            src0 - 0.34413 * src1 - 0.71414 * src2,
            src0 + 1.772 * src1,
        )
    } else {
        let green = src0 - floor_f32((src2 + src1) * 0.25);
        (src2 + green, green, src1 + green)
    }
}
