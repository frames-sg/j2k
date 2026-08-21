// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inverse JPEG 2000 color transform for final-store kernels.

use crate::sample::floor_f32;
use cuda_device::ptx_asm;

#[inline(always)]
fn fused_mul_add_f32(multiplicand: f32, multiplier: f32, addend: f32) -> f32 {
    let output: f32;
    // SAFETY: this is a pure register-only IEEE binary32 FMA with no memory,
    // control-flow, or lane-participation contract.
    unsafe {
        ptx_asm!(
            "fma.rn.f32 %0, %1, %2, %3;",
            out("=f") output,
            in("f") multiplicand,
            in("f") multiplier,
            in("f") addend,
            options(register_only),
        );
    }
    output
}

#[inline(always)]
pub(crate) fn inverse_mct_sample(
    src0: f32,
    src1: f32,
    src2: f32,
    irreversible97: u32,
) -> (f32, f32, f32) {
    if irreversible97 != 0 {
        (
            fused_mul_add_f32(src2, 1.402, src0),
            fused_mul_add_f32(src2, -0.71414, fused_mul_add_f32(src1, -0.34413, src0)),
            fused_mul_add_f32(src1, 1.772, src0),
        )
    } else {
        let green = src0 - floor_f32((src2 + src1) * 0.25);
        (src2 + green, green, src1 + green)
    }
}
