// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{simt_load, simt_store};
use cuda_device::ptx_asm;

#[inline(always)]
pub(crate) fn load_u8(ptr: *const u8, index: u64) -> u8 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
pub(crate) fn load_u32(ptr: *const u32, index: u64) -> u32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
pub(crate) fn load_f32(ptr: *const f32, index: u32) -> f32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
pub(crate) fn load_f32_u64(ptr: *const f32, index: u64) -> f32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
pub(crate) fn store_f32(ptr: *mut f32, index: u32, value: f32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
pub(crate) fn store_f32_u64(ptr: *mut f32, index: u64, value: f32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
pub(crate) fn store_i32(ptr: *mut i32, index: u64, value: i32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
pub(crate) fn store_u8(ptr: *mut u8, index: u64, value: u8) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
pub(crate) fn store_u32(ptr: *mut u32, index: u64, value: u32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
pub(crate) fn load_job<T: Copy>(ptr: *const T, index: u32) -> T {
    simt_load(ptr, index as usize)
}

#[inline(always)]
pub(crate) fn sign_extend_u32(raw: u32, bit_depth: u32) -> i32 {
    let shift = 32 - bit_depth;
    ((raw << shift) as i32) >> shift
}

#[inline(always)]
pub(crate) fn floor_f32(value: f32) -> f32 {
    // f32::floor routes through libdevice in cuda-oxide, which emits NVVM IR
    // instead of the PTX loaded by this runtime path.
    let truncated = value as i32 as f32;
    if truncated > value {
        truncated - 1.0
    } else {
        truncated
    }
}

#[inline(always)]
pub(crate) fn fused_mul_add_f32(
    multiplicand: f32,
    multiplier: f32,
    addend: f32,
) -> f32 {
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
pub(crate) fn forward_ict_rgb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Match the CPU transform's target-independent nested fused rounding.
    (
        fused_mul_add_f32(b, 0.114, fused_mul_add_f32(r, 0.299, 0.587 * g)),
        fused_mul_add_f32(b, 0.5, fused_mul_add_f32(r, -0.16875, -0.33126 * g)),
        fused_mul_add_f32(b, -0.08131, fused_mul_add_f32(r, 0.5, -0.41869 * g)),
    )
}

#[inline(always)]
pub(crate) fn abs_f32(value: f32) -> f32 {
    if value < 0.0 { -value } else { value }
}
