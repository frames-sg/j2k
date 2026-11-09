// SPDX-License-Identifier: MIT OR Apache-2.0

use cuda_device::ptx_asm;
use j2k_codec_math::transcode97::Transcode97Arithmetic;

use crate::{simt_load, simt_mut_ptr_at, simt_store};

/// Device binary32 primitives for the canonical 9/7 transcode sequence.
///
/// Each operation is one explicitly rounded PTX instruction (`.rn`), so
/// `ptxas` can neither fuse a product into a following sum nor split a fused
/// multiply-add, whatever `--fmad` setting the build uses.
pub(crate) struct PtxArithmetic;

impl Transcode97Arithmetic for PtxArithmetic {
    #[inline(always)]
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        let output: f32;
        // SAFETY: a pure register-only IEEE binary32 fused multiply-add with
        // no memory, control-flow or lane-participation contract.
        unsafe {
            ptx_asm!(
                "fma.rn.f32 %0, %1, %2, %3;",
                out("=f") output,
                in("f") a,
                in("f") b,
                in("f") c,
                options(register_only),
            );
        }
        output
    }

    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        let output: f32;
        // SAFETY: a pure register-only IEEE binary32 multiply with no memory,
        // control-flow or lane-participation contract.
        unsafe {
            ptx_asm!(
                "mul.rn.f32 %0, %1, %2;",
                out("=f") output,
                in("f") a,
                in("f") b,
                options(register_only),
            );
        }
        output
    }

    #[inline(always)]
    fn add(a: f32, b: f32) -> f32 {
        let output: f32;
        // SAFETY: a pure register-only IEEE binary32 addition with no memory,
        // control-flow or lane-participation contract.
        unsafe {
            ptx_asm!(
                "add.rn.f32 %0, %1, %2;",
                out("=f") output,
                in("f") a,
                in("f") b,
                options(register_only),
            );
        }
        output
    }
}

#[inline(always)]
pub(crate) fn load_i16(ptr: *const i16, index: u64) -> i16 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
pub(crate) fn load_i32(ptr: *const i32, index: u64) -> i32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
pub(crate) fn load_f32(ptr: *const f32, index: u64) -> f32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
pub(crate) fn store_i32(ptr: *mut i32, index: u64, value: i32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
pub(crate) fn store_f32(ptr: *mut f32, index: u64, value: f32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
pub(crate) fn offset_i32_mut(ptr: *mut i32, index: u64) -> *mut i32 {
    simt_mut_ptr_at(ptr, index as usize)
}

#[inline(always)]
pub(crate) fn offset_f32_mut(ptr: *mut f32, index: u64) -> *mut f32 {
    simt_mut_ptr_at(ptr, index as usize)
}

#[inline(always)]
pub(crate) fn floor_div_pos(a: i32, d: i32) -> i32 {
    let mut q = a / d;
    let r = a - q * d;
    if r < 0 {
        q -= 1;
    }
    q
}
