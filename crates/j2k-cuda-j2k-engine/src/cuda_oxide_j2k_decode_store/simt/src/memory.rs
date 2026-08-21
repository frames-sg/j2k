// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed SIMT memory access for final-store kernels.

use crate::{simt_load, simt_store};

#[inline(always)]
pub(crate) fn load_f32(ptr: *const f32, index: u32) -> f32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
pub(crate) fn load_job<T: Copy>(ptr: *const T) -> T {
    simt_load(ptr, 0)
}

#[inline(always)]
pub(crate) fn store_f32(ptr: *mut f32, index: u32, value: f32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
pub(crate) fn store_u8(ptr: *mut u8, index: u32, value: u8) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
pub(crate) fn store_u16(ptr: *mut u16, index: u32, value: u16) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
pub(crate) fn store_i16(ptr: *mut i16, index: u32, value: i16) {
    simt_store(ptr, index as usize, value);
}
