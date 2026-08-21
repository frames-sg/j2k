// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{simt_load, simt_mut_ptr_at, simt_store};

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
