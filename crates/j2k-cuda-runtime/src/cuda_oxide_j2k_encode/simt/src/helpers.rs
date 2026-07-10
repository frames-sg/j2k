// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{simt_load, simt_store};

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
pub(crate) fn abs_f32(value: f32) -> f32 {
    if value < 0.0 { -value } else { value }
}
