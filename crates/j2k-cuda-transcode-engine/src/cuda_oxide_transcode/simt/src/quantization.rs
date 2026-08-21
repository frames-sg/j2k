// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) fn floor_f32(value: f32) -> f32 {
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

#[inline(always)]
pub(crate) fn min_i32(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

#[inline(always)]
pub(crate) fn quantize_dwt97_deadzone(value: f32, inv_delta: f32) -> i32 {
    let sign = if value < 0.0 { -1 } else { 1 };
    sign * floor_f32(abs_f32(value) * inv_delta) as i32
}

#[inline(always)]
pub(crate) fn dwt97_codeblock_major_offset(
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    cb_width: i32,
    cb_height: i32,
) -> u64 {
    if cb_width == 64 && cb_height == 64 {
        let cbx = x >> 6;
        let cby = y >> 6;
        let local_x = x & 63;
        let local_y = y & 63;
        let block_width = min_i32(64, width - (cbx << 6));
        let block_height = min_i32(64, height - (cby << 6));
        return (cby as u64) * 64 * width as u64
            + (cbx as u64) * 64 * block_height as u64
            + (local_y as u64) * block_width as u64
            + local_x as u64;
    }
    let cbx = x / cb_width;
    let cby = y / cb_height;
    let local_x = x - cbx * cb_width;
    let local_y = y - cby * cb_height;
    let block_width = min_i32(cb_width, width - cbx * cb_width);
    let block_height = min_i32(cb_height, height - cby * cb_height);
    (cby as u64) * cb_height as u64 * width as u64
        + (cbx as u64) * cb_width as u64 * block_height as u64
        + (local_y as u64) * block_width as u64
        + local_x as u64
}
