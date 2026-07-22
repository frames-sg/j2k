// SPDX-License-Identifier: MIT OR Apache-2.0

//! Source and destination addressing for final-store kernels.

#[inline(always)]
pub(crate) fn source_index(
    input_width: u32,
    source_x: u32,
    source_y: u32,
    row: u32,
    col: u32,
) -> u32 {
    (source_y + row) * input_width + source_x + col
}

#[inline(always)]
pub(crate) fn output_pixel_index(
    output_width: u32,
    output_x: u32,
    output_y: u32,
    row: u32,
    col: u32,
) -> u32 {
    (output_y + row) * output_width + output_x + col
}

#[inline(always)]
pub(crate) fn pixel_coords(gid: u32, copy_width: u32) -> (u32, u32) {
    let row = gid / copy_width;
    (row, gid - row * copy_width)
}
