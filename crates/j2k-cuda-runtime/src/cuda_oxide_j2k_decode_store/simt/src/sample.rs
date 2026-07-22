// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native and display-width sample conversion for final-store kernels.

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
fn round_f32(value: f32) -> f32 {
    if value >= 0.0 {
        floor_f32(value + 0.5)
    } else {
        -floor_f32(-value + 0.5)
    }
}

#[inline(always)]
fn clamp_f32(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[inline(always)]
fn max_int_for_bit_depth(bit_depth: u32) -> u32 {
    if bit_depth == 0 {
        1
    } else {
        (1_u32 << bit_depth) - 1
    }
}

#[inline(always)]
pub(crate) fn sample_as_u8(sample: f32, bit_depth: u32) -> u8 {
    let rounded = round_f32(sample);
    if bit_depth >= 8 {
        return clamp_f32(rounded, 0.0, 255.0) as u8;
    }

    let max_int = (1_u32 << bit_depth) - 1;
    let max_value = if max_int > 1 { max_int as f32 } else { 1.0 };
    round_f32((clamp_f32(rounded, 0.0, max_value) / max_value) * 255.0) as u8
}

#[inline(always)]
pub(crate) fn sample_as_u16(sample: f32, bit_depth: u32) -> u16 {
    let rounded = round_f32(sample);
    if bit_depth >= 16 {
        return clamp_f32(rounded, 0.0, 65535.0) as u16;
    }

    let max_int = max_int_for_bit_depth(bit_depth);
    let max_value = if max_int > 1 { max_int as f32 } else { 1.0 };
    round_f32((clamp_f32(rounded, 0.0, max_value) / max_value) * 65535.0) as u16
}

#[inline(always)]
pub(crate) fn sample_as_native_u8(sample: f32, bit_depth: u32) -> u8 {
    let max_value = max_int_for_bit_depth(bit_depth.min(8)) as f32;
    clamp_f32(round_f32(sample), 0.0, max_value) as u8
}

#[inline(always)]
pub(crate) fn sample_as_native_u16(sample: f32, bit_depth: u32) -> u16 {
    let max_value = max_int_for_bit_depth(bit_depth.min(16)) as f32;
    clamp_f32(round_f32(sample), 0.0, max_value) as u16
}

#[inline(always)]
pub(crate) fn sample_as_native_i16(sample: f32, bit_depth: u32) -> i16 {
    let precision = bit_depth.clamp(1, 16);
    let magnitude = 1_i32 << (precision - 1);
    clamp_f32(
        round_f32(sample),
        (-magnitude) as f32,
        (magnitude - 1) as f32,
    ) as i16
}
