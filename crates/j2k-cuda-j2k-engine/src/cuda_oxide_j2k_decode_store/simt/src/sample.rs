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
pub(crate) fn round_ties_even_f32(value: f32) -> f32 {
    // Ordered comparisons also preserve NaN/infinity and integral f32 values
    // at magnitude >= 2^23. Avoid abs(): CUDA-Oxide lowers it to libdevice,
    // switching this PTX-only store build to NVVM IR.
    if !(value > -8_388_608.0 && value < 8_388_608.0) {
        return value;
    }
    let lower = floor_f32(value);
    let upper = lower + 1.0;
    let midpoint = lower + 0.5;
    if value < midpoint {
        lower
    } else if value > midpoint || (lower as i32) & 1 != 0 {
        upper
    } else {
        lower
    }
}

#[inline(always)]
pub(crate) fn round_integer_output_then_add(sample: f32, addend: f32) -> f32 {
    // Match the CPU integer-output path: round the centered 9/7 result before
    // applying the unsigned level shift. Adding a 16-bit shift first discards
    // one bit of f32 precision and can move samples across a rounding boundary.
    round_ties_even_f32(sample) + addend
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
    if bit_depth == 8 {
        return clamp_f32(rounded, 0.0, 255.0) as u8;
    }

    let max_int = max_int_for_bit_depth(bit_depth.min(16));
    let max_value = if max_int > 1 { max_int as f32 } else { 1.0 };
    round_f32((clamp_f32(rounded, 0.0, max_value) / max_value) * 255.0) as u8
}

#[inline(always)]
pub(crate) fn sample_as_u16(sample: f32, bit_depth: u32) -> u16 {
    let rounded = round_f32(sample);
    if bit_depth > 8 {
        return sample_as_native_u16(rounded, bit_depth);
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

// Synthesized alpha follows the requested U16 storage domain: widened through
// eight bits, native significant precision above eight bits.
#[inline(always)]
pub(crate) fn opaque_alpha_u16(bit_depth: u32) -> u16 {
    if bit_depth <= 8 {
        65535
    } else {
        max_int_for_bit_depth(bit_depth.min(16)) as u16
    }
}
