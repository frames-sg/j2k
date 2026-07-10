// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::helpers::{abs_f32, floor_f32};

#[inline(always)]
fn ldexp_one_f32(exponent: i32) -> f32 {
    if exponent < -149 {
        0.0
    } else if exponent < -126 {
        f32::from_bits(1_u32 << ((exponent + 149) as u32))
    } else if exponent <= 127 {
        f32::from_bits(((exponent + 127) as u32) << 23)
    } else {
        f32::INFINITY
    }
}

#[inline(always)]
pub(crate) fn j2k_quantize_sample(
    sample: f32,
    step_exponent: u32,
    step_mantissa: u32,
    range_bits: u32,
    reversible: u32,
) -> i32 {
    if reversible != 0 {
        let rounded = if sample >= 0.0 {
            floor_f32(sample + 0.5)
        } else {
            -floor_f32(-sample + 0.5)
        };
        return rounded as i32;
    }

    let exponent = range_bits as i32 - step_exponent as i32;
    let base = ldexp_one_f32(exponent);
    let delta = base * (1.0 + step_mantissa as f32 / 2048.0);
    if delta <= 0.0 {
        return 0;
    }

    let sign = if sample < 0.0 { -1 } else { 1 };
    let magnitude = floor_f32(abs_f32(sample) / delta) as i32;
    sign * magnitude
}
