// SPDX-License-Identifier: MIT OR Apache-2.0

//! CIE Lab sample scaling before application-owned ICC conversion.

use crate::error::bail;
use crate::j2c::ComponentData;
use crate::jp2::colr::CieLab;
use crate::math::{f32x8, Simd, SIMD_WIDTH};
use crate::{ColorError, Result};

fn clamped_power_of_two_u32(exponent: u8) -> u32 {
    if u32::from(exponent) >= u32::BITS {
        u32::MAX
    } else {
        1_u32 << exponent
    }
}

fn clamped_add_u32(left: u32, right: u32) -> u32 {
    if right > u32::MAX - left {
        u32::MAX
    } else {
        left + right
    }
}

fn max_value_for_bit_depth(bit_depth: u8) -> u32 {
    if u32::from(bit_depth) >= u32::BITS {
        u32::MAX
    } else {
        (1_u32 << bit_depth) - 1
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "OpenJPEG-compatible CIE Lab scaling intentionally uses f32 arithmetic"
)]
#[inline]
pub(crate) fn cielab_to_rgb<S: Simd>(
    simd: S,
    components: &mut [ComponentData],
    bit_depth: u8,
    lab: &CieLab,
) -> Result<()> {
    let (head, _) = components
        .split_at_mut_checked(3)
        .ok_or(ColorError::LabConversionFailed)?;
    let [l, a, b] = head else {
        bail!(ColorError::LabConversionFailed);
    };

    let prec0 = l.bit_depth;
    let prec1 = a.bit_depth;
    let prec2 = b.bit_depth;
    if prec0 < 4 || prec1 < 4 || prec2 < 4 {
        bail!(ColorError::LabConversionFailed);
    }

    let rl = lab.rl.unwrap_or(100);
    let ra = lab.ra.unwrap_or(170);
    let rb = lab.rb.unwrap_or(200);
    let ol = lab.ol.unwrap_or(0);
    let a_shift = bit_depth
        .checked_sub(1)
        .ok_or(ColorError::LabConversionFailed)?;
    let b_high_shift = bit_depth
        .checked_sub(2)
        .ok_or(ColorError::LabConversionFailed)?;
    let b_low_shift = bit_depth
        .checked_sub(3)
        .ok_or(ColorError::LabConversionFailed)?;
    let default_a_offset = clamped_power_of_two_u32(a_shift);
    let default_b_offset = clamped_add_u32(
        clamped_power_of_two_u32(b_high_shift),
        clamped_power_of_two_u32(b_low_shift),
    );
    let oa = lab.oa.unwrap_or(default_a_offset);
    let ob = lab.ob.unwrap_or(default_b_offset);

    let min_l = -(rl as f32 * ol as f32) / ((1_u64 << u32::from(prec0)) - 1) as f32;
    let max_l = min_l + rl as f32;
    let min_a = -(ra as f32 * oa as f32) / ((1_u64 << u32::from(prec1)) - 1) as f32;
    let max_a = min_a + ra as f32;
    let min_b = -(rb as f32 * ob as f32) / ((1_u64 << u32::from(prec2)) - 1) as f32;
    let max_b = min_b + rb as f32;
    let bit_max = max_value_for_bit_depth(bit_depth);

    let divisor_l = ((1_u64 << u32::from(prec0)) - 1) as f32;
    let divisor_a = ((1_u64 << u32::from(prec1)) - 1) as f32;
    let divisor_b = ((1_u64 << u32::from(prec2)) - 1) as f32;
    let scale_l_final = bit_max as f32 / 100.0;
    let scale_ab_final = bit_max as f32 / 255.0;

    let l_offset_v = f32x8::splat(simd, min_l * scale_l_final);
    let l_scale_v = f32x8::splat(simd, (max_l - min_l) / divisor_l * scale_l_final);
    let a_offset_v = f32x8::splat(simd, (min_a + 128.0) * scale_ab_final);
    let a_scale_v = f32x8::splat(simd, (max_a - min_a) / divisor_a * scale_ab_final);
    let b_offset_v = f32x8::splat(simd, (min_b + 128.0) * scale_ab_final);
    let b_scale_v = f32x8::splat(simd, (max_b - min_b) / divisor_b * scale_ab_final);

    for ((l_chunk, a_chunk), b_chunk) in l
        .container
        .chunks_exact_mut(SIMD_WIDTH)
        .zip(a.container.chunks_exact_mut(SIMD_WIDTH))
        .zip(b.container.chunks_exact_mut(SIMD_WIDTH))
    {
        let l_v = f32x8::from_slice(simd, l_chunk);
        let a_v = f32x8::from_slice(simd, a_chunk);
        let b_v = f32x8::from_slice(simd, b_chunk);
        l_v.mul_add(l_scale_v, l_offset_v).store(l_chunk);
        a_v.mul_add(a_scale_v, a_offset_v).store(a_chunk);
        b_v.mul_add(b_scale_v, b_offset_v).store(b_chunk);
    }

    l.integer_container = None;
    a.integer_container = None;
    b.integer_container = None;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{clamped_add_u32, clamped_power_of_two_u32, max_value_for_bit_depth};

    #[test]
    fn lab_integer_scaling_preserves_clamped_boundaries() {
        assert_eq!(clamped_power_of_two_u32(31), 1_u32 << 31);
        assert_eq!(clamped_power_of_two_u32(32), u32::MAX);
        assert_eq!(clamped_add_u32(u32::MAX, 1), u32::MAX);
        assert_eq!(max_value_for_bit_depth(31), (1_u32 << 31) - 1);
        assert_eq!(max_value_for_bit_depth(32), u32::MAX);
    }
}
