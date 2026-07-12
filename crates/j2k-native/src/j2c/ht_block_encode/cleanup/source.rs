// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use alloc::vec::Vec;

use crate::j2c::coefficient_view::CoefficientBlockView;

#[cfg(test)]
pub(in crate::j2c::ht_block_encode) fn convert_nonzero_to_aligned_sign_magnitude_and_max(
    coefficients: &[i32],
    k_max: u8,
) -> Option<(Vec<u32>, u32)> {
    let first_nonzero = coefficients
        .iter()
        .position(|&coefficient| coefficient != 0)?;
    let shift = u32::from(31_u8.saturating_sub(k_max));
    let mut aligned = Vec::with_capacity(coefficients.len());
    aligned.resize(first_nonzero, 0);
    let mut max_magnitude = 0_u32;

    for &coefficient in &coefficients[first_nonzero..] {
        let magnitude = coefficient.unsigned_abs();
        max_magnitude = max_magnitude.max(magnitude);
        if magnitude == 0 {
            aligned.push(0);
        } else {
            let sign = if coefficient < 0 { 0x8000_0000 } else { 0 };
            aligned.push(sign | (magnitude << shift));
        }
    }

    Some((aligned, max_magnitude))
}

pub(in crate::j2c::ht_block_encode) fn max_nonzero_magnitude_view(
    coefficients: CoefficientBlockView<'_, i32>,
) -> Option<u32> {
    let mut max_magnitude = 0_u32;
    for row in coefficients.rows() {
        for &coefficient in row {
            max_magnitude = max_magnitude.max(coefficient.unsigned_abs());
        }
    }
    (max_magnitude != 0).then_some(max_magnitude)
}

pub(in crate::j2c::ht_block_encode) trait CleanupCoefficientSource {
    fn aligned_value(&self, index: usize) -> u32;
}

impl CleanupCoefficientSource for [u32] {
    #[expect(
        clippy::inline_always,
        reason = "coefficient loads are fused into the per-sample cleanup hot path"
    )]
    #[inline(always)]
    fn aligned_value(&self, index: usize) -> u32 {
        self[index]
    }
}

pub(in crate::j2c::ht_block_encode) struct I32CleanupCoefficients<'a> {
    pub(in crate::j2c::ht_block_encode) coefficients: &'a [i32],
    pub(in crate::j2c::ht_block_encode) shift: u32,
}

impl CleanupCoefficientSource for I32CleanupCoefficients<'_> {
    #[expect(
        clippy::inline_always,
        reason = "coefficient conversion is fused into the per-sample cleanup hot path"
    )]
    #[inline(always)]
    fn aligned_value(&self, index: usize) -> u32 {
        aligned_sign_magnitude(self.coefficients[index], self.shift)
    }
}

pub(super) struct I32CleanupBlockView<'a> {
    coefficients: CoefficientBlockView<'a, i32>,
    shift: u32,
}

impl<'a> I32CleanupBlockView<'a> {
    pub(super) fn new(coefficients: CoefficientBlockView<'a, i32>, shift: u32) -> Self {
        Self {
            coefficients,
            shift,
        }
    }
}

impl CleanupCoefficientSource for I32CleanupBlockView<'_> {
    #[expect(
        clippy::inline_always,
        reason = "strided coefficient loads are fused into the per-sample cleanup hot path"
    )]
    #[inline(always)]
    fn aligned_value(&self, index: usize) -> u32 {
        let coefficient = self.coefficients.value_at_linear_index(index);
        aligned_sign_magnitude(coefficient, self.shift)
    }
}

#[expect(
    clippy::inline_always,
    reason = "sign-magnitude conversion runs once per sample in the cleanup hot path"
)]
#[inline(always)]
fn aligned_sign_magnitude(coefficient: i32, shift: u32) -> u32 {
    let magnitude = coefficient.unsigned_abs();
    if magnitude == 0 {
        0
    } else {
        let sign = if coefficient < 0 { 0x8000_0000 } else { 0 };
        sign | (magnitude << shift)
    }
}
