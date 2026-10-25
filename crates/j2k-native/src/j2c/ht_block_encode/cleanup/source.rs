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
    /// Exponents and `MagSgn` values, at cleanup shift `p`, of the
    /// `exponents.len()` samples from row-major index `start` of one row.
    fn fill_row(
        &self,
        start: usize,
        p: u32,
        exponents: &mut [u8],
        values: &mut [u32],
    ) -> Result<(), &'static str>;
}

/// Exponent `E` and `MagSgn` value `2 * (mu - 1) + sign` of a row's samples,
/// given each one's aligned sign-magnitude form. `E` is 0 for an
/// insignificant sample, whose value is never emitted. An indexed loop
/// without branches is what LLVM vectorizes (NEON `clz`); a zipped iterator
/// chain stays scalar.
#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "a u32 bit width fits u8, and this indexed row loop is the vectorized per-sample work"
)]
#[inline(always)]
fn fill_samples<T: Copy>(
    row: &[T],
    aligned: impl Fn(T) -> u32,
    p: u32,
    exponents: &mut [u8],
    values: &mut [u32],
) {
    let samples = row.len().min(exponents.len()).min(values.len());
    for index in 0..samples {
        let value = aligned(row[index]);
        let doubled = (((value & 0x7FFF_FFFF) << 1) >> p) & !1;
        let width = doubled.wrapping_sub(1).bit_width();
        exponents[index] = if doubled == 0 { 0 } else { width as u8 };
        values[index] = doubled.wrapping_sub(2).wrapping_add(value >> 31);
    }
}

impl CleanupCoefficientSource for [u32] {
    fn fill_row(
        &self,
        start: usize,
        p: u32,
        exponents: &mut [u8],
        values: &mut [u32],
    ) -> Result<(), &'static str> {
        let row = &self[start..start + exponents.len()];
        fill_samples(row, |value| value, p, exponents, values);
        Ok(())
    }
}

pub(in crate::j2c::ht_block_encode) struct I32CleanupCoefficients<'a> {
    pub(in crate::j2c::ht_block_encode) coefficients: &'a [i32],
    pub(in crate::j2c::ht_block_encode) shift: u32,
}

impl CleanupCoefficientSource for I32CleanupCoefficients<'_> {
    fn fill_row(
        &self,
        start: usize,
        p: u32,
        exponents: &mut [u8],
        values: &mut [u32],
    ) -> Result<(), &'static str> {
        let row = &self.coefficients[start..start + exponents.len()];
        let aligned = |value| aligned_sign_magnitude(value, self.shift);
        fill_samples(row, aligned, p, exponents, values);
        Ok(())
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
    fn fill_row(
        &self,
        start: usize,
        p: u32,
        exponents: &mut [u8],
        values: &mut [u32],
    ) -> Result<(), &'static str> {
        let row = self
            .coefficients
            .row(start / self.coefficients.width())
            .ok_or("HTJ2K cleanup row is outside its coefficient block")?;
        let aligned = |value| aligned_sign_magnitude(value, self.shift);
        fill_samples(row, aligned, p, exponents, values);
        Ok(())
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
