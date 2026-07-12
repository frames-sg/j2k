// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible preparation of padded classic Tier-1 coefficient state.

use alloc::vec::Vec;

#[cfg(test)]
use super::super::coefficient_view::legacy_coefficient_view_error;
use super::super::coefficient_view::{CoefficientBlockView, SignedCoefficient};
use super::super::encode::allocation::try_untracked_vec_filled;
use super::passes::NEGATIVE;
use crate::{EncodeError, EncodeResult};

#[cfg(test)]
pub(super) fn prepare_padded_coefficients(
    coefficients: &[i64],
    width: usize,
    height: usize,
    padded_width: usize,
) -> (Vec<u64>, Vec<u8>) {
    let view = CoefficientBlockView::try_contiguous(coefficients, width, height)
        .map_err(legacy_coefficient_view_error)
        .unwrap_or_else(|detail| panic!("invalid contiguous classic coefficient block: {detail}"));
    try_prepare_padded_coefficients_from_view(view, padded_width)
        .expect("test classic padded coefficient allocation")
}

pub(super) fn try_prepare_padded_coefficients_from_view<T: SignedCoefficient>(
    coefficients: CoefficientBlockView<'_, T>,
    padded_width: usize,
) -> EncodeResult<(Vec<u64>, Vec<u8>)> {
    let padded_height =
        coefficients
            .height()
            .checked_add(2)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "classic Tier-1 padded height",
            })?;
    let padded_coefficients =
        padded_width
            .checked_mul(padded_height)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "classic Tier-1 padded coefficient count",
            })?;
    let mut magnitudes = try_untracked_vec_filled(
        padded_coefficients,
        0_u64,
        "classic Tier-1 padded magnitudes",
    )?;
    let mut states = try_untracked_vec_filled(
        padded_coefficients,
        0_u8,
        "classic Tier-1 coefficient states",
    )?;

    for (y, source) in coefficients.rows().enumerate() {
        for (x, &coefficient) in source.iter().enumerate() {
            let idx = (y + 1) * padded_width + (x + 1);
            magnitudes[idx] = coefficient.unsigned_magnitude();
            if coefficient.is_negative() {
                states[idx] = NEGATIVE;
            }
        }
    }

    Ok((magnitudes, states))
}
