// SPDX-License-Identifier: MIT OR Apache-2.0

/// Part of the `1D_EXTR` procedure, defined in F.3.7.
///
/// Applies the period symmetric extension on the left side.
#[expect(
    clippy::inline_always,
    reason = "the boundary extension primitive is intentionally inlined into horizontal and vertical lifting loops"
)]
#[inline(always)]
pub(super) fn periodic_symmetric_extension_left(idx: usize, offset: usize) -> usize {
    offset.abs_diff(idx)
}

/// Part of the `1D_EXTR` procedure, defined in F.3.7.
///
/// Applies the period symmetric extension on the right side.
#[expect(
    clippy::inline_always,
    reason = "the boundary extension primitive is intentionally inlined into horizontal and vertical lifting loops"
)]
#[inline(always)]
pub(super) fn periodic_symmetric_extension_right(
    idx: usize,
    offset: usize,
    length: usize,
) -> usize {
    let new_idx = idx + offset;
    if new_idx >= length {
        let overshoot = new_idx - length;
        length - 2 - overshoot
    } else {
        new_idx
    }
}

/// Equation (F-5) of the reversible 5-3 synthesis on integer-valued samples:
/// `s - floor((left + right + 2) / 4)`, computed in integer arithmetic.
///
/// The `f32` decode path holds integer-valued 5-3 coefficients, so integer
/// operations give the exact result without floor or fused multiply-add
/// instructions. Baseline x86-64 has neither, and each is a library call
/// there.
#[expect(
    clippy::inline_always,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "inlined into the scalar 5-3 lifting loops; integer-valued f32 samples convert to i64 exactly, and the lifted sample is integer-valued in the same f32 range"
)]
#[inline(always)]
pub(super) fn reversible_53_predict(s: f32, left: f32, right: f32) -> f32 {
    s - ((left as i64 + right as i64 + 2) >> 2) as f32
}

/// Equation (F-6) of the reversible 5-3 synthesis on integer-valued samples:
/// `s + floor((left + right) / 2)`, computed in integer arithmetic as
/// [`reversible_53_predict`] is.
#[expect(
    clippy::inline_always,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "inlined into the scalar 5-3 lifting loops; integer-valued f32 samples convert to i64 exactly, and the lifted sample is integer-valued in the same f32 range"
)]
#[inline(always)]
pub(super) fn reversible_53_update(s: f32, left: f32, right: f32) -> f32 {
    s + ((left as i64 + right as i64) >> 1) as f32
}

#[expect(
    clippy::inline_always,
    reason = "the reversible lifting division primitive is intentionally inlined into IDWT hot loops"
)]
#[inline(always)]
pub(super) fn floor_div_i64(numerator: i64, denominator: i64) -> i64 {
    debug_assert!(denominator > 0);
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if remainder != 0 && remainder < 0 {
        quotient - 1
    } else {
        quotient
    }
}
