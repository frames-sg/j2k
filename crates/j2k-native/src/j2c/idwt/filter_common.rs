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
