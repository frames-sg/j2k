// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-pass squared-error reduction for HT PCRD candidates.

use crate::j2c::coefficient_view::CoefficientBlockView;

pub(super) fn pass_distortion_deltas(
    coefficients: CoefficientBlockView<'_, i32>,
    cleanup_bitplane: u8,
    num_coding_passes: u8,
) -> [f64; 3] {
    let cleanup_mask = !((1u64 << cleanup_bitplane) - 1);
    let refinement_mask = cleanup_bitplane
        .checked_sub(1)
        .map_or(0, |bitplane| 1u64 << bitplane);
    let mut deltas = [0.0; 3];

    for &coefficient in coefficients.rows().flatten() {
        let magnitude = u64::from(coefficient.unsigned_abs());
        let cleanup = magnitude & cleanup_mask;
        let sigprop = if cleanup == 0 {
            magnitude & refinement_mask
        } else {
            cleanup
        };
        let magref = if cleanup == 0 {
            sigprop
        } else {
            cleanup | (magnitude & refinement_mask)
        };
        deltas[0] += squared_error(magnitude, 0) - squared_error(magnitude, cleanup);
        if num_coding_passes > 1 {
            deltas[1] += squared_error(magnitude, cleanup) - squared_error(magnitude, sigprop);
        }
        if num_coding_passes > 2 {
            deltas[2] += squared_error(magnitude, sigprop) - squared_error(magnitude, magref);
        }
    }
    for delta in deltas.iter_mut().take(usize::from(num_coding_passes)) {
        *delta = delta.max(f64::EPSILON);
    }
    deltas
}

#[expect(
    clippy::cast_precision_loss,
    reason = "PCRD distortion is intentionally accumulated in f64 after bounded integer reconstruction"
)]
fn squared_error(magnitude: u64, reconstruction: u64) -> f64 {
    let error = magnitude.saturating_sub(reconstruction) as f64;
    error * error
}
