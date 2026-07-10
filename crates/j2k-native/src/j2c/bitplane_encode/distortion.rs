// SPDX-License-Identifier: MIT OR Apache-2.0

// Per-segment distortion reduction used by PCRD allocation.

pub(super) fn segment_distortion_delta(
    coefficients: &[i64],
    start_coding_pass: u8,
    end_coding_pass: u8,
    num_bitplanes: u8,
) -> f64 {
    let before =
        coefficient_distortion_after_passes(coefficients, start_coding_pass, num_bitplanes);
    let after = coefficient_distortion_after_passes(coefficients, end_coding_pass, num_bitplanes);
    (before - after).max(f64::EPSILON)
}

fn coefficient_distortion_after_passes(
    coefficients: &[i64],
    completed_passes: u8,
    num_bitplanes: u8,
) -> f64 {
    coefficients
        .iter()
        .map(|coefficient| {
            let magnitude = coefficient.unsigned_abs();
            let reconstructed =
                reconstructed_magnitude_after_passes(magnitude, completed_passes, num_bitplanes);
            let error = magnitude.saturating_sub(reconstructed) as f64;
            error * error
        })
        .sum()
}

fn reconstructed_magnitude_after_passes(
    magnitude: u64,
    completed_passes: u8,
    num_bitplanes: u8,
) -> u64 {
    if magnitude == 0 || completed_passes == 0 || num_bitplanes == 0 {
        return 0;
    }

    let deepest_coded_bitplane = completed_passes
        .saturating_sub(1)
        .div_ceil(3)
        .min(num_bitplanes.saturating_sub(1));
    let retained_bitplanes = deepest_coded_bitplane.saturating_add(1);
    if retained_bitplanes >= num_bitplanes {
        return magnitude;
    }

    let lower_bits = u32::from(num_bitplanes - retained_bitplanes);
    let mask = !((1u64 << lower_bits) - 1);
    magnitude & mask
}
