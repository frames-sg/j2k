// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::math::bit_width_u64;

/// Encode the smallest T.814 Table 4 magnitude set that covers `required_bound`.
pub(crate) fn encode_magnitude_bound(required_bound: u8) -> u16 {
    match required_bound.clamp(8, 74) {
        8 => 0,
        value @ 9..=27 => u16::from(value - 8),
        28..=31 => 20,
        value @ 32..=71 => u16::from(20 + (value - 31).div_ceil(4)),
        _ => 31,
    }
}

/// Return the smallest integer `B` satisfying the T.814 8.7.3 cleanup bound.
///
/// `maximum` is measured after quantization and ROI maxshift, at the cleanup
/// pass bit-plane. The CAP serializer rounds the result up to a legal Table 4
/// set through the internal `encode_magnitude_bound` serializer.
#[must_use]
pub fn required_magnitude_bound(maximum: u64, reversible: bool, decomposition_level: u8) -> u8 {
    let magnitude_bits = bit_width_u64(maximum);
    let required = if reversible {
        magnitude_bits
    } else {
        magnitude_bits
            .saturating_add(1)
            .saturating_sub(decomposition_level)
    };
    required.clamp(8, 74)
}

#[cfg(test)]
mod tests {
    use super::{encode_magnitude_bound, required_magnitude_bound};

    #[test]
    fn cap_parameter_selects_the_smallest_legal_magnitude_set() {
        let cases = [
            (0, 0),
            (8, 0),
            (9, 1),
            (27, 19),
            (28, 20),
            (31, 20),
            (32, 21),
            (35, 21),
            (36, 22),
            (71, 30),
            (72, 31),
            (74, 31),
            (u8::MAX, 31),
        ];

        for (required_bound, expected_parameter) in cases {
            assert_eq!(
                encode_magnitude_bound(required_bound),
                expected_parameter,
                "required B={required_bound}"
            );
        }
    }

    #[test]
    fn actual_cleanup_magnitude_selects_the_required_transform_bound() {
        let cases = [
            // (maximum cleanup magnitude, reversible, decomposition level, B)
            (0, true, 0, 8),
            (255, true, 0, 8),
            (256, true, 0, 9),
            (255, false, 0, 9),
            (255, false, 1, 8),
            (4095, false, 1, 12),
            (4095, false, 5, 8),
            (1 << 30, false, 0, 32),
        ];

        for (maximum, reversible, decomposition_level, expected) in cases {
            assert_eq!(
                required_magnitude_bound(maximum, reversible, decomposition_level),
                expected,
                "maximum={maximum}, reversible={reversible}, level={decomposition_level}"
            );
        }
    }
}
