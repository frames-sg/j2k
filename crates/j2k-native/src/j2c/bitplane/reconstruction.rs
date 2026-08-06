// SPDX-License-Identifier: MIT OR Apache-2.0

use super::state::Coefficient;
use j2k_codec_math::classic::irreversible_midpoint_bit;

/// Reconstruct an irreversible coefficient at the centre of its final
/// decoded quantization interval, as permitted by T.800 E.1.1.2.
#[expect(
    clippy::cast_precision_loss,
    reason = "irreversible JPEG 2000 coefficients enter the codec f32 domain here"
)]
pub(super) fn reconstruct_irreversible_midpoint(
    coefficient: Coefficient,
    decoded_bitplanes: u8,
    number_of_coding_passes: u8,
    roi_shift: u8,
) -> f32 {
    let signed = coefficient.get_i64();
    let magnitude = signed.unsigned_abs();
    if magnitude == 0 || decoded_bitplanes == 0 || number_of_coding_passes == 0 {
        return 0.0;
    }

    let Some(lowest_decoded_bit) = irreversible_midpoint_bit(
        magnitude,
        u32::from(decoded_bitplanes),
        u32::from(number_of_coding_passes),
    ) else {
        // Callers validate pass metadata; keep this shared arithmetic boundary total.
        return signed as f32;
    };

    // A doubled unsigned representation preserves the half-bin term and has
    // headroom for the decoder's 63-bit coefficient limit.
    let mut fixed_magnitude = (u128::from(magnitude) << 1) | (1_u128 << lowest_decoded_bit);
    if roi_shift != 0 {
        let threshold = 1_u128 << u32::from(roi_shift);
        if fixed_magnitude >= threshold {
            fixed_magnitude >>= roi_shift;
        }
    }

    let reconstructed = fixed_magnitude as f32 * 0.5;
    if signed < 0 {
        -reconstructed
    } else {
        reconstructed
    }
}
