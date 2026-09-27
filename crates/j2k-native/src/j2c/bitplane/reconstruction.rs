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

/// Per-code-block form of [`reconstruct_irreversible_midpoint`]. The pass
/// arithmetic is resolved once per block, and the common case (no ROI shift)
/// needs only a `u64` shift-or and a hardware integer-to-float conversion per
/// coefficient; `u128` has no hardware conversion. Every other case defers to
/// the reference function, so results are bit-identical.
#[derive(Clone, Copy)]
pub(crate) struct MidpointReconstructor {
    /// Lowest decoded bit and whether the final pass was significance
    /// propagation, when the fast path applies.
    fast: Option<(u32, bool)>,
    decoded_bitplanes: u8,
    number_of_coding_passes: u8,
    roi_shift: u8,
}

impl MidpointReconstructor {
    pub(crate) fn new(decoded_bitplanes: u8, number_of_coding_passes: u8, roi_shift: u8) -> Self {
        let fast = if roi_shift != 0 || decoded_bitplanes == 0 || number_of_coding_passes == 0 {
            None
        } else {
            let final_pass = u32::from(number_of_coding_passes) - 1;
            u32::from(decoded_bitplanes)
                .checked_sub(final_pass.div_ceil(3) + 1)
                .filter(|&lowest| lowest < u64::BITS)
                .map(|lowest| (lowest, final_pass % 3 == 1))
        };
        Self {
            fast,
            decoded_bitplanes,
            number_of_coding_passes,
            roi_shift,
        }
    }

    /// Reconstruct and dequantize a row: `output[i] = reconstruct(c[i]) * step`
    /// for as many elements as both slices hold.
    ///
    /// With at most 31 decoded bitplanes, magnitudes and their doubled
    /// midpoints fit in `u32`, and the per-coefficient branches become selects
    /// that vectorize. Halving is exact, so `fixed * (step / 2)` rounds like
    /// `(fixed / 2) * step`, and negation is a sign-bit flip.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "magnitudes are below 2^31 on the u32 path, which enters the f32 domain"
    )]
    pub(crate) fn dequantize_row(
        self,
        coefficients: &[Coefficient],
        output: &mut [f32],
        step: f32,
    ) {
        let Some((lowest, final_sigprop)) = self.fast.filter(|_| self.decoded_bitplanes <= 31)
        else {
            for (sample, &coefficient) in output.iter_mut().zip(coefficients) {
                *sample = self.reconstruct(coefficient) * step;
            }
            return;
        };
        let half_step = step * 0.5;
        let refine = u32::from(final_sigprop);
        for (sample, &coefficient) in output.iter_mut().zip(coefficients) {
            let bits = coefficient.to_bits();
            let magnitude = bits as u32;
            let negative = (bits >> 63) as u32;
            let bit = lowest + (refine & !(magnitude >> lowest) & 1);
            let value = ((magnitude << 1) | (1 << bit)) as f32 * half_step;
            let value = f32::from_bits(value.to_bits() ^ (negative << 31));
            *sample = if magnitude == 0 { 0.0 } else { value };
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::inline_always,
        reason = "irreversible coefficients enter the f32 domain here, once per coefficient"
    )]
    #[inline(always)]
    pub(crate) fn reconstruct(self, coefficient: Coefficient) -> f32 {
        let Some((lowest, final_sigprop)) = self.fast else {
            return reconstruct_irreversible_midpoint(
                coefficient,
                self.decoded_bitplanes,
                self.number_of_coding_passes,
                self.roi_shift,
            );
        };
        let signed = coefficient.get_i64();
        let magnitude = signed.unsigned_abs();
        if magnitude == 0 {
            return 0.0;
        }
        // A final significance pass leaves already-significant coefficients
        // unrefined; see `irreversible_midpoint_bit`.
        let bit = lowest + u32::from(final_sigprop && magnitude & (1_u64 << lowest) == 0);
        if bit >= u64::BITS {
            return signed as f32;
        }
        // Magnitudes are below 2^63, so the doubled value fits in a u64.
        let reconstructed = ((magnitude << 1) | (1_u64 << bit)) as f32 * 0.5;
        if signed < 0 {
            -reconstructed
        } else {
            reconstructed
        }
    }
}
