// SPDX-License-Identifier: MIT OR Apache-2.0

/// Whether a decoded coefficient belongs to the region of interest under ROI
/// maxshift `roi_shift` (T.800 H.2), so the decoder scales it back down by
/// `2^roi_shift`.
///
/// The encoder scales region coefficients by `2^s`, with `s` chosen so every
/// background magnitude stays below `2^s`; the region coefficients are then
/// exactly those of magnitude at least `2^s`. `magnitude` is the decoded
/// integer magnitude, with or without a midpoint offset below its lowest
/// decoded bit, which cannot carry it across `2^s`. A decoder that holds
/// magnitudes doubled, to keep a half-bin term, must still compare the
/// undoubled value; comparing the doubled one against `2^s` takes background
/// magnitudes from `2^(s-1)` for region ones.
// `#[inline]` exports the MIR the cuda-oxide device compiler needs.
#[inline]
#[must_use]
pub const fn roi_maxshift_is_region(magnitude: u64, roi_shift: u32) -> bool {
    roi_shift != 0 && roi_shift < u64::BITS && magnitude >> roi_shift != 0
}

#[cfg(test)]
mod tests {
    use super::roi_maxshift_is_region;

    #[test]
    fn region_starts_at_two_to_the_shift() {
        for shift in 1..u64::BITS {
            let threshold = 1_u64 << shift;
            assert!(!roi_maxshift_is_region(threshold - 1, shift), "{shift}");
            assert!(!roi_maxshift_is_region(threshold >> 1, shift), "{shift}");
            assert!(roi_maxshift_is_region(threshold, shift), "{shift}");
            assert!(roi_maxshift_is_region(u64::MAX, shift), "{shift}");
        }
    }

    #[test]
    fn no_shift_or_an_oversized_one_has_no_region() {
        assert!(!roi_maxshift_is_region(u64::MAX, 0));
        assert!(!roi_maxshift_is_region(u64::MAX, u64::BITS));
        assert!(!roi_maxshift_is_region(u64::MAX, u32::MAX));
    }
}
