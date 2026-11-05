// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integer "slow" IDCT (ISLOW) — Chen-Wang decomposition, 16-bit fixed point.
//! Bit-exact with libjpeg-turbo's `jidctint.c` algorithm on the same inputs.
//! The maths is `j2k_codec_math::jpeg::idct`; these are the CPU entry points.
//!
//! Input: 64 dequantized DCT coefficients in natural (row-major) order, each
//! already multiplied by its quantization entry.
//!
//! Output: 64 u8 pixel samples in natural order, level-shifted by +128 and
//! clamped to `[0, 255]`.
//!
//! Arithmetic wraps modulo 2^32, so intermediate overflow on adversarial
//! (malformed) inputs wraps rather than panicking in debug builds.
//! libjpeg-turbo exhibits the same modular behavior in release.

use j2k_codec_math::jpeg::idct;

/// Inverse DCT of a single 8×8 block, with level shift and clamping.
// Out of line: inlined as the scalar arm of the `Backend` IDCT dispatch, it
// grows the AVX2 4:4:4 decode loops and costs them ~3% (measured).
#[inline(never)]
pub(crate) fn idct_islow(input: &[i16; 64], output: &mut [u8; 64]) {
    if input[32..].iter().all(|&coeff| coeff == 0) {
        idct::islow_block_top_half(input, output);
    } else {
        idct::islow_block(input, output);
    }
}

/// Inverse DCT for blocks whose natural-order rows 4..7 are known to be zero.
// Out of line for the same reason as `idct_islow`.
#[inline(never)]
pub(crate) fn idct_islow_bottom_half_zero(input: &[i16; 64], output: &mut [u8; 64]) {
    idct::islow_block_top_half(input, output);
}

/// Bit-exact DC-only ISLOW path. Equivalent to `idct_islow` when every AC
/// coefficient is zero.
pub(crate) fn idct_islow_dc_only(dc_coeff: i16, output: &mut [u8; 64]) {
    output.fill(idct_islow_dc_only_pixel(dc_coeff));
}

/// Return the uniform output sample produced by the DC-only ISLOW path.
pub(crate) fn idct_islow_dc_only_pixel(dc_coeff: i16) -> u8 {
    idct::sample_u8(idct::islow_dc(i32::from(dc_coeff)))
}

/// Inverse DCT of a 12-bit JPEG block, returning native 0..4095 sample values.
///
/// Mirrors `jidctint.c` built with `BITS_IN_JSAMPLE == 12`: `PASS1_BITS = 1`
/// and `JLONG` (64-bit on LP64 targets) intermediates, so nothing wraps. The
/// zero-AC shortcuts of the C code give the same result as the full
/// computation here and are omitted.
pub(crate) fn idct_islow_12bit(input: &[i16; 64], output: &mut [u16; 64]) {
    idct::islow_block_12bit(input, output);
}

/// Return the uniform native 12-bit sample produced by the DC-only ISLOW path.
pub(crate) fn idct_islow_12bit_dc_only_sample(dc_coeff: i16) -> u16 {
    idct::sample_12bit(i64::from(idct::islow_dc(i32::from(dc_coeff))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zero_input_produces_level_shifted_gray_block() {
        let input = [0i16; 64];
        let mut output = [0u8; 64];
        idct_islow(&input, &mut output);
        for (i, &px) in output.iter().enumerate() {
            assert_eq!(px, 128, "pixel {i} = {px}, expected 128");
        }
    }

    #[test]
    fn dc_only_input_produces_uniform_block() {
        let mut input = [0i16; 64];
        input[0] = 8 * 8;
        let mut output = [0u8; 64];
        idct_islow(&input, &mut output);
        for &px in &output {
            assert!((i32::from(px) - 136).abs() <= 1, "got {px}");
        }
    }

    // -------------------------------------------------------------------------
    // Ground truth: exact mathematical inverse DCT.
    //
    // ISLOW is a fixed-point (Chen-Wang) approximation; nothing previously
    // pinned it to the *defining* inverse DCT, so a wrong constant or butterfly
    // could pass the sanity checks (and be faithfully copied by the CUDA port).
    // Compare every pixel to the exact double-precision IDCT. ISLOW meets the
    // IEEE-1180 peak-error bound, so a 1-LSB tolerance is the correct gate.

    /// Exact 8x8 inverse DCT (DCT-III, orthonormal) of natural-order signed
    /// coefficients, before the +128 level shift. `coeffs[v * 8 + u]` is the
    /// coefficient at horizontal frequency `u`, vertical frequency `v`.
    fn exact_idct_pixel(coeffs: &[i16; 64], x: usize, y: usize) -> f64 {
        use core::f64::consts::PI;
        let alpha = |k: usize| {
            if k == 0 {
                (1.0_f64 / 8.0).sqrt()
            } else {
                (2.0_f64 / 8.0).sqrt()
            }
        };
        let cos_term = |sample: usize, freq: usize| -> f64 {
            let s = f64::from(u8::try_from(sample).unwrap());
            let f = f64::from(u8::try_from(freq).unwrap());
            (((2.0 * s) + 1.0) * f * PI / 16.0).cos()
        };
        let mut acc = 0.0;
        for v in 0..8 {
            for u in 0..8 {
                acc += alpha(u)
                    * alpha(v)
                    * f64::from(coeffs[v * 8 + u])
                    * cos_term(x, u)
                    * cos_term(y, v);
            }
        }
        acc
    }

    /// The exact IDCT pixels with JPEG's +128 level shift and `[0, 255]` clamp.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the mathematical reference rounds and clamps each pixel to the u8 domain before conversion"
    )]
    fn exact_islow_reference(coeffs: &[i16; 64]) -> [u8; 64] {
        let mut out = [0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                let value = exact_idct_pixel(coeffs, x, y) + 128.0;
                out[y * 8 + x] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    fn next_coeff(state: &mut u64) -> i16 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Modest dequantized magnitudes so most pixels stay inside 8-bit range.
        ((*state >> 40) & 0x1ff) as i32 as i16 - 256
    }

    #[test]
    fn islow_matches_exact_idct_within_one_lsb() {
        let mut state = 0x0bad_c0de_1234_5678u64;
        for _ in 0..256 {
            let mut coeffs = [0i16; 64];
            for c in &mut coeffs {
                *c = next_coeff(&mut state);
            }
            // Keep DC moderate so the field is centered in range.
            coeffs[0] = i16::try_from(i32::from(coeffs[0]) / 4).unwrap();

            let mut got = [0u8; 64];
            idct_islow(&coeffs, &mut got);
            let want = exact_islow_reference(&coeffs);
            for i in 0..64 {
                assert!(
                    (i32::from(got[i]) - i32::from(want[i])).abs() <= 1,
                    "pixel {i}: islow={} exact={} (coeffs {coeffs:?})",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the closed-form reference rounds and clamps the expected pixel to the u8 domain"
    )]
    fn islow_dc_only_matches_closed_form() {
        // DC-only block: every pixel is F(0,0)/8 + 128.
        for dc in [-512i16, -200, -64, 8, 64, 200, 512] {
            let mut coeffs = [0i16; 64];
            coeffs[0] = dc;
            let mut got = [0u8; 64];
            idct_islow(&coeffs, &mut got);
            let expected = ((f64::from(dc) / 8.0) + 128.0).round().clamp(0.0, 255.0) as u8;
            for &px in &got {
                assert!(
                    (i32::from(px) - i32::from(expected)).abs() <= 1,
                    "dc={dc}: px={px} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn dc_only_helper_matches_full_idct() {
        let mut input = [0i16; 64];
        input[0] = 73;
        let mut full = [0u8; 64];
        let mut fast = [0u8; 64];
        idct_islow(&input, &mut full);
        idct_islow_dc_only(input[0], &mut fast);
        assert_eq!(fast, full);
        assert_eq!(idct_islow_dc_only_pixel(input[0]), full[0]);
    }

    #[test]
    fn clamps_extreme_coefficients_into_0_255() {
        let mut input = [0i16; 64];
        input[0] = i16::MAX;
        let mut output = [0u8; 64];
        idct_islow(&input, &mut output);
        assert!(output.iter().all(|&px| px == 255));

        let mut input = [0i16; 64];
        input[0] = i16::MIN;
        let mut output = [0u8; 64];
        idct_islow(&input, &mut output);
        assert!(output.iter().all(|&px| px == 0));
    }

    #[test]
    fn roundtrip_identity_basis_reconstructs_8x8_impulse() {
        let mut input = [0i16; 64];
        input[1] = 400;
        let mut output = [0u8; 64];
        idct_islow(&input, &mut output);
        let left = i32::from(output[0]);
        let right = i32::from(output[7]);
        assert!(
            (left - right).abs() > 40,
            "AC[1] basis should produce horizontal variation, got L={left} R={right}"
        );
    }

    #[test]
    fn does_not_panic_on_extreme_adversarial_coefficients() {
        // All maxed-out i16 — intermediate multiplies overflow i32. Wrapping
        // arithmetic makes this produce garbage pixels instead of panicking.
        let input = [i16::MAX; 64];
        let mut output = [0u8; 64];
        idct_islow(&input, &mut output);
        // No panic = success. Output values are not asserted (they are modular
        // garbage by design on adversarial input).
        let _ = output;
    }
}
