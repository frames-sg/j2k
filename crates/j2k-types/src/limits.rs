// SPDX-License-Identifier: MIT OR Apache-2.0

/// Maximum JPEG 2000 Part 1 component count representable by `Csiz`.
#[doc(hidden)]
pub const MAX_JPEG2000_PART1_COMPONENTS: u16 = 16_384;

/// Maximum JPEG 2000 Part 1 component precision.
#[doc(hidden)]
pub const MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH: u8 = 38;

/// The deepest sample precision at which reversible coding may run in `f32`.
///
/// The reversible colour transform and the 5/3 wavelet are exact in `f32`
/// only while every value they compute is a whole number of magnitude at
/// most 2^24. The largest is the sum a lifting step takes of two neighbours.
/// Through any number of decomposition levels that sum stays below 16 times
/// the largest magnitude entering the wavelet, and a chroma sample of
/// `P`-bit input reaches 2^P - 1, so the sums stay within 2^24 through 20
/// bits. Deeper samples take the integer path, on encode and on decode.
/// `j2k-native`'s `tests/reversible_exactness.rs` holds the limit in place
/// from both sides.
#[doc(hidden)]
pub const MAX_F32_EXACT_REVERSIBLE_SAMPLE_BIT_DEPTH: u8 = 20;

/// The deepest samples an `f32` plane holds as whole numbers: what the
/// component-plane and row outputs of at most this precision are made of.
///
/// A representation limit, not a coding rule: reversible coding of samples
/// up to this depth may still need integers
/// ([`reversible_coding_needs_integers`]).
#[doc(hidden)]
pub const MAX_F32_PLANE_SAMPLE_BIT_DEPTH: u8 = 24;

/// Whether reversible coding of `bit_depth`-bit samples needs integers,
/// because `f32` would not hold every value of its transforms.
#[doc(hidden)]
#[must_use]
pub const fn reversible_coding_needs_integers(bit_depth: u8) -> bool {
    bit_depth > MAX_F32_EXACT_REVERSIBLE_SAMPLE_BIT_DEPTH
}

/// The largest exponent a QCD/QCC step-size field signals (5 bits).
#[doc(hidden)]
pub const MAX_QUANTIZATION_EXPONENT: u16 = 31;

/// The most guard bits a QCD/QCC `Sqcd` field signals (3 bits).
#[doc(hidden)]
pub const MAX_QUANTIZATION_GUARD_BITS: u8 = 7;

/// The most magnitude bitplanes a classic reversible no-quantization
/// sub-band can signal: `Mb = G + exponent - 1` (T.800 E-2) at the field
/// maxima.
#[doc(hidden)]
pub const MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES: u16 =
    MAX_QUANTIZATION_GUARD_BITS as u16 + MAX_QUANTIZATION_EXPONENT - 1;

/// The most coded bitplanes an HTJ2K code-block carries (T.814).
#[doc(hidden)]
pub const MAX_HTJ2K_CODED_BITPLANES: u8 = 31;
