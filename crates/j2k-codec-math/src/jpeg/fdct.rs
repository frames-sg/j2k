//! Integer forward DCT and quantization of baseline 8-bit JPEG.
//!
//! This module is the workspace's one definition of the baseline encoders'
//! transform. The CPU encoder, the CUDA-Oxide encode kernel and the Metal
//! encode shaders produce exactly its coefficients; the shaders read the basis
//! from a constant fragment that `j2k-jpeg-metal`'s build script renders from
//! [`BASIS_HIGH`](crate::jpeg::fdct::BASIS_HIGH) and
//! [`BASIS_LOW`](crate::jpeg::fdct::BASIS_LOW).
//!
//! With `K_u(x) = C(u) / 2 * cos((2x + 1) u pi / 16)`, `K_u(7 - x) = (-1)^u
//! K_u(x)`, so an 8x8 block of level-shifted samples folds exactly into four
//! 4x4 integer quadrants, one per `(u, v)` parity, each within `[-512, 508]`.
//! Every coefficient is a 16-term sum of one fold against the basis
//! `K_u(x) K_v(y) * 2^32`, each entry rounded to the nearest integer, and is
//! quantized by rounding half away from zero. Integer arithmetic makes the
//! result independent of the device and of any floating-point mode:
//!
//! - every basis entry is within half a unit of the real value, so the sum is
//!   within `2^-20` of the real DCT, and a coefficient rounds as the real DCT
//!   does unless the real quotient lies within `2^-20` of a half-integer;
//! - the entries of the frequencies with `u` and `v` in `{0, 4}` are exactly
//!   `+-2^29`, the rational `+-1/8`, so their rounding ties are exact.
//!
//! The basis is computed at compile time from nested square roots in `u128`
//! fixed point, so no floating-point cosine is involved anywhere.
//!
//! Every function is `#[inline]`, which exports the MIR the cuda-oxide device
//! compiler needs.

/// Fraction bits of the fixed-point basis: each entry is
/// `K_u(x) K_v(y) * 2^FRACTION_BITS`.
pub const FRACTION_BITS: u32 = 32;

/// Signed high halves of the folded basis.
///
/// Entry `natural * 16 + y * 4 + x` belongs to natural-order coefficient
/// `natural` and fold position `(x, y)`, `x, y < 4`; the basis entry is
/// `BASIS_HIGH[i] * 2^16 + BASIS_LOW[i]`. The split keeps both 16-term sums of
/// a coefficient inside `i32` on devices without fast 64-bit multiplication.
pub const BASIS_HIGH: [i16; 1024] = basis_high_halves(&BASIS);

/// Unsigned low halves of the folded basis, in `0..2^16`; see [`BASIS_HIGH`].
pub const BASIS_LOW: [u16; 1024] = basis_low_halves(&BASIS);

/// [`BASIS_HIGH`] and [`BASIS_LOW`] as statics, which the transform indexes.
/// A static is read in place. The cuda-oxide device compiler instead copies a
/// dynamically indexed `const` array onto the stack at every use, 4 KiB of
/// local memory per coefficient.
static BASIS_HIGH_TABLE: [i16; 1024] = BASIS_HIGH;
static BASIS_LOW_TABLE: [u16; 1024] = BASIS_LOW;

/// `2^62`, the scale of the fixed-point cosines.
const UNIT: u128 = 1 << 62;

/// `cos(k * pi / 16) * 2^62`, rounded down, for `k` in `0..=8`.
const COSINES: [u128; 9] = cosines();

const BASIS: [i32; 1024] = basis();

/// The square root of a `2^62` fixed-point value, in `2^62` fixed point and
/// rounded down. Callers pass values below `2^64`.
const fn fixed_sqrt(value: u128) -> u128 {
    (value * UNIT).isqrt()
}

/// `cos(k pi / 16)` from `cos(theta / 2) = sqrt(2 + 2 cos(theta)) / 2` and
/// `sin(theta / 2) = sqrt(2 - 2 cos(theta)) / 2`, starting at `cos(pi / 4)`.
const fn cosines() -> [u128; 9] {
    let sqrt2 = fixed_sqrt(2 * UNIT);
    let two_cos_pi_8 = fixed_sqrt(2 * UNIT + sqrt2);
    let two_sin_pi_8 = fixed_sqrt(2 * UNIT - sqrt2);
    [
        UNIT,
        fixed_sqrt(2 * UNIT + two_cos_pi_8) / 2,
        two_cos_pi_8 / 2,
        fixed_sqrt(2 * UNIT + two_sin_pi_8) / 2,
        sqrt2 / 2,
        fixed_sqrt(2 * UNIT - two_sin_pi_8) / 2,
        two_sin_pi_8 / 2,
        fixed_sqrt(2 * UNIT - two_cos_pi_8) / 2,
        0,
    ]
}

/// `cos(m pi / 16)` as an index into [`COSINES`] and whether it is negative.
const fn cosine_term(m: usize) -> (usize, bool) {
    let m = m % 32;
    let m = if m > 16 { 32 - m } else { m };
    if m > 8 {
        (16 - m, true)
    } else {
        (m, false)
    }
}

/// `K_u(x) K_v(y) * 2^32`, rounded to the nearest integer, in fold order.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a basis entry is at most 2^30 in magnitude"
)]
const fn basis() -> [i32; 1024] {
    let mut basis = [0i32; 1024];
    let mut index = 0;
    while index < 1024 {
        let (natural, position) = (index / 16, index % 16);
        let (horizontal, vertical) = (natural % 8, natural / 8);
        let (column, row) = (position % 4, position / 4);
        let (first_term, first_negative) = cosine_term((2 * column + 1) * horizontal);
        let (second_term, second_negative) = cosine_term((2 * row + 1) * vertical);
        // K_u(x) K_v(y) = C(u) C(v) / 4 * cos(a) cos(b) with C(0) = cos(pi / 4)
        // multiplying the factor cos(0) = 1 of a zero frequency. Both zero
        // leaves the rational 1/8.
        basis[index] = if horizontal == 0 && vertical == 0 {
            1 << 29
        } else {
            let (first, second) = if horizontal == 0 {
                (4, second_term)
            } else if vertical == 0 {
                (first_term, 4)
            } else {
                (first_term, second_term)
            };
            // (product / 2^124) / 4 * 2^32, rounded half up.
            let product = COSINES[first] * COSINES[second];
            let magnitude = ((product + (1 << 93)) >> 94) as i32;
            if first_negative == second_negative {
                magnitude
            } else {
                -magnitude
            }
        };
        index += 1;
    }
    basis
}

const fn basis_high_halves(basis: &[i32; 1024]) -> [i16; 1024] {
    let mut halves = [0i16; 1024];
    let mut index = 0;
    while index < 1024 {
        halves[index] = (basis[index] >> 16) as i16;
        index += 1;
    }
    halves
}

#[expect(clippy::cast_sign_loss, reason = "the masked low half lies in 0..2^16")]
const fn basis_low_halves(basis: &[i32; 1024]) -> [u16; 1024] {
    let mut halves = [0u16; 1024];
    let mut index = 0;
    while index < 1024 {
        halves[index] = (basis[index] & 0xffff) as u16;
        index += 1;
    }
    halves
}

/// The fold of `block` for `u` parity `parity & 1` and `v` parity
/// `parity >> 1`, at fold positions `y * 4 + x`.
#[inline]
const fn fold(block: &[u8; 64], parity: usize) -> [i32; 16] {
    let u_sign = if parity & 1 == 0 { 1 } else { -1 };
    let v_sign = if parity & 2 == 0 { 1 } else { -1 };
    let mut folded = [0i32; 16];
    let mut position = 0;
    while position < 16 {
        let (x, y) = (position % 4, position / 4);
        let near_near = block[y * 8 + x] as i32 - 128;
        let near_far = block[y * 8 + 7 - x] as i32 - 128;
        let far_near = block[(7 - y) * 8 + x] as i32 - 128;
        let far_far = block[(7 - y) * 8 + 7 - x] as i32 - 128;
        folded[position] = (near_near + u_sign * near_far) + v_sign * (far_near + u_sign * far_far);
        position += 1;
    }
    folded
}

/// `F(u, v) * 2^32` for natural coefficient `natural` from its fold. Both
/// 16-term sums stay inside `i32`: `|fold| <= 512` and every half is below
/// `2^16` in magnitude.
#[inline]
const fn scaled_coefficient(folded: &[i32; 16], natural: usize) -> i64 {
    let mut high = 0i32;
    let mut low = 0i32;
    let mut position = 0;
    while position < 16 {
        let index = natural * 16 + position;
        high += folded[position] * BASIS_HIGH_TABLE[index] as i32;
        low += folded[position] * BASIS_LOW_TABLE[index] as i32;
        position += 1;
    }
    high as i64 * 65_536 + low as i64
}

/// Divide `F(u, v) * 2^32` by the quantizer, rounding half away from zero:
/// `floor((|S| + q * 2^31) / (q * 2^32)) = floor(floor(... / 2^32) / q)`.
/// A zero quantizer, which T.81 forbids, divides by one. The integer part is
/// at most 1151, so the division runs in `u32`, which GPUs do natively.
#[inline]
const fn quantize(scaled: i64, quantizer: u8) -> i16 {
    let divisor = if quantizer == 0 { 1 } else { quantizer as u32 };
    let half_step = (divisor as u64) << (FRACTION_BITS - 1);
    let rounded = ((scaled.unsigned_abs() + half_step) >> FRACTION_BITS) as u32;
    let quotient = (rounded / divisor) as i16;
    if scaled < 0 {
        -quotient
    } else {
        quotient
    }
}

/// `u` parity in bit 0 and `v` parity in bit 1 of a natural-order index.
#[inline]
const fn parity(natural: usize) -> usize {
    (natural & 1) | (((natural >> 3) & 1) << 1)
}

/// The quantized forward DCT of one 8x8 block of 8-bit samples.
///
/// `block` and `quant` are in natural (row-major) order, and so is the
/// result. Samples are level-shifted by 128 before the transform.
#[inline]
pub const fn fdct_quantize(block: &[u8; 64], quant: &[u8; 64]) -> [i16; 64] {
    let quadrants = [
        fold(block, 0),
        fold(block, 1),
        fold(block, 2),
        fold(block, 3),
    ];
    let mut coefficients = [0i16; 64];
    let mut natural = 0;
    while natural < 64 {
        let scaled = scaled_coefficient(&quadrants[parity(natural)], natural);
        coefficients[natural] = quantize(scaled, quant[natural]);
        natural += 1;
    }
    coefficients
}

/// One coefficient of [`fdct_quantize`]: natural-order coefficient `natural`
/// of `block`, quantized by `quantizer`. It folds only the quadrant the
/// coefficient needs.
///
/// # Panics
///
/// Panics if `natural` is not below 64.
#[inline]
pub const fn fdct_quantize_coefficient(block: &[u8; 64], quantizer: u8, natural: usize) -> i16 {
    quantize(
        scaled_coefficient(&fold(block, parity(natural)), natural),
        quantizer,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cos(k pi / 16)` for `k` in `0..8` by its Taylor series, independently of
    /// the fixed-point square roots.
    fn reference_cosines() -> [f64; 8] {
        core::array::from_fn(|k| {
            let angle =
                f64::from(u8::try_from(k).expect("index fits u8")) * core::f64::consts::PI / 16.0;
            let (mut term, mut sum) = (1.0, 1.0);
            for n in 1..24u8 {
                let n = f64::from(n);
                term *= -angle * angle / ((2.0 * n - 1.0) * (2.0 * n));
                sum += term;
            }
            sum
        })
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "the comparison tolerates the f64 rounding of 2^62 fixed point"
    )]
    fn fixed_to_f64(value: u128) -> f64 {
        value as f64 / UNIT as f64
    }

    #[test]
    fn fixed_point_cosines_match_the_reference_values() {
        for (k, reference) in reference_cosines().into_iter().enumerate() {
            let error = (fixed_to_f64(COSINES[k]) - reference).abs();
            assert!(error < 1e-15, "cos({k} pi / 16) error {error}");
        }
        assert_eq!(COSINES[8], 0);
    }

    #[test]
    fn halves_recombine_into_entries_within_half_a_unit_of_the_real_basis() {
        let cosines = reference_cosines();
        for index in 0..1024 {
            let entry = i64::from(BASIS_HIGH[index]) * 65_536 + i64::from(BASIS_LOW[index]);
            assert_eq!(entry, i64::from(BASIS[index]));
            let (natural, position) = (index / 16, index % 16);
            let (horizontal, vertical) = (natural % 8, natural / 8);
            let (column, row) = (position % 4, position / 4);
            let factor = |frequency: usize, sample: usize| {
                let (term, negative) = cosine_term((2 * sample + 1) * frequency);
                let cosine = if term == 8 { 0.0 } else { cosines[term] };
                let normalization = if frequency == 0 { cosines[4] } else { 1.0 };
                0.5 * normalization * if negative { -cosine } else { cosine }
            };
            let real = factor(horizontal, column) * factor(vertical, row) * 4_294_967_296.0;
            let error = (f64::from(BASIS[index]) - real).abs();
            assert!(
                error <= 0.5 + 1e-6,
                "entry {index}: {} vs {real}",
                BASIS[index]
            );
        }
    }

    #[test]
    fn rational_frequencies_have_the_exact_basis_one_eighth() {
        for natural in [0, 4, 32, 36] {
            for position in 0..16 {
                assert_eq!(BASIS[natural * 16 + position].unsigned_abs(), 1 << 29);
            }
        }
    }

    /// The exact quantized coefficient, decided in the Q-basis
    /// `{cos(k pi / 16)}_{k < 8}` of the DCT's number field: a rational value
    /// is rounded exactly, and an irrational one, which is never a tie, by
    /// f64 far from any tie.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Exact {
        Value(i16),
        /// A tie at a frequency whose basis entries are all rational.
        RationalBasisTie(i16),
        /// A tie whose irrational basis terms cancel.
        IrrationalBasisTie(i16),
    }

    fn small_f64(value: i64) -> f64 {
        f64::from(i32::try_from(value).expect("DCT component fits i32"))
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::many_single_char_names,
        reason = "the rounded coefficient of an eight-bit block fits i16; u, v, x, y name DCT indices"
    )]
    fn exact_coefficient(
        block: &[u8; 64],
        quantizer: u8,
        natural: usize,
        cosines: &[f64; 8],
    ) -> Exact {
        let (u, v) = (natural % 8, natural / 8);
        let mut components = [0i64; 9];
        for (index, &sample) in block.iter().enumerate() {
            let (x, y) = (index % 8, index / 8);
            let centered = i64::from(sample) - 128;
            let (a, b) = ((2 * x + 1) * u, (2 * y + 1) * v);
            // cos(a) cos(b) = (cos(a - b) + cos(a + b)) / 2.
            for m in [a + 128 - b, a + b] {
                let (k, negative) = cosine_term(m);
                components[k] += if negative { -centered } else { centered };
            }
        }
        // F = C(u) C(v) / 8 * sum n_k c_k; C(0) = cos(4 pi / 16) re-expands
        // when exactly one frequency is zero.
        let (components, denominator) = if (u == 0) == (v == 0) {
            (components, if u == 0 { 16 } else { 8 })
        } else {
            let mut expanded = [0i64; 9];
            for (k, &n) in components.iter().enumerate() {
                for m in [k + 32 - 4, k + 4] {
                    let (j, negative) = cosine_term(m);
                    expanded[j] += if negative { -n } else { n };
                }
            }
            (expanded, 16)
        };
        let divisor = denominator * i64::from(quantizer);
        if components[1..8].iter().all(|&n| n == 0) {
            let twice = 2 * components[0].abs();
            let magnitude =
                i16::try_from((twice + divisor) / (2 * divisor)).expect("coefficient fits i16");
            let value = if components[0] < 0 {
                -magnitude
            } else {
                magnitude
            };
            let tie = twice % divisor == 0 && (twice / divisor) % 2 == 1;
            return match (tie, u % 4 == 0 && v % 4 == 0) {
                (false, _) => Exact::Value(value),
                (true, true) => Exact::RationalBasisTie(value),
                (true, false) => Exact::IrrationalBasisTie(value),
            };
        }
        let value = (0..8)
            .map(|k| small_f64(components[k]) * cosines[k])
            .sum::<f64>()
            / small_f64(divisor);
        assert!(
            (value.abs() % 1.0 - 0.5).abs() > 1e-9,
            "corpus coefficient {natural} is too close to a tie for an f64 reference"
        );
        let rounded = (value.abs() + 0.5) as i16;
        Exact::Value(if value < 0.0 { -rounded } else { rounded })
    }

    struct Lcg(u64);

    impl Lcg {
        fn next_byte(&mut self) -> u8 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0.to_be_bytes()[0]
        }
    }

    /// Uniform noise, smooth gradients and two-level blocks; the last produce
    /// exact ties at many frequencies, rational and irrational.
    fn corpus_block(rng: &mut Lcg, family: usize) -> [u8; 64] {
        match family {
            0 => core::array::from_fn(|_| rng.next_byte()),
            1 => {
                let base = i32::from(rng.next_byte());
                let gx = i32::from(rng.next_byte() % 17) - 8;
                let gy = i32::from(rng.next_byte() % 17) - 8;
                core::array::from_fn(|index| {
                    let x = i32::try_from(index % 8).expect("column fits i32");
                    let y = i32::try_from(index / 8).expect("row fits i32");
                    let jitter = i32::from(rng.next_byte() % 5) - 2;
                    u8::try_from((base + gx * x + gy * y + jitter).clamp(0, 255))
                        .expect("clamped sample fits u8")
                })
            }
            _ => {
                let (high, low) = (rng.next_byte(), rng.next_byte());
                core::array::from_fn(|_| if rng.next_byte() & 1 == 1 { high } else { low })
            }
        }
    }

    /// T.81 Annex K.1 luminance table, in natural order.
    const ANNEX_K_LUMA: [u8; 64] = [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ];

    #[test]
    fn fdct_quantize_rounds_the_exact_dct_half_away_from_zero() {
        let tables: [[u8; 64]; 3] = [
            [1; 64],
            ANNEX_K_LUMA,
            core::array::from_fn(|index| {
                u8::try_from(1 + (index * 37) % 255).expect("quantizer fits u8")
            }),
        ];
        let cosines = reference_cosines();
        let mut rng = Lcg(0x5eed_f00d);
        let (mut rational_ties, mut irrational_ties) = (0usize, 0usize);
        for family in 0..3 {
            for _ in 0..1024 {
                let block = corpus_block(&mut rng, family);
                for quant in &tables {
                    let actual = fdct_quantize(&block, quant);
                    for natural in 0..64 {
                        let coefficient = actual[natural];
                        assert_eq!(
                            fdct_quantize_coefficient(&block, quant[natural], natural),
                            coefficient
                        );
                        match exact_coefficient(&block, quant[natural], natural, &cosines) {
                            Exact::Value(value) => {
                                assert_eq!(coefficient, value, "{block:?} coefficient {natural}");
                            }
                            Exact::RationalBasisTie(value) => {
                                rational_ties += 1;
                                assert_eq!(
                                    coefficient, value,
                                    "tie {block:?} coefficient {natural}"
                                );
                            }
                            Exact::IrrationalBasisTie(value) => {
                                // Within 2^-20 of the tie, the fixed-point
                                // sum may land on either side.
                                irrational_ties += 1;
                                assert!(
                                    coefficient == value || coefficient == value - value.signum(),
                                    "tie {block:?} coefficient {natural}: {coefficient} vs {value}"
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(
            rational_ties > 1000,
            "corpus exercises exact ties: {rational_ties}"
        );
        assert!(
            irrational_ties > 0,
            "corpus exercises irrational-basis ties"
        );
    }

    #[test]
    fn extreme_blocks_quantize_within_the_coefficient_range() {
        let checker: [u8; 64] = core::array::from_fn(|index| {
            if (index % 8 + index / 8) % 2 == 0 {
                255
            } else {
                0
            }
        });
        let ones = [1u8; 64];
        assert_eq!(fdct_quantize(&[0; 64], &ones)[0], -1024);
        assert_eq!(fdct_quantize(&[255; 64], &ones)[0], 1016);
        let coefficients = fdct_quantize(&checker, &ones);
        assert!(coefficients
            .iter()
            .all(|value| value.unsigned_abs() <= 1024));
        assert_eq!(fdct_quantize(&[200; 64], &[0; 64])[0], 576);
    }
}
