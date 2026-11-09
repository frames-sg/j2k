//! Canonical binary32 arithmetic for the JPEG-to-HTJ2K irreversible 9/7
//! transcode.
//!
//! Every backend that produces default-precision 9/7 transcode coefficients
//! (CPU scalar and SIMD, Metal, and CUDA) evaluates exactly the IEEE-754
//! binary32 operation sequence defined here, so all of them emit identical
//! quantized coefficients and therefore identical codestream bytes:
//!
//! 1. **IDCT.** Each dequantized coefficient is converted exactly to `f32`.
//!    A block is reconstructed separably, columns first: the vertical value
//!    `T[y][u]` is [`idct8_dot`] of basis row `y` with coefficient column `u`,
//!    and the sample `s[y][x]` is [`idct8_dot`] of basis row `x` with `T[y]`.
//! 2. **Lifting.** Rows are lifted first, then the columns of the row-low and
//!    row-high planes. Each signal runs the four [`DWT97_LIFT_STEPS_F32`]
//!    steps with [`lift97_update`] and whole-sample symmetric extension, then
//!    scales even samples by [`DWT97_LOW_SCALE_F32`] and odd samples by
//!    [`DWT97_HIGH_SCALE_F32`]. Later decomposition levels lift the previous
//!    LL band the same way.
//! 3. **Quantization.** [`quantize97`] multiplies the magnitude by the
//!    host-rounded reciprocal step from [`quantize97_inv_delta`] and truncates.
//!
//! Additions and multiplications are single-rounded and never contracted;
//! [`Transcode97Arithmetic::fma`] is the only fused operation. Backends supply
//! the primitives through [`Transcode97Arithmetic`]; shader languages that
//! cannot share this Rust source mirror it and are checked for bit equality.
//!
//! [`idct8_dot`]: crate::transcode97::idct8_dot
//! [`DWT97_LIFT_STEPS_F32`]: crate::transcode97::DWT97_LIFT_STEPS_F32
//! [`lift97_update`]: crate::transcode97::lift97_update
//! [`DWT97_LOW_SCALE_F32`]: crate::transcode97::DWT97_LOW_SCALE_F32
//! [`DWT97_HIGH_SCALE_F32`]: crate::transcode97::DWT97_HIGH_SCALE_F32
//! [`quantize97`]: crate::transcode97::quantize97
//! [`quantize97_inv_delta`]: crate::transcode97::quantize97_inv_delta
//! [`Transcode97Arithmetic::fma`]: crate::transcode97::Transcode97Arithmetic::fma
//! [`Transcode97Arithmetic`]: crate::transcode97::Transcode97Arithmetic

use crate::dwt::{
    DWT97_ALPHA_F32, DWT97_BETA_F32, DWT97_DELTA_F32, DWT97_GAMMA_F32, DWT97_INV_KAPPA_F32,
    DWT97_KAPPA_F32,
};

/// IEEE-754 binary32 primitives used by the canonical transcode sequence.
///
/// Implementations must round every operation exactly once to nearest-even
/// and must not let a compiler fuse [`mul`](Self::mul) with a following
/// [`add`](Self::add) or reassociate either.
pub trait Transcode97Arithmetic {
    /// `a * b + c` with a single rounding.
    fn fma(a: f32, b: f32, c: f32) -> f32;
    /// `a * b`, rounded once.
    fn mul(a: f32, b: f32) -> f32;
    /// `a + b`, rounded once.
    fn add(a: f32, b: f32) -> f32;
}

/// Scaled 8-point IDCT cosines: `C[0] = 1/(2*sqrt(2))` and
/// `C[k] = cos(k*pi/16) / 2` for `k` in `1..8`, each rounded to nearest `f32`.
pub const IDCT8_COS_F32: [f32; 8] = [
    0.353_553_38,
    0.490_392_63,
    0.461_939_75,
    0.415_734_8,
    0.353_553_38,
    0.277_785_12,
    0.191_341_71,
    0.097_545_16,
];

/// Orthonormal 8-point IDCT basis indexed `[sample][frequency]`.
///
/// Entries are the [`IDCT8_COS_F32`] cosines with the signs of
/// `cos((2 * sample + 1) * frequency * pi / 16)`, so symmetric entries are
/// exact negations of each other.
pub const IDCT8_BASIS_F32: [[f32; 8]; 8] = idct8_basis_table();

/// The four forward 9/7 lifting coefficients in application order: alpha
/// (odd samples), beta (even), gamma (odd), delta (even).
pub const DWT97_LIFT_STEPS_F32: [f32; 4] = [
    DWT97_ALPHA_F32,
    DWT97_BETA_F32,
    DWT97_GAMMA_F32,
    DWT97_DELTA_F32,
];

/// Scale applied to even (low-pass) samples after the lifting steps.
pub const DWT97_LOW_SCALE_F32: f32 = DWT97_INV_KAPPA_F32;

/// Scale applied to odd (high-pass) samples after the lifting steps.
pub const DWT97_HIGH_SCALE_F32: f32 = DWT97_KAPPA_F32;

const fn idct8_basis_table() -> [[f32; 8]; 8] {
    let mut table = [[0.0; 8]; 8];
    let mut sample = 0;
    while sample < 8 {
        let mut frequency = 0;
        while frequency < 8 {
            table[sample][frequency] = idct8_basis_entry(sample, frequency);
            frequency += 1;
        }
        sample += 1;
    }
    table
}

const fn idct8_basis_entry(sample: usize, frequency: usize) -> f32 {
    if frequency == 0 {
        return IDCT8_COS_F32[0];
    }
    // Fold the angle (2n + 1) * k * pi / 16 onto the first quadrant. For
    // k in 1..8 the folded index is always in 1..8.
    let angle = ((2 * sample + 1) * frequency) % 32;
    let (index, negative) = match angle {
        0..=8 => (angle, false),
        9..=16 => (16 - angle, true),
        17..=24 => (angle - 16, true),
        _ => (32 - angle, false),
    };
    if negative {
        -IDCT8_COS_F32[index]
    } else {
        IDCT8_COS_F32[index]
    }
}

/// Canonical 8-term IDCT dot product `sum(basis[k] * values[k])`.
///
/// The first product is rounded on its own; terms `1..8` are accumulated in
/// ascending order with one fused multiply-add each.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "the canonical dot product must inline into CUDA-Oxide device loops"
)]
pub fn idct8_dot<A: Transcode97Arithmetic>(basis: &[f32; 8], values: &[f32; 8]) -> f32 {
    let mut accumulator = A::mul(basis[0], values[0]);
    accumulator = A::fma(basis[1], values[1], accumulator);
    accumulator = A::fma(basis[2], values[2], accumulator);
    accumulator = A::fma(basis[3], values[3], accumulator);
    accumulator = A::fma(basis[4], values[4], accumulator);
    accumulator = A::fma(basis[5], values[5], accumulator);
    accumulator = A::fma(basis[6], values[6], accumulator);
    A::fma(basis[7], values[7], accumulator)
}

/// Reconstruct one natural-order 8x8 block of `f32` coefficients into
/// row-major samples with the canonical separable IDCT.
///
/// Every output is the [`idct8_dot`] chain of the module documentation; the
/// loops are ordered so that eight independent chains advance together, which
/// lets SIMD targets evaluate one chain per lane without changing any
/// operation.
#[must_use]
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "the block IDCT must inline into the CPU's SIMD-dispatched loops"
)]
pub fn idct8x8<A: Transcode97Arithmetic>(coefficients: &[f32; 64]) -> [f32; 64] {
    // Vertical pass: T[y][u] = sum_k basis[y][k] * coefficients[k][u].
    let mut vertical = [[0.0f32; 8]; 8];
    for (y, vertical_row) in vertical.iter_mut().enumerate() {
        let basis = &IDCT8_BASIS_F32[y];
        for (u, value) in vertical_row.iter_mut().enumerate() {
            *value = A::mul(basis[0], coefficients[u]);
        }
        for k in 1..8 {
            for (u, value) in vertical_row.iter_mut().enumerate() {
                *value = A::fma(basis[k], coefficients[k * 8 + u], *value);
            }
        }
    }
    // Horizontal pass: s[y][x] = sum_k basis[x][k] * T[y][k].
    let mut samples = [0.0f32; 64];
    for (y, vertical_row) in vertical.iter().enumerate() {
        let row = &mut samples[y * 8..y * 8 + 8];
        for (x, value) in row.iter_mut().enumerate() {
            *value = A::mul(IDCT8_BASIS_T_F32[0][x], vertical_row[0]);
        }
        for k in 1..8 {
            for (x, value) in row.iter_mut().enumerate() {
                *value = A::fma(IDCT8_BASIS_T_F32[k][x], vertical_row[k], *value);
            }
        }
    }
    samples
}

/// [`IDCT8_BASIS_F32`] transposed, `[frequency][sample]`.
const IDCT8_BASIS_T_F32: [[f32; 8]; 8] = {
    let mut table = [[0.0; 8]; 8];
    let mut frequency = 0;
    while frequency < 8 {
        let mut sample = 0;
        while sample < 8 {
            table[frequency][sample] = IDCT8_BASIS_F32[sample][frequency];
            sample += 1;
        }
        frequency += 1;
    }
    table
};

/// One lifting update: `center + step * (left + right)` evaluated as
/// `fma(step, left + right, center)`.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "the canonical lifting update must inline into CUDA-Oxide device loops"
)]
pub fn lift97_update<A: Transcode97Arithmetic>(
    center: f32,
    left: f32,
    right: f32,
    step: f32,
) -> f32 {
    A::fma(step, A::add(left, right), center)
}

/// Canonical in-place forward 9/7 lifting of one signal.
///
/// Boundary samples use whole-sample symmetric extension: the missing left
/// neighbour of sample 0 is sample 1, and a missing right neighbour equals
/// the left neighbour. After lifting, even samples hold the scaled low-pass
/// band and odd samples the scaled high-pass band. Signals shorter than two
/// samples are left unchanged.
pub fn forward_lift97<A: Transcode97Arithmetic>(data: &mut [f32]) {
    let len = data.len();
    if len < 2 {
        return;
    }
    for (step_index, &step) in DWT97_LIFT_STEPS_F32.iter().enumerate() {
        let mut index = usize::from(step_index % 2 == 0);
        while index < len {
            let left = if index > 0 {
                data[index - 1]
            } else {
                data[index + 1]
            };
            let right = if index + 1 < len {
                data[index + 1]
            } else {
                left
            };
            data[index] = lift97_update::<A>(data[index], left, right, step);
            index += 2;
        }
    }
    for (index, value) in data.iter_mut().enumerate() {
        let scale = if index % 2 == 0 {
            DWT97_LOW_SCALE_F32
        } else {
            DWT97_HIGH_SCALE_F32
        };
        *value = A::mul(*value, scale);
    }
}

/// Host-side reciprocal of a deadzone quantization step, rounded once to
/// `f32`. Every backend multiplies by this value instead of dividing.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the reciprocal step is deliberately rounded once to the binary32 quantizer operand"
)]
pub fn quantize97_inv_delta(delta: f64) -> f32 {
    (1.0 / delta) as f32
}

/// Canonical deadzone quantizer: `sign(value) * trunc(|value| * inv_delta)`.
///
/// Magnitudes that exceed `i32` saturate; the HTJ2K bitplane validation keeps
/// valid inputs far below that range. The magnitude is taken by an exact
/// negation rather than `f32::abs`, so device backends need no float
/// intrinsic; the result is identical for every input.
#[inline(always)]
#[expect(
    clippy::inline_always,
    clippy::cast_possible_truncation,
    reason = "the canonical quantizer inlines into device loops and truncates the non-negative scaled magnitude"
)]
pub fn quantize97<A: Transcode97Arithmetic>(value: f32, inv_delta: f32) -> i32 {
    let negative = value < 0.0;
    let magnitude = if negative { -value } else { value };
    let quantized = A::mul(magnitude, inv_delta) as i32;
    if negative {
        -quantized
    } else {
        quantized
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    struct HostArithmetic;

    impl Transcode97Arithmetic for HostArithmetic {
        fn fma(a: f32, b: f32, c: f32) -> f32 {
            a.mul_add(b, c)
        }
        fn mul(a: f32, b: f32) -> f32 {
            a * b
        }
        fn add(a: f32, b: f32) -> f32 {
            a + b
        }
    }

    /// CUDA-Oxide lowers float intrinsics such as `abs` or `floor` to libdevice
    /// calls, which stop it from emitting the transcode PTX. The canonical
    /// sequence must use only the arithmetic primitives, comparisons and casts.
    #[test]
    fn canonical_sequence_avoids_device_float_intrinsics() {
        let source = include_str!("transcode97.rs");
        let module = source
            .split("#[cfg(test)]")
            .next()
            .expect("module source precedes its tests");
        for intrinsic in [
            ".abs()",
            ".floor()",
            ".ceil()",
            ".trunc()",
            ".round()",
            ".sqrt()",
            ".signum()",
            ".fract()",
            ".mul_add(",
            ".copysign(",
        ] {
            assert!(
                !module.contains(intrinsic),
                "transcode97 calls `{intrinsic}`"
            );
        }
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the reference truncates the scaled magnitude like the quantizer"
    )]
    fn quantizer_matches_the_absolute_value_definition() {
        let inv_delta = quantize97_inv_delta(1.0 / 3.0);
        for value in [
            0.0f32,
            -0.0,
            0.33,
            -0.33,
            2.999_999_8,
            -2.999_999_8,
            1.0e9,
            -1.0e9,
            f32::NAN,
        ] {
            let expected = {
                let magnitude = HostArithmetic::mul(value.abs(), inv_delta) as i32;
                if value < 0.0 {
                    -magnitude
                } else {
                    magnitude
                }
            };
            assert_eq!(quantize97::<HostArithmetic>(value, inv_delta), expected);
        }
    }

    /// Exact cosine `cos(index * pi / 16)` scaled like [`IDCT8_COS_F32`].
    fn reference_cosine(index: usize) -> f64 {
        if index == 0 {
            (1.0_f64 / 8.0).sqrt()
        } else {
            0.5 * (f64::from(u32::try_from(index).unwrap()) * core::f64::consts::PI / 16.0).cos()
        }
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the f64 reference is rounded once to compare against the f32 constant"
    )]
    fn cosines_are_the_nearest_f32_to_the_exact_values() {
        for (index, &cosine) in IDCT8_COS_F32.iter().enumerate() {
            let exact = reference_cosine(index);
            assert_eq!(cosine.to_bits(), (exact as f32).to_bits(), "C[{index}]");
            // The f64 reference error (about 1e-17) is far below its distance
            // from an f32 rounding boundary, so it cannot flip the result.
            // C[1] is the closest case at 0.499 ulp from the constant.
            let ulp = f64::from(cosine.next_up()) - f64::from(cosine);
            let boundary_distance = (exact - f64::from(cosine)).abs();
            assert!(
                boundary_distance < 0.5 * ulp - 1e-12,
                "C[{index}] is within the f64 reference error of a tie"
            );
        }
        assert_eq!(IDCT8_COS_F32[0].to_bits(), IDCT8_COS_F32[4].to_bits());
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "IDCT indices are below eight and exact in f64"
    )]
    fn basis_matches_signed_cosines() {
        for (sample, row) in IDCT8_BASIS_F32.iter().enumerate() {
            for (frequency, &value) in row.iter().enumerate() {
                let scale = if frequency == 0 {
                    (1.0_f64 / 8.0).sqrt()
                } else {
                    0.5
                };
                let exact = scale
                    * (((2 * sample + 1) * frequency) as f64 * core::f64::consts::PI / 16.0).cos();
                assert!(
                    (f64::from(value) - exact).abs() < 1e-7,
                    "basis[{sample}][{frequency}] = {value}, exact {exact}"
                );
            }
        }
    }

    #[test]
    fn basis_rows_are_orthonormal_to_f32_precision() {
        for a in 0..8 {
            for b in 0..8 {
                let dot: f64 = IDCT8_BASIS_F32
                    .iter()
                    .map(|row| f64::from(row[a]) * f64::from(row[b]))
                    .sum();
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-6, "columns {a},{b}: {dot}");
            }
        }
    }

    #[test]
    fn dc_only_block_reconstructs_a_flat_block() {
        let mut block = [0.0f32; 64];
        block[0] = 320.0;
        let samples = idct8x8::<HostArithmetic>(&block);
        let first = samples[0];
        assert!((first - 40.0).abs() < 1e-4);
        assert!(samples
            .iter()
            .all(|sample| sample.to_bits() == first.to_bits()));
    }

    #[test]
    fn dot_product_is_the_documented_ascending_fma_chain() {
        let basis = IDCT8_BASIS_F32[3];
        let values = [12.0, -7.5, 3.25, 100.0, -0.125, 9.0, -64.0, 1.5];
        let mut expected = basis[0] * values[0];
        for k in 1..8 {
            expected = basis[k].mul_add(values[k], expected);
        }
        assert_eq!(
            idct8_dot::<HostArithmetic>(&basis, &values).to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    fn block_idct_is_the_documented_dot_product_composition() {
        let coefficients: [f32; 64] = core::array::from_fn(|index| {
            f32::from(i16::try_from((index * 37) % 211).unwrap() - 105) * 1.75
        });
        let mut vertical = [[0.0f32; 8]; 8];
        for (y, row) in vertical.iter_mut().enumerate() {
            for (u, value) in row.iter_mut().enumerate() {
                let column: [f32; 8] = core::array::from_fn(|k| coefficients[k * 8 + u]);
                *value = idct8_dot::<HostArithmetic>(&IDCT8_BASIS_F32[y], &column);
            }
        }
        let samples = idct8x8::<HostArithmetic>(&coefficients);
        for y in 0..8 {
            for x in 0..8 {
                let expected = idct8_dot::<HostArithmetic>(&IDCT8_BASIS_F32[x], &vertical[y]);
                assert_eq!(
                    samples[y * 8 + x].to_bits(),
                    expected.to_bits(),
                    "({x},{y})"
                );
            }
        }
    }

    #[test]
    fn fused_update_differs_from_an_unfused_one() {
        // (1 + 2^-23) * (1 - 2^-23) = 1 - 2^-46: the fused update keeps the
        // tail that a separately rounded product would lose.
        let step = 1.0 + f32::EPSILON;
        let half_sum = (1.0 - f32::EPSILON) / 2.0;
        let fused = lift97_update::<HostArithmetic>(-1.0, half_sum, half_sum, step);
        assert_eq!(fused, -(f32::EPSILON * f32::EPSILON));
        assert_eq!(step * (half_sum + half_sum) - 1.0, 0.0);
    }

    #[test]
    fn lifting_constant_signal_keeps_dc_in_low_band() {
        for len in [2usize, 3, 8, 9, 64, 65] {
            let mut signal = std::vec![50.0f32; len];
            forward_lift97::<HostArithmetic>(&mut signal);
            for (index, value) in signal.iter().enumerate() {
                let expected = if index % 2 == 0 { 50.0 } else { 0.0 };
                assert!(
                    (value - expected).abs() < 1e-3,
                    "len {len} index {index}: {value}"
                );
            }
        }
    }

    #[test]
    fn single_sample_signal_is_unchanged() {
        let mut signal = [7.25f32];
        forward_lift97::<HostArithmetic>(&mut signal);
        assert_eq!(signal, [7.25]);
    }

    #[test]
    fn quantizer_truncates_magnitude_and_keeps_sign() {
        assert_eq!(quantize97::<HostArithmetic>(2.9, 1.0), 2);
        assert_eq!(quantize97::<HostArithmetic>(-2.9, 1.0), -2);
        assert_eq!(quantize97::<HostArithmetic>(-0.0, 4.0), 0);
        assert_eq!(quantize97::<HostArithmetic>(0.49, 2.0), 0);
        assert_eq!(quantize97::<HostArithmetic>(f32::MAX, 2.0), i32::MAX);
    }
}
