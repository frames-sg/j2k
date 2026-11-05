//! Integer "slow" inverse DCT (ISLOW) of baseline and 12-bit JPEG.
//!
//! This module is the workspace's one implementation of the ISLOW maths,
//! bit-exact with libjpeg-turbo's `jidctint.c` (8x8) and `jidctred.c` (4x4,
//! 2x2, 1x1). The CPU scalar decoder, the CPU transcode oracle, and the
//! CUDA-Oxide decode and transcode kernels call it; the CPU SIMD kernels take
//! their constants and [`ISLOW_GAINS`] from it; the Metal shaders include a
//! constant fragment that `j2k-jpeg-metal`'s build script renders from it.
//!
//! Inputs are dequantized coefficients in natural (row-major) order. The 8-bit
//! transforms compute in `i32` with wrapping arithmetic, so intermediate
//! overflow on malformed coefficients wraps modulo 2^32 as in libjpeg-turbo
//! instead of panicking. The 12-bit transforms compute in `i64` (`JLONG`),
//! where `i16` inputs cannot overflow.
//!
//! Every transform is `#[inline]`, which exports the MIR the cuda-oxide device
//! compiler needs; rustc exports it unprompted only for leaf functions. The
//! passes and row loads the block loops call are `#[inline(always)]`: left to
//! its cost model, LLVM calls them out of line from the unrolled loops and
//! moves every `[i32; 8]` through memory.

/// Number of fractional bits in the integer IDCT constants.
pub const CONST_BITS: usize = 13;
/// First-pass post-IDCT scaling bits.
pub const PASS1_BITS: usize = 2;

/// Fixed-point integer approximation of 0.298631336.
pub const FIX_0_298631336: i32 = 2_446;
/// Fixed-point integer approximation of 0.390180644.
pub const FIX_0_390180644: i32 = 3_196;
/// Fixed-point integer approximation of 0.541196100.
pub const FIX_0_541196100: i32 = 4_433;
/// Fixed-point integer approximation of 0.765366865.
pub const FIX_0_765366865: i32 = 6_270;
/// Fixed-point integer approximation of 0.899976223.
pub const FIX_0_899976223: i32 = 7_373;
/// Fixed-point integer approximation of 1.175875602.
pub const FIX_1_175875602: i32 = 9_633;
/// Fixed-point integer approximation of 1.501321110.
pub const FIX_1_501321110: i32 = 12_299;
/// Fixed-point integer approximation of 1.847759065.
pub const FIX_1_847759065: i32 = 15_137;
/// Fixed-point integer approximation of 1.961570560.
pub const FIX_1_961570560: i32 = 16_069;
/// Fixed-point integer approximation of 2.053119869.
pub const FIX_2_053119869: i32 = 16_819;
/// Fixed-point integer approximation of 2.562915447.
pub const FIX_2_562915447: i32 = 20_995;
/// Fixed-point integer approximation of 3.072711026.
pub const FIX_3_072711026: i32 = 25_172;

/// Fixed-point integer approximation of 0.211164243 (reduced-size IDCTs).
pub const FIX_0_211164243: i32 = 1_730;
/// Fixed-point integer approximation of 0.509795579 (reduced-size IDCTs).
pub const FIX_0_509795579: i32 = 4_176;
/// Fixed-point integer approximation of 0.601344887 (reduced-size IDCTs).
pub const FIX_0_601344887: i32 = 4_926;
/// Fixed-point integer approximation of 0.720959822 (reduced-size IDCTs).
pub const FIX_0_720959822: i32 = 5_906;
/// Fixed-point integer approximation of 0.850430095 (reduced-size IDCTs).
pub const FIX_0_850430095: i32 = 6_967;
/// Fixed-point integer approximation of 1.061594337 (reduced-size IDCTs).
pub const FIX_1_061594337: i32 = 8_697;
/// Fixed-point integer approximation of 1.272758580 (reduced-size IDCTs).
pub const FIX_1_272758580: i32 = 10_426;
/// Fixed-point integer approximation of 1.451774981 (reduced-size IDCTs).
pub const FIX_1_451774981: i32 = 11_893;
/// Fixed-point integer approximation of 2.172734803 (reduced-size IDCTs).
pub const FIX_2_172734803: i32 = 17_799;
/// Fixed-point integer approximation of 3.624509785 (reduced-size IDCTs).
pub const FIX_3_624509785: i32 = 29_692;

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "IDCT bit counts are small positive compile-time values"
)]
const fn bits(count: usize) -> i32 {
    count as i32
}

/// Right shift ending the 8-bit column pass.
pub const PASS1_SHIFT: i32 = bits(CONST_BITS) - bits(PASS1_BITS);
/// Right shift ending the 8-bit row pass, including the 1/8 normalisation.
pub const PASS2_SHIFT: i32 = bits(CONST_BITS) + bits(PASS1_BITS) + 3;
/// Round-half-up addend of the 8-bit column pass.
pub const PASS1_ROUNDING: i32 = 1 << (PASS1_SHIFT - 1);
/// Round-half-up addend of the 8-bit row pass.
pub const PASS2_ROUNDING: i32 = 1 << (PASS2_SHIFT - 1);
/// The row pass's DC-only shortcut is `(p0 + DC_ROUNDING) >> DC_SHIFT`; it
/// equals the full row transform while `p0 << CONST_BITS` fits in `i32`.
pub const DC_SHIFT: i32 = bits(PASS1_BITS) + 3;
/// Round-half-up addend of the row pass's DC-only shortcut.
pub const DC_ROUNDING: i32 = 1 << (DC_SHIFT - 1);

/// Fraction bits the 12-bit transforms keep between their passes:
/// libjpeg-turbo's `jidctint.c` keeps one instead of [`PASS1_BITS`] to leave
/// headroom.
pub const PASS1_BITS_12: i32 = 1;
/// Right shift ending the 12-bit column pass.
pub const PASS1_SHIFT_12: i32 = bits(CONST_BITS) - PASS1_BITS_12;
/// Right shift ending the 12-bit row pass, including the 1/8 normalisation.
pub const PASS2_SHIFT_12: i32 = bits(CONST_BITS) + PASS1_BITS_12 + 3;

/// The matrix of [`islow_1d`], in units of `2^-CONST_BITS`.
///
/// `islow_1d(p)[j] == sum_k ISLOW_GAINS[j][k] * p[k]`: it is the transform's
/// impulse response, for SIMD kernels that multiply inputs by per-output
/// gains instead of sharing the `z1..z5` products.
pub const ISLOW_GAINS: [[i32; 8]; 8] = islow_gains();

const fn islow_gains() -> [[i32; 8]; 8] {
    let mut gains = [[0; 8]; 8];
    let mut k = 0;
    while k < 8 {
        let mut impulse = [0; 8];
        impulse[k] = 1;
        let response = islow_1d(impulse);
        let mut j = 0;
        while j < 8 {
            gains[j][k] = response[j];
            j += 1;
        }
        k += 1;
    }
    gains
}

/// `(max_gain, max_l1)` of [`ISLOW_GAINS`].
///
/// These are the largest gain magnitude of any input into any output (11,363)
/// and the largest L1 norm of one output's gains (61,214). A 1-D output is at
/// most `max_gain * sum(|p|)` and at most `max_l1 * max(|p|)` before
/// rounding; SIMD kernels derive their overflow bounds from these.
#[must_use]
pub const fn islow_gain_bounds() -> (i32, i32) {
    let (mut max_gain, mut max_l1) = (0, 0);
    let mut j = 0;
    while j < 8 {
        let mut l1 = 0;
        let mut k = 0;
        while k < 8 {
            let gain = ISLOW_GAINS[j][k].abs();
            if gain > max_gain {
                max_gain = gain;
            }
            l1 += gain;
            k += 1;
        }
        if l1 > max_l1 {
            max_l1 = l1;
        }
        j += 1;
    }
    (max_gain, max_l1)
}

#[inline]
const fn fix_i32(constant: i32) -> i32 {
    constant
}

#[inline]
const fn fix_i64(constant: i32) -> i64 {
    constant as i64
}

// Stamps out the 1-D transforms for one arithmetic type. Only ring
// operations (`+`, `-`, `*`, `<<`) appear, so the `i32` instances agree bit
// for bit with any evaluation order of the same products modulo 2^32.
macro_rules! islow_butterflies {
    (
        $t:ty, $fix:ident,
        $(#[$full_meta:meta])* fn $full:ident,
        $(#[$half_meta:meta])* fn $half:ident,
        $(#[$quarter_meta:meta])* fn $quarter:ident $(,)?
    ) => {
        $(#[$full_meta])*
        #[inline]
        #[must_use]
        pub const fn $full(p: [$t; 8]) -> [$t; 8] {
            // Even part: rotation of p2/p6 and the p0/p4 butterfly.
            let z1 = p[2].wrapping_add(p[6]).wrapping_mul($fix(FIX_0_541196100));
            let tmp2 = z1.wrapping_add(p[6].wrapping_mul($fix(-FIX_1_847759065)));
            let tmp3 = z1.wrapping_add(p[2].wrapping_mul($fix(FIX_0_765366865)));
            let tmp0 = p[0].wrapping_add(p[4]) << CONST_BITS;
            let tmp1 = p[0].wrapping_sub(p[4]) << CONST_BITS;
            let tmp10 = tmp0.wrapping_add(tmp3);
            let tmp13 = tmp0.wrapping_sub(tmp3);
            let tmp11 = tmp1.wrapping_add(tmp2);
            let tmp12 = tmp1.wrapping_sub(tmp2);

            // Odd part: Chen-Wang rotations sharing the z1..z5 products.
            let (tmp0, tmp1, tmp2, tmp3) = (p[7], p[5], p[3], p[1]);
            let z1 = tmp0.wrapping_add(tmp3);
            let z2 = tmp1.wrapping_add(tmp2);
            let z3 = tmp0.wrapping_add(tmp2);
            let z4 = tmp1.wrapping_add(tmp3);
            let z5 = z3.wrapping_add(z4).wrapping_mul($fix(FIX_1_175875602));
            let tmp0 = tmp0.wrapping_mul($fix(FIX_0_298631336));
            let tmp1 = tmp1.wrapping_mul($fix(FIX_2_053119869));
            let tmp2 = tmp2.wrapping_mul($fix(FIX_3_072711026));
            let tmp3 = tmp3.wrapping_mul($fix(FIX_1_501321110));
            let z1 = z1.wrapping_mul($fix(-FIX_0_899976223));
            let z2 = z2.wrapping_mul($fix(-FIX_2_562915447));
            let z3 = z3.wrapping_mul($fix(-FIX_1_961570560)).wrapping_add(z5);
            let z4 = z4.wrapping_mul($fix(-FIX_0_390180644)).wrapping_add(z5);
            let tmp0 = tmp0.wrapping_add(z1).wrapping_add(z3);
            let tmp1 = tmp1.wrapping_add(z2).wrapping_add(z4);
            let tmp2 = tmp2.wrapping_add(z2).wrapping_add(z3);
            let tmp3 = tmp3.wrapping_add(z1).wrapping_add(z4);

            [
                tmp10.wrapping_add(tmp3),
                tmp11.wrapping_add(tmp2),
                tmp12.wrapping_add(tmp1),
                tmp13.wrapping_add(tmp0),
                tmp13.wrapping_sub(tmp0),
                tmp12.wrapping_sub(tmp1),
                tmp11.wrapping_sub(tmp2),
                tmp10.wrapping_sub(tmp3),
            ]
        }

        $(#[$half_meta])*
        #[inline]
        #[must_use]
        pub const fn $half(p: [$t; 8]) -> [$t; 4] {
            let tmp0 = p[0] << (CONST_BITS + 1);
            let tmp2 = p[2]
                .wrapping_mul($fix(FIX_1_847759065))
                .wrapping_add(p[6].wrapping_mul($fix(-FIX_0_765366865)));
            let tmp10 = tmp0.wrapping_add(tmp2);
            let tmp12 = tmp0.wrapping_sub(tmp2);

            let (z1, z2, z3, z4) = (p[7], p[5], p[3], p[1]);
            let tmp0 = z1
                .wrapping_mul($fix(-FIX_0_211164243))
                .wrapping_add(z2.wrapping_mul($fix(FIX_1_451774981)))
                .wrapping_add(z3.wrapping_mul($fix(-FIX_2_172734803)))
                .wrapping_add(z4.wrapping_mul($fix(FIX_1_061594337)));
            let tmp2 = z1
                .wrapping_mul($fix(-FIX_0_509795579))
                .wrapping_add(z2.wrapping_mul($fix(-FIX_0_601344887)))
                .wrapping_add(z3.wrapping_mul($fix(FIX_0_899976223)))
                .wrapping_add(z4.wrapping_mul($fix(FIX_2_562915447)));

            [
                tmp10.wrapping_add(tmp2),
                tmp12.wrapping_add(tmp0),
                tmp12.wrapping_sub(tmp0),
                tmp10.wrapping_sub(tmp2),
            ]
        }

        $(#[$quarter_meta])*
        #[inline]
        #[must_use]
        pub const fn $quarter(p: [$t; 8]) -> [$t; 2] {
            let tmp10 = p[0] << (CONST_BITS + 2);
            let tmp0 = p[7]
                .wrapping_mul($fix(-FIX_0_720959822))
                .wrapping_add(p[5].wrapping_mul($fix(FIX_0_850430095)))
                .wrapping_add(p[3].wrapping_mul($fix(-FIX_1_272758580)))
                .wrapping_add(p[1].wrapping_mul($fix(FIX_3_624509785)));
            [tmp10.wrapping_add(tmp0), tmp10.wrapping_sub(tmp0)]
        }
    };
}

islow_butterflies!(
    i32,
    fix_i32,
    /// The 8-point ISLOW transform (`jidctint.c`).
    ///
    /// It transforms one row or column `p` of the 8-bit block in wrapping
    /// `i32` arithmetic. Outputs are scaled by `2^CONST_BITS` and not yet
    /// rounded or shifted.
    fn islow_1d,
    /// The 4-point reduced ISLOW (`jpeg_idct_4x4` in `jidctred.c`).
    ///
    /// It transforms one row or column `p` of the 8-bit block; `p[4]` is
    /// unused. Outputs are scaled by `2^(CONST_BITS + 1)` and not yet rounded
    /// or shifted.
    fn islow_1d_4,
    /// The 2-point reduced ISLOW (`jpeg_idct_2x2` in `jidctred.c`).
    ///
    /// It transforms one row or column `p` of the 8-bit block; `p[2]`, `p[4]`
    /// and `p[6]` are unused. Outputs are scaled by `2^(CONST_BITS + 2)` and
    /// not yet rounded or shifted.
    fn islow_1d_2,
);

islow_butterflies!(
    i64,
    fix_i64,
    /// [`islow_1d`] in `i64` (`JLONG`) arithmetic, for the 12-bit transforms.
    fn islow_1d_i64,
    /// [`islow_1d_4`] in `i64` (`JLONG`) arithmetic, for the 12-bit
    /// transforms.
    fn islow_1d_4_i64,
    /// [`islow_1d_2`] in `i64` (`JLONG`) arithmetic, for the 12-bit
    /// transforms.
    fn islow_1d_2_i64,
);

/// libjpeg's `DESCALE`: add half of `2^shift`, then shift right
/// arithmetically (round half up).
#[inline]
#[must_use]
pub const fn descale(value: i32, shift: i32) -> i32 {
    value.wrapping_add(1 << (shift - 1)) >> shift
}

#[inline]
const fn descale_i64(value: i64, shift: i32) -> i64 {
    (value + (1 << (shift - 1))) >> shift
}

#[inline]
const fn descale_8(values: [i32; 8], shift: i32) -> [i32; 8] {
    let mut out = [0; 8];
    let mut i = 0;
    while i < 8 {
        out[i] = descale(values[i], shift);
        i += 1;
    }
    out
}

/// The column (first) pass of the 8-bit transform, without zero shortcuts:
/// [`islow_1d`] rounded and shifted by [`PASS1_SHIFT`].
#[inline]
#[must_use]
pub const fn islow_pass1(p: [i32; 8]) -> [i32; 8] {
    descale_8(islow_1d(p), PASS1_SHIFT)
}

/// The row (second) pass of the 8-bit transform, without zero shortcuts:
/// [`islow_1d`] rounded and shifted by [`PASS2_SHIFT`]. Samples are signed;
/// [`sample_u8`] level-shifts and clamps them.
#[inline]
#[must_use]
pub const fn islow_pass2(p: [i32; 8]) -> [i32; 8] {
    descale_8(islow_1d(p), PASS2_SHIFT)
}

#[inline]
const fn ac_is_zero(p: &[i32; 8]) -> bool {
    p[1] == 0 && p[2] == 0 && p[3] == 0 && p[4] == 0 && p[5] == 0 && p[6] == 0 && p[7] == 0
}

/// [`islow_pass1`] with `jidctint.c`'s AC-zero column shortcut.
///
/// A column whose AC terms are zero gives `p[0] << PASS1_BITS` in every
/// output. The shortcut equals the full pass while `p[0] << CONST_BITS` fits
/// in `i32`, which holds for every `i16` input.
#[expect(
    clippy::inline_always,
    reason = "measured: out of line, the 8x8 block loops pass every column and row through memory"
)]
#[inline(always)]
#[must_use]
pub const fn islow_column(p: [i32; 8]) -> [i32; 8] {
    if ac_is_zero(&p) {
        [p[0] << PASS1_BITS; 8]
    } else {
        islow_pass1(p)
    }
}

/// [`islow_pass2`] with `jidctint.c`'s AC-zero row shortcut.
///
/// A row whose AC terms are zero gives `(p[0] + DC_ROUNDING) >> DC_SHIFT` in
/// every output. The shortcut equals the full pass while `p[0] << CONST_BITS`
/// fits in `i32`; beyond that it avoids the overflow the full pass would wrap
/// through.
#[expect(
    clippy::inline_always,
    reason = "measured: out of line, the 8x8 block loops pass every column and row through memory"
)]
#[inline(always)]
#[must_use]
pub const fn islow_row(p: [i32; 8]) -> [i32; 8] {
    if ac_is_zero(&p) {
        [p[0].wrapping_add(DC_ROUNDING) >> DC_SHIFT; 8]
    } else {
        islow_pass2(p)
    }
}

/// The output of a block whose AC coefficients are zero: `DESCALE(dc, 3)`.
///
/// Every output sample takes this value, before the level shift, at every
/// output size. It is also the whole 1x1 reduced IDCT (`jpeg_idct_1x1`).
#[inline]
#[must_use]
pub const fn islow_dc(dc: i32) -> i32 {
    descale(dc, 3)
}

/// An 8-bit output sample: `value + 128` clamped to `0..=255`.
#[inline]
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the sample is clamped to the u8 range before conversion"
)]
pub const fn sample_u8(value: i32) -> u8 {
    // Two selects on the value, then one cast: the shape LLVM lowers to
    // branchless max/min.
    let shifted = value.wrapping_add(128);
    let low = if shifted < 0 { 0 } else { shifted };
    let clamped = if low > 255 { 255 } else { low };
    clamped as u8
}

/// An 8-bit output sample centred on zero, as the JPEG 2000 DC level shift
/// leaves it: `value` clamped to `-128..=127`.
///
/// It equals `i32::from(sample_u8(value)) - 128` for every value the row
/// passes produce.
#[inline]
#[must_use]
pub const fn sample_centered(value: i32) -> i32 {
    let low = if value < -128 { -128 } else { value };
    if low > 127 {
        127
    } else {
        low
    }
}

/// A 12-bit output sample: `value + 2048` clamped to `0..=4095`.
#[inline]
#[must_use]
#[expect(
    clippy::cast_sign_loss,
    reason = "the sample is clamped to the 12-bit range before conversion"
)]
pub fn sample_12bit(value: i64) -> u16 {
    (value + 2048).clamp(0, 4095) as u16
}

/// The 8-bit 8x8 ISLOW of one block of dequantized coefficients to
/// level-shifted, clamped samples, with `jidctint.c`'s zero shortcuts
/// ([`islow_column`], [`islow_row`]).
#[inline]
pub fn islow_block<T: Copy + Into<i32>>(input: &[T; 64], output: &mut [u8; 64]) {
    islow_block_rows::<T, 8>(input, output);
}

/// [`islow_block`] for a block whose coefficient rows 4..7 (`input[32..]`)
/// are zero. It reads only `input[..32]` and skips the products of the zero
/// rows; the result equals [`islow_block`]'s.
#[inline]
pub fn islow_block_top_half<T: Copy + Into<i32>>(input: &[T; 64], output: &mut [u8; 64]) {
    islow_block_rows::<T, 4>(input, output);
}

/// The 8x8 ISLOW of a block whose coefficient rows `ROWS..8` are zero.
#[inline]
fn islow_block_rows<T: Copy + Into<i32>, const ROWS: usize>(
    input: &[T; 64],
    output: &mut [u8; 64],
) {
    let mut work = [0i32; 64];
    let mut col = 0;
    while col < 8 {
        let mut p = [0; 8];
        let mut row = 0;
        while row < ROWS {
            p[row] = input[row * 8 + col].into();
            row += 1;
        }
        let out = islow_column(p);
        let mut row = 0;
        while row < 8 {
            work[row * 8 + col] = out[row];
            row += 1;
        }
        col += 1;
    }
    let mut row = 0;
    while row < 8 {
        let base = row * 8;
        let out = islow_row(row_of(&work, base));
        let mut i = 0;
        while i < 8 {
            output[base + i] = sample_u8(out[i]);
            i += 1;
        }
        row += 1;
    }
}

#[expect(
    clippy::inline_always,
    reason = "measured: left to LLVM, the 12-bit 8x8 row loop calls this out of line"
)]
#[inline(always)]
const fn row_of<T: Copy>(work: &[T], base: usize) -> [T; 8] {
    [
        work[base],
        work[base + 1],
        work[base + 2],
        work[base + 3],
        work[base + 4],
        work[base + 5],
        work[base + 6],
        work[base + 7],
    ]
}

#[inline]
fn column_of(input: &[i16; 64], col: usize) -> [i32; 8] {
    let mut p = [0; 8];
    let mut row = 0;
    while row < 8 {
        p[row] = i32::from(input[row * 8 + col]);
        row += 1;
    }
    p
}

#[inline]
fn column_of_i64(input: &[i16; 64], col: usize) -> [i64; 8] {
    let mut p = [0; 8];
    let mut row = 0;
    while row < 8 {
        p[row] = i64::from(input[row * 8 + col]);
        row += 1;
    }
    p
}

/// The 12-bit 8x8 ISLOW (`jidctint.c` with `BITS_IN_JSAMPLE == 12`) of one
/// block to native `0..=4095` samples. The C code's zero shortcuts give the
/// same result as the full computation in `i64` and are omitted.
#[inline]
pub fn islow_block_12bit(input: &[i16; 64], output: &mut [u16; 64]) {
    let mut work = [0i64; 64];
    let mut col = 0;
    while col < 8 {
        let out = islow_1d_i64(column_of_i64(input, col));
        let mut row = 0;
        while row < 8 {
            work[row * 8 + col] = descale_i64(out[row], PASS1_SHIFT_12);
            row += 1;
        }
        col += 1;
    }
    let mut row = 0;
    while row < 8 {
        let base = row * 8;
        let out = islow_1d_i64(row_of(&work, base));
        let mut i = 0;
        while i < 8 {
            output[base + i] = sample_12bit(descale_i64(out[i], PASS2_SHIFT_12));
            i += 1;
        }
        row += 1;
    }
}

/// Whether the inputs of [`islow_1d_4`] other than `p[0]` are zero.
#[inline]
const fn ac_is_zero_4(p: &[i32; 8]) -> bool {
    p[1] == 0 && p[2] == 0 && p[3] == 0 && p[5] == 0 && p[6] == 0 && p[7] == 0
}

/// Whether the inputs of [`islow_1d_2`] other than `p[0]` are zero.
#[inline]
const fn ac_is_zero_2(p: &[i32; 8]) -> bool {
    p[1] == 0 && p[3] == 0 && p[5] == 0 && p[7] == 0
}

/// The reduced 4x4 ISLOW (`jpeg_idct_4x4`) of an 8-bit block to
/// level-shifted, clamped samples, with `jidctred.c`'s zero shortcuts.
#[inline]
pub fn islow_block_4x4(input: &[i16; 64], output: &mut [u8; 16]) {
    let mut work = [0i32; 32];
    for col in 0..8 {
        // Column 4 feeds only the unused `p[4]` of the row pass.
        if col != 4 {
            column_4x4(input, &mut work, col);
        }
    }
    for row in 0..4 {
        row_4x4(&work, output, row);
    }
}

#[expect(
    clippy::inline_always,
    reason = "measured: out of line, the 4x4 block loops pass every column and row through memory"
)]
#[inline(always)]
fn column_4x4(input: &[i16; 64], work: &mut [i32; 32], col: usize) {
    let at = |row: usize| i32::from(input[row * 8 + col]);
    // `p[4]` is unused by the 4-point transform.
    let p = [at(0), at(1), at(2), at(3), 0, at(5), at(6), at(7)];
    if ac_is_zero_4(&p) {
        let dc = p[0] << PASS1_BITS;
        work[col] = dc;
        work[8 + col] = dc;
        work[16 + col] = dc;
        work[24 + col] = dc;
        return;
    }
    let out = islow_1d_4(p);
    let shift = PASS1_SHIFT + 1;
    work[col] = descale(out[0], shift);
    work[24 + col] = descale(out[3], shift);
    work[8 + col] = descale(out[1], shift);
    work[16 + col] = descale(out[2], shift);
}

#[expect(
    clippy::inline_always,
    reason = "measured: out of line, the 4x4 block loops pass every column and row through memory"
)]
#[inline(always)]
fn row_4x4(work: &[i32; 32], output: &mut [u8; 16], row: usize) {
    let at = |col: usize| work[row * 8 + col];
    let p = [at(0), at(1), at(2), at(3), 0, at(5), at(6), at(7)];
    let out = row * 4;
    if ac_is_zero_4(&p) {
        output[out..out + 4].fill(sample_u8(descale(p[0], DC_SHIFT)));
        return;
    }
    let samples = islow_1d_4(p);
    let shift = PASS2_SHIFT + 1;
    output[out] = sample_u8(descale(samples[0], shift));
    output[out + 3] = sample_u8(descale(samples[3], shift));
    output[out + 1] = sample_u8(descale(samples[1], shift));
    output[out + 2] = sample_u8(descale(samples[2], shift));
}

/// The columns and rows the 2-point reduced transforms read: the even
/// indices other than 0 feed only unused inputs.
const REDUCED_2X2_INPUTS: [usize; 5] = [0, 1, 3, 5, 7];

/// The reduced 2x2 ISLOW (`jpeg_idct_2x2`) of an 8-bit block to
/// level-shifted, clamped samples, with `jidctred.c`'s zero shortcuts.
#[inline]
pub fn islow_block_2x2(input: &[i16; 64], output: &mut [u8; 4]) {
    let mut work = [0i32; 16];
    for col in REDUCED_2X2_INPUTS {
        let p = column_of(input, col);
        let (top, bottom) = if ac_is_zero_2(&p) {
            let dc = p[0] << PASS1_BITS;
            (dc, dc)
        } else {
            let out = islow_1d_2(p);
            (
                descale(out[0], PASS1_SHIFT + 2),
                descale(out[1], PASS1_SHIFT + 2),
            )
        };
        work[col] = top;
        work[8 + col] = bottom;
    }
    for row in 0..2 {
        let p = row_of(&work, row * 8);
        let (left, right) = if ac_is_zero_2(&p) {
            let dc = sample_u8(descale(p[0], DC_SHIFT));
            (dc, dc)
        } else {
            let out = islow_1d_2(p);
            (
                sample_u8(descale(out[0], PASS2_SHIFT + 2)),
                sample_u8(descale(out[1], PASS2_SHIFT + 2)),
            )
        };
        output[row * 2] = left;
        output[row * 2 + 1] = right;
    }
}

/// The reduced 4x4 ISLOW of a 12-bit block (`jpeg_idct_4x4` with
/// `BITS_IN_JSAMPLE == 12`) to native samples. The zero shortcuts give the
/// same result as the full computation in `i64` and are omitted.
#[inline]
pub fn islow_block_12bit_4x4(input: &[i16; 64], output: &mut [u16; 16]) {
    let mut work = [0i64; 32];
    let mut col = 0;
    while col < 8 {
        if col != 4 {
            let out = islow_1d_4_i64(column_of_i64(input, col));
            let mut row = 0;
            while row < 4 {
                work[row * 8 + col] = descale_i64(out[row], PASS1_SHIFT_12 + 1);
                row += 1;
            }
        }
        col += 1;
    }
    let mut row = 0;
    while row < 4 {
        let out = islow_1d_4_i64(row_of(&work, row * 8));
        let mut i = 0;
        while i < 4 {
            output[row * 4 + i] = sample_12bit(descale_i64(out[i], PASS2_SHIFT_12 + 1));
            i += 1;
        }
        row += 1;
    }
}

/// The reduced 2x2 ISLOW of a 12-bit block (`jpeg_idct_2x2` with
/// `BITS_IN_JSAMPLE == 12`) to native samples.
#[inline]
pub fn islow_block_12bit_2x2(input: &[i16; 64], output: &mut [u16; 4]) {
    let mut work = [0i64; 16];
    for col in REDUCED_2X2_INPUTS {
        let at = |row: usize| i64::from(input[row * 8 + col]);
        // The 2-point transform does not read `p[2]`, `p[4]` or `p[6]`.
        let out = islow_1d_2_i64([at(0), at(1), 0, at(3), 0, at(5), 0, at(7)]);
        work[col] = descale_i64(out[0], PASS1_SHIFT_12 + 2);
        work[8 + col] = descale_i64(out[1], PASS1_SHIFT_12 + 2);
    }
    for row in 0..2 {
        let at = |col: usize| work[row * 8 + col];
        let out = islow_1d_2_i64([at(0), at(1), 0, at(3), 0, at(5), 0, at(7)]);
        output[row * 2] = sample_12bit(descale_i64(out[0], PASS2_SHIFT_12 + 2));
        output[row * 2 + 1] = sample_12bit(descale_i64(out[1], PASS2_SHIFT_12 + 2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_shifts_match_libjpeg() {
        assert_eq!(
            (
                PASS1_SHIFT,
                PASS2_SHIFT,
                DC_SHIFT,
                PASS1_SHIFT_12,
                PASS2_SHIFT_12
            ),
            (11, 18, 5, 12, 17)
        );
        assert_eq!(
            (PASS1_ROUNDING, PASS2_ROUNDING, DC_ROUNDING),
            (1 << 10, 1 << 17, 1 << 4)
        );
    }

    #[test]
    fn reduced_constants_match_libjpeg_fix() {
        // FIX(x) = round(x * 2^CONST_BITS), as in jidctred.c.
        for (constant, value) in [
            (FIX_0_211164243, 0.211_164_243),
            (FIX_0_509795579, 0.509_795_579),
            (FIX_0_601344887, 0.601_344_887),
            (FIX_0_720959822, 0.720_959_822),
            (FIX_0_850430095, 0.850_430_095),
            (FIX_1_061594337, 1.061_594_337),
            (FIX_1_272758580, 1.272_758_580),
            (FIX_1_451774981, 1.451_774_981),
            (FIX_2_172734803, 2.172_734_803),
            (FIX_3_624509785, 3.624_509_785),
        ] {
            let error = f64::from(constant) - value * f64::from(1u32 << CONST_BITS);
            assert!((-0.5..=0.5).contains(&error), "{value}");
        }
    }

    #[test]
    fn islow_gains_match_the_column_transform() {
        // An impulse of ±2^PASS1_SHIFT in input k makes column output j
        // exactly ±gain[j][k]: (g * 2048 + 1024) >> 11 = g.
        let impulse = 1i32 << PASS1_SHIFT;
        for k in 0..8 {
            for sign in [1, -1] {
                let mut input = [0; 8];
                input[k] = sign * impulse;
                let output = islow_column(input);
                for (j, gains) in ISLOW_GAINS.iter().enumerate() {
                    assert_eq!(output[j], sign * gains[k], "output {j}, input {k}");
                }
            }
        }
    }

    #[test]
    fn islow_gain_bounds_match_the_documented_values() {
        assert_eq!(islow_gain_bounds(), (11_363, 61_214));
    }

    #[test]
    fn i64_transforms_agree_with_i32_without_overflow() {
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..1_000 {
            let p: [i32; 8] = core::array::from_fn(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (i32::try_from(state >> 48).unwrap() - 32_768) / 16
            });
            let wide = p.map(i64::from);
            assert_eq!(islow_1d_i64(wide), islow_1d(p).map(i64::from));
            assert_eq!(islow_1d_4_i64(wide), islow_1d_4(p).map(i64::from));
            assert_eq!(islow_1d_2_i64(wide), islow_1d_2(p).map(i64::from));
        }
    }

    #[test]
    fn shortcut_passes_equal_full_passes_on_i16_dc() {
        for dc in [
            i32::from(i16::MIN),
            -1_000,
            -1,
            0,
            1,
            777,
            i32::from(i16::MAX),
        ] {
            let p = [dc, 0, 0, 0, 0, 0, 0, 0];
            assert_eq!(islow_column(p), islow_pass1(p), "column dc={dc}");
            let q = [dc << PASS1_BITS, 0, 0, 0, 0, 0, 0, 0];
            assert_eq!(islow_row(q), islow_pass2(q), "row dc={dc}");
            assert_eq!(islow_row(q)[0], islow_dc(dc), "dc-only dc={dc}");
        }
    }

    #[test]
    fn row_shortcut_avoids_the_full_pass_overflow() {
        // p0 << CONST_BITS wraps for p0 >= 2^18; the shortcut does not.
        let p = [1 << 20, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(islow_row(p), [((1 << 20) + DC_ROUNDING) >> DC_SHIFT; 8]);
        assert_ne!(islow_row(p), islow_pass2(p));
    }

    #[test]
    fn top_half_block_matches_full_block() {
        let mut input = [0i16; 64];
        for (i, value) in input.iter_mut().take(32).enumerate() {
            *value = i16::try_from(i * 37 % 211).unwrap() - 105;
        }
        let mut full = [0u8; 64];
        let mut top = [0u8; 64];
        islow_block(&input, &mut full);
        islow_block_top_half(&input, &mut top);
        assert_eq!(full, top);
    }

    #[test]
    fn i32_and_i16_block_inputs_agree() {
        let input: [i16; 64] = core::array::from_fn(|i| i16::try_from(i * 53 % 401).unwrap() - 200);
        let mut narrow = [0u8; 64];
        let mut wide = [0u8; 64];
        islow_block(&input, &mut narrow);
        islow_block(&input.map(i32::from), &mut wide);
        assert_eq!(narrow, wide);
    }

    #[test]
    fn sample_conversions_level_shift_and_clamp() {
        assert_eq!(
            [-200, -128, 0, 127, 300].map(sample_u8),
            [0, 0, 128, 255, 255]
        );
        assert_eq!(
            [-5_000, -2_048, 0, 2_047, 9_000].map(sample_12bit),
            [0, 0, 2_048, 4_095, 4_095]
        );
        assert_eq!(sample_u8(i32::MAX), 0, "wrapping level shift");
        // Row-pass outputs are shifted right by at least 3, so |value| < 2^28.
        for value in (-(1 << 28)..(1 << 28)).step_by(4_093).chain(-300..300) {
            assert_eq!(
                sample_centered(value),
                i32::from(sample_u8(value)) - 128,
                "{value}"
            );
        }
    }

    #[test]
    fn dc_only_blocks_are_uniform_at_every_size() {
        for dc in [-1_024i16, -37, -5, 0, 4, 12, 300, 1_023] {
            let mut input = [0i16; 64];
            input[0] = dc;
            let expected = sample_u8(islow_dc(i32::from(dc)));
            let mut out8 = [0u8; 64];
            let mut out4 = [0u8; 16];
            let mut out2 = [0u8; 4];
            islow_block(&input, &mut out8);
            islow_block_4x4(&input, &mut out4);
            islow_block_2x2(&input, &mut out2);
            assert!(out8.iter().all(|&px| px == expected), "8x8 dc={dc}");
            assert!(out4.iter().all(|&px| px == expected), "4x4 dc={dc}");
            assert!(out2.iter().all(|&px| px == expected), "2x2 dc={dc}");

            let expected12 = sample_12bit(i64::from(islow_dc(i32::from(dc))));
            let mut out8 = [0u16; 64];
            let mut out4 = [0u16; 16];
            let mut out2 = [0u16; 4];
            islow_block_12bit(&input, &mut out8);
            islow_block_12bit_4x4(&input, &mut out4);
            islow_block_12bit_2x2(&input, &mut out2);
            assert!(
                out8.iter().all(|&px| px == expected12),
                "12-bit 8x8 dc={dc}"
            );
            assert!(
                out4.iter().all(|&px| px == expected12),
                "12-bit 4x4 dc={dc}"
            );
            assert!(
                out2.iter().all(|&px| px == expected12),
                "12-bit 2x2 dc={dc}"
            );
        }
    }
}
