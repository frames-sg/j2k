// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reduced-size ISLOW IDCTs derived from libjpeg-turbo's `jidctred.c`.

use core::num::Wrapping;

const CONST_BITS: usize = 13;
const PASS1_BITS: usize = 2;

const FIX_0_211164243: Wrapping<i32> = Wrapping(1730);
const FIX_0_509795579: Wrapping<i32> = Wrapping(4176);
const FIX_0_601344887: Wrapping<i32> = Wrapping(4926);
const FIX_0_720959822: Wrapping<i32> = Wrapping(5906);
const FIX_0_765366865: Wrapping<i32> = Wrapping(6270);
const FIX_0_850430095: Wrapping<i32> = Wrapping(6967);
const FIX_0_899976223: Wrapping<i32> = Wrapping(7373);
const FIX_1_061594337: Wrapping<i32> = Wrapping(8697);
const FIX_1_272758580: Wrapping<i32> = Wrapping(10426);
const FIX_1_451774981: Wrapping<i32> = Wrapping(11893);
const FIX_1_847759065: Wrapping<i32> = Wrapping(15137);
const FIX_2_172734803: Wrapping<i32> = Wrapping(17799);
const FIX_2_562915447: Wrapping<i32> = Wrapping(20995);
const FIX_3_624509785: Wrapping<i32> = Wrapping(29692);

pub(crate) fn idct_islow_4x4(input: &[i16; 64], output: &mut [u8; 16]) {
    let mut work = [Wrapping(0i32); 32];
    for col in 0..8 {
        if col == 4 {
            continue;
        }
        idct_4x4_column(input, &mut work, col);
    }
    for row in 0..4 {
        idct_4x4_row(&work, output, row);
    }
}

pub(crate) fn idct_islow_4x4_dc_only(dc_coeff: i16, output: &mut [u8; 16]) {
    output.fill(dc_only_pixel(dc_coeff));
}

pub(crate) fn idct_islow_2x2(input: &[i16; 64], output: &mut [u8; 4]) {
    idct_islow_2x2_scalar(input, output);
}

pub(crate) fn idct_islow_2x2_dc_only(dc_coeff: i16, output: &mut [u8; 4]) {
    output.fill(dc_only_pixel(dc_coeff));
}

pub(crate) fn idct_islow_2x2_scalar(input: &[i16; 64], output: &mut [u8; 4]) {
    let mut work = [Wrapping(0i32); 16];
    for col in 0..8 {
        if col == 2 || col == 4 || col == 6 {
            continue;
        }
        idct_2x2_column(input, &mut work, col);
    }
    for row in 0..2 {
        idct_2x2_row(&work, output, row);
    }
}

pub(crate) fn idct_islow_1x1(input: &[i16; 64]) -> u8 {
    descale_and_clamp(Wrapping(i32::from(input[0])), 3)
}

#[inline]
fn dc_only_pixel(dc_coeff: i16) -> u8 {
    descale_and_clamp(Wrapping(i32::from(dc_coeff)), 3)
}

fn idct_4x4_column(input: &[i16; 64], work: &mut [Wrapping<i32>; 32], col: usize) {
    let p0 = Wrapping(i32::from(input[col]));
    let p1 = Wrapping(i32::from(input[col + 8]));
    let p2 = Wrapping(i32::from(input[col + 16]));
    let p3 = Wrapping(i32::from(input[col + 24]));
    let p5 = Wrapping(i32::from(input[col + 40]));
    let p6 = Wrapping(i32::from(input[col + 48]));
    let p7 = Wrapping(i32::from(input[col + 56]));

    if p1.0 == 0 && p2.0 == 0 && p3.0 == 0 && p5.0 == 0 && p6.0 == 0 && p7.0 == 0 {
        let dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[8 + col] = dc;
        work[16 + col] = dc;
        work[24 + col] = dc;
        return;
    }

    let mut tmp0 = p0 << (CONST_BITS + 1);
    let z2 = p2;
    let z3 = p6;
    let tmp2 = z2 * FIX_1_847759065 + z3 * (-FIX_0_765366865);
    let tmp10 = tmp0 + tmp2;
    let tmp12 = tmp0 - tmp2;

    let z1 = p7;
    let z2 = p5;
    let z3 = p3;
    let z4 = p1;
    tmp0 = z1 * (-FIX_0_211164243)
        + z2 * FIX_1_451774981
        + z3 * (-FIX_2_172734803)
        + z4 * FIX_1_061594337;
    let tmp2 = z1 * (-FIX_0_509795579)
        + z2 * (-FIX_0_601344887)
        + z3 * FIX_0_899976223
        + z4 * FIX_2_562915447;

    let shift = CONST_BITS - PASS1_BITS + 1;
    work[col] = descale(tmp10 + tmp2, shift);
    work[24 + col] = descale(tmp10 - tmp2, shift);
    work[8 + col] = descale(tmp12 + tmp0, shift);
    work[16 + col] = descale(tmp12 - tmp0, shift);
}

fn idct_4x4_row(work: &[Wrapping<i32>; 32], output: &mut [u8; 16], row: usize) {
    let base = row * 8;
    let p0 = work[base];
    let p1 = work[base + 1];
    let p2 = work[base + 2];
    let p3 = work[base + 3];
    let p5 = work[base + 5];
    let p6 = work[base + 6];
    let p7 = work[base + 7];

    if p1.0 == 0 && p2.0 == 0 && p3.0 == 0 && p5.0 == 0 && p6.0 == 0 && p7.0 == 0 {
        let dc = descale_and_clamp(p0, PASS1_BITS + 3);
        let out = row * 4;
        output[out..out + 4].fill(dc);
        return;
    }

    let mut tmp0 = p0 << (CONST_BITS + 1);
    let tmp2 = p2 * FIX_1_847759065 + p6 * (-FIX_0_765366865);
    let tmp10 = tmp0 + tmp2;
    let tmp12 = tmp0 - tmp2;

    tmp0 = p7 * (-FIX_0_211164243)
        + p5 * FIX_1_451774981
        + p3 * (-FIX_2_172734803)
        + p1 * FIX_1_061594337;
    let tmp2 = p7 * (-FIX_0_509795579)
        + p5 * (-FIX_0_601344887)
        + p3 * FIX_0_899976223
        + p1 * FIX_2_562915447;

    let shift = CONST_BITS + PASS1_BITS + 3 + 1;
    let out = row * 4;
    output[out] = descale_and_clamp(tmp10 + tmp2, shift);
    output[out + 3] = descale_and_clamp(tmp10 - tmp2, shift);
    output[out + 1] = descale_and_clamp(tmp12 + tmp0, shift);
    output[out + 2] = descale_and_clamp(tmp12 - tmp0, shift);
}

fn idct_2x2_column(input: &[i16; 64], work: &mut [Wrapping<i32>; 16], col: usize) {
    let p0 = Wrapping(i32::from(input[col]));
    let p1 = Wrapping(i32::from(input[col + 8]));
    let p3 = Wrapping(i32::from(input[col + 24]));
    let p5 = Wrapping(i32::from(input[col + 40]));
    let p7 = Wrapping(i32::from(input[col + 56]));

    if p1.0 == 0 && p3.0 == 0 && p5.0 == 0 && p7.0 == 0 {
        let dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[8 + col] = dc;
        return;
    }

    let tmp10 = p0 << (CONST_BITS + 2);
    let tmp0 = p7 * (-FIX_0_720959822)
        + p5 * FIX_0_850430095
        + p3 * (-FIX_1_272758580)
        + p1 * FIX_3_624509785;

    let shift = CONST_BITS - PASS1_BITS + 2;
    work[col] = descale(tmp10 + tmp0, shift);
    work[8 + col] = descale(tmp10 - tmp0, shift);
}

fn idct_2x2_row(work: &[Wrapping<i32>; 16], output: &mut [u8; 4], row: usize) {
    let base = row * 8;
    let p0 = work[base];
    let p1 = work[base + 1];
    let p3 = work[base + 3];
    let p5 = work[base + 5];
    let p7 = work[base + 7];

    if p1.0 == 0 && p3.0 == 0 && p5.0 == 0 && p7.0 == 0 {
        let dc = descale_and_clamp(p0, PASS1_BITS + 3);
        let out = row * 2;
        output[out] = dc;
        output[out + 1] = dc;
        return;
    }

    let tmp10 = p0 << (CONST_BITS + 2);
    let tmp0 = p7 * (-FIX_0_720959822)
        + p5 * FIX_0_850430095
        + p3 * (-FIX_1_272758580)
        + p1 * FIX_3_624509785;

    let shift = CONST_BITS + PASS1_BITS + 3 + 2;
    let out = row * 2;
    output[out] = descale_and_clamp(tmp10 + tmp0, shift);
    output[out + 1] = descale_and_clamp(tmp10 - tmp0, shift);
}

/// 12-bit `jidctred.c`: `BITS_IN_JSAMPLE == 12` keeps `PASS1_BITS = 1` for
/// headroom and computes in 64-bit `JLONG`. The C code's zero-AC shortcuts give
/// the same results as the full computation and are omitted, as in
/// [`super::scalar::idct_islow_12bit`].
const PASS1_BITS_12: usize = 1;

/// Reduced 4x4 IDCT of a 12-bit block (`jpeg_idct_4x4`).
pub(crate) fn idct_islow_12bit_4x4(input: &[i16; 64], output: &mut [u16; 16]) {
    let fix = |value: Wrapping<i32>| i64::from(value.0);
    let odd = |z1: i64, z2: i64, z3: i64, z4: i64| {
        (
            z1 * -fix(FIX_0_211164243)
                + z2 * fix(FIX_1_451774981)
                + z3 * -fix(FIX_2_172734803)
                + z4 * fix(FIX_1_061594337),
            z1 * -fix(FIX_0_509795579)
                + z2 * -fix(FIX_0_601344887)
                + z3 * fix(FIX_0_899976223)
                + z4 * fix(FIX_2_562915447),
        )
    };
    let even = |p0: i64, p2: i64, p6: i64| {
        let tmp0 = p0 << (CONST_BITS + 1);
        let tmp2 = p2 * fix(FIX_1_847759065) - p6 * fix(FIX_0_765366865);
        (tmp0 + tmp2, tmp0 - tmp2)
    };
    let mut work = [0i64; 32];
    for col in (0..8).filter(|&col| col != 4) {
        let p = |row: usize| i64::from(input[row * 8 + col]);
        let (tmp10, tmp12) = even(p(0), p(2), p(6));
        let (tmp0, tmp2) = odd(p(7), p(5), p(3), p(1));
        let shift = CONST_BITS - PASS1_BITS_12 + 1;
        work[col] = descale_i64(tmp10 + tmp2, shift);
        work[24 + col] = descale_i64(tmp10 - tmp2, shift);
        work[8 + col] = descale_i64(tmp12 + tmp0, shift);
        work[16 + col] = descale_i64(tmp12 - tmp0, shift);
    }
    for row in 0..4 {
        let w = |col: usize| work[row * 8 + col];
        let (tmp10, tmp12) = even(w(0), w(2), w(6));
        let (tmp0, tmp2) = odd(w(7), w(5), w(3), w(1));
        let shift = CONST_BITS + PASS1_BITS_12 + 3 + 1;
        let out = &mut output[row * 4..row * 4 + 4];
        out[0] = level_shift_12bit(descale_i64(tmp10 + tmp2, shift));
        out[3] = level_shift_12bit(descale_i64(tmp10 - tmp2, shift));
        out[1] = level_shift_12bit(descale_i64(tmp12 + tmp0, shift));
        out[2] = level_shift_12bit(descale_i64(tmp12 - tmp0, shift));
    }
}

/// Reduced 2x2 IDCT of a 12-bit block (`jpeg_idct_2x2`).
pub(crate) fn idct_islow_12bit_2x2(input: &[i16; 64], output: &mut [u16; 4]) {
    let fix = |value: Wrapping<i32>| i64::from(value.0);
    let odd = |p7: i64, p5: i64, p3: i64, p1: i64| {
        p7 * -fix(FIX_0_720959822)
            + p5 * fix(FIX_0_850430095)
            + p3 * -fix(FIX_1_272758580)
            + p1 * fix(FIX_3_624509785)
    };
    let mut work = [0i64; 16];
    for col in [0, 1, 3, 5, 7] {
        let p = |row: usize| i64::from(input[row * 8 + col]);
        let tmp10 = p(0) << (CONST_BITS + 2);
        let tmp0 = odd(p(7), p(5), p(3), p(1));
        let shift = CONST_BITS - PASS1_BITS_12 + 2;
        work[col] = descale_i64(tmp10 + tmp0, shift);
        work[8 + col] = descale_i64(tmp10 - tmp0, shift);
    }
    for row in 0..2 {
        let w = |col: usize| work[row * 8 + col];
        let tmp10 = w(0) << (CONST_BITS + 2);
        let tmp0 = odd(w(7), w(5), w(3), w(1));
        let shift = CONST_BITS + PASS1_BITS_12 + 3 + 2;
        output[row * 2] = level_shift_12bit(descale_i64(tmp10 + tmp0, shift));
        output[row * 2 + 1] = level_shift_12bit(descale_i64(tmp10 - tmp0, shift));
    }
}

/// Reduced 1x1 IDCT of a 12-bit block (`jpeg_idct_1x1`).
pub(crate) fn idct_islow_12bit_1x1(input: &[i16; 64]) -> u16 {
    level_shift_12bit(descale_i64(i64::from(input[0]), 3))
}

const fn descale_i64(value: i64, shift: usize) -> i64 {
    (value + (1 << (shift - 1))) >> shift
}

#[expect(
    clippy::cast_sign_loss,
    reason = "reduced 12-bit IDCT samples are clamped to 0..=4095 before conversion"
)]
fn level_shift_12bit(value: i64) -> u16 {
    (value + 2048).clamp(0, 4095) as u16
}

/// libjpeg's `DESCALE`: arithmetic right shift that rounds half up. Every
/// stage of `jidctred.c` (and libjpeg-turbo's NEON/SSE2 ports) rounds, so a
/// plain shift biases scaled output low by up to one level per component.
fn descale(value: Wrapping<i32>, shift: usize) -> Wrapping<i32> {
    Wrapping((value + Wrapping(1 << (shift - 1))).0 >> shift)
}

#[expect(
    clippy::cast_sign_loss,
    reason = "reduced IDCT samples are clamped to the u8 output range before conversion"
)]
fn descale_and_clamp(value: Wrapping<i32>, shift: usize) -> u8 {
    let shifted = descale(value, shift).0;
    let level_shifted = shifted.wrapping_add(128);
    level_shifted.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_idcts_preserve_zero_block_level_shift() {
        let input = [0i16; 64];
        let mut out4 = [0u8; 16];
        let mut out2 = [0u8; 4];
        idct_islow_4x4(&input, &mut out4);
        idct_islow_2x2(&input, &mut out2);
        assert!(out4.iter().all(|&px| px == 128));
        assert!(out2.iter().all(|&px| px == 128));
        assert_eq!(idct_islow_1x1(&input), 128);
    }

    #[test]
    fn reduced_dc_only_helpers_match_full_reduced_idcts() {
        for dc in [-300, -37, 0, 37, 300] {
            let mut input = [0i16; 64];
            input[0] = dc;
            let mut expected4 = [0u8; 16];
            let mut actual4 = [0u8; 16];
            let mut expected2 = [0u8; 4];
            let mut actual2 = [0u8; 4];

            idct_islow_4x4(&input, &mut expected4);
            idct_islow_4x4_dc_only(input[0], &mut actual4);
            idct_islow_2x2(&input, &mut expected2);
            idct_islow_2x2_dc_only(input[0], &mut actual2);

            assert_eq!(actual4, expected4);
            assert_eq!(actual2, expected2);
        }
    }

    #[test]
    fn twelve_bit_reduced_idcts_of_dc_blocks_are_uniform_rounded_samples() {
        // DESCALE(dc, 3) + 2048: the DC-only result every reduced size shares.
        for (dc, expected) in [
            (4i16, 2049u16),
            (-4, 2048),
            (12, 2050),
            (-5, 2047),
            (32767, 4095),
        ] {
            let mut input = [0i16; 64];
            input[0] = dc;
            let mut out4 = [0u16; 16];
            let mut out2 = [0u16; 4];
            idct_islow_12bit_4x4(&input, &mut out4);
            idct_islow_12bit_2x2(&input, &mut out2);
            assert_eq!(idct_islow_12bit_1x1(&input), expected, "1x1 dc={dc}");
            assert!(out4.iter().all(|&px| px == expected), "4x4 dc={dc}");
            assert!(out2.iter().all(|&px| px == expected), "2x2 dc={dc}");
        }
    }

    #[test]
    fn reduced_idcts_round_dc_half_up_like_libjpeg_descale() {
        // jpeg_idct_1x1 and the DC-only paths of jpeg_idct_{4x4,2x2} compute
        // DESCALE(dc, 3) = (dc + 4) >> 3. A plain shift gives 128, 127, 129 for
        // the first three inputs.
        for (dc, expected) in [(4i16, 129u8), (-4, 128), (12, 130), (-5, 127), (3, 128)] {
            let mut input = [0i16; 64];
            input[0] = dc;
            let mut out4 = [0u8; 16];
            let mut out2 = [0u8; 4];
            idct_islow_4x4(&input, &mut out4);
            idct_islow_2x2(&input, &mut out2);

            assert_eq!(idct_islow_1x1(&input), expected, "1x1 dc={dc}");
            assert!(out4.iter().all(|&px| px == expected), "4x4 dc={dc}");
            assert!(out2.iter().all(|&px| px == expected), "2x2 dc={dc}");
        }
    }
}
