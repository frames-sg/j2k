// SPDX-License-Identifier: MIT OR Apache-2.0

//! `x86_64` AVX2 integer ISLOW IDCT. Bit-exact with `super::scalar::idct_islow`
//! on every input, including crafted out-of-range coefficient blocks.
//!
//! Two kernels share the dispatch in [`idct_islow`]:
//!
//! - The hot kernel keeps samples in 16-bit lanes of 256-bit registers, as
//!   libjpeg-turbo's `jidctint-avx2` does. `vpmaddwd` multiplies interleaved
//!   pairs of 16-bit inputs by 16-bit gains and adds the two products in
//!   32 bits, so every product and sum of the butterfly is exact. Unlike
//!   turbo it never adds two inputs in 16 bits: each scalar shared product
//!   such as `(p7 + p3) * FIX` is expanded into per-input gains. The only
//!   narrowing left is the 16-bit hand-off between the passes, which
//!   [`first_pass_fits_i16`] proves exact before the kernel runs.
//! - Blocks that check rejects run the 32-bit-lane port of the NEON kernel,
//!   whose wrapping arithmetic reproduces the scalar `Wrapping<i32>` path for
//!   any input.

use core::arch::x86_64::{
    __m128i, __m256i, _mm256_abs_epi16, _mm256_add_epi32, _mm256_adds_epu16, _mm256_blend_epi16,
    _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_madd_epi16, _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute4x64_epi64,
    _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256, _mm256_shuffle_epi8,
    _mm256_slli_epi32, _mm256_srai_epi32, _mm256_sub_epi32, _mm256_unpackhi_epi16,
    _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32,
    _mm256_unpacklo_epi64, _mm_add_epi32, _mm_adds_epu16, _mm_blendv_epi8, _mm_cmpeq_epi32,
    _mm_cvtepi16_epi32, _mm_mullo_epi32, _mm_or_si128, _mm_packs_epi32, _mm_packus_epi16,
    _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi8, _mm_setzero_si128, _mm_slli_epi32,
    _mm_srai_epi32, _mm_srli_si128, _mm_sub_epi32, _mm_subs_epu16, _mm_testz_si128,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi32, _mm_unpacklo_epi64,
};
use j2k_codec_math::jpeg::idct;

use super::scalar::ISLOW_GAINS;

use crate::simd::x86::{exact_avx2_kernel, madd_pair};
use crate::simd::x86_memory;

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "shared IDCT shift constants are small positive compile-time values"
)]
const CONST_BITS: i32 = idct::CONST_BITS as i32;
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "shared IDCT shift constants are small positive compile-time values"
)]
const PASS1_BITS: i32 = idct::PASS1_BITS as i32;
const PASS1_SHIFT: i32 = CONST_BITS - PASS1_BITS;
const PASS2_SHIFT: i32 = CONST_BITS + PASS1_BITS + 3;

const FIX_0_298631336: i32 = idct::FIX_0_298631336;
const FIX_0_390180644: i32 = idct::FIX_0_390180644;
const FIX_0_541196100: i32 = idct::FIX_0_541196100;
const FIX_0_765366865: i32 = idct::FIX_0_765366865;
const FIX_0_899976223: i32 = idct::FIX_0_899976223;
const FIX_1_175875602: i32 = idct::FIX_1_175875602;
const FIX_1_501321110: i32 = idct::FIX_1_501321110;
const FIX_1_847759065: i32 = idct::FIX_1_847759065;
const FIX_1_961570560: i32 = idct::FIX_1_961570560;
const FIX_2_053119869: i32 = idct::FIX_2_053119869;
const FIX_2_562915447: i32 = idct::FIX_2_562915447;
const FIX_3_072711026: i32 = idct::FIX_3_072711026;

/// Largest gain magnitude of any input in any 1-D output (11,363).
const MAX_GAIN: i32 = {
    let mut max = 0;
    let mut j = 0;
    while j < 8 {
        let mut k = 0;
        while k < 8 {
            if ISLOW_GAINS[j][k].abs() > max {
                max = ISLOW_GAINS[j][k].abs();
            }
            k += 1;
        }
        j += 1;
    }
    max
};

/// Largest L1 norm of one output's gains (61,214).
const MAX_GAIN_L1: i32 = {
    let mut max = 0;
    let mut j = 0;
    while j < 8 {
        let mut sum = 0;
        let mut k = 0;
        while k < 8 {
            sum += ISLOW_GAINS[j][k].abs();
            k += 1;
        }
        if sum > max {
            max = sum;
        }
        j += 1;
    }
    max
};

const PASS1_ROUNDING: i32 = 1 << (PASS1_SHIFT - 1);
/// Second-pass rounding with the +128 level shift folded in: adding
/// `128 << PASS2_SHIFT` before the arithmetic shift adds exactly 128 after it.
const PASS2_ROUNDING: i32 = (1 << (PASS2_SHIFT - 1)) + (128 << PASS2_SHIFT);

/// Largest per-column `sum(|coefficient|)` the 16-bit kernel accepts (5,905).
///
/// Derivation. Every 1-D output of both passes is the exact integer linear
/// form `sum_k ISLOW_GAINS[j][k] * p[k]`, rounded and shifted:
///
/// 1. Nothing leaves `i32`. `vpmaddwd` sums two `i16 * i16` products exactly
///    (it wraps only for two `-32768 * -32768` products, and no gain is
///    `-32768`). Any output's gains have L1 norm at most `MAX_GAIN_L1`
///    = 61,214, and `61,214 * 32,768 + PASS2_ROUNDING < 2^31`, so no
///    first- or second-pass sum overflows for any `i16` inputs (asserted
///    below). The scalar's wrapping `i32` sums are therefore the exact
///    integers as well.
/// 2. The first-pass outputs fit in `i16`, the only narrowing. For column
///    `c`, `|sum_k gain[k] * x[k][c]| <= MAX_GAIN * S_c`, where `MAX_GAIN` =
///    11,363 and `S_c = sum_k |x[k][c]|`. The output is
///    `floor((sum + 2^10) / 2^11)`; `S_c <= 5,905` gives
///    `|sum| <= 67,098,515 <= 32,767 * 2^11 - 2^10`, so it lies in
///    `[-32,764, 32,763]`. The bound is tight: `x[1][c] = 5,906` alone (gain
///    11,363 into output row 0) gives 32,769.
/// 3. The scalar's DC-only row shortcut `(p0 + 16) >> 5` equals the full row
///    transform `(p0 * 2^13 + 2^17) >> 18` whenever `p0 * 2^13 + 2^17` fits in
///    `i32`, which (2) guarantees; its column shortcuts always equal the full
///    transform.
///
/// The second pass needs no further bound: its inputs are `i16` by
/// construction, and narrowing its results `i32 -> i16 -> u8` with saturation
/// at each step is the monotone clamp `clamp(v + 128, 0, 255)` the scalar
/// applies. Decoded photographs stay far below the bound; a block above it is
/// still exact, it only takes the 32-bit kernel.
const MAX_COLUMN_ABS_SUM: i32 = (32_767 * (1 << PASS1_SHIFT) - PASS1_ROUNDING) / MAX_GAIN;
#[expect(
    clippy::cast_possible_truncation,
    reason = "the column bound is 5,905, asserted below to fit in i16"
)]
const MAX_COLUMN_ABS_SUM_I16: i16 = MAX_COLUMN_ABS_SUM as i16;

const _: () = {
    assert!(MAX_GAIN == 11_363);
    assert!(MAX_GAIN_L1 == 61_214);
    assert!(MAX_COLUMN_ABS_SUM == 5_905);
    // (2): the worst column sum stays inside the i16 output range.
    assert!(MAX_GAIN * MAX_COLUMN_ABS_SUM + PASS1_ROUNDING < 32_767 * (1 << PASS1_SHIFT) + 1);
    // (1): no second-pass sum overflows for any i16 inputs.
    assert!(match MAX_GAIN_L1.checked_mul(1 << 15) {
        Some(sum) => sum.checked_add(PASS2_ROUNDING).is_some(),
        None => false,
    });
};

/// The two gains `vpmaddwd` applies to the interleaved `(p[k], p[k + 4])`
/// input pair for 1-D output `output`, packed into one 32-bit lane.
const fn gain_pair(output: usize, k: usize) -> i32 {
    madd_pair(ISLOW_GAINS[output][k], ISLOW_GAINS[output][k + 4])
}

// The kernels use the even/odd butterfly: for j < 4, output j is
// `even[j] + odd[j]` and output 7 - j is `even[j] - odd[j]`, where even[0..4]
// are tmp10 = tmp0 + tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2 and
// tmp13 = tmp0 - tmp3. Output 0's `(p0, p4)` and `(p2, p6)` gains are tmp0
// and tmp3; output 1's are tmp1 and tmp2.
const TMP0_PAIR: i32 = gain_pair(0, 0);
const TMP1_PAIR: i32 = gain_pair(1, 0);
const TMP3_PAIR: i32 = gain_pair(0, 2);
const TMP2_PAIR: i32 = gain_pair(1, 2);
/// `[(p1, p5), (p3, p7)]` gain pairs of the odd term of outputs j and 7 - j.
const ODD_PAIRS: [[i32; 2]; 4] = [
    [gain_pair(0, 1), gain_pair(0, 3)],
    [gain_pair(1, 1), gain_pair(1, 3)],
    [gain_pair(2, 1), gain_pair(2, 3)],
    [gain_pair(3, 1), gain_pair(3, 3)],
];

const _: () = {
    // The butterfly the kernels rely on, checked against the shared table:
    // outputs j and 7 - j share even gains and negate odd ones, tmp13 and
    // tmp12 reuse tmp0/tmp1 and negate tmp3/tmp2, and tmp0/tmp1 are p0 ± p4.
    let mut k = 0;
    while k < 8 {
        let mut j = 0;
        while j < 4 {
            let (gain, mirrored) = (ISLOW_GAINS[j][k], ISLOW_GAINS[7 - j][k]);
            assert!(if k % 2 == 0 {
                gain == mirrored
            } else {
                gain == -mirrored
            });
            j += 1;
        }
        if k % 4 == 0 {
            assert!(
                ISLOW_GAINS[3][k] == ISLOW_GAINS[0][k] && ISLOW_GAINS[2][k] == ISLOW_GAINS[1][k]
            );
        } else if k % 2 == 0 {
            assert!(
                ISLOW_GAINS[3][k] == -ISLOW_GAINS[0][k] && ISLOW_GAINS[2][k] == -ISLOW_GAINS[1][k]
            );
        }
        k += 1;
    }
    let one = ISLOW_GAINS[0][0];
    assert!(ISLOW_GAINS[0][4] == one && ISLOW_GAINS[1][0] == one && ISLOW_GAINS[1][4] == -one);
};

exact_avx2_kernel! {
    /// Inverse DCT of one 8×8 block. Output is level-shifted (+128) and
    /// saturated to `[0, 255]`, matching the scalar path byte-for-byte on
    /// every input.
    pub(crate) fn idct_islow(avx2: ExactAvx2, input: &[i16; 64], output: &mut [u8; 64]) {
        idct_islow_shape::<8, 8>(input, output);
    }
}

exact_avx2_kernel! {
    /// [`idct_islow`] for blocks whose natural-order rows 4..7 are zero.
    pub(crate) fn idct_islow_bottom_half_zero(
        avx2: ExactAvx2,
        input: &[i16; 64],
        output: &mut [u8; 64],
    ) {
        idct_islow_shape::<4, 8>(input, output);
    }
}

exact_avx2_kernel! {
    /// [`idct_islow`] for blocks whose natural-order columns 4..7 are zero.
    pub(crate) fn idct_islow_right_half_zero(
        avx2: ExactAvx2,
        input: &[i16; 64],
        output: &mut [u8; 64],
    ) {
        idct_islow_shape::<8, 4>(input, output);
    }
}

exact_avx2_kernel! {
    /// [`idct_islow`] for blocks with coefficients only in the natural-order
    /// top-left 4x4 quadrant.
    pub(crate) fn idct_islow_top_left_quadrant(
        avx2: ExactAvx2,
        input: &[i16; 64],
        output: &mut [u8; 64],
    ) {
        idct_islow_shape::<4, 4>(input, output);
    }
}

exact_avx2_kernel! {
    /// [`idct_islow`] for blocks with coefficients only in the natural-order
    /// top-left 2x2 corner.
    pub(crate) fn idct_islow_top_left_2x2(
        avx2: ExactAvx2,
        input: &[i16; 64],
        output: &mut [u8; 64],
    ) {
        idct_islow_shape::<2, 2>(input, output);
    }
}

/// Inverse DCT of a block whose coefficients outside natural-order rows
/// `0..ROWS` and columns `0..COLS` are zero (`ROWS`, `COLS` in 2, 4, 8). The
/// 16-bit kernel skips the work those zeros make constant; the 32-bit
/// fallback handles any block.
#[inline]
#[target_feature(enable = "avx2")]
fn idct_islow_shape<const ROWS: usize, const COLS: usize>(
    input: &[i16; 64],
    output: &mut [u8; 64],
) {
    if first_pass_fits_i16::<ROWS>(input) {
        idct_islow_i16::<ROWS, COLS>(input, output);
    } else {
        idct_islow_i32(input, output);
    }
}

/// Whether every column of `input` satisfies the bound documented on
/// [`MAX_COLUMN_ABS_SUM`], given that rows `ROWS..8` are zero. Saturating
/// unsigned adds keep the column sums exact up to 65,535 (`|i16::MIN|` =
/// 32,768 is exact in `u16`) and above the bound beyond it, so this accepts
/// exactly the blocks within the bound.
#[inline]
#[target_feature(enable = "avx2")]
fn first_pass_fits_i16<const ROWS: usize>(input: &[i16; 64]) -> bool {
    let (row_pairs, _) = input.as_chunks::<16>();
    let rows01 = _mm256_abs_epi16(x86_memory::load_i16x16(&row_pairs[0]));
    let rows23 = _mm256_abs_epi16(x86_memory::load_i16x16(&row_pairs[1]));
    // Lanes: [sum over rows 0, 2, 4, 6 | sum over rows 1, 3, 5, 7].
    let halves = if ROWS > 4 {
        let rows45 = _mm256_abs_epi16(x86_memory::load_i16x16(&row_pairs[2]));
        let rows67 = _mm256_abs_epi16(x86_memory::load_i16x16(&row_pairs[3]));
        _mm256_adds_epu16(
            _mm256_adds_epu16(rows01, rows23),
            _mm256_adds_epu16(rows45, rows67),
        )
    } else {
        _mm256_adds_epu16(rows01, rows23)
    };
    let columns = _mm_adds_epu16(
        _mm256_castsi256_si128(halves),
        _mm256_extracti128_si256::<1>(halves),
    );
    let excess = _mm_subs_epu16(columns, _mm_set1_epi16(MAX_COLUMN_ABS_SUM_I16));
    _mm_testz_si128(excess, excess) == 1
}

/// 16-bit-lane ISLOW IDCT. Exact only for blocks [`first_pass_fits_i16`]
/// accepts.
///
/// Pass 1 runs on the natural layout: lanes are columns. Each input register
/// holds `(row k, row k + 4)` pairs of four columns in both 128-bit lanes,
/// and lane-specific gains make the low lane compute one output row and the
/// high lane another. Narrowing pairs column `m` with column `m + 4`, so a
/// 32-bit transpose with in-lane unpacks hands pass 2 the same pairing with
/// lanes as rows, in the order `[0, 3, 4, 7 | 1, 2, 5, 6]`; the final byte
/// transpose restores natural order.
///
/// `ROWS` and `COLS` are as in [`idct_islow_shape`]: zero rows are not
/// loaded, columns 4..7 skip half of pass 1 when zero, and pass 2 drops the
/// products of zero first-pass columns.
#[inline]
#[target_feature(enable = "avx2")]
fn idct_islow_i16<const ROWS: usize, const COLS: usize>(input: &[i16; 64], output: &mut [u8; 64]) {
    let (rows, _) = input.as_chunks::<8>();
    let zero = _mm256_setzero_si256();
    let r0 = x86_memory::broadcast_i16x8(&rows[0]);
    let r1 = x86_memory::broadcast_i16x8(&rows[1]);
    let [r2, r3] = if ROWS > 2 {
        [
            x86_memory::broadcast_i16x8(&rows[2]),
            x86_memory::broadcast_i16x8(&rows[3]),
        ]
    } else {
        [zero; 2]
    };
    let [r4, r5, r6, r7] = if ROWS > 4 {
        [
            x86_memory::broadcast_i16x8(&rows[4]),
            x86_memory::broadcast_i16x8(&rows[5]),
            x86_memory::broadcast_i16x8(&rows[6]),
            x86_memory::broadcast_i16x8(&rows[7]),
        ]
    } else {
        [zero; 4]
    };

    // Columns 0..3 and 4..7.
    let lo = pass1_four_columns::<ROWS>([
        _mm256_unpacklo_epi16(r0, r4),
        _mm256_unpacklo_epi16(r1, r5),
        _mm256_unpacklo_epi16(r2, r6),
        _mm256_unpacklo_epi16(r3, r7),
    ]);
    let hi = if COLS > 4 {
        pass1_four_columns::<ROWS>([
            _mm256_unpackhi_epi16(r0, r4),
            _mm256_unpackhi_epi16(r1, r5),
            _mm256_unpackhi_epi16(r2, r6),
            _mm256_unpackhi_epi16(r3, r7),
        ])
    } else {
        [zero; 4]
    };
    let pass1 = [
        narrow_pass1(lo[0], hi[0]),
        narrow_pass1(lo[1], hi[1]),
        narrow_pass1(lo[2], hi[2]),
        narrow_pass1(lo[3], hi[3]),
    ];
    store_block(output, pass2_rows::<COLS>(transpose_pass1::<COLS>(pass1)));
}

/// `vpmaddwd` gain pairs: the low 128-bit lane applies `low`, the high lane
/// `high`.
#[inline]
#[target_feature(enable = "avx2")]
fn lane_gains(low: i32, high: i32) -> __m256i {
    _mm256_setr_epi32(low, low, low, low, high, high, high, high)
}

/// Odd term `low` (low lane) and `high` (high lane) from the `(p1, p5)` and
/// `(p3, p7)` input pairs; `None` means the `(p3, p7)` pair is zero.
#[inline]
#[target_feature(enable = "avx2")]
fn odd_terms(p15: __m256i, p37: Option<__m256i>, low: [i32; 2], high: [i32; 2]) -> __m256i {
    let p15_terms = _mm256_madd_epi16(p15, lane_gains(low[0], high[0]));
    p37.map_or(p15_terms, |p37| {
        _mm256_add_epi32(
            p15_terms,
            _mm256_madd_epi16(p37, lane_gains(low[1], high[1])),
        )
    })
}

/// First-pass 1-D IDCT of four columns. `pairs` are the `(p0, p4)`,
/// `(p1, p5)`, `(p2, p6)` and `(p3, p7)` registers, identical in both lanes;
/// with `ROWS` at most 2 the last two are zero and skipped. Returns the
/// rounded, unshifted outputs as `[out0 | out1]`, `[out3 | out2]`,
/// `[out4 | out5]` and `[out7 | out6]`.
#[inline]
#[target_feature(enable = "avx2")]
fn pass1_four_columns<const ROWS: usize>([p04, p15, p26, p37]: [__m256i; 4]) -> [__m256i; 4] {
    let tmp0_1 = _mm256_add_epi32(
        _mm256_madd_epi16(p04, lane_gains(TMP0_PAIR, TMP1_PAIR)),
        _mm256_set1_epi32(PASS1_ROUNDING),
    );
    let (tmp10_11, tmp13_12) = if ROWS > 2 {
        let tmp3_2 = _mm256_madd_epi16(p26, lane_gains(TMP3_PAIR, TMP2_PAIR));
        (
            _mm256_add_epi32(tmp0_1, tmp3_2),
            _mm256_sub_epi32(tmp0_1, tmp3_2),
        )
    } else {
        (tmp0_1, tmp0_1)
    };
    let p37 = (ROWS > 2).then_some(p37);
    let odd0_1 = odd_terms(p15, p37, ODD_PAIRS[0], ODD_PAIRS[1]);
    let odd3_2 = odd_terms(p15, p37, ODD_PAIRS[3], ODD_PAIRS[2]);
    [
        _mm256_add_epi32(tmp10_11, odd0_1),
        _mm256_add_epi32(tmp13_12, odd3_2),
        _mm256_sub_epi32(tmp13_12, odd3_2),
        _mm256_sub_epi32(tmp10_11, odd0_1),
    ]
}

/// Descale two first-pass output rows to 16 bits. `lo` holds columns 0..3
/// and `hi` columns 4..7; 32-bit lane `m` of each half becomes
/// `(out[m], out[m + 4])`. `x >> 11` leaves the result's low 16 bits in
/// place and `x << 5` moves the same bits to the high half; the range check
/// guarantees those 16 bits are the whole output.
#[inline]
#[target_feature(enable = "avx2")]
fn narrow_pass1(lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_blend_epi16::<0xaa>(
        _mm256_srai_epi32::<PASS1_SHIFT>(lo),
        _mm256_slli_epi32::<{ 16 - PASS1_SHIFT }>(hi),
    )
}

/// Turn `[row a | row b]` registers (rows as `(y[m], y[m + 4])` pairs, in
/// the order rows 0|1, 3|2, 4|5, 7|6) into the `(q0, q4)`, `(q1, q5)`,
/// `(q2, q6)` and `(q3, q7)` registers of pass 2, whose 32-bit lanes are rows
/// `[0, 3, 4, 7 | 1, 2, 5, 6]`. With `COLS` at most 2 the last two are zero
/// and not formed.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose_pass1<const COLS: usize>(
    [rows01, rows32, rows45, rows76]: [__m256i; 4],
) -> [__m256i; 4] {
    let pairs01 = _mm256_unpacklo_epi32(rows01, rows32);
    let pairs01_tail = _mm256_unpacklo_epi32(rows45, rows76);
    let [q26, q37] = if COLS > 2 {
        let pairs23 = _mm256_unpackhi_epi32(rows01, rows32);
        let pairs23_tail = _mm256_unpackhi_epi32(rows45, rows76);
        [
            _mm256_unpacklo_epi64(pairs23, pairs23_tail),
            _mm256_unpackhi_epi64(pairs23, pairs23_tail),
        ]
    } else {
        [_mm256_setzero_si256(); 2]
    };
    [
        _mm256_unpacklo_epi64(pairs01, pairs01_tail),
        _mm256_unpackhi_epi64(pairs01, pairs01_tail),
        q26,
        q37,
    ]
}

/// Second-pass 1-D IDCT of all eight rows at once. Returns output column `j`
/// at index `j`, descaled and level-shifted, lanes in pass-2 row order. With
/// `COLS` at most 2 the `(q2, q6)` and `(q3, q7)` pairs are zero and skipped.
#[inline]
#[target_feature(enable = "avx2")]
fn pass2_rows<const COLS: usize>([q04, q15, q26, q37]: [__m256i; 4]) -> [__m256i; 8] {
    let rounding = _mm256_set1_epi32(PASS2_ROUNDING);
    let tmp0 = _mm256_add_epi32(
        _mm256_madd_epi16(q04, _mm256_set1_epi32(TMP0_PAIR)),
        rounding,
    );
    let tmp1 = _mm256_add_epi32(
        _mm256_madd_epi16(q04, _mm256_set1_epi32(TMP1_PAIR)),
        rounding,
    );
    let [tmp10, tmp11, tmp12, tmp13] = if COLS > 2 {
        let tmp3 = _mm256_madd_epi16(q26, _mm256_set1_epi32(TMP3_PAIR));
        let tmp2 = _mm256_madd_epi16(q26, _mm256_set1_epi32(TMP2_PAIR));
        [
            _mm256_add_epi32(tmp0, tmp3),
            _mm256_add_epi32(tmp1, tmp2),
            _mm256_sub_epi32(tmp1, tmp2),
            _mm256_sub_epi32(tmp0, tmp3),
        ]
    } else {
        [tmp0, tmp1, tmp1, tmp0]
    };
    let q37 = (COLS > 2).then_some(q37);
    let odd0 = odd_terms(q15, q37, ODD_PAIRS[0], ODD_PAIRS[0]);
    let odd1 = odd_terms(q15, q37, ODD_PAIRS[1], ODD_PAIRS[1]);
    let odd2 = odd_terms(q15, q37, ODD_PAIRS[2], ODD_PAIRS[2]);
    let odd3 = odd_terms(q15, q37, ODD_PAIRS[3], ODD_PAIRS[3]);
    [
        _mm256_srai_epi32::<PASS2_SHIFT>(_mm256_add_epi32(tmp10, odd0)),
        _mm256_srai_epi32::<PASS2_SHIFT>(_mm256_add_epi32(tmp11, odd1)),
        _mm256_srai_epi32::<PASS2_SHIFT>(_mm256_add_epi32(tmp12, odd2)),
        _mm256_srai_epi32::<PASS2_SHIFT>(_mm256_add_epi32(tmp13, odd3)),
        _mm256_srai_epi32::<PASS2_SHIFT>(_mm256_sub_epi32(tmp13, odd3)),
        _mm256_srai_epi32::<PASS2_SHIFT>(_mm256_sub_epi32(tmp12, odd2)),
        _mm256_srai_epi32::<PASS2_SHIFT>(_mm256_sub_epi32(tmp11, odd1)),
        _mm256_srai_epi32::<PASS2_SHIFT>(_mm256_sub_epi32(tmp10, odd0)),
    ]
}

/// Saturate the eight output columns to bytes, transpose them to rows and
/// store the block.
#[inline]
#[target_feature(enable = "avx2")]
fn store_block(output: &mut [u8; 64], columns: [__m256i; 8]) {
    // Each 128-bit lane: columns 0..3 (or 4..7) of four rows, column-major.
    let left = _mm256_packus_epi16(
        _mm256_packs_epi32(columns[0], columns[1]),
        _mm256_packs_epi32(columns[2], columns[3]),
    );
    let right = _mm256_packus_epi16(
        _mm256_packs_epi32(columns[4], columns[5]),
        _mm256_packs_epi32(columns[6], columns[7]),
    );
    let transpose_4x4 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
    ));
    // 32-bit lanes now hold four bytes of one row: rows [0, 3, 4, 7 | 1, 2, 5, 6].
    let left = _mm256_shuffle_epi8(left, transpose_4x4);
    let right = _mm256_shuffle_epi8(right, transpose_4x4);
    // 64-bit lanes hold whole rows: [0, 3 | 1, 2] and [4, 7 | 5, 6].
    let top = _mm256_unpacklo_epi32(left, right);
    let bottom = _mm256_unpackhi_epi32(left, right);
    let (halves, _) = output.as_chunks_mut::<32>();
    x86_memory::store_u8x32(&mut halves[0], _mm256_permute4x64_epi64::<0x78>(top));
    x86_memory::store_u8x32(&mut halves[1], _mm256_permute4x64_epi64::<0x78>(bottom));
}

/// 32-bit-lane ISLOW IDCT, exact for every input. Only blocks outside the
/// 16-bit kernel's bound reach it.
#[cold]
#[inline(never)]
#[target_feature(enable = "avx2")]
fn idct_islow_i32(input: &[i16; 64], output: &mut [u8; 64]) {
    let (rows, _) = input.as_chunks::<8>();
    let (
        (r0l, r0h),
        (r1l, r1h),
        (r2l, r2h),
        (r3l, r3h),
        (r4l, r4h),
        (r5l, r5h),
        (r6l, r6h),
        (r7l, r7h),
    ) = (
        widen(&rows[0]),
        widen(&rows[1]),
        widen(&rows[2]),
        widen(&rows[3]),
        widen(&rows[4]),
        widen(&rows[5]),
        widen(&rows[6]),
        widen(&rows[7]),
    );

    let round1 = _mm_set1_epi32(1 << (PASS1_SHIFT - 1));
    let cw_lo = idct_1d_x4::<PASS1_SHIFT>([r0l, r1l, r2l, r3l, r4l, r5l, r6l, r7l], round1);
    let cw_hi = idct_1d_x4::<PASS1_SHIFT>([r0h, r1h, r2h, r3h, r4h, r5h, r6h, r7h], round1);

    let [q0l, q1l, q2l, q3l] = transpose_4x4_i32(cw_lo[0], cw_lo[1], cw_lo[2], cw_lo[3]);
    let [q4l, q5l, q6l, q7l] = transpose_4x4_i32(cw_hi[0], cw_hi[1], cw_hi[2], cw_hi[3]);
    let [q0h, q1h, q2h, q3h] = transpose_4x4_i32(cw_lo[4], cw_lo[5], cw_lo[6], cw_lo[7]);
    let [q4h, q5h, q6h, q7h] = transpose_4x4_i32(cw_hi[4], cw_hi[5], cw_hi[6], cw_hi[7]);

    let round2 = _mm_set1_epi32(1 << (PASS2_SHIFT - 1));
    let rw_lo = idct_1d_x4::<PASS2_SHIFT>([q0l, q1l, q2l, q3l, q4l, q5l, q6l, q7l], round2);
    let rw_hi = idct_1d_x4::<PASS2_SHIFT>([q0h, q1h, q2h, q3h, q4h, q5h, q6h, q7h], round2);

    let bias = _mm_set1_epi32(128);
    let [fll0, fll1, fll2, fll3] = transpose_4x4_i32(
        _mm_add_epi32(rw_lo[0], bias),
        _mm_add_epi32(rw_lo[1], bias),
        _mm_add_epi32(rw_lo[2], bias),
        _mm_add_epi32(rw_lo[3], bias),
    );
    let [flh0, flh1, flh2, flh3] = transpose_4x4_i32(
        _mm_add_epi32(rw_lo[4], bias),
        _mm_add_epi32(rw_lo[5], bias),
        _mm_add_epi32(rw_lo[6], bias),
        _mm_add_epi32(rw_lo[7], bias),
    );
    let [fhl0, fhl1, fhl2, fhl3] = transpose_4x4_i32(
        _mm_add_epi32(rw_hi[0], bias),
        _mm_add_epi32(rw_hi[1], bias),
        _mm_add_epi32(rw_hi[2], bias),
        _mm_add_epi32(rw_hi[3], bias),
    );
    let [fhh0, fhh1, fhh2, fhh3] = transpose_4x4_i32(
        _mm_add_epi32(rw_hi[4], bias),
        _mm_add_epi32(rw_hi[5], bias),
        _mm_add_epi32(rw_hi[6], bias),
        _mm_add_epi32(rw_hi[7], bias),
    );

    let (rows, _) = output.as_chunks_mut::<8>();
    store_row(&mut rows[0], fll0, flh0);
    store_row(&mut rows[1], fll1, flh1);
    store_row(&mut rows[2], fll2, flh2);
    store_row(&mut rows[3], fll3, flh3);
    store_row(&mut rows[4], fhl0, fhh0);
    store_row(&mut rows[5], fhl1, fhh1);
    store_row(&mut rows[6], fhl2, fhh2);
    store_row(&mut rows[7], fhl3, fhh3);
}

/// Load 8 `i16` values from `src` and sign-extend them to a pair of
/// `__m128i` each carrying 4 `i32` lanes (low 4, high 4).
#[inline]
#[target_feature(enable = "avx2")]
fn widen(src: &[i16; 8]) -> (__m128i, __m128i) {
    let full = x86_memory::load_i16x8(src);
    let lo = _mm_cvtepi16_epi32(full);
    let hi_shuffled = _mm_srli_si128::<8>(full);
    let hi = _mm_cvtepi16_epi32(hi_shuffled);
    (lo, hi)
}

/// Saturating narrow an `(i32x4, i32x4)` pair to `u8x8` and store at `dst`.
#[inline]
#[target_feature(enable = "avx2")]
fn store_row(dst: &mut [u8; 8], lo: __m128i, hi: __m128i) {
    // Lanes are [lo0..3, hi0..3] as i16.
    let i16_packed = _mm_packs_epi32(lo, hi);
    // The low eight lanes are the saturated output row.
    let u8_packed = _mm_packus_epi16(i16_packed, i16_packed);
    x86_memory::store_u8x8(dst, u8_packed);
}

/// 1D IDCT pass over 4 i32 lanes. Mirrors `idct::neon::idct_1d_x4`,
/// including the scalar row transform's DC-only shortcut.
#[target_feature(enable = "avx2")]
fn idct_1d_x4<const SHIFT: i32>(
    [p0, p1, p2, p3, p4, p5, p6, p7]: [__m128i; 8],
    rounding: __m128i,
) -> [__m128i; 8] {
    let mul = |v, c: i32| _mm_mullo_epi32(v, _mm_set1_epi32(c));

    let z1 = mul(_mm_add_epi32(p2, p6), FIX_0_541196100);
    let tmp2 = _mm_add_epi32(z1, mul(p6, -FIX_1_847759065));
    let tmp3 = _mm_add_epi32(z1, mul(p2, FIX_0_765366865));
    let tmp0 = _mm_slli_epi32::<CONST_BITS>(_mm_add_epi32(p0, p4));
    let tmp1 = _mm_slli_epi32::<CONST_BITS>(_mm_sub_epi32(p0, p4));
    let tmp10 = _mm_add_epi32(tmp0, tmp3);
    let tmp13 = _mm_sub_epi32(tmp0, tmp3);
    let tmp11 = _mm_add_epi32(tmp1, tmp2);
    let tmp12 = _mm_sub_epi32(tmp1, tmp2);

    let z1o = _mm_add_epi32(p7, p1);
    let z2o = _mm_add_epi32(p5, p3);
    let z3o = _mm_add_epi32(p7, p3);
    let z4o = _mm_add_epi32(p5, p1);
    let z5 = mul(_mm_add_epi32(z3o, z4o), FIX_1_175875602);

    let o0 = mul(p7, FIX_0_298631336);
    let o1 = mul(p5, FIX_2_053119869);
    let o2 = mul(p3, FIX_3_072711026);
    let o3 = mul(p1, FIX_1_501321110);
    let z1m = mul(z1o, -FIX_0_899976223);
    let z2m = mul(z2o, -FIX_2_562915447);
    let z3m = mul(z3o, -FIX_1_961570560);
    let z4m = mul(z4o, -FIX_0_390180644);
    let z3f = _mm_add_epi32(z3m, z5);
    let z4f = _mm_add_epi32(z4m, z5);

    let k0 = _mm_add_epi32(_mm_add_epi32(o0, z1m), z3f);
    let k1 = _mm_add_epi32(_mm_add_epi32(o1, z2m), z4f);
    let k2 = _mm_add_epi32(_mm_add_epi32(o2, z2m), z3f);
    let k3 = _mm_add_epi32(_mm_add_epi32(o3, z1m), z4f);

    let shift = |v| _mm_srai_epi32::<SHIFT>(_mm_add_epi32(v, rounding));
    let outputs = [
        shift(_mm_add_epi32(tmp10, k3)),
        shift(_mm_add_epi32(tmp11, k2)),
        shift(_mm_add_epi32(tmp12, k1)),
        shift(_mm_add_epi32(tmp13, k0)),
        shift(_mm_sub_epi32(tmp13, k0)),
        shift(_mm_sub_epi32(tmp12, k1)),
        shift(_mm_sub_epi32(tmp11, k2)),
        shift(_mm_sub_epi32(tmp10, k3)),
    ];
    if SHIFT == PASS2_SHIFT {
        // The scalar row transform descales `p0` directly when every AC input
        // of the row is zero, avoiding the overflow `p0 << CONST_BITS` hits
        // for first-pass values of 2^18 and more.
        let ac = _mm_or_si128(
            _mm_or_si128(_mm_or_si128(p1, p2), _mm_or_si128(p3, p4)),
            _mm_or_si128(_mm_or_si128(p5, p6), p7),
        );
        let dc_only = _mm_cmpeq_epi32(ac, _mm_setzero_si128());
        let dc = _mm_srai_epi32::<{ PASS1_BITS + 3 }>(_mm_add_epi32(p0, _mm_set1_epi32(16)));
        outputs.map(|value| _mm_blendv_epi8(value, dc, dc_only))
    } else {
        outputs
    }
}

/// 4×4 i32 transpose via SSE2 unpack intrinsics.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose_4x4_i32(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> [__m128i; 4] {
    // Stage 1: pairwise interleave i32 lanes.
    let t0 = _mm_unpacklo_epi32(a, b); // [a0, b0, a1, b1]
    let t1 = _mm_unpackhi_epi32(a, b); // [a2, b2, a3, b3]
    let t2 = _mm_unpacklo_epi32(c, d); // [c0, d0, c1, d1]
    let t3 = _mm_unpackhi_epi32(c, d); // [c2, d2, c3, d3]

    // Stage 2: combine halves via i64 unpack — maps 2×2 quadrants to columns.
    let col0 = _mm_unpacklo_epi64(t0, t2); // [a0, b0, c0, d0]
    let col1 = _mm_unpackhi_epi64(t0, t2); // [a1, b1, c1, d1]
    let col2 = _mm_unpacklo_epi64(t1, t3); // [a2, b2, c2, d2]
    let col3 = _mm_unpackhi_epi64(t1, t3); // [a3, b3, c3, d3]
    [col0, col1, col2, col3]
}

#[cfg(test)]
mod tests;
