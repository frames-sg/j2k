// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guarded `AArch64` NEON AAN forward DCT with exact fallback.

use core::arch::aarch64::{
    float32x4_t, float64x2_t, uint16x8_t, uint32x4_t, uint8x16_t, uint8x16x4_t, vabsq_f32,
    vaddq_f32, vaddq_f64, vaddv_u8, vand_u8, vandq_u8, vceqq_u8, vcltq_f32, vcombine_f32,
    vcvtaq_s32_f32, vdupq_n_f32, vdupq_n_u8, vget_high_f32, vget_low_f32, vld1q_f32, vld1q_f64,
    vld1q_u8_x4, vminvq_u8, vmovn_u16, vmulq_f32, vmulq_n_f32, vmulq_n_f64, vorrq_u8, vqtbl4q_u8,
    vreinterpretq_u16_u8, vrndnq_f32, vst1q_f64, vst1q_u16, vst1q_u32, vsubq_f32, vsubq_f64,
    vtrn1q_f32, vtrn2q_f32, vtstq_u16,
};

use crate::baseline_entropy::{PACKED_AC_TOKEN_COEFFICIENT_MASK, PACKED_AC_TOKEN_ZRL};

use super::QuantizedBlock;

const ZIGZAG_BYTE_INDICES: [u8; 192] = zigzag_byte_indices();
const ZIGZAG_LOOKUP_ROWS: [usize; 12] = [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7];
const ZIGZAG_LOOKUP_BOTTOM: [bool; 12] = [
    false, false, true, false, true, false, true, false, true, false, true, true,
];

#[repr(align(64))]
struct AlignedAcPackIndices([[u8; 8]; 256]);

static AC_PACK_INDICES: AlignedAcPackIndices = AlignedAcPackIndices(ac_pack_indices());
static AC_PACK_METADATA: [u8; 256] = ac_pack_metadata();
const AC_BYTE_INDEX_OFFSETS: [u8; 16] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];

// Inclusive integer Q14 distance from a half-integer that requires exact
// fallback. The proof bounds each linear form over centered bytes with
// B=127.5*L1+0.5*abs(sum). Fixed error includes the exact Q31 constant
// difference and every half-Q16 multiply error; reference error includes the
// retained f64 operations, division, and the 1e-9 direct-tie tolerance. An
// additional 2^-40 quantized-unit allowance covers stored-reciprocal rounding
// on the fast reference path and does not change any integer guard. Each
// entry is floor(max_bound * 2^14) over q=1..=255. Since the measured Q14
// distance d is an integer, an unguarded d > floor(max_bound * 2^14) implies
// d / 2^14 > max_bound. This discretization keeps the complete error bound;
// it does not round that bound downward. The largest fixed node is 844,746,781
// (<2^31), and the largest rounded AC is 1020, so the existing baseline
// coefficient and entropy contracts remain unchanged.
#[rustfmt::skip]
const Q16_QUANTIZATION_GUARD_Q14: [i32; 64] = [
    0, 0, 0, 0, 0, 1, 0, 1,
    0, 0, 0, 1, 0, 1, 0, 2,
    0, 0, 0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 1, 0, 2,
    0, 0, 0, 0, 0, 1, 0, 1,
    0, 0, 0, 1, 0, 1, 1, 2,
    0, 1, 0, 1, 0, 1, 1, 3,
    0, 2, 1, 2, 0, 3, 2, 9,
];

// Map each inclusive distance guard G to the unsigned magnitude threshold
// used after shifting the low Q14 fraction into the signed high bits. Every
// retained guard is in 0..=9. For G=0, wrapping the shift preserves the
// unsigned half-center threshold 0x8000_0000 in its i32 storage.
const fn q16_guard_magnitude_thresholds() -> [i32; 64] {
    let mut thresholds = [0i32; 64];
    let mut natural = 0;
    while natural < 64 {
        thresholds[natural] = ((1i32 << 13) - Q16_QUANTIZATION_GUARD_Q14[natural]).wrapping_shl(18);
        natural += 1;
    }
    thresholds
}

const Q16_QUANTIZATION_GUARD_MAGNITUDE_Q18: [i32; 64] = q16_guard_magnitude_thresholds();

#[expect(
    clippy::cast_possible_truncation,
    reason = "the generated lane index is in 0..8"
)]
const fn ac_pack_indices() -> [[u8; 8]; 256] {
    // A row mask's high bit describes lane zero. Selected lane indices stay
    // ordered, and padding lane eight becomes an out-of-range byte pair.
    let mut table = [[8u8; 8]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut lane = 0usize;
        let mut output = 0usize;
        while lane < 8 {
            if mask & (0x80 >> lane) != 0 {
                table[mask][output] = lane as u8;
                output += 1;
            }
            lane += 1;
        }
        mask += 1;
    }
    table
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the generated lane index is in 0..8"
)]
const fn ac_pack_metadata() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut mask = 1usize;
    while mask < 256 {
        let mut lane = 0usize;
        let mut count = 0u8;
        let mut last = 0u8;
        while lane < 8 {
            if mask & (0x80 >> lane) != 0 {
                count += 1;
                last = lane as u8;
            }
            lane += 1;
        }
        table[mask] = (last << 4) | count;
        mask += 1;
    }
    table
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "local zigzag byte offsets are reduced modulo 64 before conversion to u8"
)]
const fn zigzag_byte_indices() -> [u8; 192] {
    let mut indices = [u8::MAX; 192];
    let mut lookup = 0usize;
    while lookup < 12 {
        let mut natural = 0usize;
        while natural < 64 {
            let output = super::NATURAL_TO_ZIGZAG[natural] * 2;
            let input_byte = natural * 2;
            if output / 16 == ZIGZAG_LOOKUP_ROWS[lookup]
                && (input_byte >= 64) == ZIGZAG_LOOKUP_BOTTOM[lookup]
            {
                let output_in_row = output % 16;
                let local_input = input_byte % 64;
                indices[lookup * 16 + output_in_row] = local_input as u8;
                indices[lookup * 16 + output_in_row + 1] = (local_input + 1) as u8;
            }
            natural += 1;
        }
        lookup += 1;
    }
    indices
}

pub(super) fn fdct_values(neon: fearless_simd::Neon, block: &[u8; 64]) -> [f64; 64] {
    fdct_values_kernel(neon, block)
}

#[inline]
pub(super) fn is_constant_block(neon: fearless_simd::Neon, block: &[u8; 64]) -> bool {
    is_constant_block_kernel(neon, block)
}

#[inline]
pub(super) fn ac_zero_energy_centered_sum(
    neon: fearless_simd::Neon,
    block: &[u8; 64],
    threshold: u32,
) -> Option<i16> {
    ac_zero_energy_centered_sum_kernel(neon, block, threshold)
}

#[inline(never)]
pub(super) fn fdct_quantize_q16(
    neon: fearless_simd::Neon,
    block: &[u8; 64],
    q31_quarter_reciprocals: &[i32; 64],
    output: &mut QuantizedBlock,
) -> u64 {
    fdct_quantize_q16_kernel(neon, block, q31_quarter_reciprocals, output)
}

#[inline(never)]
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the fixed sample reference preserves the shared repeated-axis kernel ABI"
)]
pub(super) fn fdct_quantize_repeated_axis_f32<const AXIS: u8>(
    neon: fearless_simd::Neon,
    samples: &[u8; 8],
    reciprocals: &[f32; 64],
    boundary_thresholds: &[f32; 64],
    output: &mut QuantizedBlock,
) -> bool {
    match AXIS {
        1 => fdct_quantize_repeated_rows_kernel(
            neon,
            samples,
            reciprocals,
            boundary_thresholds,
            output,
        ),
        2 => fdct_quantize_repeated_columns_kernel(
            neon,
            samples,
            reciprocals,
            boundary_thresholds,
            output,
        ),
        _ => unreachable!("repeated JPEG FDCT axis is rows or columns"),
    }
}

#[inline(never)]
/// Compact an eligible block with at least 24 nonzero AC coefficients.
pub(super) fn compact_entropy_ac(
    neon: fearless_simd::Neon,
    ac_coefficients: &mut [i16; 64],
    nonzero_ac: u64,
) -> u8 {
    compact_entropy_ac_kernel(neon, ac_coefficients, nonzero_ac)
}

fearless_simd::kernel! {
    fn is_constant_block_kernel(neon: Neon, block: &[u8; 64]) -> bool {
        is_constant_block_neon(block)
    }
}

fearless_simd::kernel! {
    fn compact_entropy_ac_kernel(
        neon: Neon,
        ac_coefficients: &mut [i16; 64],
        nonzero_ac: u64,
    ) -> u8 {
        compact_entropy_ac_neon(ac_coefficients, nonzero_ac)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the row mask is one byte and the proven token count is at most 63"
)]
#[target_feature(enable = "neon")]
fn compact_entropy_ac_neon(ac_coefficients: &mut [i16; 64], nonzero_ac: u64) -> u8 {
    use core::arch::aarch64::{
        vandq_u16, vcombine_u8, vdupq_n_u16, vextq_u16, vget_lane_u8, vld1_u8, vld1q_u8, vmovl_u8,
        vorrq_u16, vorrq_u8, vqtbl1q_u8, vreinterpretq_u16_u8, vsetq_lane_u16, vshlq_n_u16,
        vshlq_n_u8, vst1q_u16, vsubq_u16, vzip1q_u8,
    };

    debug_assert_eq!(nonzero_ac & 1, 0);
    debug_assert!(nonzero_ac.count_ones() >= 24);
    let output = ac_coefficients.as_mut_ptr().cast::<u16>();
    // Restoring the unused DC bit aligns each byte with one eight-slot row.
    let mut row_bits = nonzero_ac >> 1;
    let mut zero_run = 0u8;
    let mut token_count = 0usize;

    for row_index in 0..8 {
        let row_mask = (row_bits >> 56) as u8;
        row_bits <<= 8;
        if row_mask == 0 {
            zero_run += if row_index == 0 { 7 } else { 8 };
            continue;
        }

        // SAFETY: load the complete source row before any in-place ZRL or
        // token write can touch its storage.
        let source = unsafe { vld1q_u8(ac_coefficients.as_ptr().add(row_index * 8).cast()) };
        let metadata = AC_PACK_METADATA[usize::from(row_mask)];
        // SAFETY: the mask selects one complete fixed eight-byte table row.
        let packed_lanes = unsafe { vld1_u8(AC_PACK_INDICES.0[usize::from(row_mask)].as_ptr()) };
        // Each selected i16 needs its two byte indices. Padding lane eight
        // becomes indices 16 and 17, which TBL maps to zero.
        let lane_bytes = vcombine_u8(packed_lanes, packed_lanes);
        let duplicated_lanes = vzip1q_u8(lane_bytes, lane_bytes);
        // SAFETY: the fixed array supplies all sixteen byte offsets.
        let byte_index_offsets = unsafe { vld1q_u8(AC_BYTE_INDEX_OFFSETS.as_ptr()) };
        let byte_indices = vorrq_u8(vshlq_n_u8::<1>(duplicated_lanes), byte_index_offsets);
        let packed_coefficients = vreinterpretq_u16_u8(vqtbl1q_u8(source, byte_indices));
        let count = usize::from(metadata & 0x0f);
        let last_lane = metadata >> 4;
        let first_lane = vget_lane_u8::<0>(packed_lanes);
        let leading_zeros = if row_index == 0 {
            debug_assert!(first_lane != 0);
            first_lane - 1
        } else {
            first_lane
        };
        zero_run += leading_zeros;
        // Eligibility leaves at most 39 zero AC positions in the whole block,
        // so a run can require at most two ZRLs.
        if zero_run >= 16 {
            debug_assert!(token_count < 64);
            // SAFETY: every ZRL accounts for 16 already visited zero slots, so
            // the actual token prefix cannot overtake the loaded source row.
            unsafe { output.add(token_count).write(PACKED_AC_TOKEN_ZRL) };
            token_count += 1;
            zero_run -= 16;
            if zero_run >= 16 {
                debug_assert!(token_count < 64);
                // SAFETY: the second ZRL accounts for another 16 already
                // visited zero slots and cannot overtake the loaded row.
                unsafe { output.add(token_count).write(PACKED_AC_TOKEN_ZRL) };
                token_count += 1;
                zero_run -= 16;
            }
        }

        let positions = vmovl_u8(packed_lanes);
        let previous = vextq_u16::<7>(vdupq_n_u16(0), positions);
        let mut runs = vsubq_u16(vsubq_u16(positions, previous), vdupq_n_u16(1));
        runs = vsetq_lane_u16::<0>(u16::from(zero_run), runs);
        let tokens = vorrq_u16(
            vshlq_n_u16::<4>(vandq_u16(
                packed_coefficients,
                vdupq_n_u16(PACKED_AC_TOKEN_COEFFICIENT_MASK),
            )),
            vandq_u16(runs, vdupq_n_u16(0x0f)),
        );
        debug_assert!(token_count + 8 <= 64);
        // SAFETY: even with two ZRLs, the actual prefix is no farther than the
        // start of this already loaded row: every ZRL accounts for 16 zeros,
        // while a row contributes at most seven leading zeros. The full
        // eight-token store therefore remains inside the 64-slot coefficient
        // array; only `count` tokens commit.
        unsafe { vst1q_u16(output.add(token_count), tokens) };
        token_count += count;
        zero_run = 7 - last_lane;
    }

    if zero_run != 0 {
        debug_assert!(token_count < 63);
        // SAFETY: an EOB accounts for at least one trailing AC position, so
        // the complete token stream contains at most 63 entries.
        unsafe { output.add(token_count).write(0) };
        token_count += 1;
    }
    debug_assert!(token_count <= 63);
    token_count as u8
}

fearless_simd::kernel! {
    fn ac_zero_energy_centered_sum_kernel(
        neon: Neon,
        block: &[u8; 64],
        threshold: u32,
    ) -> Option<i16> {
        ac_zero_energy_centered_sum_neon(block, threshold)
    }
}

fearless_simd::kernel! {
    fn fdct_quantize_repeated_rows_kernel(
        neon: Neon,
        samples: &[u8; 8],
        reciprocals: &[f32; 64],
        boundary_thresholds: &[f32; 64],
        output: &mut QuantizedBlock,
    ) -> bool {
        fdct_quantize_repeated_axis_neon::<1>(samples, reciprocals, boundary_thresholds, output)
    }
}

fearless_simd::kernel! {
    fn fdct_quantize_repeated_columns_kernel(
        neon: Neon,
        samples: &[u8; 8],
        reciprocals: &[f32; 64],
        boundary_thresholds: &[f32; 64],
        output: &mut QuantizedBlock,
    ) -> bool {
        fdct_quantize_repeated_axis_neon::<2>(samples, reciprocals, boundary_thresholds, output)
    }
}

fearless_simd::kernel! {
    fn fdct_values_kernel(neon: Neon, block: &[u8; 64]) -> [f64; 64] {
        fdct_values_neon(block)
    }
}

fearless_simd::kernel! {
    fn fdct_quantize_q16_kernel(
        neon: Neon,
        block: &[u8; 64],
        q31_quarter_reciprocals: &[i32; 64],
        output: &mut QuantizedBlock,
    ) -> u64 {
        fdct_quantize_q16_neon(block, q31_quarter_reciprocals, output)
    }
}

#[target_feature(enable = "neon")]
fn is_constant_block_neon(block: &[u8; 64]) -> bool {
    let reference = vdupq_n_u8(block[0]);
    // SAFETY: the x4 load reads exactly the fixed 64-byte block. AArch64
    // vector loads support unaligned addresses.
    let rows = unsafe { vld1q_u8_x4(block.as_ptr()) };
    let equal = vandq_u8(
        vandq_u8(vceqq_u8(rows.0, reference), vceqq_u8(rows.1, reference)),
        vandq_u8(vceqq_u8(rows.2, reference), vceqq_u8(rows.3, reference)),
    );
    vminvq_u8(equal) == u8::MAX
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::useless_conversion,
    reason = "the sum fits i16 and explicit u32 conversions document both exact energy products"
)]
#[target_feature(enable = "neon")]
fn ac_zero_energy_centered_sum_neon(block: &[u8; 64], threshold: u32) -> Option<i16> {
    use core::arch::aarch64::{
        vaddlvq_u16, vaddq_u16, vaddvq_u32, vdupq_n_u32, vget_high_u8, vget_low_u8, vmull_u8,
        vpadalq_u16, vpaddlq_u8,
    };

    // SAFETY: the x4 load reads exactly the fixed 64-byte block. AArch64
    // vector loads support unaligned addresses.
    let rows = unsafe { vld1q_u8_x4(block.as_ptr()) };
    let pair_sums = vaddq_u16(
        vaddq_u16(vpaddlq_u8(rows.0), vpaddlq_u8(rows.1)),
        vaddq_u16(vpaddlq_u8(rows.2), vpaddlq_u8(rows.3)),
    );
    let sum = vaddlvq_u16(pair_sums);

    let mut square_lanes = vdupq_n_u32(0);
    for row in [rows.0, rows.1, rows.2, rows.3] {
        let low = vget_low_u8(row);
        let high = vget_high_u8(row);
        square_lanes = vpadalq_u16(square_lanes, vmull_u8(low, low));
        square_lanes = vpadalq_u16(square_lanes, vmull_u8(high, high));
    }
    let square_sum = vaddvq_u32(square_lanes);

    // This is 64 times the energy about the block mean. Both products are at
    // most 266,342,400, so every operation is exact in u32.
    let energy = 64 * square_sum - u32::from(sum) * u32::from(sum);
    (energy < threshold).then_some(sum as i16 - 8_192)
}

#[target_feature(enable = "neon")]
fn fdct_quantize_repeated_axis_neon<const AXIS: u8>(
    samples: &[u8; 8],
    reciprocals: &[f32; 64],
    boundary_thresholds: &[f32; 64],
    output: &mut QuantizedBlock,
) -> bool {
    use core::arch::aarch64::{vcombine_s16, vgetq_lane_s16, vmovn_s32, vst1q_s16};

    debug_assert!(AXIS == 1 || AXIS == 2);
    let input = core::array::from_fn(|index| {
        let centered = f32::from(samples[index]) - 128.0;
        let value = if AXIS == 2 { centered * 8.0 } else { centered };
        vdupq_n_f32(value)
    });
    let transformed = fdct_8x4(input);
    let groups = [
        transpose_4x4([
            transformed[0],
            transformed[1],
            transformed[2],
            transformed[3],
        ])[0],
        transpose_4x4([
            transformed[4],
            transformed[5],
            transformed[6],
            transformed[7],
        ])[0],
    ];
    let mut rounded_groups = [core::arch::aarch64::vdupq_n_s32(0); 2];
    for (group, mut values) in groups.into_iter().enumerate() {
        if AXIS == 1 {
            values = vmulq_n_f32(values, 8.0);
        }
        let natural = core::array::from_fn::<_, 4, _>(|lane| {
            let frequency = group * 4 + lane;
            if AXIS == 1 {
                frequency
            } else {
                frequency * 8
            }
        });
        let reciprocal_values = natural.map(|index| reciprocals[index]);
        let threshold_values = natural.map(|index| boundary_thresholds[index]);
        let quantized = vmulq_f32(values, load_f32x4(&reciprocal_values));
        let nearest_fraction = vabsq_f32(vsubq_f32(quantized, vrndnq_f32(quantized)));
        let mask = lanes_u32(vcltq_f32(load_f32x4(&threshold_values), nearest_fraction));
        if mask.into_iter().any(|value| value != 0) {
            return true;
        }
        rounded_groups[group] = vcvtaq_s32_f32(quantized);
    }

    // Only eight coefficients survive a repeated axis. Narrow them once,
    // then scatter the raw values to their fixed zigzag positions.
    let coefficients = vcombine_s16(vmovn_s32(rounded_groups[0]), vmovn_s32(rounded_groups[1]));
    let mut coefficient_lanes = [0i16; 8];
    // SAFETY: the fixed array contains eight writable i16 lanes, and this
    // function executes with NEON enabled.
    unsafe { vst1q_s16(coefficient_lanes.as_mut_ptr(), coefficients) };
    output.dc = vgetq_lane_s16::<0>(coefficients);
    for (frequency, coefficient) in coefficient_lanes.into_iter().enumerate() {
        let natural = if AXIS == 1 { frequency } else { frequency * 8 };
        let zigzag = super::NATURAL_TO_ZIGZAG[natural];
        if zigzag != 0 {
            output.ac_coefficients[zigzag] = coefficient;
            if coefficient != 0 {
                output.nonzero_ac |= 1u64 << (64 - zigzag);
            }
        }
    }
    false
}

#[target_feature(enable = "neon")]
fn fdct_quantize_q16_neon(
    block: &[u8; 64],
    q31_quarter_reciprocals: &[i32; 64],
    output: &mut QuantizedBlock,
) -> u64 {
    use core::arch::aarch64::{
        vabsq_s32, vandq_s32, vcgeq_u32, vdup_n_u16, vmovn_u32, vqrdmulhq_s32, vqrshrn_n_s32,
        vreinterpretq_u32_s32, vset_lane_u16, vshlq_n_s32, vshrq_n_s32, vsri_n_u16, vsubq_s32,
    };

    let mut rounded_natural = [0i16; 64];
    let mut ambiguity_columns = vdup_n_u16(0);
    let (top, bottom) = fdct_first_pass_q16(block);

    for u in (0..8).step_by(4) {
        let top = transpose_4x4_q16([top[u], top[u + 1], top[u + 2], top[u + 3]]);
        let bottom = transpose_4x4_q16([bottom[u], bottom[u + 1], bottom[u + 2], bottom[u + 3]]);
        let transformed = fdct_8x4_q16([
            top[0], top[1], top[2], top[3], bottom[0], bottom[1], bottom[2], bottom[3],
        ]);
        for (v, values) in transformed.into_iter().enumerate() {
            let natural = v * 8 + u;
            let reciprocals = load_i32x4(&q31_quarter_reciprocals[natural..natural + 4]);
            // A Q16 transform value times a Q31 quarter-reciprocal is the
            // quantized coefficient represented in Q14.
            let quantized_q14 = vqrdmulhq_s32(values, reciprocals);
            // SHL retains exactly the low Q14 fraction and maps a half-unit
            // fraction to INT_MIN. Wrapping ABS keeps that center bit pattern
            // as unsigned 0x8000_0000 and decreases by 2^18 per unit of
            // distance from half. The unsigned inclusive comparison therefore
            // selects exactly the original `half_distance <= guard` interval.
            let shifted_fraction = vshlq_n_s32::<18>(quantized_q14);
            let half_proximity = vreinterpretq_u32_s32(vabsq_s32(shifted_fraction));
            let thresholds = vreinterpretq_u32_s32(load_i32x4(
                &Q16_QUANTIZATION_GUARD_MAGNITUDE_Q18[natural..natural + 4],
            ));
            let mut mask = vmovn_u32(vcgeq_u32(half_proximity, thresholds));

            // SQRSHRN adds its half-unit before the arithmetic shift. The
            // upward DC reciprocal keeps positive ties outward; subtracting
            // one Q14 unit only from a negative DC lane makes its tie round
            // away from zero as well. AC lanes retain unbiased rounding.
            let rounded_input = if natural == 0 {
                let negative = vshrq_n_s32::<31>(quantized_q14);
                let dc_lane = load_i32x4(&[1, 0, 0, 0]);
                vsubq_s32(quantized_q14, vandq_s32(negative, dc_lane))
            } else {
                quantized_q14
            };
            let rounded = vqrshrn_n_s32::<14>(rounded_input);
            if natural == 0 {
                // The directed reciprocal and signed bias above are proven
                // exhaustively for every centered sum and q=1..=255. Clear
                // DC only after computing that exact integer result.
                mask = vset_lane_u16::<0>(0, mask);
            }
            // Insert this row at the top while shifting earlier rows toward
            // bit zero. The two u groups therefore fill each lane's low and
            // high byte with one column of the natural-order bitmap.
            ambiguity_columns = vsri_n_u16::<1>(mask, ambiguity_columns);
            // SAFETY: `natural` advances in four-coefficient groups inside
            // the fixed 64-element output; the store writes four i16s.
            unsafe {
                core::arch::aarch64::vst1_s16(rounded_natural.as_mut_ptr().add(natural), rounded);
            };
        }
    }
    output.dc = rounded_natural[0];
    output.nonzero_ac = zigzag_and_store_nonzero(&rounded_natural, &mut output.ac_coefficients);
    ambiguity_bitmap(ambiguity_columns)
}

#[inline]
#[target_feature(enable = "neon")]
fn ambiguity_bitmap(columns: core::arch::aarch64::uint16x4_t) -> u64 {
    use core::arch::aarch64::{
        vget_lane_u64, vld1_u8, vreinterpret_u64_u8, vreinterpret_u8_u16, vtbl1_u8,
    };

    // The accumulator's bytes are [u0, u4, u1, u5, u2, u6, u3, u7].
    // Reorder them into [u0, ..., u7], where each byte's bit v describes
    // natural coefficient 8*v + u.
    const COLUMN_ORDER: [u8; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    // SAFETY: the fixed-size array provides all eight table indices.
    let order = unsafe { vld1_u8(COLUMN_ORDER.as_ptr()) };
    let columns = vtbl1_u8(vreinterpret_u8_u16(columns), order);
    let mut bitmap = vget_lane_u64::<0>(vreinterpret_u64_u8(columns));

    // Transpose the eight column bytes into eight natural-order row bytes.
    let mut swap = (bitmap ^ (bitmap >> 7)) & 0x00aa_00aa_00aa_00aa;
    bitmap ^= swap ^ (swap << 7);
    swap = (bitmap ^ (bitmap >> 14)) & 0x0000_cccc_0000_cccc;
    bitmap ^= swap ^ (swap << 14);
    swap = (bitmap ^ (bitmap >> 28)) & 0x0000_0000_f0f0_f0f0;
    bitmap ^ swap ^ (swap << 28)
}

// Load all eight rows together and keep the exact linear prefix of the first
// AAN pass in i16. The widest nodes are DC0 in [-1024, 1016] and the rotation,
// cross, and DC4 inputs in [-1020, 1020]. Little-endian targets widen by
// zipping each node above zero halfwords; big-endian targets use a numeric
// signed widen and shift so both form the same exact Q16 value.
#[inline]
#[target_feature(enable = "neon")]
#[expect(
    clippy::too_many_lines,
    reason = "the measured first-pass kernel keeps one linear NEON dataflow without helper boundaries"
)]
fn fdct_first_pass_q16(
    block: &[u8; 64],
) -> (
    [core::arch::aarch64::int32x4_t; 8],
    [core::arch::aarch64::int32x4_t; 8],
) {
    use core::arch::aarch64::{
        vaddq_s16, vaddq_s32, vdupq_n_s32, vdupq_n_u8, veorq_u8, vget_low_s8, vld4q_u8, vmovl_s8,
        vqrdmulhq_s32, vreinterpretq_s8_u8, vsubq_s16, vsubq_s32, vuzp1q_u8, vuzp2q_u8,
    };

    const Q31_1_SQRT_2: i32 = 1_518_500_250;
    const Q31_0_382: i32 = 821_806_413;
    const Q31_0_541: i32 = 1_162_209_775;
    const Q31_0_306: i32 = 658_338_954;

    // SAFETY: vld4q_u8 reads exactly 64 initialized bytes from the fixed
    // array; AArch64 vector loads accept unaligned addresses.
    let loaded = unsafe { vld4q_u8(block.as_ptr()) };
    let center = vdupq_n_u8(0x80);
    let columns = [
        veorq_u8(loaded.0, center),
        veorq_u8(loaded.1, center),
        veorq_u8(loaded.2, center),
        veorq_u8(loaded.3, center),
    ];
    let input = [
        vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(vuzp1q_u8(
            columns[0], columns[0],
        )))),
        vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(vuzp1q_u8(
            columns[1], columns[1],
        )))),
        vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(vuzp1q_u8(
            columns[2], columns[2],
        )))),
        vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(vuzp1q_u8(
            columns[3], columns[3],
        )))),
        vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(vuzp2q_u8(
            columns[0], columns[0],
        )))),
        vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(vuzp2q_u8(
            columns[1], columns[1],
        )))),
        vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(vuzp2q_u8(
            columns[2], columns[2],
        )))),
        vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(vuzp2q_u8(
            columns[3], columns[3],
        )))),
    ];

    let sum = [
        vaddq_s16(input[0], input[7]),
        vaddq_s16(input[1], input[6]),
        vaddq_s16(input[2], input[5]),
        vaddq_s16(input[3], input[4]),
    ];
    let diff = [
        vsubq_s16(input[0], input[7]),
        vsubq_s16(input[1], input[6]),
        vsubq_s16(input[2], input[5]),
        vsubq_s16(input[3], input[4]),
    ];
    let outer = vaddq_s16(sum[0], sum[3]);
    let inner = vaddq_s16(sum[1], sum[2]);
    let even = vsubq_s16(sum[0], sum[3]);
    let rotation_input = vsubq_s16(vaddq_s16(even, sum[1]), sum[2]);
    let left = vaddq_s16(diff[3], diff[2]);
    let right = vaddq_s16(diff[1], diff[0]);
    let cross_input = vsubq_s16(left, right);
    let middle_input = vaddq_s16(diff[2], diff[1]);
    let dc0 = vaddq_s16(outer, inner);
    let dc4 = vsubq_s16(outer, inner);

    #[cfg(target_endian = "little")]
    let zero = core::arch::aarch64::vdupq_n_s16(0);
    macro_rules! widen_top_q16 {
        ($value:expr) => {{
            #[cfg(target_endian = "little")]
            {
                core::arch::aarch64::vreinterpretq_s32_s16(core::arch::aarch64::vzip1q_s16(
                    zero, $value,
                ))
            }
            #[cfg(target_endian = "big")]
            {
                core::arch::aarch64::vshlq_n_s32::<16>(core::arch::aarch64::vmovl_s16(
                    core::arch::aarch64::vget_low_s16($value),
                ))
            }
        }};
    }
    macro_rules! widen_bottom_q16 {
        ($value:expr) => {{
            #[cfg(target_endian = "little")]
            {
                core::arch::aarch64::vreinterpretq_s32_s16(core::arch::aarch64::vzip2q_s16(
                    zero, $value,
                ))
            }
            #[cfg(target_endian = "big")]
            {
                core::arch::aarch64::vshlq_n_s32::<16>(core::arch::aarch64::vmovl_s16(
                    core::arch::aarch64::vget_high_s16($value),
                ))
            }
        }};
    }
    let q31_1_sqrt_2 = vdupq_n_s32(Q31_1_SQRT_2);
    let q31_0_382 = vdupq_n_s32(Q31_0_382);
    let q31_0_541 = vdupq_n_s32(Q31_0_541);
    let q31_0_306 = vdupq_n_s32(Q31_0_306);

    let top_even = widen_top_q16!(even);
    let top_rotation = vqrdmulhq_s32(widen_top_q16!(rotation_input), q31_1_sqrt_2);
    let top_left = widen_top_q16!(left);
    let top_right = widen_top_q16!(right);
    let top_cross = vqrdmulhq_s32(widen_top_q16!(cross_input), q31_0_382);
    let top_low = vaddq_s32(vqrdmulhq_s32(top_left, q31_0_541), top_cross);
    // Preserve 1.306... as right + 0.306...*right before the cross term;
    // signed Q31 cannot represent the combined multiplier.
    let top_high_scaled = vaddq_s32(top_right, vqrdmulhq_s32(top_right, q31_0_306));
    let top_high = vaddq_s32(top_high_scaled, top_cross);
    let top_middle = vqrdmulhq_s32(widen_top_q16!(middle_input), q31_1_sqrt_2);
    let top_diff0 = widen_top_q16!(diff[0]);
    let top = [
        widen_top_q16!(dc0),
        vaddq_s32(vaddq_s32(top_diff0, top_middle), top_high),
        vaddq_s32(top_even, top_rotation),
        vsubq_s32(vsubq_s32(top_diff0, top_middle), top_low),
        widen_top_q16!(dc4),
        vaddq_s32(vsubq_s32(top_diff0, top_middle), top_low),
        vsubq_s32(top_even, top_rotation),
        vsubq_s32(vaddq_s32(top_diff0, top_middle), top_high),
    ];

    let bottom_even = widen_bottom_q16!(even);
    let bottom_rotation = vqrdmulhq_s32(widen_bottom_q16!(rotation_input), q31_1_sqrt_2);
    let bottom_left = widen_bottom_q16!(left);
    let bottom_right = widen_bottom_q16!(right);
    let bottom_cross = vqrdmulhq_s32(widen_bottom_q16!(cross_input), q31_0_382);
    let bottom_low = vaddq_s32(vqrdmulhq_s32(bottom_left, q31_0_541), bottom_cross);
    let bottom_high_scaled = vaddq_s32(bottom_right, vqrdmulhq_s32(bottom_right, q31_0_306));
    let bottom_high = vaddq_s32(bottom_high_scaled, bottom_cross);
    let bottom_middle = vqrdmulhq_s32(widen_bottom_q16!(middle_input), q31_1_sqrt_2);
    let bottom_diff0 = widen_bottom_q16!(diff[0]);
    let bottom = [
        widen_bottom_q16!(dc0),
        vaddq_s32(vaddq_s32(bottom_diff0, bottom_middle), bottom_high),
        vaddq_s32(bottom_even, bottom_rotation),
        vsubq_s32(vsubq_s32(bottom_diff0, bottom_middle), bottom_low),
        widen_bottom_q16!(dc4),
        vaddq_s32(vsubq_s32(bottom_diff0, bottom_middle), bottom_low),
        vsubq_s32(bottom_even, bottom_rotation),
        vsubq_s32(vaddq_s32(bottom_diff0, bottom_middle), bottom_high),
    ];

    (top, bottom)
}

#[inline]
#[target_feature(enable = "neon")]
fn zigzag_and_store_nonzero(natural: &[i16; 64], ac_coefficients: &mut [i16; 64]) -> u64 {
    // SAFETY: each x4 load reads exactly 64 bytes within a 128-byte coefficient
    // array or the fixed 192-byte index table. AArch64 loads are unaligned.
    let (top, bottom, indices) = unsafe {
        (
            vld1q_u8_x4(natural.as_ptr().cast()),
            vld1q_u8_x4(natural.as_ptr().add(32).cast()),
            (
                vld1q_u8_x4(ZIGZAG_BYTE_INDICES.as_ptr()),
                vld1q_u8_x4(ZIGZAG_BYTE_INDICES.as_ptr().add(64)),
                vld1q_u8_x4(ZIGZAG_BYTE_INDICES.as_ptr().add(128)),
            ),
        )
    };
    let (indices_0_3, indices_4_7, indices_8_11) = indices;
    let rows = [
        vreinterpretq_u16_u8(vqtbl4q_u8(top, indices_0_3.0)),
        zigzag_row(top, bottom, indices_0_3.1, indices_0_3.2),
        zigzag_row(top, bottom, indices_0_3.3, indices_4_7.0),
        vreinterpretq_u16_u8(vqtbl4q_u8(top, indices_4_7.1)),
        vreinterpretq_u16_u8(vqtbl4q_u8(bottom, indices_4_7.2)),
        zigzag_row(top, bottom, indices_4_7.3, indices_8_11.0),
        zigzag_row(top, bottom, indices_8_11.1, indices_8_11.2),
        vreinterpretq_u16_u8(vqtbl4q_u8(bottom, indices_8_11.3)),
    ];
    let bitmap_weights = [0x80u8, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
    // SAFETY: the fixed array supplies all eight initialized bitmap weights.
    let weights = unsafe { core::arch::aarch64::vld1_u8(bitmap_weights.as_ptr()) };
    let mut nonzero = 0u64;
    for (row_index, row) in rows.into_iter().enumerate() {
        // SAFETY: each store writes one eight-i16 row within ac_coefficients.
        unsafe {
            vst1q_u16(
                ac_coefficients
                    .as_mut_ptr()
                    .add(row_index * 8)
                    .cast::<u16>(),
                row,
            );
        };
        let flags = vmovn_u16(vtstq_u16(row, row));
        nonzero = (nonzero << 8) | u64::from(vaddv_u8(vand_u8(flags, weights)));
    }
    ac_coefficients[0] = 0;
    // Drop the DC flag and align coefficient 1 with the most-significant bit.
    nonzero << 1
}

#[inline]
#[target_feature(enable = "neon")]
fn zigzag_row(
    top: uint8x16x4_t,
    bottom: uint8x16x4_t,
    top_indices: uint8x16_t,
    bottom_indices: uint8x16_t,
) -> uint16x8_t {
    vreinterpretq_u16_u8(vorrq_u8(
        vqtbl4q_u8(top, top_indices),
        vqtbl4q_u8(bottom, bottom_indices),
    ))
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4_q16(
    rows: [core::arch::aarch64::int32x4_t; 4],
) -> [core::arch::aarch64::int32x4_t; 4] {
    use core::arch::aarch64::{vcombine_s32, vget_high_s32, vget_low_s32, vtrn1q_s32, vtrn2q_s32};
    let rows_01_low = vtrn1q_s32(rows[0], rows[1]);
    let rows_01_high = vtrn2q_s32(rows[0], rows[1]);
    let rows_23_low = vtrn1q_s32(rows[2], rows[3]);
    let rows_23_high = vtrn2q_s32(rows[2], rows[3]);
    [
        vcombine_s32(vget_low_s32(rows_01_low), vget_low_s32(rows_23_low)),
        vcombine_s32(vget_low_s32(rows_01_high), vget_low_s32(rows_23_high)),
        vcombine_s32(vget_high_s32(rows_01_low), vget_high_s32(rows_23_low)),
        vcombine_s32(vget_high_s32(rows_01_high), vget_high_s32(rows_23_high)),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn fdct_8x4_q16(input: [core::arch::aarch64::int32x4_t; 8]) -> [core::arch::aarch64::int32x4_t; 8] {
    use core::arch::aarch64::{vaddq_s32, vdupq_n_s32, vqrdmulhq_s32, vsubq_s32};

    const Q31_1_SQRT_2: i32 = 1_518_500_250;
    const Q31_0_382: i32 = 821_806_413;
    const Q31_0_541: i32 = 1_162_209_775;
    const Q31_0_306: i32 = 658_338_954;

    let sum: [core::arch::aarch64::int32x4_t; 4] =
        core::array::from_fn(|i| vaddq_s32(input[i], input[7 - i]));
    let diff: [core::arch::aarch64::int32x4_t; 4] =
        core::array::from_fn(|i| vsubq_s32(input[i], input[7 - i]));
    let outer = vaddq_s32(sum[0], sum[3]);
    let inner = vaddq_s32(sum[1], sum[2]);
    let even = vsubq_s32(sum[0], sum[3]);
    let rotation = vqrdmulhq_s32(
        vsubq_s32(vaddq_s32(even, sum[1]), sum[2]),
        vdupq_n_s32(Q31_1_SQRT_2),
    );
    let left = vaddq_s32(diff[3], diff[2]);
    let right = vaddq_s32(diff[1], diff[0]);
    let cross = vqrdmulhq_s32(vsubq_s32(left, right), vdupq_n_s32(Q31_0_382));
    let low = vaddq_s32(vqrdmulhq_s32(left, vdupq_n_s32(Q31_0_541)), cross);
    // 1.306... cannot be represented in signed Q31. Preserve the established
    // graph as right + 0.306...*right before adding the cross term.
    let high_scaled = vaddq_s32(right, vqrdmulhq_s32(right, vdupq_n_s32(Q31_0_306)));
    let high = vaddq_s32(high_scaled, cross);
    let middle = vqrdmulhq_s32(vaddq_s32(diff[2], diff[1]), vdupq_n_s32(Q31_1_SQRT_2));
    [
        vaddq_s32(outer, inner),
        vaddq_s32(vaddq_s32(diff[0], middle), high),
        vaddq_s32(even, rotation),
        vsubq_s32(vsubq_s32(diff[0], middle), low),
        vsubq_s32(outer, inner),
        vaddq_s32(vsubq_s32(diff[0], middle), low),
        vsubq_s32(even, rotation),
        vsubq_s32(vaddq_s32(diff[0], middle), high),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4(rows: [float32x4_t; 4]) -> [float32x4_t; 4] {
    let rows_01_low = vtrn1q_f32(rows[0], rows[1]);
    let rows_01_high = vtrn2q_f32(rows[0], rows[1]);
    let rows_23_low = vtrn1q_f32(rows[2], rows[3]);
    let rows_23_high = vtrn2q_f32(rows[2], rows[3]);
    [
        vcombine_f32(vget_low_f32(rows_01_low), vget_low_f32(rows_23_low)),
        vcombine_f32(vget_low_f32(rows_01_high), vget_low_f32(rows_23_high)),
        vcombine_f32(vget_high_f32(rows_01_low), vget_high_f32(rows_23_low)),
        vcombine_f32(vget_high_f32(rows_01_high), vget_high_f32(rows_23_high)),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
#[expect(
    clippy::excessive_precision,
    reason = "the scalar and NEON AAN paths retain the same established source constants before f32 rounding"
)]
fn fdct_8x4(input: [float32x4_t; 8]) -> [float32x4_t; 8] {
    let sum: [float32x4_t; 4] = core::array::from_fn(|i| vaddq_f32(input[i], input[7 - i]));
    let diff: [float32x4_t; 4] = core::array::from_fn(|i| vsubq_f32(input[i], input[7 - i]));
    let outer = vaddq_f32(sum[0], sum[3]);
    let inner = vaddq_f32(sum[1], sum[2]);
    let even = vsubq_f32(sum[0], sum[3]);
    let rotation = vmulq_n_f32(
        vsubq_f32(vaddq_f32(even, sum[1]), sum[2]),
        core::f32::consts::FRAC_1_SQRT_2,
    );
    let left = vaddq_f32(diff[3], diff[2]);
    let right = vaddq_f32(diff[1], diff[0]);
    let cross = vmulq_n_f32(vsubq_f32(left, right), 0.382_683_432_365_089_8_f32);
    let low = vaddq_f32(vmulq_n_f32(left, 0.541_196_100_146_197_f32), cross);
    let high = vaddq_f32(vmulq_n_f32(right, 1.306_562_964_876_376_6_f32), cross);
    let middle = vmulq_n_f32(
        vaddq_f32(diff[2], diff[1]),
        core::f32::consts::FRAC_1_SQRT_2,
    );
    [
        vaddq_f32(outer, inner),
        vaddq_f32(vaddq_f32(diff[0], middle), high),
        vaddq_f32(even, rotation),
        vsubq_f32(vsubq_f32(diff[0], middle), low),
        vsubq_f32(outer, inner),
        vaddq_f32(vsubq_f32(diff[0], middle), low),
        vsubq_f32(even, rotation),
        vsubq_f32(vaddq_f32(diff[0], middle), high),
    ]
}

#[target_feature(enable = "neon")]
fn fdct_values_neon(block: &[u8; 64]) -> [f64; 64] {
    let mut columns = [[0.0; 8]; 8];
    for y in (0..8).step_by(2) {
        let row = fdct_8x2(core::array::from_fn(|x| {
            pair(
                f64::from(block[y * 8 + x]) - 128.0,
                f64::from(block[(y + 1) * 8 + x]) - 128.0,
            )
        }));
        for (u, values) in row.into_iter().enumerate() {
            let values = lanes(values);
            columns[u][y] = values[0];
            columns[u][y + 1] = values[1];
        }
    }

    let mut output = [0.0; 64];
    for u in (0..8).step_by(2) {
        let columns = fdct_8x2(core::array::from_fn(|y| {
            pair(columns[u][y], columns[u + 1][y])
        }));
        for (v, values) in columns.into_iter().enumerate() {
            let values = lanes(values);
            output[v * 8 + u] = values[0];
            output[v * 8 + u + 1] = values[1];
        }
    }
    output
}

#[inline]
#[target_feature(enable = "neon")]
fn fdct_8x2(input: [float64x2_t; 8]) -> [float64x2_t; 8] {
    let sum: [float64x2_t; 4] = core::array::from_fn(|i| vaddq_f64(input[i], input[7 - i]));
    let diff: [float64x2_t; 4] = core::array::from_fn(|i| vsubq_f64(input[i], input[7 - i]));
    let outer = vaddq_f64(sum[0], sum[3]);
    let inner = vaddq_f64(sum[1], sum[2]);
    let even = vsubq_f64(sum[0], sum[3]);
    let rotation = vmulq_n_f64(
        vsubq_f64(vaddq_f64(even, sum[1]), sum[2]),
        core::f64::consts::FRAC_1_SQRT_2,
    );
    let left = vaddq_f64(diff[3], diff[2]);
    let right = vaddq_f64(diff[1], diff[0]);
    let cross = vmulq_n_f64(vsubq_f64(left, right), 0.382_683_432_365_089_8);
    let low = vaddq_f64(vmulq_n_f64(left, 0.541_196_100_146_197), cross);
    let high = vaddq_f64(vmulq_n_f64(right, 1.306_562_964_876_376_6), cross);
    let middle = vmulq_n_f64(
        vaddq_f64(diff[2], diff[1]),
        core::f64::consts::FRAC_1_SQRT_2,
    );
    [
        vaddq_f64(outer, inner),
        vaddq_f64(vaddq_f64(diff[0], middle), high),
        vaddq_f64(even, rotation),
        vsubq_f64(vsubq_f64(diff[0], middle), low),
        vsubq_f64(outer, inner),
        vaddq_f64(vsubq_f64(diff[0], middle), low),
        vsubq_f64(even, rotation),
        vsubq_f64(vaddq_f64(diff[0], middle), high),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn load_f32x4(values: &[f32]) -> float32x4_t {
    debug_assert!(values.len() >= 4);
    // SAFETY: callers pass four-element ranges from fixed-size arrays, AArch64
    // vector loads support unaligned addresses, and this runs in a Neon kernel.
    unsafe { vld1q_f32(values.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i32x4(values: &[i32]) -> core::arch::aarch64::int32x4_t {
    debug_assert!(values.len() >= 4);
    // SAFETY: callers pass four-element ranges from fixed-size arrays, AArch64
    // vector loads support unaligned addresses, and this runs in a NEON kernel.
    unsafe { core::arch::aarch64::vld1q_s32(values.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn lanes_u32(values: uint32x4_t) -> [u32; 4] {
    let mut output = [0u32; 4];
    // SAFETY: the fixed-size array proves four writable lanes and this runs in
    // a Neon kernel. AArch64 vector stores support unaligned addresses.
    unsafe { vst1q_u32(output.as_mut_ptr(), values) };
    output
}

#[inline]
#[target_feature(enable = "neon")]
fn pair(low: f64, high: f64) -> float64x2_t {
    let values = [low, high];
    // SAFETY: the fixed-size array proves two readable f64 values, AArch64
    // vector loads support unaligned addresses, and this runs in a Neon kernel.
    unsafe { vld1q_f64(values.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn lanes(values: float64x2_t) -> [f64; 2] {
    let mut output = [0.0; 2];
    // SAFETY: the fixed-size array proves two writable f64 values, AArch64
    // vector stores support unaligned addresses, and this runs in a Neon kernel.
    unsafe { vst1q_f64(output.as_mut_ptr(), values) };
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neon_fdct_values_match_scalar_bits() {
        let Some(neon) = fearless_simd::Level::new().as_neon() else {
            return;
        };
        let mut seed = 17u32;
        for case in 0..256 {
            let block = core::array::from_fn(|index| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                match case {
                    0 => 0,
                    1 => 255,
                    2 => [0, 255][index & 1],
                    _ => (seed >> 24) as u8,
                }
            });
            let expected = super::super::fdct_values_scalar(&block);
            let actual = fdct_values(neon, &block);
            for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "case={case} coefficient={index} actual={actual} expected={expected}"
                );
            }
        }
    }
}
