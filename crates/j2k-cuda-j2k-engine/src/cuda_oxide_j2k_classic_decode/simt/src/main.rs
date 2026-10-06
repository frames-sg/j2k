#![allow(
    static_mut_refs,
    reason = "CUDA shared-memory state is scoped to one classic Tier-1 codeblock"
)]

use cuda_device::{SharedArray, kernel, thread};
use cuda_host::cuda_module;
use j2k_codec_math::classic::irreversible_midpoint_bit;
include!("../../../cuda_oxide_simt_prelude.rs");

// One u32 describes a four-row stripe column. Bits 0..17 cache the
// 3-column by 6-row significance window; bits 18,19,22,25,28,31 cache signs;
// bits 20,23,26,29 mark magnitude refinement; bits 21,24,27,30 mark the
// significance-propagation visit for the current bitplane. Shifting by
// 3*row aligns the same masks for each coefficient in the stripe.
// Retain side padding; cross-stripe updates are bounded to real stripes.
const MAX_FLAG_WORDS: usize = 66 * 16;
const SIGMA_THIS: u32 = 1 << 4;
const SIGMA_NEIGHBOURS: u32 = 0x1ef;
const SIGMA_ROWS: u32 = SIGMA_THIS | (SIGMA_THIS << 3) | (SIGMA_THIS << 6) | (SIGMA_THIS << 9);
const MU_THIS: u32 = 1 << 20;
const PI_THIS: u32 = 1 << 21;
const PI_ALL: u32 = PI_THIS | (PI_THIS << 3) | (PI_THIS << 6) | (PI_THIS << 9);
const CLASSIC_DECODE_THREADS: u32 = 32;
const STYLE_RESET_CONTEXTS: u32 = 1 << 0;
const STYLE_VERTICALLY_CAUSAL: u32 = 1 << 2;
const STYLE_SEGMENTATION_SYMBOLS: u32 = 1 << 3;
const KNOWN_STYLE_FLAGS: u32 = 0x1f;
const STATUS_OK: u32 = 0;
const STATUS_FAILED: u32 = 1;
const STATUS_UNSUPPORTED: u32 = 2;
const RAW_READ_FAILED: u32 = u32::MAX;
const MQ_QE_WORD_OFFSET: usize = 0;
const MQ_TRANSITION_WORD_OFFSET: usize = 47;
const SIGN_CONTEXT_HALFWORD_OFFSET: usize = 188;
const ZERO_CONTEXT_LL_LH_BYTE_OFFSET: usize = 888;
const ZERO_CONTEXT_HL_BYTE_OFFSET: usize = 1400;
const ZERO_CONTEXT_HH_BYTE_OFFSET: usize = 1912;

#[repr(C)]
#[derive(Clone, Copy)]
struct ClassicJob {
    output_ptr: u64,
    coded_offset: u32,
    coded_len: u32,
    segment_offset: u32,
    segment_count: u32,
    scratch_offset: u32,
    width: u32,
    height: u32,
    output_stride: u32,
    output_offset: u32,
    missing_msbs: u32,
    total_bitplanes: u32,
    number_of_coding_passes: u32,
    sub_band_type: u32,
    style_flags: u32,
    strict: u32,
    irreversible_midpoint: u32,
    dequantization_step: f32,
    roi_shift: u32,
    status_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClassicSegment {
    data_offset: u32,
    data_length: u32,
    start_coding_pass: u32,
    end_coding_pass: u32,
    use_arithmetic: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClassicStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

struct ArithmeticDecoder {
    mq_tables: *const u32,
    data: *const u8,
    data_len: u32,
    c: u32,
    a: u32,
    base_pointer: u32,
    byte: u8,
    shift_count: u32,
}

struct BypassDecoder {
    data: *const u8,
    data_len: u32,
    bit_pos: u32,
    strict: bool,
}

#[inline(always)]
fn coefficient_index(padded_width: u32, x: u32, y: u32) -> u32 {
    x + y * padded_width
}

#[inline(always)]
fn flag_index(stride: u32, stripe: u32, x: u32) -> u32 {
    stripe * stride + x + 1
}

#[inline(always)]
fn load_flag(flags: *const u32, index: u32) -> u32 {
    simt_load(flags, index as usize)
}

#[inline(always)]
fn or_flag(flags: *mut u32, index: u32, bits: u32) {
    simt_store(
        flags,
        index as usize,
        load_flag(flags.cast_const(), index) | bits,
    );
}

#[inline(always)]
fn push_coefficient_bit(coefficients: *mut u32, index: u32, position: u32) {
    let value = simt_load(coefficients.cast_const(), index as usize);
    simt_store(coefficients, index as usize, value | (1 << position));
}

#[inline(always)]
fn initialize_significant_coefficient(
    coefficients: *mut u32,
    index: u32,
    position: u32,
    negative: u32,
) {
    simt_store(
        coefficients,
        index as usize,
        (1 << position) | (negative << 31),
    );
}

#[inline(always)]
fn reconstructed_classic_sample(coefficient: u32, job: ClassicJob) -> f32 {
    let magnitude = coefficient & 0x7fff_ffff;
    let mut reconstructed = magnitude as f32;
    let decoded_bitplanes = job.total_bitplanes + job.roi_shift - job.missing_msbs;
    if job.irreversible_midpoint != 0 {
        if let Some(lowest_decoded_bit) = irreversible_midpoint_bit(
            u64::from(magnitude),
            decoded_bitplanes,
            job.number_of_coding_passes,
        ) {
            let mut fixed_magnitude = (magnitude << 1) | (1 << lowest_decoded_bit);
            if job.roi_shift != 0 && fixed_magnitude >= 1 << job.roi_shift {
                fixed_magnitude >>= job.roi_shift;
            }
            reconstructed = fixed_magnitude as f32 * 0.5;
        }
    } else if job.roi_shift != 0 && magnitude >= 1 << job.roi_shift {
        reconstructed = (magnitude >> job.roi_shift) as f32;
    }
    if coefficient & 0x8000_0000 != 0 {
        -reconstructed
    } else {
        reconstructed
    }
}

#[inline(always)]
fn zero_context_table(tables: *const u8, sub_band_type: u32) -> *const u8 {
    let offset = match sub_band_type {
        1 => ZERO_CONTEXT_HL_BYTE_OFFSET,
        3 => ZERO_CONTEXT_HH_BYTE_OFFSET,
        _ => ZERO_CONTEXT_LL_LH_BYTE_OFFSET,
    };
    simt_mut_ptr_at(tables.cast_mut(), offset).cast_const()
}

#[inline(always)]
fn zero_context(zero_contexts: *const u8, window: u32) -> u8 {
    simt_load(zero_contexts, (window & SIGMA_NEIGHBOURS) as usize)
}

#[inline(always)]
fn magnitude_context(window: u32) -> u8 {
    if window & MU_THIS != 0 {
        16
    } else {
        14 + u8::from(window & SIGMA_NEIGHBOURS != 0)
    }
}

#[inline(always)]
fn sign_context(tables: *const u8, own: u32, left: u32, right: u32, row: u32) -> (u8, u8) {
    let window = own >> (3 * row);
    let significances = (((window >> 1) & 1) << 6)
        | (((window >> 3) & 1) << 4)
        | (((window >> 5) & 1) << 2)
        | ((window >> 7) & 1);
    let north_sign_bit = if row == 0 { 18 } else { 16 + 3 * row };
    let signs = (((own >> north_sign_bit) & 1) << 6)
        | (((left >> (19 + 3 * row)) & 1) << 4)
        | (((right >> (19 + 3 * row)) & 1) << 2)
        | (((own >> (22 + 3 * row)) & 1) << 0);
    let negative = significances & signs;
    let positive = significances & !signs;
    let packed = simt_load(
        tables.cast::<u16>(),
        SIGN_CONTEXT_HALFWORD_OFFSET + (((negative << 1) | positive) as usize),
    );
    (packed as u8, (packed >> 8) as u8)
}

#[inline(always)]
fn set_significant(
    flags: *mut u32,
    own: &mut u32,
    index: u32,
    stride: u32,
    flag_count: u32,
    row: u32,
    negative: u32,
    vertically_causal: bool,
) {
    let shift = 3 * row;
    *own |= (SIGMA_THIS | (negative << 19)) << shift;
    or_flag(flags, index - 1, (1 << 5) << shift);
    or_flag(flags, index + 1, (1 << 3) << shift);
    if row == 0 && !vertically_causal && index >= stride {
        let north = index - stride;
        or_flag(flags, north - 1, 1 << 17);
        or_flag(flags, north, (1 << 16) | (negative << 31));
        or_flag(flags, north + 1, 1 << 15);
    } else if row == 3 && index + stride < flag_count {
        let south = index + stride;
        or_flag(flags, south - 1, 1 << 2);
        or_flag(flags, south, (1 << 1) | (negative << 18));
        or_flag(flags, south + 1, 1 << 0);
    }
}

// Context words contain Qe in bits 0..15, NMPS in 16..21, NLPS in
// 22..27, the MPS switch in bit 28, and the current MPS in bit 31.
// Caching the probability with its transitions avoids a dependent table
// lookup on every decoded bit.
#[inline(always)]
fn reset_contexts(contexts: *mut u32, mq_tables: *const u32) {
    let mut index = 0;
    let initial = simt_load(mq_tables, 0);
    while index < 19 {
        simt_store(contexts, index, initial);
        index += 1;
    }
    simt_store(contexts, 0, simt_load(mq_tables, 4));
    simt_store(contexts, 17, simt_load(mq_tables, 3));
    simt_store(contexts, 18, simt_load(mq_tables, 46));
}

#[inline(always)]
fn next_byte(decoder: &ArithmeticDecoder) -> u8 {
    if decoder.base_pointer + 1 < decoder.data_len {
        simt_load(decoder.data, decoder.base_pointer as usize + 1)
    } else {
        0xff
    }
}

#[inline(always)]
fn arithmetic_read_byte(decoder: &mut ArithmeticDecoder) {
    if decoder.byte == 0xff {
        let next = next_byte(decoder);
        if next > 0x8f {
            decoder.shift_count = 8;
        } else {
            decoder.base_pointer += 1;
            decoder.byte = next;
            decoder.c = decoder
                .c
                .wrapping_add(0xfe00)
                .wrapping_sub((next as u32) << 9);
            decoder.shift_count = 7;
        }
    } else {
        decoder.byte = next_byte(decoder);
        decoder.base_pointer += 1;
        decoder.c = decoder
            .c
            .wrapping_add(0xff00)
            .wrapping_sub((decoder.byte as u32) << 8);
        decoder.shift_count = 8;
    }
}

#[inline(always)]
fn arithmetic_initialize(decoder: &mut ArithmeticDecoder) {
    decoder.byte = if decoder.data_len == 0 {
        0xff
    } else {
        simt_load(decoder.data, 0)
    };
    decoder.c = ((decoder.byte ^ 0xff) as u32) << 16;
    arithmetic_read_byte(decoder);
    decoder.c <<= 7;
    decoder.shift_count -= 7;
    decoder.a = 0x8000;
}

#[inline(always)]
fn arithmetic_renormalize(decoder: &mut ArithmeticDecoder) {
    let mut remaining = decoder.a.leading_zeros().saturating_sub(16);
    while remaining != 0 {
        if decoder.shift_count == 0 {
            arithmetic_read_byte(decoder);
        }
        // Stop exactly at the next byte boundary so byte stuffing and the MQ
        // marker rules see the same accumulator as one-bit renormalization.
        let shift = remaining.min(decoder.shift_count);
        decoder.a <<= shift;
        decoder.c <<= shift;
        decoder.shift_count -= shift;
        remaining -= shift;
    }
}

#[inline(always)]
fn arithmetic_decode_bit(decoder: &mut ArithmeticDecoder, contexts: *mut u32, label: u8) -> u32 {
    let context = simt_load(contexts.cast_const(), label as usize);
    let qe = context & 0xffff;
    let mps = context >> 31;
    decoder.a -= qe;
    let lower_interval = decoder.c >> 16 < decoder.a;
    if lower_interval && decoder.a & 0x8000 != 0 {
        return mps;
    }
    // Conditional exchange reverses MPS/LPS when the reduced interval is
    // smaller than Qe.
    let lps = lower_interval == (decoder.a < qe);
    let decoded = mps ^ u32::from(lps);
    if !lower_interval {
        decoder.c -= decoder.a << 16;
        decoder.a = qe;
    }
    let next_state = if lps {
        (context >> 22) & 0x3f
    } else {
        (context >> 16) & 0x3f
    };
    let switch = (context >> 28) & 1;
    let next_mps = mps ^ (u32::from(lps) & switch);
    let next_context = simt_load(decoder.mq_tables, next_state as usize);
    simt_store(contexts, label as usize, next_context | (next_mps << 31));
    let shifts_needed = decoder.a.leading_zeros() - 16;
    if shifts_needed <= decoder.shift_count {
        decoder.a <<= shifts_needed;
        decoder.c <<= shifts_needed;
        decoder.shift_count -= shifts_needed;
    } else {
        arithmetic_renormalize(decoder);
    }
    decoded
}

#[inline(always)]
fn raw_read_bit(decoder: &mut BypassDecoder) -> u32 {
    let byte_position = decoder.bit_pos / 8;
    if byte_position >= decoder.data_len {
        // T.800 D.4.1 extends a cleanly exhausted terminated segment with
        // 0xFF bytes. Strict mode still rejects a malformed stuffed bit below.
        decoder.bit_pos += 1;
        return 1;
    }
    let bit_position = decoder.bit_pos % 8;
    let byte = simt_load(decoder.data, byte_position as usize);
    decoder.bit_pos += 1;
    ((byte as u32) >> (7 - bit_position)) & 1
}

#[inline(always)]
fn bypass_read_bit(decoder: &mut BypassDecoder) -> u32 {
    let byte_position = decoder.bit_pos / 8;
    let bit_position = decoder.bit_pos % 8;
    let bit = raw_read_bit(decoder);
    if bit == RAW_READ_FAILED {
        return RAW_READ_FAILED;
    }
    if bit_position == 7
        && byte_position < decoder.data_len
        && simt_load(decoder.data, byte_position as usize) == 0xff
    {
        let stuffed = raw_read_bit(decoder);
        if decoder.strict && stuffed != 0 {
            return RAW_READ_FAILED;
        }
    }
    bit
}

#[inline(always)]
fn decode_sign_arithmetic(
    decoder: &mut ArithmeticDecoder,
    contexts: *mut u32,
    tables: *const u8,
    flags: *mut u32,
    own: &mut u32,
    flag_index: u32,
    flag_stride: u32,
    flag_count: u32,
    row: u32,
    coefficients: *mut u32,
    coefficient: u32,
    position: u32,
    vertically_causal: bool,
) {
    let (label, xor) = sign_context(
        tables,
        *own,
        load_flag(flags.cast_const(), flag_index - 1),
        load_flag(flags.cast_const(), flag_index + 1),
        row,
    );
    let negative = arithmetic_decode_bit(decoder, contexts, label) ^ xor as u32;
    initialize_significant_coefficient(coefficients, coefficient, position, negative);
    set_significant(
        flags,
        own,
        flag_index,
        flag_stride,
        flag_count,
        row,
        negative,
        vertically_causal,
    );
}

#[inline(always)]
fn decode_sign_bypass(
    decoder: &mut BypassDecoder,
    flags: *mut u32,
    own: &mut u32,
    flag_index: u32,
    flag_stride: u32,
    flag_count: u32,
    row: u32,
    coefficients: *mut u32,
    coefficient: u32,
    position: u32,
    vertically_causal: bool,
) -> bool {
    let negative = bypass_read_bit(decoder);
    if negative == RAW_READ_FAILED {
        return false;
    }
    initialize_significant_coefficient(coefficients, coefficient, position, negative);
    set_significant(
        flags,
        own,
        flag_index,
        flag_stride,
        flag_count,
        row,
        negative,
        vertically_causal,
    );
    true
}

#[inline(always)]
fn set_status(statuses: *mut ClassicStatus, index: u32, code: u32, detail: u32) {
    simt_store(
        statuses,
        index as usize,
        ClassicStatus {
            code,
            detail,
            reserved0: 0,
            reserved1: 0,
        },
    );
}

#[inline(always)]
fn fail(statuses: *mut ClassicStatus, index: u32, code: u32, detail: u32) -> bool {
    set_status(statuses, index, code, detail);
    false
}

#[inline(always)]
fn validate_job_header(job: ClassicJob, statuses: *mut ClassicStatus, job_index: u32) -> bool {
    if job.width == 0
        || job.height == 0
        || job.width > 64
        || job.height > 64
        || job.output_stride < job.width
    {
        return fail(statuses, job_index, STATUS_UNSUPPORTED, 1);
    }
    if job.total_bitplanes == 0
        || job.total_bitplanes > 31
        || job.roi_shift > 31 - job.total_bitplanes
        || job.sub_band_type > 3
        || job.style_flags & !KNOWN_STYLE_FLAGS != 0
    {
        return fail(statuses, job_index, STATUS_UNSUPPORTED, 2);
    }
    let coded_bitplanes = job.total_bitplanes + job.roi_shift;
    if job.missing_msbs >= coded_bitplanes {
        return fail(statuses, job_index, STATUS_UNSUPPORTED, 2);
    }
    let bitplanes = coded_bitplanes - job.missing_msbs;
    let max_passes = 1 + 3 * (bitplanes - 1);
    if job.number_of_coding_passes > max_passes {
        return fail(statuses, job_index, STATUS_UNSUPPORTED, 3);
    }
    if job.number_of_coding_passes != 0 && job.segment_count == 0 {
        return fail(statuses, job_index, STATUS_UNSUPPORTED, 4);
    }
    true
}

#[inline(always)]
fn decode_pass<const PASS_TYPE: u32, const COMMON_STYLE: bool>(
    job: ClassicJob,
    arithmetic: &mut ArithmeticDecoder,
    bypass: &mut BypassDecoder,
    use_arithmetic: bool,
    contexts: *mut u32,
    tables: *const u8,
    flags: *mut u32,
    coefficients: *mut u32,
    current_position: u32,
    statuses: *mut ClassicStatus,
    job_index: u32,
) -> bool {
    let padded_width = job.width + 2;
    let flag_stride = job.width + 2;
    let zero_contexts = zero_context_table(tables, job.sub_band_type);
    let style_flags = if COMMON_STYLE { 0 } else { job.style_flags };
    let vertically_causal = style_flags & STYLE_VERTICALLY_CAUSAL != 0;
    let stripe_count = (job.height + 3) / 4;
    let flag_count = flag_stride * stripe_count;
    let mut stripe = 0;
    while stripe < stripe_count {
        let base_row = 4 * stripe;
        let rows = (job.height - base_row).min(4);
        let mut x = 0;
        while x < job.width {
            let flag_index = flag_index(flag_stride, stripe, x);
            let mut own = load_flag(flags.cast_const(), flag_index);
            let initial_own = own;
            let own_sigma = own & SIGMA_ROWS;
            let top_coefficient = coefficient_index(padded_width, x + 1, base_row + 1);

            // These skips preserve scan order because none of their rows can
            // consume a symbol or update a neighbor word in this pass.
            if PASS_TYPE == 1 && (own == 0 || (rows == 4 && own_sigma == SIGMA_ROWS)) {
                x += 1;
                continue;
            }
            if PASS_TYPE == 2 && own_sigma == 0 {
                x += 1;
                continue;
            }
            if PASS_TYPE == 0 && rows == 4 && own_sigma == SIGMA_ROWS {
                // Even a fully significant cleanup column must retire visit
                // bits carried from this bitplane's significance pass.
                own &= !PI_ALL;
                if own != initial_own {
                    simt_store(flags, flag_index as usize, own);
                }
                x += 1;
                continue;
            }

            if PASS_TYPE == 0 && rows == 4 && own == 0 {
                let bit = arithmetic_decode_bit(arithmetic, contexts, 17);
                if bit != 0 {
                    let first = (arithmetic_decode_bit(arithmetic, contexts, 18) << 1)
                        | arithmetic_decode_bit(arithmetic, contexts, 18);
                    let coefficient = top_coefficient + first * padded_width;
                    decode_sign_arithmetic(
                        arithmetic,
                        contexts,
                        tables,
                        flags,
                        &mut own,
                        flag_index,
                        flag_stride,
                        flag_count,
                        first,
                        coefficients,
                        coefficient,
                        current_position,
                        vertically_causal,
                    );
                    let mut row = first + 1;
                    while row < 4 {
                        let window = own >> (3 * row);
                        let bit = arithmetic_decode_bit(
                            arithmetic,
                            contexts,
                            zero_context(zero_contexts, window),
                        );
                        if bit != 0 {
                            decode_sign_arithmetic(
                                arithmetic,
                                contexts,
                                tables,
                                flags,
                                &mut own,
                                flag_index,
                                flag_stride,
                                flag_count,
                                row,
                                coefficients,
                                top_coefficient + row * padded_width,
                                current_position,
                                vertically_causal,
                            );
                        }
                        row += 1;
                    }
                }
                // PI is known clear when the aggregate run-length path applies.
                if own != initial_own {
                    simt_store(flags, flag_index as usize, own);
                }
                x += 1;
                continue;
            }

            if PASS_TYPE == 2 {
                // Sigma and PI use the same three-bit row spacing. Refinement
                // visits only coefficients significant before this bitplane;
                // enumerate those rows directly in top-to-bottom scan order.
                let valid_rows = SIGMA_ROWS & ((1 << (3 * rows + 2)) - 1);
                let mut eligible = own_sigma & !((own & PI_ALL) >> 17) & valid_rows;
                while eligible != 0 {
                    let sigma_bit = eligible.trailing_zeros();
                    let row = (sigma_bit - SIGMA_THIS.trailing_zeros()) / 3;
                    let shift = 3 * row;
                    let window = own >> shift;
                    let coefficient = top_coefficient + row * padded_width;
                    let label = magnitude_context(window);
                    let bit = if use_arithmetic {
                        arithmetic_decode_bit(arithmetic, contexts, label)
                    } else {
                        let bit = bypass_read_bit(bypass);
                        if bit == RAW_READ_FAILED {
                            return fail(statuses, job_index, STATUS_FAILED, 10);
                        }
                        bit
                    };
                    if bit != 0 {
                        push_coefficient_bit(coefficients, coefficient, current_position);
                    }
                    own |= MU_THIS << shift;
                    eligible &= eligible - 1;
                }
                if own != initial_own {
                    simt_store(flags, flag_index as usize, own);
                }
                x += 1;
                continue;
            }

            let mut row = 0;
            while row < rows {
                let shift = 3 * row;
                let window = own >> shift;
                let coefficient = top_coefficient + row * padded_width;
                if PASS_TYPE == 0 {
                    if window & (SIGMA_THIS | PI_THIS) == 0 {
                        let bit = arithmetic_decode_bit(
                            arithmetic,
                            contexts,
                            zero_context(zero_contexts, window),
                        );
                        if bit != 0 {
                            decode_sign_arithmetic(
                                arithmetic,
                                contexts,
                                tables,
                                flags,
                                &mut own,
                                flag_index,
                                flag_stride,
                                flag_count,
                                row,
                                coefficients,
                                coefficient,
                                current_position,
                                vertically_causal,
                            );
                        }
                    }
                } else if PASS_TYPE == 1 {
                    if window & (SIGMA_THIS | PI_THIS) == 0 && window & SIGMA_NEIGHBOURS != 0 {
                        let bit = if use_arithmetic {
                            arithmetic_decode_bit(
                                arithmetic,
                                contexts,
                                zero_context(zero_contexts, window),
                            )
                        } else {
                            let bit = bypass_read_bit(bypass);
                            if bit == RAW_READ_FAILED {
                                return fail(statuses, job_index, STATUS_FAILED, 8);
                            }
                            bit
                        };
                        if bit != 0 {
                            if use_arithmetic {
                                decode_sign_arithmetic(
                                    arithmetic,
                                    contexts,
                                    tables,
                                    flags,
                                    &mut own,
                                    flag_index,
                                    flag_stride,
                                    flag_count,
                                    row,
                                    coefficients,
                                    coefficient,
                                    current_position,
                                    vertically_causal,
                                );
                            } else if !decode_sign_bypass(
                                bypass,
                                flags,
                                &mut own,
                                flag_index,
                                flag_stride,
                                flag_count,
                                row,
                                coefficients,
                                coefficient,
                                current_position,
                                vertically_causal,
                            ) {
                                return fail(statuses, job_index, STATUS_FAILED, 9);
                            }
                        }
                        own |= PI_THIS << shift;
                    }
                }
                row += 1;
            }
            if PASS_TYPE == 0 {
                // Significance-propagation visits belong to exactly one
                // bitplane, even when an MQ segment boundary splits passes.
                own &= !PI_ALL;
            }
            if own != initial_own {
                simt_store(flags, flag_index as usize, own);
            }
            x += 1;
        }
        stripe += 1;
    }
    true
}

#[inline(always)]
fn decode_job<const COMMON_STYLE: bool>(
    job: ClassicJob,
    coded_data: *const u8,
    segments: *const ClassicSegment,
    tables: *const u8,
    mq_tables: *const u32,
    coefficients: *mut u32,
    contexts: *mut u32,
    flags: *mut u32,
    statuses: *mut ClassicStatus,
    job_index: u32,
) -> bool {
    let style_flags = if COMMON_STYLE { 0 } else { job.style_flags };
    let bitplanes = job.total_bitplanes + job.roi_shift - job.missing_msbs;
    if job.number_of_coding_passes == 0 {
        return true;
    }

    let coded_end = job.coded_offset as u64 + job.coded_len as u64;
    reset_contexts(contexts, mq_tables);
    let mut expected_pass = 0;
    let mut expected_offset = job.coded_offset as u64;

    let mut segment_index = 0;
    while segment_index < job.segment_count {
        let segment = simt_load(segments, (job.segment_offset + segment_index) as usize);
        if segment.start_coding_pass != expected_pass
            || segment.start_coding_pass > segment.end_coding_pass
            || segment.end_coding_pass > job.number_of_coding_passes
            || segment.data_offset as u64 != expected_offset
        {
            return fail(statuses, job_index, STATUS_UNSUPPORTED, 5);
        }
        let segment_end = segment.data_offset as u64 + segment.data_length as u64;
        if segment.data_offset < job.coded_offset || segment_end > coded_end {
            return fail(statuses, job_index, STATUS_UNSUPPORTED, 6);
        }
        expected_pass = segment.end_coding_pass;
        expected_offset = segment_end;
        if segment.start_coding_pass == segment.end_coding_pass {
            segment_index += 1;
            continue;
        }

        let segment_data = if segment.data_length == 0 {
            coded_data
        } else {
            simt_mut_ptr_at(coded_data.cast_mut(), segment.data_offset as usize).cast_const()
        };
        let mut arithmetic = ArithmeticDecoder {
            mq_tables,
            data: segment_data,
            data_len: segment.data_length,
            c: 0,
            a: 0,
            base_pointer: 0,
            byte: 0xff,
            shift_count: 0,
        };
        let mut bypass = BypassDecoder {
            data: segment_data,
            data_len: segment.data_length,
            bit_pos: 0,
            strict: job.strict != 0,
        };
        let use_arithmetic = COMMON_STYLE || segment.use_arithmetic != 0;
        if use_arithmetic {
            arithmetic_initialize(&mut arithmetic);
        }

        let mut coding_pass = segment.start_coding_pass;
        while coding_pass < segment.end_coding_pass {
            let current_bitplane = (coding_pass + 2) / 3;
            let current_position = bitplanes - 1 - current_bitplane;
            let pass_type = coding_pass % 3;
            if pass_type == 0 && !use_arithmetic {
                return fail(statuses, job_index, STATUS_UNSUPPORTED, 7);
            }

            let decoded = match pass_type {
                0 => decode_pass::<0, COMMON_STYLE>(
                    job,
                    &mut arithmetic,
                    &mut bypass,
                    use_arithmetic,
                    contexts,
                    tables,
                    flags,
                    coefficients,
                    current_position,
                    statuses,
                    job_index,
                ),
                1 => decode_pass::<1, COMMON_STYLE>(
                    job,
                    &mut arithmetic,
                    &mut bypass,
                    use_arithmetic,
                    contexts,
                    tables,
                    flags,
                    coefficients,
                    current_position,
                    statuses,
                    job_index,
                ),
                _ => decode_pass::<2, COMMON_STYLE>(
                    job,
                    &mut arithmetic,
                    &mut bypass,
                    use_arithmetic,
                    contexts,
                    tables,
                    flags,
                    coefficients,
                    current_position,
                    statuses,
                    job_index,
                ),
            };
            if !decoded {
                return false;
            }

            if pass_type == 0 {
                if style_flags & STYLE_SEGMENTATION_SYMBOLS != 0 {
                    let b0 = arithmetic_decode_bit(&mut arithmetic, contexts, 18);
                    let b1 = arithmetic_decode_bit(&mut arithmetic, contexts, 18);
                    let b2 = arithmetic_decode_bit(&mut arithmetic, contexts, 18);
                    let b3 = arithmetic_decode_bit(&mut arithmetic, contexts, 18);
                    let valid = b0 == 1 && b1 == 0 && b2 == 1 && b3 == 0;
                    if !valid && job.strict != 0 {
                        return fail(statuses, job_index, STATUS_FAILED, 11);
                    }
                }
            }
            if style_flags & STYLE_RESET_CONTEXTS != 0 {
                reset_contexts(contexts, mq_tables);
            }
            coding_pass += 1;
        }
        segment_index += 1;
    }

    if expected_pass != job.number_of_coding_passes || expected_offset != coded_end {
        return fail(statuses, job_index, STATUS_UNSUPPORTED, 12);
    }
    true
}

#[cuda_module]
mod kernels {
    use super::*;

    #[expect(
        static_mut_refs,
        reason = "CUDA block-shared state belongs exclusively to one launched codeblock"
    )]
    #[kernel]
    pub unsafe fn j2k_decode_classic_codeblocks_multi(
        coded_data: *const u8,
        jobs: *const ClassicJob,
        segments: *const ClassicSegment,
        tables: *const u8,
        statuses: *mut ClassicStatus,
        coefficient_scratch: *mut u32,
    ) {
        static mut MQ_TABLES: SharedArray<u32, 47> = SharedArray::UNINIT;
        static mut FLAGS: SharedArray<u32, MAX_FLAG_WORDS> = SharedArray::UNINIT;
        static mut CONTEXTS: SharedArray<u32, 19> = SharedArray::UNINIT;

        let lane = thread::threadIdx_x();
        let job = simt_load(jobs, thread::blockIdx_x() as usize);
        let job_index = job.status_index as u32;
        let mq_tables = unsafe { MQ_TABLES.as_mut_ptr() };
        let mut table_index = lane;
        while table_index < 47 {
            let qe = simt_load(
                tables.cast::<u32>(),
                MQ_QE_WORD_OFFSET + table_index as usize,
            );
            let transition = simt_load(
                tables.cast::<u32>(),
                MQ_TRANSITION_WORD_OFFSET + table_index as usize,
            );
            let packed = qe
                | ((transition & 0x3f) << 16)
                | (((transition >> 8) & 0x3f) << 22)
                | (((transition >> 16) & 1) << 28);
            simt_store(mq_tables, table_index as usize, packed);
            table_index += CLASSIC_DECODE_THREADS;
        }
        if lane == 0 {
            set_status(statuses, job_index, STATUS_OK, 0);
            validate_job_header(job, statuses, job_index);
        }
        thread::sync_threads();
        if simt_load(statuses.cast_const(), job_index as usize).code != STATUS_OK {
            return;
        }

        let padded_width = job.width + 2;
        let coefficient_count = padded_width * (job.height + 2);
        let coefficients = simt_mut_ptr_at(coefficient_scratch, job.scratch_offset as usize);
        let flags = unsafe { FLAGS.as_mut_ptr() };
        let contexts = unsafe { CONTEXTS.as_mut_ptr() };
        let flag_count = (job.width + 2) * ((job.height + 3) / 4);

        let mut index = lane;
        while index < coefficient_count {
            simt_store(coefficients, index as usize, 0);
            index += CLASSIC_DECODE_THREADS;
        }
        let mut flag = lane;
        while flag < flag_count {
            simt_store(flags, flag as usize, 0);
            flag += CLASSIC_DECODE_THREADS;
        }
        thread::sync_threads();

        if lane == 0 {
            // Keep optional coding-style branches out of the ordinary MQ
            // inner loop, while retaining the general path for all other jobs.
            let common_style = job.style_flags == 0
                && job.segment_count == 1
                && simt_load(segments, job.segment_offset as usize).use_arithmetic != 0;
            let decoded = if common_style {
                decode_job::<true>(
                    job,
                    coded_data,
                    segments,
                    tables,
                    mq_tables,
                    coefficients,
                    contexts,
                    flags,
                    statuses,
                    job_index,
                )
            } else {
                decode_job::<false>(
                    job,
                    coded_data,
                    segments,
                    tables,
                    mq_tables,
                    coefficients,
                    contexts,
                    flags,
                    statuses,
                    job_index,
                )
            };
            if !decoded && simt_load(statuses.cast_const(), job_index as usize).code == STATUS_OK {
                set_status(statuses, job_index, STATUS_FAILED, 0);
            }
        }
        thread::sync_threads();

        if simt_load(statuses.cast_const(), job_index as usize).code != STATUS_OK {
            return;
        }
        let output = job.output_ptr as usize as *mut f32;
        let sample_count = job.width * job.height;
        let mut sample = lane;
        while sample < sample_count {
            let x = sample % job.width;
            let y = sample / job.width;
            let packed = simt_load(
                coefficients.cast_const(),
                coefficient_index(padded_width, x + 1, y + 1) as usize,
            );
            simt_store(
                output,
                job.output_offset as usize + y as usize * job.output_stride as usize + x as usize,
                reconstructed_classic_sample(packed, job) * job.dequantization_step,
            );
            sample += CLASSIC_DECODE_THREADS;
        }
    }
}

fn main() {}
