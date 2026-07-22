#![allow(
    static_mut_refs,
    reason = "CUDA shared-memory state is scoped to one classic Tier-1 codeblock"
)]

use cuda_device::{kernel, thread, SharedArray};
use cuda_host::cuda_module;
include!("../../../cuda_oxide_simt_prelude.rs");

const MAX_PADDED_COEFFICIENTS: usize = 66 * 66;
const CLASSIC_DECODE_THREADS: u32 = 32;
const SIGNIFICANT_SHIFT: u8 = 7;
const MAGNITUDE_REFINED_SHIFT: u8 = 6;
const SIGN_SHIFT: u8 = 5;
const ZERO_CODED_MASK: u8 = 0x1f;
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
const ZERO_CONTEXT_HL_BYTE_OFFSET: usize = 1144;
const ZERO_CONTEXT_HH_BYTE_OFFSET: usize = 1400;

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
    dequantization_step: f32,
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
    data: *const u8,
    data_len: u32,
    c: u32,
    a: u32,
    base_pointer: u32,
    shift_count: u32,
}

struct BypassDecoder {
    data: *const u8,
    data_len: u32,
    bit_pos: u32,
    strict: bool,
}

#[inline(always)]
fn state_bit(states: *const u8, index: u32, shift: u8) -> u8 {
    (simt_load(states, index as usize) >> shift) & 1
}

#[inline(always)]
fn set_state_bit(states: *mut u8, index: u32, shift: u8, value: u8) {
    let state = simt_load(states.cast_const(), index as usize);
    simt_store(
        states,
        index as usize,
        (state & !(1 << shift)) | ((value & 1) << shift),
    );
}

#[inline(always)]
fn coefficient_index(padded_width: u32, x: u32, y: u32) -> u32 {
    x + y * padded_width
}

#[inline(always)]
fn is_significant(states: *const u8, index: u32) -> bool {
    state_bit(states, index, SIGNIFICANT_SHIFT) != 0
}

#[inline(always)]
fn sign(states: *const u8, index: u32) -> u32 {
    state_bit(states, index, SIGN_SHIFT) as u32
}

#[inline(always)]
fn zero_coded(states: *const u8, index: u32, marker: u8) -> bool {
    marker != 0 && (simt_load(states, index as usize) & ZERO_CODED_MASK) == marker
}

#[inline(always)]
fn set_zero_coded(states: *mut u8, index: u32, marker: u8) {
    let state = simt_load(states.cast_const(), index as usize);
    simt_store(
        states,
        index as usize,
        (state & !ZERO_CODED_MASK) | (marker & ZERO_CODED_MASK),
    );
}

#[inline(always)]
fn set_significant(states: *mut u8, index: u32) {
    set_state_bit(states, index, SIGNIFICANT_SHIFT, 1);
}

#[inline(always)]
fn is_magnitude_refined(states: *const u8, index: u32) -> bool {
    state_bit(states, index, MAGNITUDE_REFINED_SHIFT) != 0
}

#[inline(always)]
fn set_magnitude_refined(states: *mut u8, index: u32) {
    set_state_bit(states, index, MAGNITUDE_REFINED_SHIFT, 1);
}

#[inline(always)]
fn push_coefficient_bit(coefficients: *mut u32, index: u32, position: u32) {
    let value = simt_load(coefficients.cast_const(), index as usize);
    simt_store(coefficients, index as usize, value | (1 << position));
}

#[inline(always)]
fn set_coefficient_sign(coefficients: *mut u32, index: u32, negative: u32) {
    let value = simt_load(coefficients.cast_const(), index as usize) & 0x7fff_ffff;
    simt_store(
        coefficients,
        index as usize,
        if negative != 0 {
            value | 0x8000_0000
        } else {
            value
        },
    );
}

#[inline(always)]
fn neighbor_in_next_stripe(y: u32, height: u32) -> bool {
    let real_y = y - 1;
    real_y + 1 < height && ((real_y + 1) >> 2) > (real_y >> 2)
}

#[inline(always)]
fn neighborhood(states: *const u8, padded_width: u32, x: u32, y: u32) -> u8 {
    (is_significant(states, coefficient_index(padded_width, x, y + 1)) as u8)
        | ((is_significant(states, coefficient_index(padded_width, x + 1, y + 1)) as u8)
            << 1)
        | ((is_significant(states, coefficient_index(padded_width, x + 1, y)) as u8) << 2)
        | ((is_significant(states, coefficient_index(padded_width, x - 1, y + 1)) as u8)
            << 3)
        | ((is_significant(states, coefficient_index(padded_width, x - 1, y)) as u8) << 4)
        | ((is_significant(states, coefficient_index(padded_width, x + 1, y - 1)) as u8)
            << 5)
        | ((is_significant(states, coefficient_index(padded_width, x, y - 1)) as u8) << 6)
        | ((is_significant(states, coefficient_index(padded_width, x - 1, y - 1)) as u8)
            << 7)
}

#[inline(always)]
fn effective_neighborhood(
    states: *const u8,
    padded_width: u32,
    x: u32,
    y: u32,
    height: u32,
    style_flags: u32,
) -> u8 {
    let mut result = neighborhood(states, padded_width, x, y);
    if style_flags & STYLE_VERTICALLY_CAUSAL != 0 && neighbor_in_next_stripe(y, height) {
        result &= 0b1111_0100;
    }
    result
}

#[inline(always)]
fn zero_context(tables: *const u8, neighbors: u8, sub_band_type: u32) -> u8 {
    let offset = match sub_band_type {
        1 => ZERO_CONTEXT_HL_BYTE_OFFSET,
        3 => ZERO_CONTEXT_HH_BYTE_OFFSET,
        _ => ZERO_CONTEXT_LL_LH_BYTE_OFFSET,
    };
    simt_load(tables, offset + neighbors as usize)
}

#[inline(always)]
fn magnitude_context(
    states: *const u8,
    padded_width: u32,
    x: u32,
    y: u32,
    height: u32,
    style_flags: u32,
) -> u8 {
    let index = coefficient_index(padded_width, x, y);
    if is_magnitude_refined(states, index) {
        16
    } else {
        14 + effective_neighborhood(states, padded_width, x, y, height, style_flags).min(1)
    }
}

#[inline(always)]
fn sign_context(
    tables: *const u8,
    states: *const u8,
    padded_width: u32,
    x: u32,
    y: u32,
    height: u32,
    style_flags: u32,
) -> (u8, u8) {
    let significances =
        effective_neighborhood(states, padded_width, x, y, height, style_flags) & 0b0101_0101;
    let top = sign(states, coefficient_index(padded_width, x, y - 1));
    let left = sign(states, coefficient_index(padded_width, x - 1, y));
    let right = sign(states, coefficient_index(padded_width, x + 1, y));
    let bottom = if style_flags & STYLE_VERTICALLY_CAUSAL != 0
        && neighbor_in_next_stripe(y, height)
    {
        0
    } else {
        sign(states, coefficient_index(padded_width, x, y + 1))
    };
    let signs = ((top << 6) | (left << 4) | (right << 2) | bottom) as u8;
    let negative = significances & signs;
    let positive = significances & !signs;
    let packed = simt_load(
        tables.cast::<u16>(),
        SIGN_CONTEXT_HALFWORD_OFFSET + ((negative << 1) | positive) as usize,
    );
    (packed as u8, (packed >> 8) as u8)
}

#[inline(always)]
fn reset_contexts(contexts: *mut u8) {
    let mut index = 0;
    while index < 19 {
        simt_store(contexts, index, 0);
        index += 1;
    }
    simt_store(contexts, 0, 4);
    simt_store(contexts, 17, 3);
    simt_store(contexts, 18, 46);
}

#[inline(always)]
fn current_byte(decoder: &ArithmeticDecoder) -> u8 {
    if decoder.base_pointer < decoder.data_len {
        simt_load(decoder.data, decoder.base_pointer as usize)
    } else {
        0xff
    }
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
    if current_byte(decoder) == 0xff {
        let next = next_byte(decoder);
        if next > 0x8f {
            decoder.shift_count = 8;
        } else {
            decoder.base_pointer += 1;
            decoder.c = decoder
                .c
                .wrapping_add(0xfe00)
                .wrapping_sub((current_byte(decoder) as u32) << 9);
            decoder.shift_count = 7;
        }
    } else {
        decoder.base_pointer += 1;
        decoder.c = decoder
            .c
            .wrapping_add(0xff00)
            .wrapping_sub((current_byte(decoder) as u32) << 8);
        decoder.shift_count = 8;
    }
}

#[inline(always)]
fn arithmetic_initialize(decoder: &mut ArithmeticDecoder) {
    decoder.c = ((current_byte(decoder) ^ 0xff) as u32) << 16;
    arithmetic_read_byte(decoder);
    decoder.c <<= 7;
    decoder.shift_count -= 7;
    decoder.a = 0x8000;
}

#[inline(always)]
fn arithmetic_renormalize(decoder: &mut ArithmeticDecoder) {
    while decoder.a & 0x8000 == 0 {
        if decoder.shift_count == 0 {
            arithmetic_read_byte(decoder);
        }
        decoder.a <<= 1;
        decoder.c <<= 1;
        decoder.shift_count -= 1;
    }
}

#[inline(always)]
fn arithmetic_decode_bit(
    decoder: &mut ArithmeticDecoder,
    contexts: *mut u8,
    label: u8,
    tables: *const u8,
) -> u32 {
    let mut context = simt_load(contexts.cast_const(), label as usize);
    let state = (context & 0x7f) as usize;
    let qe = simt_load(tables.cast::<u32>(), MQ_QE_WORD_OFFSET + state);
    let transition = simt_load(
        tables.cast::<u32>(),
        MQ_TRANSITION_WORD_OFFSET + state,
    );
    let nmps = transition as u8;
    let nlps = (transition >> 8) as u8;
    let switch = transition & (1 << 16) != 0;
    decoder.a -= qe;

    let decoded;
    if decoder.c >> 16 < decoder.a {
        if decoder.a & 0x8000 != 0 {
            return (context >> 7) as u32;
        }
        if decoder.a < qe {
            decoded = ((context >> 7) ^ 1) as u32;
            if switch {
                context ^= 0x80;
            }
            context = (context & 0x80) | nlps;
        } else {
            decoded = (context >> 7) as u32;
            context = (context & 0x80) | nmps;
        }
    } else {
        decoder.c -= decoder.a << 16;
        if decoder.a < qe {
            decoder.a = qe;
            decoded = (context >> 7) as u32;
            context = (context & 0x80) | nmps;
        } else {
            decoder.a = qe;
            decoded = ((context >> 7) ^ 1) as u32;
            if switch {
                context ^= 0x80;
            }
            context = (context & 0x80) | nlps;
        }
    }
    simt_store(contexts, label as usize, context);
    arithmetic_renormalize(decoder);
    decoded
}

#[inline(always)]
fn raw_read_bit(decoder: &mut BypassDecoder) -> u32 {
    let byte_position = decoder.bit_pos / 8;
    if byte_position >= decoder.data_len {
        if decoder.strict {
            return RAW_READ_FAILED;
        }
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
    contexts: *mut u8,
    tables: *const u8,
    states: *mut u8,
    coefficients: *mut u32,
    padded_width: u32,
    x: u32,
    y: u32,
    height: u32,
    style_flags: u32,
) {
    let (label, xor) = sign_context(
        tables,
        states.cast_const(),
        padded_width,
        x,
        y,
        height,
        style_flags,
    );
    let negative = arithmetic_decode_bit(decoder, contexts, label, tables) ^ xor as u32;
    let index = coefficient_index(padded_width, x, y);
    set_state_bit(states, index, SIGN_SHIFT, negative as u8);
    set_coefficient_sign(coefficients, index, negative);
    set_significant(states, index);
}

#[inline(always)]
fn decode_sign_bypass(
    decoder: &mut BypassDecoder,
    states: *mut u8,
    coefficients: *mut u32,
    padded_width: u32,
    x: u32,
    y: u32,
) -> bool {
    let negative = bypass_read_bit(decoder);
    if negative == RAW_READ_FAILED {
        return false;
    }
    let index = coefficient_index(padded_width, x, y);
    set_state_bit(states, index, SIGN_SHIFT, negative as u8);
    set_coefficient_sign(coefficients, index, negative);
    set_significant(states, index);
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
fn validate_job_header(
    job: ClassicJob,
    statuses: *mut ClassicStatus,
    job_index: u32,
) -> bool {
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
        || job.missing_msbs >= job.total_bitplanes
        || job.sub_band_type > 3
        || job.style_flags & !KNOWN_STYLE_FLAGS != 0
    {
        return fail(statuses, job_index, STATUS_UNSUPPORTED, 2);
    }
    let bitplanes = job.total_bitplanes - job.missing_msbs;
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
fn decode_job(
    job: ClassicJob,
    coded_data: *const u8,
    segments: *const ClassicSegment,
    tables: *const u8,
    coefficients: *mut u32,
    contexts: *mut u8,
    states: *mut u8,
    statuses: *mut ClassicStatus,
    job_index: u32,
) -> bool {
    let bitplanes = job.total_bitplanes - job.missing_msbs;
    if job.number_of_coding_passes == 0 {
        return true;
    }

    let padded_width = job.width + 2;
    let coded_end = job.coded_offset as u64 + job.coded_len as u64;
    reset_contexts(contexts);
    let mut expected_pass = 0;
    let mut expected_offset = job.coded_offset as u64;

    let mut segment_index = 0;
    while segment_index < job.segment_count {
        let segment = simt_load(
            segments,
            (job.segment_offset + segment_index) as usize,
        );
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
            data: segment_data,
            data_len: segment.data_length,
            c: 0,
            a: 0,
            base_pointer: 0,
            shift_count: 0,
        };
        let mut bypass = BypassDecoder {
            data: segment_data,
            data_len: segment.data_length,
            bit_pos: 0,
            strict: job.strict != 0,
        };
        let use_arithmetic = segment.use_arithmetic != 0;
        if use_arithmetic {
            arithmetic_initialize(&mut arithmetic);
        }

        let mut zero_epoch = ((segment.start_coding_pass + 2) / 3) as u8;
        let mut coding_pass = segment.start_coding_pass;
        while coding_pass < segment.end_coding_pass {
            let current_bitplane = (coding_pass + 2) / 3;
            let current_position = bitplanes - 1 - current_bitplane;
            let pass_type = coding_pass % 3;
            if pass_type == 0 && !use_arithmetic {
                return fail(statuses, job_index, STATUS_UNSUPPORTED, 7);
            }

            let mut base_row = 0;
            while base_row < job.height {
                let stripe_end = (base_row + 4).min(job.height);
                let mut x = 0;
                while x < job.width {
                    let index_x = x + 1;
                    let mut index_y = base_row + 1;
                    while index_y < stripe_end + 1 {
                        let index = coefficient_index(padded_width, index_x, index_y);
                        if pass_type == 0 {
                            if !is_significant(states.cast_const(), index)
                                && !zero_coded(states.cast_const(), index, zero_epoch)
                            {
                                let run_length = (index_y - 1) % 4 == 0
                                    && job.height - (index_y - 1) >= 4
                                    && effective_neighborhood(
                                        states.cast_const(),
                                        padded_width,
                                        index_x,
                                        index_y,
                                        job.height,
                                        job.style_flags,
                                    ) == 0
                                    && effective_neighborhood(
                                        states.cast_const(),
                                        padded_width,
                                        index_x,
                                        index_y + 1,
                                        job.height,
                                        job.style_flags,
                                    ) == 0
                                    && effective_neighborhood(
                                        states.cast_const(),
                                        padded_width,
                                        index_x,
                                        index_y + 2,
                                        job.height,
                                        job.style_flags,
                                    ) == 0
                                    && effective_neighborhood(
                                        states.cast_const(),
                                        padded_width,
                                        index_x,
                                        index_y + 3,
                                        job.height,
                                        job.style_flags,
                                    ) == 0;
                                let bit;
                                if run_length {
                                    bit = arithmetic_decode_bit(&mut arithmetic, contexts, 17, tables);
                                    if bit == 0 {
                                        index_y += 4;
                                        continue;
                                    }
                                    let first =
                                        arithmetic_decode_bit(&mut arithmetic, contexts, 18, tables);
                                    let second =
                                        arithmetic_decode_bit(&mut arithmetic, contexts, 18, tables);
                                    index_y += (first << 1) | second;
                                } else {
                                    let label = zero_context(
                                        tables,
                                        effective_neighborhood(
                                            states.cast_const(),
                                            padded_width,
                                            index_x,
                                            index_y,
                                            job.height,
                                            job.style_flags,
                                        ),
                                        job.sub_band_type,
                                    );
                                    bit =
                                        arithmetic_decode_bit(&mut arithmetic, contexts, label, tables);
                                }
                                if bit != 0 {
                                    let actual =
                                        coefficient_index(padded_width, index_x, index_y);
                                    push_coefficient_bit(coefficients, actual, current_position);
                                    decode_sign_arithmetic(
                                        &mut arithmetic,
                                        contexts,
                                        tables,
                                        states,
                                        coefficients,
                                        padded_width,
                                        index_x,
                                        index_y,
                                        job.height,
                                        job.style_flags,
                                    );
                                }
                            }
                        } else if pass_type == 1 {
                            let neighbors = effective_neighborhood(
                                states.cast_const(),
                                padded_width,
                                index_x,
                                index_y,
                                job.height,
                                job.style_flags,
                            );
                            if !is_significant(states.cast_const(), index) && neighbors != 0 {
                                let label = zero_context(tables, neighbors, job.sub_band_type);
                                let bit = if use_arithmetic {
                                    arithmetic_decode_bit(&mut arithmetic, contexts, label, tables)
                                } else {
                                    let bit = bypass_read_bit(&mut bypass);
                                    if bit == RAW_READ_FAILED {
                                        return fail(statuses, job_index, STATUS_FAILED, 8);
                                    }
                                    bit
                                };
                                set_zero_coded(states, index, zero_epoch);
                                if bit != 0 {
                                    push_coefficient_bit(coefficients, index, current_position);
                                    if use_arithmetic {
                                        decode_sign_arithmetic(
                                            &mut arithmetic,
                                            contexts,
                                            tables,
                                            states,
                                            coefficients,
                                            padded_width,
                                            index_x,
                                            index_y,
                                            job.height,
                                            job.style_flags,
                                        );
                                    } else if !decode_sign_bypass(
                                        &mut bypass,
                                        states,
                                        coefficients,
                                        padded_width,
                                        index_x,
                                        index_y,
                                    ) {
                                        return fail(statuses, job_index, STATUS_FAILED, 9);
                                    }
                                }
                            }
                        } else if is_significant(states.cast_const(), index)
                            && !zero_coded(states.cast_const(), index, zero_epoch)
                        {
                            let label = magnitude_context(
                                states.cast_const(),
                                padded_width,
                                index_x,
                                index_y,
                                job.height,
                                job.style_flags,
                            );
                            let bit = if use_arithmetic {
                                arithmetic_decode_bit(&mut arithmetic, contexts, label, tables)
                            } else {
                                let bit = bypass_read_bit(&mut bypass);
                                if bit == RAW_READ_FAILED {
                                    return fail(statuses, job_index, STATUS_FAILED, 10);
                                }
                                bit
                            };
                            if bit != 0 {
                                push_coefficient_bit(coefficients, index, current_position);
                            }
                            set_magnitude_refined(states, index);
                        }
                        index_y += 1;
                    }
                    x += 1;
                }
                base_row += 4;
            }

            if pass_type == 0 {
                if job.style_flags & STYLE_SEGMENTATION_SYMBOLS != 0 {
                    let b0 = arithmetic_decode_bit(&mut arithmetic, contexts, 18, tables);
                    let b1 = arithmetic_decode_bit(&mut arithmetic, contexts, 18, tables);
                    let b2 = arithmetic_decode_bit(&mut arithmetic, contexts, 18, tables);
                    let b3 = arithmetic_decode_bit(&mut arithmetic, contexts, 18, tables);
                    let valid = b0 == 1 && b1 == 0 && b2 == 1 && b3 == 0;
                    if !valid && job.strict != 0 {
                        return fail(statuses, job_index, STATUS_FAILED, 11);
                    }
                }
                zero_epoch = zero_epoch.saturating_add(1).min(ZERO_CODED_MASK);
            }
            if job.style_flags & STYLE_RESET_CONTEXTS != 0 {
                reset_contexts(contexts);
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
        static mut STATES: SharedArray<u8, MAX_PADDED_COEFFICIENTS> = SharedArray::UNINIT;
        static mut CONTEXTS: SharedArray<u8, 19> = SharedArray::UNINIT;

        let job_index = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        let job = simt_load(jobs, job_index as usize);
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
        let coefficients =
            simt_mut_ptr_at(coefficient_scratch, job.scratch_offset as usize);
        let states = unsafe { STATES.as_mut_ptr() };
        let contexts = unsafe { CONTEXTS.as_mut_ptr() };

        let mut index = lane;
        while index < coefficient_count {
            simt_store(coefficients, index as usize, 0);
            simt_store(states, index as usize, 0);
            index += CLASSIC_DECODE_THREADS;
        }
        thread::sync_threads();

        if lane == 0
            && !decode_job(
                job,
                coded_data,
                segments,
                tables,
                coefficients,
                contexts,
                states,
                statuses,
                job_index,
            )
            && simt_load(statuses.cast_const(), job_index as usize).code == STATUS_OK
        {
            set_status(statuses, job_index, STATUS_FAILED, 0);
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
            let magnitude = (packed & 0x7fff_ffff) as i32;
            let signed = if packed & 0x8000_0000 != 0 {
                -magnitude
            } else {
                magnitude
            };
            simt_store(
                output,
                job.output_offset as usize
                    + y as usize * job.output_stride as usize
                    + x as usize,
                signed as f32 * job.dequantization_step,
            );
            sample += CLASSIC_DECODE_THREADS;
        }
    }
}

fn main() {}
