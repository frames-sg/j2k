#![allow(
    static_mut_refs,
    reason = "CUDA shared-memory statics are accessed through device-scoped references"
)]

use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32};
use cuda_device::{kernel, thread, warp, SharedArray};
use cuda_host::cuda_module;
use j2k_codec_math::htj2k::HT_MAX_SCRATCH;

include!("../../../cuda_oxide_simt_prelude.rs");

const HT_STATUS_OK: u32 = 0;
const HT_STATUS_FAIL: u32 = 1;
const HT_STATUS_UNSUPPORTED: u32 = 2;
const HT_ROI_SHIFT_MASK: u32 = 0xff;
const HT_IRREVERSIBLE_MIDPOINT_FLAG: u32 = 1 << 8;

const HT_MAX_WIDTH: u32 = 256;
const HT_MAX_HEIGHT: u32 = 256;
const HT_MAX_COEFFICIENTS: u32 = 4096;
const HT_MAX_SSTR: u32 = 264;
const HT_MAX_VN: usize = 130;
const HT_MAX_MSTR: u32 = 72;
const HT_MAX_SIGMA: usize = 528;
const HT_MAX_PREV_ROW_SIG: usize = 72;

const SIGPROP_SPREAD_MASKS: [u32; 16] = [
    0x33, 0x76, 0xEC, 0xC8, 0x330, 0x760, 0xEC0, 0xC80, 0x3300, 0x7600, 0xEC00, 0xC800, 0x33000,
    0x76000, 0xEC000, 0xC8000,
];

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtCleanupParams {
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    reconstruction: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtCleanupBatchJob {
    coded_offset: u32,
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    reconstruction: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtCleanupMultiBatchJob {
    output_ptr: u64,
    coded_offset: u32,
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
    reconstruction: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kCoefficientClearTarget {
    output_ptr: u64,
    words: u64,
}

const COEFFICIENT_CLEAR_TARGETS_PER_BATCH: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kCoefficientClearBatch {
    targets: [J2kCoefficientClearTarget; COEFFICIENT_CLEAR_TARGETS_PER_BATCH],
    target_count: u32,
    blocks_per_target: u32,
}

#[derive(Clone, Copy)]
struct MelDecoder {
    data: *const u8,
    pos: u32,
    remaining: u32,
    unstuff: bool,
    current_byte: u8,
    bits_left: u8,
    k: u32,
    num_runs: u32,
    runs: u64,
}

#[derive(Clone, Copy)]
struct ForwardBitReader {
    data: *const u8,
    data_len: u32,
    pos: u32,
    tmp: u64,
    bits: u32,
    unstuff: bool,
    pad: u8,
}

#[derive(Clone, Copy)]
struct ReverseBitReader {
    data: *const u8,
    pos: i32,
    remaining: u32,
    tmp: u64,
    bits: u32,
    unstuff: bool,
}

#[inline(always)]
fn min_u32(a: u32, b: u32) -> u32 {
    if a < b {
        a
    } else {
        b
    }
}

#[inline(always)]
fn load_u8(ptr: *const u8, index: u32) -> u8 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn load_u16(ptr: *const u16, index: u32) -> u16 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn load_job<T: Copy>(ptr: *const T, index: u32) -> T {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn store_status(status: *mut J2kHtStatus, code: u32, detail: u32) {
    unsafe {
        (*status).code = code;
        (*status).detail = detail;
        (*status).reserved0 = 0;
        (*status).reserved1 = 0;
    }
}

#[inline(always)]
fn popcount32(value: u32) -> u32 {
    value.count_ones()
}

#[inline(always)]
fn trailing_zeros32(value: u32) -> u32 {
    value.trailing_zeros()
}

#[inline(always)]
fn floor_log2_nonzero(value: u32) -> u32 {
    31 - value.max(1).leading_zeros()
}

#[inline(always)]
fn read_u32_pair(values: &[u16], index: u32) -> u32 {
    values[index as usize] as u32 | ((values[index as usize + 1] as u32) << 16)
}

#[inline(always)]
fn sample_mask(bit: u32) -> u32 {
    1 << (4 + bit)
}

#[inline(always)]
fn coefficient_to_i32(value: u32, k_max: u32) -> i32 {
    let shift = 31 - k_max;
    let magnitude = ((value & 0x7fff_ffff) >> shift) as i32;
    if (value & 0x8000_0000) != 0 {
        -magnitude
    } else {
        magnitude
    }
}

#[inline(always)]
fn coefficient_to_float_bits(value: u32, k_max: u32, scale: f32, reconstruction: u32) -> u32 {
    let roi_shift = reconstruction & HT_ROI_SHIFT_MASK;
    let irreversible_midpoint = reconstruction & HT_IRREVERSIBLE_MIDPOINT_FLAG != 0;
    if !irreversible_midpoint {
        let coefficient = coefficient_to_i32(value, k_max);
        let magnitude = if coefficient < 0 {
            (-coefficient) as u32
        } else {
            coefficient as u32
        };
        let shifted = if roi_shift != 0 && magnitude >= 1 << roi_shift {
            (magnitude >> roi_shift) as i32
        } else {
            magnitude as i32
        };
        return ((if coefficient < 0 { -shifted } else { shifted }) as f32 * scale).to_bits();
    }

    let fixed_scale = f32::from_bits((k_max + 96) << 23);
    let magnitude = (value & 0x7fff_ffff) as f32 * fixed_scale;
    let roi_scale = f32::from_bits((127 - roi_shift) << 23);
    let roi_threshold = f32::from_bits((127 + roi_shift) << 23);
    let reconstructed = if roi_shift != 0 && magnitude >= roi_threshold {
        magnitude * roi_scale
    } else {
        magnitude
    };
    let coefficient = if value & 0x8000_0000 != 0 {
        -reconstructed
    } else {
        reconstructed
    };
    (coefficient * scale).to_bits()
}

#[inline(always)]
fn decoded_cleanup_sample_bits(value: u32, params: J2kHtCleanupParams, dequantize: bool) -> u32 {
    if dequantize {
        coefficient_to_float_bits(
            value,
            params.num_bitplanes + (params.reconstruction & HT_ROI_SHIFT_MASK),
            params.dequantization_step,
            params.reconstruction,
        )
    } else {
        value
    }
}

#[inline(always)]
fn store_decoded_sample(
    decoded_data: *mut u32,
    index: u32,
    value: u32,
    params: J2kHtCleanupParams,
    dequantize: bool,
) {
    simt_store(
        decoded_data,
        index as usize,
        decoded_cleanup_sample_bits(value, params, dequantize),
    );
}

#[inline(always)]
fn xor_decoded_sample(decoded_data: *mut u32, index: u32, value: u32) {
    unsafe {
        let ptr = simt_mut_ptr_at(decoded_data, index as usize);
        *ptr ^= value;
    }
}

#[inline(always)]
fn mel_decoder_new(data: *const u8, lcup: u32, scup: u32) -> MelDecoder {
    MelDecoder {
        data,
        pos: lcup - scup,
        remaining: scup - 1,
        unstuff: false,
        current_byte: 0,
        bits_left: 0,
        k: 0,
        num_runs: 0,
        runs: 0,
    }
}

#[inline(always)]
fn mel_read_bit(decoder: &mut MelDecoder, bit: &mut u32) -> bool {
    if decoder.bits_left == 0 {
        let mut byte = if decoder.remaining > 0 {
            let byte = load_u8(decoder.data, decoder.pos);
            decoder.pos += 1;
            decoder.remaining -= 1;
            byte
        } else {
            0xff
        };
        if decoder.remaining == 0 {
            byte |= 0x0f;
        }
        decoder.current_byte = byte;
        decoder.bits_left = 8 - decoder.unstuff as u8;
        decoder.unstuff = byte == 0xff;
    }

    decoder.bits_left -= 1;
    *bit = ((decoder.current_byte >> decoder.bits_left) & 1) as u32;
    true
}

#[inline(always)]
fn mel_read_bits(decoder: &mut MelDecoder, count: u32, value: &mut u32) -> bool {
    *value = 0;
    let mut idx = 0;
    while idx < count {
        let mut bit = 0;
        if !mel_read_bit(decoder, &mut bit) {
            return false;
        }
        *value = (*value << 1) | bit;
        idx += 1;
    }
    true
}

#[inline(always)]
fn mel_decode_more_runs(decoder: &mut MelDecoder) -> bool {
    const MEL_EXP: [u32; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];
    while decoder.num_runs < 8 {
        let eval = MEL_EXP[decoder.k as usize];
        let mut first = 0;
        if !mel_read_bit(decoder, &mut first) {
            return false;
        }
        let run = if first == 1 {
            decoder.k = min_u32(decoder.k + 1, 12);
            ((1 << eval) - 1) << 1
        } else {
            if decoder.k != 0 {
                decoder.k -= 1;
            }
            let mut bits = 0;
            if !mel_read_bits(decoder, eval, &mut bits) {
                return false;
            }
            (bits << 1) | 1
        };
        decoder.runs |= (run as u64) << (decoder.num_runs * 7);
        decoder.num_runs += 1;
        if eval == 5 && first == 0 && decoder.num_runs >= 8 {
            break;
        }
    }
    true
}

#[inline(always)]
fn mel_get_run(decoder: &mut MelDecoder, run: &mut i32) -> bool {
    if decoder.num_runs == 0 && !mel_decode_more_runs(decoder) {
        return false;
    }
    *run = (decoder.runs & 0x7f) as i32;
    decoder.runs >>= 7;
    decoder.num_runs -= 1;
    true
}

#[inline(always)]
fn forward_reader_new(data: *const u8, data_len: u32, pad: u8) -> ForwardBitReader {
    ForwardBitReader {
        data,
        data_len,
        pos: 0,
        tmp: 0,
        bits: 0,
        unstuff: false,
        pad,
    }
}

#[inline(always)]
fn forward_reader_fill(reader: &mut ForwardBitReader) {
    while reader.bits <= 32 {
        let byte = if reader.pos < reader.data_len {
            let byte = load_u8(reader.data, reader.pos);
            reader.pos += 1;
            byte
        } else {
            reader.pad
        };
        let valid_bits = 8 - reader.unstuff as u32;
        let next_unstuff = byte == 0xff;
        let byte = if reader.unstuff { byte & 0x7f } else { byte };
        reader.tmp |= (byte as u64) << reader.bits;
        reader.bits += valid_bits;
        reader.unstuff = next_unstuff;
    }
}

#[inline(always)]
fn forward_reader_fetch(reader: &mut ForwardBitReader) -> u32 {
    if reader.bits < 32 {
        forward_reader_fill(reader);
    }
    reader.tmp as u32
}

#[inline(always)]
fn forward_reader_advance(reader: &mut ForwardBitReader, count: u32) {
    reader.tmp >>= count;
    reader.bits -= count;
}

#[inline(always)]
fn reverse_reader_new_vlc(data: *const u8, lcup: u32, scup: u32) -> ReverseBitReader {
    let d = load_u8(data, lcup - 2);
    let tmp = (d >> 4) as u64;
    ReverseBitReader {
        data,
        pos: lcup as i32 - 3,
        remaining: scup - 2,
        tmp,
        bits: 4 - ((tmp & 0x7) == 0x7) as u32,
        unstuff: (d | 0x0f) > 0x8f,
    }
}

#[inline(always)]
fn reverse_reader_new_mrp(data: *const u8, lcup: u32, len2: u32) -> ReverseBitReader {
    ReverseBitReader {
        data,
        pos: (lcup + len2) as i32 - 1,
        remaining: len2,
        tmp: 0,
        bits: 0,
        unstuff: true,
    }
}

#[inline(always)]
fn reverse_reader_fill(reader: &mut ReverseBitReader) {
    while reader.bits <= 32 {
        let byte = if reader.remaining > 0 {
            let byte = load_u8(reader.data, reader.pos as u32);
            reader.pos -= 1;
            reader.remaining -= 1;
            byte
        } else {
            0
        };
        let stuffed = reader.unstuff && (byte & 0x7f) == 0x7f;
        let d_bits = 8 - stuffed as u32;
        let next_unstuff = byte > 0x8f;
        let byte = if stuffed { byte & 0x7f } else { byte };
        reader.tmp |= (byte as u64) << reader.bits;
        reader.bits += d_bits;
        reader.unstuff = next_unstuff;
    }
}

#[inline(always)]
fn reverse_reader_fetch(reader: &mut ReverseBitReader) -> u32 {
    if reader.bits < 32 {
        reverse_reader_fill(reader);
    }
    reader.tmp as u32
}

#[inline(always)]
fn reverse_reader_advance(reader: &mut ReverseBitReader, count: u32) -> u32 {
    reader.tmp >>= count;
    reader.bits -= count;
    reader.tmp as u32
}

#[inline(always)]
fn decode_mag_sgn_sample_with_vn(
    magsgn: &mut ForwardBitReader,
    inf: u32,
    bit: u32,
    uq: u32,
    p: u32,
    value: &mut u32,
    v_n: &mut u32,
) {
    if (inf & sample_mask(bit)) == 0 {
        *value = 0;
        *v_n = 0;
        return;
    }

    let ms_val = forward_reader_fetch(magsgn);
    let m_n = uq - ((inf >> (12 + bit)) & 1);
    forward_reader_advance(magsgn, m_n);
    (*value, *v_n) = mag_sgn_value(ms_val, m_n, inf, bit, p);
}

/// Decoded sample bits and `v_n` for one significant sample from its `m_n`
/// MagSgn bits `ms`. Both cleanup routes use this so they stay bit-exact.
#[inline(always)]
fn mag_sgn_value(ms: u32, m_n: u32, inf: u32, bit: u32, p: u32) -> (u32, u32) {
    let mask = if m_n == 0 { 0 } else { (1 << m_n) - 1 };
    let v_n = (ms & mask) | (((inf >> (8 + bit)) & 1) << m_n) | 1;
    ((ms << 31) | ((v_n + 2) << (p - 1)), v_n)
}

#[inline(always)]
fn decode_cleanup_symbols_first_row(
    mel: &mut MelDecoder,
    vlc: &mut ReverseBitReader,
    run: &mut i32,
    scratch: &mut [u16],
    width: u32,
    vlc_table0: *const u16,
    uvlc_table0: *const u16,
) -> u32 {
    let mut c_q = 0;
    let mut row_offset = 0;
    let mut x = 0;
    while x < width {
        let mut vlc_val = reverse_reader_fetch(vlc);
        let mut t0 = load_u16(vlc_table0, c_q + (vlc_val & 0x7f)) as u32;
        if c_q == 0 {
            *run -= 2;
            t0 = if *run == -1 { t0 } else { 0 };
            if *run < 0 && !mel_get_run(mel, run) {
                return 7;
            }
        }
        scratch[row_offset as usize] = t0 as u16;
        x += 2;
        c_q = ((t0 & 0x10) << 3) | ((t0 & 0xe0) << 2);
        vlc_val = reverse_reader_advance(vlc, t0 & 0x7);

        let mut t1 = load_u16(vlc_table0, c_q + (vlc_val & 0x7f)) as u32;
        if c_q == 0 && x < width {
            *run -= 2;
            t1 = if *run == -1 { t1 } else { 0 };
            if *run < 0 && !mel_get_run(mel, run) {
                return 8;
            }
        }
        if x >= width {
            t1 = 0;
        }
        scratch[row_offset as usize + 2] = t1 as u16;
        x += 2;
        c_q = ((t1 & 0x10) << 3) | ((t1 & 0xe0) << 2);
        vlc_val = reverse_reader_advance(vlc, t1 & 0x7);

        let mut uvlc_mode = ((t0 & 0x8) << 3) | ((t1 & 0x8) << 4);
        if uvlc_mode == 0xc0 {
            *run -= 2;
            if *run == -1 {
                uvlc_mode += 0x40;
            }
            if *run < 0 && !mel_get_run(mel, run) {
                return 9;
            }
        }

        let mut uvlc_entry = load_u16(uvlc_table0, uvlc_mode + (vlc_val & 0x3f)) as u32;
        vlc_val = reverse_reader_advance(vlc, uvlc_entry & 0x7);
        uvlc_entry >>= 3;
        let mut len = uvlc_entry & 0xf;
        let tmp = vlc_val & ((1 << len) - 1);
        let _ = reverse_reader_advance(vlc, len);
        uvlc_entry >>= 4;
        len = uvlc_entry & 0x7;
        uvlc_entry >>= 3;
        scratch[row_offset as usize + 1] = (1 + (uvlc_entry & 0x7) + (tmp & !(0xff << len))) as u16;
        scratch[row_offset as usize + 3] = (1 + (uvlc_entry >> 3) + (tmp >> len)) as u16;
        row_offset += 4;
    }
    scratch[row_offset as usize] = 0;
    scratch[row_offset as usize + 1] = 0;
    0
}

#[inline(always)]
fn decode_cleanup_symbols_row(
    mel: &mut MelDecoder,
    vlc: &mut ReverseBitReader,
    run: &mut i32,
    scratch: &mut [u16],
    width: u32,
    row_base: u32,
    prev_base: u32,
    vlc_table1: *const u16,
    uvlc_table1: *const u16,
) -> u32 {
    let mut local_x = 0;
    let mut local_c_q = 0;
    let mut row_offset = row_base;
    while local_x < width {
        let delta = row_offset - row_base;
        local_c_q |= (scratch[(prev_base + delta) as usize] as u32 & 0xa0) << 2;
        local_c_q |= (scratch[(prev_base + delta + 2) as usize] as u32 & 0x20) << 4;

        let mut vlc_val = reverse_reader_fetch(vlc);
        let mut t0 = load_u16(vlc_table1, local_c_q + (vlc_val & 0x7f)) as u32;
        if local_c_q == 0 {
            *run -= 2;
            t0 = if *run == -1 { t0 } else { 0 };
            if *run < 0 && !mel_get_run(mel, run) {
                return 10;
            }
        }
        scratch[row_offset as usize] = t0 as u16;
        local_x += 2;

        local_c_q = ((t0 & 0x40) << 2) | ((t0 & 0x80) << 1);
        local_c_q |= scratch[(prev_base + delta) as usize] as u32 & 0x80;
        local_c_q |= (scratch[(prev_base + delta + 2) as usize] as u32 & 0xa0) << 2;
        local_c_q |= (scratch[(prev_base + delta + 4) as usize] as u32 & 0x20) << 4;
        vlc_val = reverse_reader_advance(vlc, t0 & 0x7);

        let mut t1 = load_u16(vlc_table1, local_c_q + (vlc_val & 0x7f)) as u32;
        if local_c_q == 0 && local_x < width {
            *run -= 2;
            t1 = if *run == -1 { t1 } else { 0 };
            if *run < 0 && !mel_get_run(mel, run) {
                return 11;
            }
        }
        if local_x >= width {
            t1 = 0;
        }
        scratch[row_offset as usize + 2] = t1 as u16;
        local_x += 2;

        local_c_q = ((t1 & 0x40) << 2) | ((t1 & 0x80) << 1);
        local_c_q |= scratch[(prev_base + delta + 2) as usize] as u32 & 0x80;
        vlc_val = reverse_reader_advance(vlc, t1 & 0x7);

        let uvlc_mode = ((t0 & 0x8) << 3) | ((t1 & 0x8) << 4);
        let mut uvlc_entry = load_u16(uvlc_table1, uvlc_mode + (vlc_val & 0x3f)) as u32;
        vlc_val = reverse_reader_advance(vlc, uvlc_entry & 0x7);
        uvlc_entry >>= 3;
        let mut len = uvlc_entry & 0xf;
        let tmp = vlc_val & ((1 << len) - 1);
        let _ = reverse_reader_advance(vlc, len);
        uvlc_entry >>= 4;
        len = uvlc_entry & 0x7;
        uvlc_entry >>= 3;
        scratch[row_offset as usize + 1] = ((uvlc_entry & 0x7) + (tmp & !(0xff << len))) as u16;
        scratch[row_offset as usize + 3] = ((uvlc_entry >> 3) + (tmp >> len)) as u16;
        row_offset += 4;
    }
    scratch[row_offset as usize] = 0;
    scratch[row_offset as usize + 1] = 0;
    0
}

#[inline(always)]
fn decode_cleanup_symbols_remaining_rows(
    coded_data: *const u8,
    lcup: u32,
    scup: u32,
    scratch: &mut [u16; HT_MAX_SCRATCH],
    width: u32,
    height: u32,
    sstr: u32,
    vlc_table0: *const u16,
    vlc_table1: *const u16,
    uvlc_table0: *const u16,
    uvlc_table1: *const u16,
) -> u32 {
    let mut mel = mel_decoder_new(coded_data, lcup, scup);
    let mut vlc = reverse_reader_new_vlc(coded_data, lcup, scup);
    let mut run = 0;
    if !mel_get_run(&mut mel, &mut run) {
        return 6;
    }
    let first = decode_cleanup_symbols_first_row(
        &mut mel,
        &mut vlc,
        &mut run,
        scratch,
        width,
        vlc_table0,
        uvlc_table0,
    );
    if first != 0 {
        return first;
    }

    let mut y = 2;
    while y < height {
        let row_base = (y >> 1) * sstr;
        let prev_base = row_base - sstr;
        let detail = decode_cleanup_symbols_row(
            &mut mel,
            &mut vlc,
            &mut run,
            scratch,
            width,
            row_base,
            prev_base,
            vlc_table1,
            uvlc_table1,
        );
        if detail != 0 {
            return detail;
        }
        y += 2;
    }
    0
}

#[inline(always)]
fn decode_magnitude_sign_pair(
    magsgn: &mut ForwardBitReader,
    decoded_data: *mut u32,
    v_n_scratch: &mut [u32; HT_MAX_VN],
    inf: u32,
    uq: u32,
    p: u32,
    params: J2kHtCleanupParams,
    second_row_present: bool,
    x: &mut u32,
    dp: &mut u32,
    vp: &mut u32,
    prev_v_n: &mut u32,
    dequantize: bool,
) -> bool {
    if uq > params.missing_msbs + 2 {
        return false;
    }

    let mut value = 0;
    let mut ignored_vn = 0;
    decode_mag_sgn_sample_with_vn(magsgn, inf, 0, uq, p, &mut value, &mut ignored_vn);
    store_decoded_sample(decoded_data, *dp, value, params, dequantize);

    let mut v_n = 0;
    decode_mag_sgn_sample_with_vn(magsgn, inf, 1, uq, p, &mut value, &mut v_n);
    if second_row_present {
        store_decoded_sample(
            decoded_data,
            *dp + params.output_stride,
            value,
            params,
            dequantize,
        );
    }
    v_n_scratch[*vp as usize] = *prev_v_n | v_n;
    *prev_v_n = 0;
    *dp += 1;
    *x += 1;

    if *x >= params.width {
        *vp += 1;
        return true;
    }

    decode_mag_sgn_sample_with_vn(magsgn, inf, 2, uq, p, &mut value, &mut ignored_vn);
    store_decoded_sample(decoded_data, *dp, value, params, dequantize);

    decode_mag_sgn_sample_with_vn(magsgn, inf, 3, uq, p, &mut value, &mut v_n);
    if second_row_present {
        store_decoded_sample(
            decoded_data,
            *dp + params.output_stride,
            value,
            params,
            dequantize,
        );
    }
    *prev_v_n = v_n;
    *dp += 1;
    *x += 1;
    *vp += 1;
    true
}

#[inline(always)]
fn decode_magnitude_sign_row(
    magsgn: &mut ForwardBitReader,
    scratch: &[u16],
    row_base: u32,
    y: u32,
    decoded_data: *mut u32,
    params: J2kHtCleanupParams,
    v_n_scratch: &mut [u32; HT_MAX_VN],
    dequantize: bool,
) -> u32 {
    let p = 30 - params.missing_msbs;
    let mut x = 0;
    let mut sp = row_base;
    let mut vp = 0;
    let mut dp = params.output_offset + y * params.output_stride;
    let mut prev_v_n = 0;
    while x < params.width {
        let inf = scratch[sp as usize] as u32;
        let mut uq = scratch[sp as usize + 1] as u32;
        if y != 0 {
            let mut gamma = inf & 0xf0;
            gamma &= gamma.wrapping_sub(0x10);
            let emax =
                floor_log2_nonzero((v_n_scratch[vp as usize] | v_n_scratch[vp as usize + 1]) | 2);
            uq += if gamma != 0 { emax } else { 1 };
        }
        if !decode_magnitude_sign_pair(
            magsgn,
            decoded_data,
            v_n_scratch,
            inf,
            uq,
            p,
            params,
            y + 1 < params.height,
            &mut x,
            &mut dp,
            &mut vp,
            &mut prev_v_n,
            dequantize,
        ) {
            return if y == 0 { 13 } else { 14 };
        }
        sp += 2;
    }
    v_n_scratch[vp as usize] = prev_v_n;
    0
}

#[inline(always)]
fn decode_magnitude_sign_phase(
    coded_data: *const u8,
    lcup: u32,
    scup: u32,
    scratch: &[u16; HT_MAX_SCRATCH],
    decoded_data: *mut u32,
    params: J2kHtCleanupParams,
    sstr: u32,
    v_n_scratch: &mut [u32; HT_MAX_VN],
    dequantize: bool,
) -> u32 {
    let v_n_width = ((params.width + 1) / 2) + 2;
    if v_n_width as usize > HT_MAX_VN {
        return 12;
    }
    let mut clear = 0;
    while clear < v_n_width {
        v_n_scratch[clear as usize] = 0;
        clear += 1;
    }
    let mut magsgn = forward_reader_new(coded_data, lcup - scup, 0xff);
    let mut y = 0;
    while y < params.height {
        let detail = decode_magnitude_sign_row(
            &mut magsgn,
            scratch,
            (y >> 1) * sstr,
            y,
            decoded_data,
            params,
            v_n_scratch,
            dequantize,
        );
        if detail != 0 {
            return detail;
        }
        y += 2;
    }
    0
}

#[inline(always)]
fn build_sigma_from_cleanup(
    cleanup: &[u16; HT_MAX_SCRATCH],
    sigma: &mut [u16; HT_MAX_SIGMA],
    width: u32,
    height: u32,
    sstr: u32,
    mstr: u32,
) {
    let mut y = 0;
    while y < height {
        let sp_base = (y >> 1) * sstr;
        let dp_base = (y >> 2) * mstr;
        let mut x = 0;
        let mut sp = sp_base;
        let mut dp = dp_base;
        while x < width {
            let mut t0 = ((cleanup[sp as usize] as u32 & 0x30) >> 4)
                | ((cleanup[sp as usize] as u32 & 0xc0) >> 2);
            t0 |= ((cleanup[sp as usize + 2] as u32 & 0x30) << 4)
                | ((cleanup[sp as usize + 2] as u32 & 0xc0) << 6);
            let mut t1 = ((cleanup[(sp + sstr) as usize] as u32 & 0x30) >> 2)
                | (cleanup[(sp + sstr) as usize] as u32 & 0xc0);
            t1 |= ((cleanup[(sp + sstr + 2) as usize] as u32 & 0x30) << 6)
                | ((cleanup[(sp + sstr + 2) as usize] as u32 & 0xc0) << 8);
            sigma[dp as usize] = (t0 | t1) as u16;
            x += 4;
            sp += 4;
            dp += 1;
        }
        sigma[dp as usize] = 0;
        y += 4;
    }

    let tail = ((height + 3) / 4) * mstr;
    let mut idx = 0;
    while idx <= (width + 3) / 4 {
        sigma[(tail + idx) as usize] = 0;
        idx += 1;
    }
}

#[inline(always)]
fn apply_significance_propagation(
    coded_data: *const u8,
    sigma: &[u16; HT_MAX_SIGMA],
    decoded_data: *mut u32,
    params: J2kHtCleanupParams,
    mstr: u32,
    p: u32,
    prev_row_sig: &mut [u16; HT_MAX_PREV_ROW_SIG],
) -> u32 {
    if ((params.width + 3) / 4 + 8) as usize > HT_MAX_PREV_ROW_SIG {
        return 15;
    }
    let mut clear = 0;
    while clear < (params.width + 3) / 4 + 8 {
        prev_row_sig[clear as usize] = 0;
        clear += 1;
    }

    let mut sigprop = forward_reader_new(
        unsafe { coded_data.add(params.cleanup_length as usize) },
        params.refinement_length,
        0,
    );
    let mut y = 0;
    while y < params.height {
        let mut pattern = 0xffff;
        if params.height - y < 4 {
            pattern = 0x7777;
            if params.height - y < 3 {
                pattern = 0x3333;
                if params.height - y < 2 {
                    pattern = 0x1111;
                }
            }
        }

        let mut prev = 0;
        let cur_row = (y >> 2) * mstr;
        let next_row = cur_row + mstr;
        let dpp = params.output_offset + y * params.output_stride;
        let mut x = 0;
        while x < params.width {
            let mut col_pattern = pattern;
            let s = if x + 4 > params.width {
                x + 4 - params.width
            } else {
                0
            };
            col_pattern >>= s * 4;

            let idx = x >> 2;
            let ps =
                prev_row_sig[idx as usize] as u32 | ((prev_row_sig[idx as usize + 1] as u32) << 16);
            let ns = read_u32_pair(sigma, next_row + idx);
            let mut u = (ps & 0x8888_8888) >> 3;
            if params.stripe_causal == 0 {
                u |= (ns & 0x1111_1111) << 3;
            }
            let cs = read_u32_pair(sigma, cur_row + idx);
            let mut mbr = cs;
            mbr |= (cs & 0x7777_7777) << 1;
            mbr |= (cs & 0xeeee_eeee) >> 1;
            mbr |= u;
            let t = mbr;
            mbr |= t << 4;
            mbr |= t >> 4;
            mbr |= prev >> 12;
            mbr &= col_pattern;
            mbr &= !cs;

            let mut new_sig = 0;
            if mbr != 0 {
                let mut cwd = forward_reader_fetch(&mut sigprop);
                let mut cnt = 0;
                let inv_sig = !cs & col_pattern;
                let mut candidates = mbr;
                let mut processed = 0;
                while candidates != 0 {
                    let bit = trailing_zeros32(candidates);
                    let mask = 1 << bit;
                    candidates &= !mask;
                    processed |= mask;
                    if (cwd & 1) != 0 {
                        new_sig |= mask;
                        candidates |= SIGPROP_SPREAD_MASKS[bit as usize] & inv_sig & !processed;
                    }
                    cwd >>= 1;
                    cnt += 1;
                }

                if new_sig != 0 {
                    let value = 3 << (p - 2);
                    let block_base = dpp + x;
                    let mut sign_bits = new_sig;
                    while sign_bits != 0 {
                        let bit = trailing_zeros32(sign_bits);
                        let sample = 1 << bit;
                        sign_bits &= !sample;
                        let offset = (bit >> 2) + ((bit & 3) * params.output_stride);
                        store_decoded_sample(
                            decoded_data,
                            block_base + offset,
                            (cwd << 31) | value,
                            params,
                            false,
                        );
                        cwd >>= 1;
                        cnt += 1;
                    }
                }
                forward_reader_advance(&mut sigprop, cnt);
            }

            let combined_sig = new_sig | (cs & 0xffff);
            prev_row_sig[idx as usize] = combined_sig as u16;

            let combined = combined_sig;
            let mut next_prev = combined_sig;
            next_prev |= (combined & 0x7777) << 1;
            next_prev |= (combined & 0xeeee) >> 1;
            prev = (next_prev | u) & 0xf000;
            x += 4;
        }
        y += 4;
    }
    0
}

#[inline(always)]
fn apply_magnitude_refinement(
    coded_data: *const u8,
    sigma: &[u16; HT_MAX_SIGMA],
    decoded_data: *mut u32,
    params: J2kHtCleanupParams,
    mstr: u32,
    p: u32,
) {
    let mut magref =
        reverse_reader_new_mrp(coded_data, params.cleanup_length, params.refinement_length);
    let half_value = 1 << (p - 2);
    let mut y = 0;
    while y < params.height {
        let mut cur_sig_idx = (y >> 2) * mstr;
        let dpp = params.output_offset + y * params.output_stride;
        let mut x8 = 0;
        while x8 < params.width {
            let mut cwd = reverse_reader_fetch(&mut magref);
            let sig = read_u32_pair(sigma, cur_sig_idx);
            cur_sig_idx += 2;
            let mut col_mask = 0xf;
            if sig != 0 {
                let mut column = 0;
                while column < 8 {
                    if (sig & col_mask) != 0 {
                        let mut mag_dp = dpp + x8 + column;
                        let mut sample_mask = 0x1111_1111 & col_mask;
                        let mut row = 0;
                        while row < 4 {
                            if (sig & sample_mask) != 0 {
                                let mut sym = cwd & 1;
                                sym = (1 - sym) << (p - 1);
                                sym |= half_value;
                                xor_decoded_sample(decoded_data, mag_dp, sym);
                                cwd >>= 1;
                            }
                            sample_mask <<= 1;
                            mag_dp += params.output_stride;
                            row += 1;
                        }
                    }
                    col_mask <<= 4;
                    column += 1;
                }
            }
            reverse_reader_advance(&mut magref, popcount32(sig));
            x8 += 8;
        }
        y += 4;
    }
}

#[inline(always)]
fn cleanup_scup(
    coded_data: *const u8,
    params: J2kHtCleanupParams,
    status: *mut J2kHtStatus,
    cleanup_only: bool,
    dequantize: bool,
) -> u32 {
    store_status(status, HT_STATUS_OK, 0);

    let mut num_passes = params.number_of_coding_passes;
    if num_passes > 1 && params.refinement_length == 0 {
        num_passes = 1;
    }
    if cleanup_only && params.refinement_length != 0 {
        store_status(status, HT_STATUS_UNSUPPORTED, 17);
        return 0;
    }
    if dequantize
        && (!cleanup_only || params.number_of_coding_passes > 1 || params.refinement_length != 0)
    {
        store_status(status, HT_STATUS_UNSUPPORTED, 18);
        return 0;
    }
    if params.width == 0 || params.height == 0 {
        return 0;
    }
    if params.width > HT_MAX_WIDTH
        || params.height > HT_MAX_HEIGHT
        || params.width * params.height > HT_MAX_COEFFICIENTS
    {
        store_status(status, HT_STATUS_UNSUPPORTED, 1);
        return 0;
    }
    let roi_shift = params.reconstruction & HT_ROI_SHIFT_MASK;
    if params.num_bitplanes == 0
        || params.num_bitplanes > 31
        || roi_shift > 31 - params.num_bitplanes
    {
        store_status(status, HT_STATUS_FAIL, 2);
        return 0;
    }
    if num_passes > 3 || params.missing_msbs >= 30 {
        store_status(status, HT_STATUS_FAIL, 3);
        return 0;
    }
    let lcup = params.cleanup_length;
    if lcup < 2 || params.coded_len < lcup + params.refinement_length {
        store_status(status, HT_STATUS_FAIL, 4);
        return 0;
    }
    let scup = ((load_u8(coded_data, lcup - 1) as u32) << 4)
        + (load_u8(coded_data, lcup - 2) as u32 & 0x0f);
    if scup < 2 || scup > lcup || scup > 4079 {
        store_status(status, HT_STATUS_FAIL, 5);
        return 0;
    }

    let quad_rows = (params.height + 1) / 2;
    let sstr = (params.width + 9) & !7;
    if sstr > HT_MAX_SSTR || (sstr * (quad_rows + 1)) as usize > HT_MAX_SCRATCH {
        store_status(status, HT_STATUS_UNSUPPORTED, 6);
        return 0;
    }

    scup
}

#[inline(always)]
fn decode_ht_cleanup_impl(
    coded_data: *const u8,
    decoded_data: *mut u32,
    params: J2kHtCleanupParams,
    vlc_table0: *const u16,
    vlc_table1: *const u16,
    uvlc_table0: *const u16,
    uvlc_table1: *const u16,
    status: *mut J2kHtStatus,
) {
    let scup = cleanup_scup(coded_data, params, status, false, false);
    if scup == 0 {
        return;
    }
    let mut num_passes = params.number_of_coding_passes;
    if num_passes > 1 && (params.refinement_length == 0 || params.missing_msbs == 29) {
        num_passes = 1;
    }
    let lcup = params.cleanup_length;
    let sstr = (params.width + 9) & !7;

    let mut scratch = [0u16; HT_MAX_SCRATCH];
    let cleanup_detail = decode_cleanup_symbols_remaining_rows(
        coded_data,
        lcup,
        scup,
        &mut scratch,
        params.width,
        params.height,
        sstr,
        vlc_table0,
        vlc_table1,
        uvlc_table0,
        uvlc_table1,
    );
    if cleanup_detail != 0 {
        store_status(status, HT_STATUS_FAIL, cleanup_detail);
        return;
    }

    let mut v_n_scratch = [0u32; HT_MAX_VN];
    let magsgn_detail = decode_magnitude_sign_phase(
        coded_data,
        lcup,
        scup,
        &scratch,
        decoded_data,
        params,
        sstr,
        &mut v_n_scratch,
        false,
    );
    if magsgn_detail != 0 {
        store_status(status, HT_STATUS_FAIL, magsgn_detail);
        return;
    }
    if num_passes == 1 {
        return;
    }

    let mstr = (((params.width + 3) / 4) + 9) & !7;
    let sigma_rows = (params.height + 3) / 4 + 1;
    if mstr > HT_MAX_MSTR || (mstr * sigma_rows) as usize > HT_MAX_SIGMA {
        store_status(status, HT_STATUS_UNSUPPORTED, 16);
        return;
    }
    let p = 30 - params.missing_msbs;
    let mut sigma = [0u16; HT_MAX_SIGMA];
    build_sigma_from_cleanup(
        &scratch,
        &mut sigma,
        params.width,
        params.height,
        sstr,
        mstr,
    );

    let mut prev_row_sig = [0u16; HT_MAX_PREV_ROW_SIG];
    let sigprop_detail = apply_significance_propagation(
        coded_data,
        &sigma,
        decoded_data,
        params,
        mstr,
        p,
        &mut prev_row_sig,
    );
    if sigprop_detail != 0 {
        store_status(status, HT_STATUS_UNSUPPORTED, sigprop_detail);
        return;
    }

    if num_passes > 2 {
        apply_magnitude_refinement(coded_data, &sigma, decoded_data, params, mstr, p);
    }
}

// One warp consumes at most 32 quads (3968 bits) per prefix sum. A circular
// 8192-bit window retains that chunk, fetch lookahead, and the final byte group
// without storing the entire MagSgn segment in shared memory.
const HT_MAG_WORDS: usize = 256;
const HT_VN_ROW: usize = HT_MAX_WIDTH as usize + 2;

#[inline(always)]
fn warp_prefix_sum(mut value: u32, lane: u32) -> u32 {
    let mut delta = 1;
    while delta < 32 {
        let previous = warp::shuffle_up(value, delta);
        if lane >= delta {
            value += previous;
        }
        delta *= 2;
    }
    value
}

#[inline(always)]
fn fill_magsgn_window(
    data: *const u8,
    data_len: u32,
    words: *mut u32,
    byte_cursor: &mut u32,
    loaded_bits: &mut u32,
    needed_bits: u32,
    lane: u32,
) {
    while *byte_cursor < data_len && *loaded_bits < needed_bits {
        let byte_index = *byte_cursor + lane;
        let present = byte_index < data_len;
        let stuffed = present && byte_index != 0 && load_u8(data, byte_index - 1) == 0xff;
        let stuffed_mask = warp::ballot(stuffed);
        let offset = *loaded_bits + lane * 8 - (stuffed_mask & warp::lanemask_lt()).count_ones();
        let end = *loaded_bits + 8 * (data_len - *byte_cursor).min(32) - stuffed_mask.count_ones();
        // Each lane clears a distinct new word; preserve the partial word from
        // the previous group, then merge adjacent bytes atomically.
        let clear = (*loaded_bits + 31) / 32 + lane;
        if clear < (end + 31) / 32 {
            simt_store(words, clear as usize % HT_MAG_WORDS, 0);
        }
        warp::sync_mask(u32::MAX);
        if present {
            let byte = load_u8(data, byte_index) as u32 & if stuffed { 0x7f } else { 0xff };
            let word = (offset / 32) as usize;
            let shift = offset % 32;
            // SAFETY: `words` is this block's `HT_MAG_WORDS` shared-memory
            // window; the index is reduced modulo its length and is only ever
            // accessed atomically while lanes merge bytes into shared words.
            unsafe {
                BlockAtomicU32::from_ptr(simt_mut_ptr_at(words, word % HT_MAG_WORDS))
                    .fetch_or(byte << shift, AtomicOrdering::Relaxed);
            }
            if shift + 8 - stuffed as u32 > 32 {
                // SAFETY: as above, the wrapped index stays inside the window.
                unsafe {
                    BlockAtomicU32::from_ptr(simt_mut_ptr_at(words, (word + 1) % HT_MAG_WORDS))
                        .fetch_or(byte >> (32 - shift), AtomicOrdering::Relaxed);
                }
            }
        }
        warp::sync_mask(u32::MAX);
        *loaded_bits = end;
        *byte_cursor += 32;
    }
}

#[inline(always)]
fn magsgn_word(words: *const u32, index: u32, valid_bits: u32) -> u32 {
    let start = index * 32;
    if start >= valid_bits {
        return u32::MAX;
    }
    let value = simt_load(words, index as usize % HT_MAG_WORDS);
    if valid_bits - start < 32 {
        value | (u32::MAX << (valid_bits - start))
    } else {
        value
    }
}

#[inline(always)]
fn magsgn_fetch(words: *const u32, offset: u32, valid_bits: u32) -> u32 {
    let word = offset / 32;
    let shift = offset % 32;
    let value = magsgn_word(words, word, valid_bits) >> shift;
    if shift == 0 {
        value
    } else {
        value | (magsgn_word(words, word + 1, valid_bits) << (32 - shift))
    }
}

#[inline(always)]
fn decode_cleanup_magsgn_warp(
    coded_data: *const u8,
    symbols: *const u16,
    words: *mut u32,
    vn_rows: *mut u32,
    params: J2kHtCleanupParams,
    status: *mut J2kHtStatus,
    output: *mut u32,
    dequantize: bool,
) {
    let lane = thread::threadIdx_x();
    let scup = ((load_u8(coded_data, params.cleanup_length - 1) as u32) << 4)
        + (load_u8(coded_data, params.cleanup_length - 2) as u32 & 0xf);
    let mut byte_cursor = 0;
    let mut valid_bits = 0;
    let mut index = lane as usize;
    while index < 2 * HT_VN_ROW {
        simt_store(vn_rows, index, 0);
        index += 32;
    }
    warp::sync_mask(u32::MAX);
    let p = 30 - params.missing_msbs;
    let sstr = (params.width + 9) & !7;
    let quads = (params.width + 1) / 2;
    let mut bit_base = 0;
    let mut y = 0;
    while y < params.height {
        let row = ((y / 2) & 1) as usize * HT_VN_ROW;
        let prev = HT_VN_ROW - row;
        let mut quad_base = 0;
        while quad_base < quads {
            let quad = quad_base + lane;
            let x = quad * 2;
            let active = quad < quads;
            let mut inf = 0;
            let mut uq = 0;
            if active {
                let symbol = (y / 2 * sstr + quad * 2) as usize;
                inf = simt_load(symbols, symbol) as u32;
                uq = simt_load(symbols, symbol + 1) as u32;
                if y != 0 {
                    let previous = simt_load(vn_rows.cast_const(), prev + x as usize)
                        | simt_load(vn_rows.cast_const(), prev + x as usize + 1)
                        | simt_load(vn_rows.cast_const(), prev + x as usize + 2)
                        | simt_load(vn_rows.cast_const(), prev + x as usize + 3);
                    let gamma = inf & 0xf0;
                    uq += if gamma & gamma.wrapping_sub(0x10) != 0 {
                        floor_log2_nonzero(previous | 2)
                    } else {
                        1
                    };
                }
            }
            if warp::ballot(active && uq > params.missing_msbs + 2) != 0 {
                if lane == 0 {
                    store_status(status, HT_STATUS_FAIL, if y == 0 { 13 } else { 14 });
                }
                return;
            }
            let sample_count = if x + 1 < params.width { 4 } else { 2 };
            let mut count = 0;
            let mut bit = 0;
            while bit < sample_count {
                if inf & sample_mask(bit) != 0 {
                    count += uq - ((inf >> (12 + bit)) & 1);
                }
                bit += 1;
            }
            let prefix = warp_prefix_sum(count, lane);
            let mut offset = bit_base + prefix - count;
            bit_base += warp::shuffle(prefix, 31);
            fill_magsgn_window(
                coded_data,
                params.cleanup_length - scup,
                words,
                &mut byte_cursor,
                &mut valid_bits,
                bit_base + 32,
                lane,
            );
            if active {
                bit = 0;
                while bit < sample_count {
                    let mut value = 0;
                    let mut vn = 0;
                    if inf & sample_mask(bit) != 0 {
                        let ms = magsgn_fetch(words.cast_const(), offset, valid_bits);
                        let length = uq - ((inf >> (12 + bit)) & 1);
                        offset += length;
                        (value, vn) = mag_sgn_value(ms, length, inf, bit, p);
                    }
                    let sample_x = x + bit / 2;
                    let sample_y = y + bit % 2;
                    if sample_y < params.height {
                        store_decoded_sample(
                            output,
                            params.output_offset + sample_y * params.output_stride + sample_x,
                            value,
                            params,
                            dequantize,
                        );
                    }
                    if bit & 1 != 0 {
                        simt_store(vn_rows, row + sample_x as usize + 1, vn);
                    }
                    bit += 1;
                }
            }
            quad_base += 32;
        }
        warp::sync_mask(u32::MAX);
        y += 2;
    }
}

#[inline(always)]
fn params_from_batch_job(job: J2kHtCleanupBatchJob) -> J2kHtCleanupParams {
    J2kHtCleanupParams {
        width: job.width,
        height: job.height,
        coded_len: job.coded_len,
        cleanup_length: job.cleanup_length,
        refinement_length: job.refinement_length,
        missing_msbs: job.missing_msbs,
        num_bitplanes: job.num_bitplanes,
        reconstruction: job.reconstruction,
        number_of_coding_passes: job.number_of_coding_passes,
        output_stride: job.output_stride,
        output_offset: job.output_offset,
        dequantization_step: job.dequantization_step,
        stripe_causal: job.stripe_causal,
    }
}

#[inline(always)]
fn params_from_multi_job(job: J2kHtCleanupMultiBatchJob) -> J2kHtCleanupParams {
    J2kHtCleanupParams {
        width: job.width,
        height: job.height,
        coded_len: job.coded_len,
        cleanup_length: job.cleanup_length,
        refinement_length: job.refinement_length,
        missing_msbs: job.missing_msbs,
        num_bitplanes: job.num_bitplanes,
        reconstruction: job.reconstruction,
        number_of_coding_passes: job.number_of_coding_passes,
        output_stride: job.output_stride,
        output_offset: job.output_offset,
        dequantization_step: job.dequantization_step,
        stripe_causal: job.stripe_causal,
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn j2k_htj2k_decode_cleanup_symbols(
        coded_data: *const u8,
        jobs: *const J2kHtCleanupMultiBatchJob,
        vlc_table0: *const u16,
        vlc_table1: *const u16,
        uvlc_table0: *const u16,
        uvlc_table1: *const u16,
        status: *mut J2kHtStatus,
        job_count: u32,
        scratch: *mut u16,
        dequantize: u32,
        jobs_per_block: u32,
    ) {
        static mut TABLES: SharedArray<u16, 2624> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x();
        let tables = unsafe { TABLES.as_mut_ptr() };
        let mut index = lane;
        while index < 2624 {
            let value = if index < 1024 {
                load_u16(vlc_table0, index)
            } else if index < 2048 {
                load_u16(vlc_table1, index - 1024)
            } else if index < 2368 {
                load_u16(uvlc_table0, index - 2048)
            } else {
                load_u16(uvlc_table1, index - 2368)
            };
            simt_store(tables, index as usize, value);
            index += 32;
        }
        thread::sync_threads();
        let gid = thread::blockIdx_x() * jobs_per_block + lane;
        if lane >= jobs_per_block || gid >= job_count {
            return;
        }
        let job = load_job(jobs, gid);
        let params = params_from_multi_job(job);
        let data = simt_const_ptr_at(coded_data, job.coded_offset as usize);
        let status = simt_mut_ptr_at(status, gid as usize);
        let scup = cleanup_scup(data, params, status, true, dequantize != 0);
        if scup == 0 {
            return;
        }
        // SAFETY: the host allocates `HT_MAX_SCRATCH` u16 entries per job
        // (`cleanup_scratch`), `gid < job_count`, and only this thread touches
        // job `gid`'s slice.
        let scratch = unsafe {
            &mut *simt_mut_ptr_at(scratch, gid as usize * HT_MAX_SCRATCH)
                .cast::<[u16; HT_MAX_SCRATCH]>()
        };
        let detail = decode_cleanup_symbols_remaining_rows(
            data,
            params.cleanup_length,
            scup,
            scratch,
            params.width,
            params.height,
            (params.width + 9) & !7,
            tables,
            simt_const_ptr_at(tables, 1024),
            simt_const_ptr_at(tables, 2048),
            simt_const_ptr_at(tables, 2368),
        );
        if detail != 0 {
            store_status(status, HT_STATUS_FAIL, detail);
        }
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_decode_cleanup_magsgn(
        coded_data: *const u8,
        jobs: *const J2kHtCleanupMultiBatchJob,
        statuses: *mut J2kHtStatus,
        job_count: u32,
        scratch: *const u16,
        dequantize: u32,
    ) {
        static mut WORDS: SharedArray<u32, HT_MAG_WORDS> = SharedArray::UNINIT;
        static mut VN: SharedArray<u32, { 2 * HT_VN_ROW }> = SharedArray::UNINIT;
        let gid = thread::blockIdx_x();
        if gid >= job_count || simt_load(statuses.cast_const(), gid as usize).code != HT_STATUS_OK {
            return;
        }
        let job = load_job(jobs, gid);
        if job.width == 0 || job.height == 0 {
            return;
        }
        decode_cleanup_magsgn_warp(
            simt_const_ptr_at(coded_data, job.coded_offset as usize),
            simt_const_ptr_at(scratch, gid as usize * HT_MAX_SCRATCH),
            unsafe { WORDS.as_mut_ptr() },
            unsafe { VN.as_mut_ptr() },
            params_from_multi_job(job),
            simt_mut_ptr_at(statuses, gid as usize),
            job.output_ptr as usize as *mut u32,
            dequantize != 0,
        );
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_decode_codeblocks(
        coded_data: *const u8,
        decoded_data: *mut u32,
        jobs: *const J2kHtCleanupBatchJob,
        vlc_table0: *const u16,
        vlc_table1: *const u16,
        uvlc_table0: *const u16,
        uvlc_table1: *const u16,
        status: *mut J2kHtStatus,
        job_count: u32,
    ) {
        let gid = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if gid >= job_count {
            return;
        }
        let job = load_job(jobs, gid);
        decode_ht_cleanup_impl(
            simt_const_ptr_at(coded_data, job.coded_offset as usize),
            decoded_data,
            params_from_batch_job(job),
            vlc_table0,
            vlc_table1,
            uvlc_table0,
            uvlc_table1,
            simt_mut_ptr_at(status, gid as usize),
        );
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_decode_codeblocks_multi(
        coded_data: *const u8,
        jobs: *const J2kHtCleanupMultiBatchJob,
        vlc_table0: *const u16,
        vlc_table1: *const u16,
        uvlc_table0: *const u16,
        uvlc_table1: *const u16,
        status: *mut J2kHtStatus,
        job_count: u32,
    ) {
        let gid = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if gid >= job_count {
            return;
        }
        let job = load_job(jobs, gid);
        decode_ht_cleanup_impl(
            simt_const_ptr_at(coded_data, job.coded_offset as usize),
            job.output_ptr as usize as *mut u32,
            params_from_multi_job(job),
            vlc_table0,
            vlc_table1,
            uvlc_table0,
            uvlc_table1,
            simt_mut_ptr_at(status, gid as usize),
        );
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_clear_coefficient_targets(params: J2kCoefficientClearBatch) {
        let target_index = thread::blockIdx_y();
        if target_index >= params.target_count {
            return;
        }
        let target = params.targets[target_index as usize];
        let output = target.output_ptr as usize as *mut u32;
        let mut word = u64::from(thread::blockIdx_x() * thread::blockDim_x())
            + u64::from(thread::threadIdx_x());
        let stride = u64::from(params.blocks_per_target * thread::blockDim_x());
        while word < target.words {
            simt_store(output, word as usize, 0);
            word += stride;
        }
    }
}

fn main() {}
