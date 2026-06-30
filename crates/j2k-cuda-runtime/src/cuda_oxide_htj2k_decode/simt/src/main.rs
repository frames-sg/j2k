use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

const HT_STATUS_OK: u32 = 0;
const HT_STATUS_FAIL: u32 = 1;
const HT_STATUS_UNSUPPORTED: u32 = 2;

const HT_MAX_WIDTH: u32 = 256;
const HT_MAX_HEIGHT: u32 = 256;
const HT_MAX_COEFFICIENTS: u32 = 4096;
const HT_MAX_SSTR: u32 = 264;
const HT_MAX_SCRATCH: usize = 3096;
const HT_MAX_VN: usize = 130;
const HT_MAX_MSTR: u32 = 72;
const HT_MAX_SIGMA: usize = 528;
const HT_MAX_PREV_ROW_SIG: usize = 72;

const SIGPROP_SPREAD_MASKS: [u32; 16] = [
    0x33, 0x76, 0xEC, 0xC8, 0x330, 0x760, 0xEC0, 0xC80, 0x3300, 0x7600, 0xEC00, 0xC800,
    0x33000, 0x76000, 0xEC000, 0xC8000,
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
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
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
    if a < b { a } else { b }
}

#[inline(always)]
fn load_u8(ptr: *const u8, index: u32) -> u8 {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn load_u16(ptr: *const u16, index: u32) -> u16 {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn load_job<T: Copy>(ptr: *const T, index: u32) -> T {
    unsafe { *ptr.add(index as usize) }
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
fn popcount32(mut value: u32) -> u32 {
    let mut count = 0;
    while value != 0 {
        value &= value - 1;
        count += 1;
    }
    count
}

#[inline(always)]
fn trailing_zeros32(value: u32) -> u32 {
    let mut count = 0;
    let mut bits = value;
    while count < 32 && (bits & 1) == 0 {
        count += 1;
        bits >>= 1;
    }
    count
}

#[inline(always)]
fn floor_log2_nonzero(mut value: u32) -> u32 {
    let mut log = 0;
    while value > 1 {
        value >>= 1;
        log += 1;
    }
    log
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
fn coefficient_to_float_bits(value: u32, k_max: u32, scale: f32) -> u32 {
    ((coefficient_to_i32(value, k_max) as f32) * scale).to_bits()
}

#[inline(always)]
fn decoded_cleanup_sample_bits(value: u32, params: J2kHtCleanupParams, dequantize: bool) -> u32 {
    if dequantize {
        coefficient_to_float_bits(value, params.num_bitplanes, params.dequantization_step)
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
    unsafe {
        *decoded_data.add(index as usize) = decoded_cleanup_sample_bits(value, params, dequantize);
    }
}

#[inline(always)]
fn xor_decoded_sample(decoded_data: *mut u32, index: u32, value: u32) {
    unsafe {
        let ptr = decoded_data.add(index as usize);
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
        reader.tmp |= (byte as u64) << reader.bits;
        reader.bits += 8 - reader.unstuff as u32;
        reader.unstuff = byte == 0xff;
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
        let d_bits = 8 - (reader.unstuff && (byte & 0x7f) == 0x7f) as u32;
        reader.tmp |= (byte as u64) << reader.bits;
        reader.bits += d_bits;
        reader.unstuff = byte > 0x8f;
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

    *value = ms_val << 31;
    let mask = if m_n == 0 { 0 } else { (1 << m_n) - 1 };
    *v_n = ms_val & mask;
    *v_n |= ((inf >> (8 + bit)) & 1) << m_n;
    *v_n |= 1;
    *value |= (*v_n + 2) << (p - 1);
}

#[inline(always)]
fn decode_cleanup_symbols_first_row(
    mel: &mut MelDecoder,
    vlc: &mut ReverseBitReader,
    run: &mut i32,
    scratch: &mut [u16; HT_MAX_SCRATCH],
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
        scratch[row_offset as usize + 1] =
            (1 + (uvlc_entry & 0x7) + (tmp & !(0xff << len))) as u16;
        scratch[row_offset as usize + 3] = (1 + (uvlc_entry >> 3) + (tmp >> len)) as u16;
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
        let mut local_x = 0;
        let mut local_c_q = 0;
        let mut row_offset = row_base;
        while local_x < width {
            let delta = row_offset - row_base;
            local_c_q |= (scratch[(prev_base + delta) as usize] as u32 & 0xa0) << 2;
            local_c_q |= (scratch[(prev_base + delta + 2) as usize] as u32 & 0x20) << 4;

            let mut vlc_val = reverse_reader_fetch(&mut vlc);
            let mut t0 = load_u16(vlc_table1, local_c_q + (vlc_val & 0x7f)) as u32;
            if local_c_q == 0 {
                run -= 2;
                t0 = if run == -1 { t0 } else { 0 };
                if run < 0 && !mel_get_run(&mut mel, &mut run) {
                    return 10;
                }
            }
            scratch[row_offset as usize] = t0 as u16;
            local_x += 2;

            local_c_q = ((t0 & 0x40) << 2) | ((t0 & 0x80) << 1);
            local_c_q |= scratch[(prev_base + delta) as usize] as u32 & 0x80;
            local_c_q |= (scratch[(prev_base + delta + 2) as usize] as u32 & 0xa0) << 2;
            local_c_q |= (scratch[(prev_base + delta + 4) as usize] as u32 & 0x20) << 4;
            vlc_val = reverse_reader_advance(&mut vlc, t0 & 0x7);

            let mut t1 = load_u16(vlc_table1, local_c_q + (vlc_val & 0x7f)) as u32;
            if local_c_q == 0 && local_x < width {
                run -= 2;
                t1 = if run == -1 { t1 } else { 0 };
                if run < 0 && !mel_get_run(&mut mel, &mut run) {
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
            vlc_val = reverse_reader_advance(&mut vlc, t1 & 0x7);

            let uvlc_mode = ((t0 & 0x8) << 3) | ((t1 & 0x8) << 4);
            let mut uvlc_entry = load_u16(uvlc_table1, uvlc_mode + (vlc_val & 0x3f)) as u32;
            vlc_val = reverse_reader_advance(&mut vlc, uvlc_entry & 0x7);
            uvlc_entry >>= 3;
            let mut len = uvlc_entry & 0xf;
            let tmp = vlc_val & ((1 << len) - 1);
            let _ = reverse_reader_advance(&mut vlc, len);
            uvlc_entry >>= 4;
            len = uvlc_entry & 0x7;
            uvlc_entry >>= 3;
            scratch[row_offset as usize + 1] =
                ((uvlc_entry & 0x7) + (tmp & !(0xff << len))) as u16;
            scratch[row_offset as usize + 3] = ((uvlc_entry >> 3) + (tmp >> len)) as u16;
            row_offset += 4;
        }
        scratch[row_offset as usize] = 0;
        scratch[row_offset as usize + 1] = 0;
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

    let p = 30 - params.missing_msbs;
    let mut magsgn = forward_reader_new(coded_data, lcup - scup, 0xff);
    let mut prev_v_n = 0;
    let mut x = 0;
    let mut sp = 0;
    let mut vp = 0;
    let mut dp = params.output_offset;
    let second_row_present = params.height > 1;

    while x < params.width {
        let inf = scratch[sp as usize] as u32;
        let uq = scratch[sp as usize + 1] as u32;
        if !decode_magnitude_sign_pair(
            &mut magsgn,
            decoded_data,
            v_n_scratch,
            inf,
            uq,
            p,
            params,
            second_row_present,
            &mut x,
            &mut dp,
            &mut vp,
            &mut prev_v_n,
            dequantize,
        ) {
            return 13;
        }
        sp += 2;
    }
    v_n_scratch[vp as usize] = prev_v_n;

    let mut y = 2;
    while y < params.height {
        let row_base = (y >> 1) * sstr;
        let mut local_x = 0;
        let mut local_sp = row_base;
        let mut local_vp = 0;
        let mut local_dp = params.output_offset + y * params.output_stride;
        let mut local_prev_v_n = 0;
        let local_second_row_present = y + 1 < params.height;

        while local_x < params.width {
            let inf = scratch[local_sp as usize] as u32;
            let u_q = scratch[local_sp as usize + 1] as u32;
            let mut gamma = inf & 0xf0;
            gamma &= gamma.wrapping_sub(0x10);
            let emax = floor_log2_nonzero((v_n_scratch[local_vp as usize]
                | v_n_scratch[local_vp as usize + 1])
                | 2);
            let kappa = if gamma != 0 { emax } else { 1 };
            let uq = u_q + kappa;
            if !decode_magnitude_sign_pair(
                &mut magsgn,
                decoded_data,
                v_n_scratch,
                inf,
                uq,
                p,
                params,
                local_second_row_present,
                &mut local_x,
                &mut local_dp,
                &mut local_vp,
                &mut local_prev_v_n,
                dequantize,
            ) {
                return 14;
            }
            local_sp += 2;
        }
        v_n_scratch[local_vp as usize] = local_prev_v_n;
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
            let ps = prev_row_sig[idx as usize] as u32
                | ((prev_row_sig[idx as usize + 1] as u32) << 16);
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
                        store_decoded_sample(decoded_data, block_base + offset, (cwd << 31) | value, params, false);
                        cwd >>= 1;
                        cnt += 1;
                    }
                }
                forward_reader_advance(&mut sigprop, cnt);
            }

            let combined_sig = new_sig | cs;
            prev_row_sig[idx as usize] = combined_sig as u16;
            prev_row_sig[idx as usize + 1] = (combined_sig >> 16) as u16;

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
    let mut magref = reverse_reader_new_mrp(coded_data, params.cleanup_length, params.refinement_length);
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
fn decode_ht_cleanup_impl(
    coded_data: *const u8,
    decoded_data: *mut u32,
    params: J2kHtCleanupParams,
    vlc_table0: *const u16,
    vlc_table1: *const u16,
    uvlc_table0: *const u16,
    uvlc_table1: *const u16,
    status: *mut J2kHtStatus,
    cleanup_only: bool,
    dequantize: bool,
) {
    store_status(status, HT_STATUS_OK, 0);

    let mut num_passes = params.number_of_coding_passes;
    if num_passes > 1 && params.refinement_length == 0 {
        num_passes = 1;
    }
    if cleanup_only && params.refinement_length != 0 {
        store_status(status, HT_STATUS_UNSUPPORTED, 17);
        return;
    }
    if dequantize && (!cleanup_only || params.number_of_coding_passes > 1 || params.refinement_length != 0) {
        store_status(status, HT_STATUS_UNSUPPORTED, 18);
        return;
    }
    if params.width == 0 || params.height == 0 {
        return;
    }
    if params.width > HT_MAX_WIDTH
        || params.height > HT_MAX_HEIGHT
        || params.width * params.height > HT_MAX_COEFFICIENTS
    {
        store_status(status, HT_STATUS_UNSUPPORTED, 1);
        return;
    }
    if params.num_bitplanes == 0 || params.num_bitplanes > 31 {
        store_status(status, HT_STATUS_FAIL, 2);
        return;
    }
    if num_passes > 3 || params.missing_msbs >= 30 {
        store_status(status, HT_STATUS_FAIL, 3);
        return;
    }
    if params.missing_msbs == 29 && num_passes > 1 {
        num_passes = 1;
    }

    let lcup = params.cleanup_length;
    if lcup < 2 || params.coded_len < lcup + params.refinement_length {
        store_status(status, HT_STATUS_FAIL, 4);
        return;
    }
    let scup = ((load_u8(coded_data, lcup - 1) as u32) << 4) + (load_u8(coded_data, lcup - 2) as u32 & 0x0f);
    if scup < 2 || scup > lcup || scup > 4079 {
        store_status(status, HT_STATUS_FAIL, 5);
        return;
    }

    let quad_rows = (params.height + 1) / 2;
    let sstr = (params.width + 9) & !7;
    if sstr > HT_MAX_SSTR || (sstr * (quad_rows + 1)) as usize > HT_MAX_SCRATCH {
        store_status(status, HT_STATUS_UNSUPPORTED, 6);
        return;
    }

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
        dequantize,
    );
    if magsgn_detail != 0 {
        store_status(status, HT_STATUS_FAIL, magsgn_detail);
        return;
    }
    if cleanup_only || num_passes == 1 {
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
    build_sigma_from_cleanup(&scratch, &mut sigma, params.width, params.height, sstr, mstr);

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
            coded_data.add(job.coded_offset as usize),
            decoded_data,
            params_from_batch_job(job),
            vlc_table0,
            vlc_table1,
            uvlc_table0,
            uvlc_table1,
            status.add(gid as usize),
            false,
            false,
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
            coded_data.add(job.coded_offset as usize),
            job.output_ptr as usize as *mut u32,
            params_from_multi_job(job),
            vlc_table0,
            vlc_table1,
            uvlc_table0,
            uvlc_table1,
            status.add(gid as usize),
            false,
            false,
        );
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_decode_codeblocks_multi_cleanup_only(
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
            coded_data.add(job.coded_offset as usize),
            job.output_ptr as usize as *mut u32,
            params_from_multi_job(job),
            vlc_table0,
            vlc_table1,
            uvlc_table0,
            uvlc_table1,
            status.add(gid as usize),
            true,
            false,
        );
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize(
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
            coded_data.add(job.coded_offset as usize),
            job.output_ptr as usize as *mut u32,
            params_from_multi_job(job),
            vlc_table0,
            vlc_table1,
            uvlc_table0,
            uvlc_table1,
            status.add(gid as usize),
            true,
            true,
        );
    }
}

fn main() {}
