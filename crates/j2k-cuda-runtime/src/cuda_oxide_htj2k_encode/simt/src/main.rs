#![allow(
    clippy::manual_div_ceil,
    reason = "CUDA device toolchain compatibility requires explicit integer ceiling division"
)]
#![allow(
    clippy::too_many_arguments,
    reason = "flat device helpers mirror CUDA ABI buffers and launch metadata"
)]
#![allow(
    clippy::too_many_lines,
    reason = "HT device kernels keep bitstream state local to preserve control flow and register layout"
)]

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

include!("../../../cuda_oxide_simt_prelude.rs");

const ENCODE_STATUS_OK: u32 = 0;
const ENCODE_STATUS_FAIL: u32 = 1;
const ENCODE_STATUS_UNSUPPORTED: u32 = 2;

const HT_MAX_BITPLANES: u32 = 30;
const HT_MEL_SIZE: u32 = 192;
const HT_VLC_SIZE: u32 = 3072 - HT_MEL_SIZE;
const HT_MS_SIZE: u32 = ((16384 * 16) + 14) / 15;
const HT_MEL_OFFSET: u32 = HT_MS_SIZE;
const HT_VLC_OFFSET: u32 = HT_MS_SIZE + HT_MEL_SIZE;
const HT_SIGPROP_SCRATCH: usize = 513;
const HT_MAX_CODEBLOCK_WIDTH: u32 = 1024;
const HT_MAX_CODEBLOCK_SAMPLES: u32 = 4096;
const HT_COMPACT_ASSEMBLE_FLAG: u32 = 0x8000_0000;
const HT_COMPACT_LENGTH_MASK: u32 = 0x7fff;

const SIGPROP_SPREAD_MASKS: [u32; 16] = [
    0x33, 0x76, 0xEC, 0xC8, 0x330, 0x760, 0xEC0, 0xC80, 0x3300, 0x7600, 0xEC00, 0xC800, 0x33000,
    0x76000, 0xEC000, 0xC8000,
];

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtEncodeParams {
    width: u32,
    height: u32,
    coefficient_stride: u32,
    total_bitplanes: u32,
    output_capacity: u32,
    target_coding_passes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtEncodeStatus {
    code: u32,
    detail: u32,
    data_len: u32,
    num_coding_passes: u32,
    num_zero_bitplanes: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtEncodeJob {
    coefficient_offset: u32,
    coefficient_stride: u32,
    width: u32,
    height: u32,
    total_bitplanes: u32,
    output_offset: u32,
    output_capacity: u32,
    target_coding_passes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtEncodeMultiInputJob {
    coefficient_ptr: u64,
    coefficient_offset: u32,
    coefficient_stride: u32,
    width: u32,
    height: u32,
    total_bitplanes: u32,
    output_offset: u32,
    output_capacity: u32,
    target_coding_passes: u32,
}

#[derive(Clone, Copy)]
struct MelEncoder {
    pos: u32,
    remaining_bits: u32,
    tmp: u8,
    run: u32,
    k: u32,
    threshold: u32,
    failed: u32,
}

#[derive(Clone, Copy)]
struct VlcEncoder {
    pos: u32,
    used_bits: u32,
    tmp: u8,
    last_greater_than_8f: u32,
    failed: u32,
}

#[derive(Clone, Copy)]
struct MagSgnEncoder {
    pos: u32,
    max_bits: u32,
    used_bits: u32,
    tmp: u32,
    failed: u32,
}

#[derive(Clone, Copy)]
struct SigPropWriter {
    pos: u32,
    used_bits: u32,
    previous_was_ff: u32,
    capacity: u32,
    tmp: u8,
    failed: u32,
}

#[inline(always)]
fn min_u32(a: u32, b: u32) -> u32 {
    if a < b { a } else { b }
}

#[inline(always)]
fn max_u32(a: u32, b: u32) -> u32 {
    if a > b { a } else { b }
}

#[inline(always)]
fn max_i32(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

#[inline(always)]
fn max_u8(a: u8, b: u8) -> u8 {
    if a > b { a } else { b }
}

#[inline(always)]
fn unsigned_magnitude(value: i32) -> u32 {
    let bits = value as u32;
    if value < 0 {
        (!bits).wrapping_add(1)
    } else {
        bits
    }
}

#[inline(always)]
fn leading_zeros32(value: u32) -> u32 {
    let mut count = 0;
    let mut mask = 0x8000_0000;
    while mask != 0 && (value & mask) == 0 {
        count += 1;
        mask >>= 1;
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
fn load_i32(ptr: *const i32, index: u32) -> i32 {
    simt_load(ptr, index as usize)
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
fn store_u8(ptr: *mut u8, index: u32, value: u8) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn set_status(
    status: *mut J2kHtEncodeStatus,
    code: u32,
    detail: u32,
    data_len: u32,
    passes: u32,
    zbp: u32,
) {
    unsafe {
        (*status).code = code;
        (*status).detail = detail;
        (*status).data_len = data_len;
        (*status).num_coding_passes = passes;
        (*status).num_zero_bitplanes = zbp;
        (*status).reserved0 = 0;
        (*status).reserved1 = 0;
        (*status).reserved2 = 0;
    }
}

#[inline(always)]
fn set_status_with_segments(
    status: *mut J2kHtEncodeStatus,
    code: u32,
    detail: u32,
    data_len: u32,
    passes: u32,
    zbp: u32,
    cleanup_len: u32,
    refinement_len: u32,
    reserved: u32,
) {
    unsafe {
        (*status).code = code;
        (*status).detail = detail;
        (*status).data_len = data_len;
        (*status).num_coding_passes = passes;
        (*status).num_zero_bitplanes = zbp;
        (*status).reserved0 = cleanup_len;
        (*status).reserved1 = refinement_len;
        (*status).reserved2 = reserved;
    }
}

#[inline(always)]
fn pack_compact_assembly_lengths(mel_len: u32, vlc_len: u32) -> u32 {
    HT_COMPACT_ASSEMBLE_FLAG
        | (mel_len & HT_COMPACT_LENGTH_MASK)
        | ((vlc_len & HT_COMPACT_LENGTH_MASK) << 15)
}

#[inline(always)]
fn aligned_sign_magnitude(coefficient: i32, total_bitplanes: u32) -> u32 {
    if coefficient == 0 {
        return 0;
    }
    let sign = if coefficient < 0 { 0x8000_0000 } else { 0 };
    let magnitude = unsigned_magnitude(coefficient) << (31 - total_bitplanes);
    sign | magnitude
}

#[inline(always)]
fn mel_exp(k: u32) -> u32 {
    if k < 3 {
        0
    } else if k < 6 {
        1
    } else if k < 9 {
        2
    } else if k < 11 {
        3
    } else if k == 11 {
        4
    } else {
        5
    }
}

#[inline(always)]
fn sigprop_spread_mask(bit: u32) -> u32 {
    if bit < 16 {
        SIGPROP_SPREAD_MASKS[bit as usize]
    } else {
        0
    }
}

#[inline(always)]
fn sigprop_writer_init(writer: &mut SigPropWriter, capacity: u32) {
    writer.pos = 0;
    writer.used_bits = 0;
    writer.previous_was_ff = 0;
    writer.capacity = capacity;
    writer.tmp = 0;
    writer.failed = 0;
}

#[inline(always)]
fn sigprop_write_bit(writer: &mut SigPropWriter, out: *mut u8, bit: u32) {
    let max_bits = if writer.previous_was_ff != 0 { 7 } else { 8 };
    writer.tmp |= ((bit & 1) << writer.used_bits) as u8;
    writer.used_bits += 1;
    if writer.used_bits < max_bits {
        return;
    }
    if writer.pos >= writer.capacity {
        writer.failed = 1;
        return;
    }
    if !out.is_null() {
        store_u8(out, writer.pos, writer.tmp);
    }
    writer.previous_was_ff = if writer.tmp == 0xff { 1 } else { 0 };
    writer.tmp = 0;
    writer.used_bits = 0;
    writer.pos += 1;
}

#[inline(always)]
fn sigprop_finish(writer: &mut SigPropWriter, out: *mut u8) {
    if writer.used_bits == 0 {
        return;
    }
    if writer.pos >= writer.capacity {
        writer.failed = 1;
        return;
    }
    if !out.is_null() {
        store_u8(out, writer.pos, writer.tmp);
    }
    writer.pos += 1;
    writer.tmp = 0;
    writer.used_bits = 0;
}

fn sigprop_cleanup_sig16(
    coefficients: *const i32,
    coefficient_stride: u32,
    width: u32,
    height: u32,
    x_base: u32,
    y_base: u32,
) -> u32 {
    let mut mask = 0;
    let mut col = 0;
    while col < 4 {
        let x = x_base + col;
        if x < width {
            let mut row = 0;
            while row < 4 {
                let y = y_base + row;
                if y < height {
                    let magnitude =
                        unsigned_magnitude(load_i32(coefficients, y * coefficient_stride + x));
                    if magnitude >= 5 && (magnitude & 1) != 0 {
                        mask |= 1 << (col * 4 + row);
                    }
                }
                row += 1;
            }
        }
        col += 1;
    }
    mask
}

fn sigprop_target_sig16(
    coefficients: *const i32,
    coefficient_stride: u32,
    width: u32,
    height: u32,
    x_base: u32,
    y_base: u32,
) -> u32 {
    let mut mask = 0;
    let mut col = 0;
    while col < 4 {
        let x = x_base + col;
        if x < width {
            let mut row = 0;
            while row < 4 {
                let y = y_base + row;
                if y < height {
                    let magnitude =
                        unsigned_magnitude(load_i32(coefficients, y * coefficient_stride + x));
                    if magnitude == 3 {
                        mask |= 1 << (col * 4 + row);
                    }
                }
                row += 1;
            }
        }
        col += 1;
    }
    mask
}

#[inline(always)]
fn sigprop_coefficient_sign(
    coefficients: *const i32,
    coefficient_stride: u32,
    x_base: u32,
    y_base: u32,
    bit: u32,
) -> u32 {
    let col = bit >> 2;
    let row = bit & 3;
    if load_i32(
        coefficients,
        (y_base + row) * coefficient_stride + x_base + col,
    ) < 0
    {
        1
    } else {
        0
    }
}

fn write_sigprop_segment(
    coefficients: *const i32,
    coefficient_stride: u32,
    width: u32,
    height: u32,
    out: *mut u8,
    capacity: u32,
    bytes_written: &mut u32,
) -> u32 {
    let group_count = (width + 3) >> 2;
    if group_count + 8 > HT_SIGPROP_SCRATCH as u32 {
        return 0;
    }
    let mut prev_row_sig = [0u16; HT_SIGPROP_SCRATCH];
    let mut writer = SigPropWriter {
        pos: 0,
        used_bits: 0,
        previous_was_ff: 0,
        capacity: 0,
        tmp: 0,
        failed: 0,
    };
    sigprop_writer_init(&mut writer, capacity);

    let mut y = 0;
    while y < height {
        let mut pattern = 0xffff;
        if height - y < 4 {
            pattern = 0x7777;
            if height - y < 3 {
                pattern = 0x3333;
                if height - y < 2 {
                    pattern = 0x1111;
                }
            }
        }

        let mut prev = 0;
        let mut x = 0;
        while x < width {
            let mut col_pattern = pattern;
            if x + 4 > width {
                col_pattern >>= (x + 4 - width) * 4;
            }

            let idx = x >> 2;
            let ps = (prev_row_sig[idx as usize] as u32)
                | ((prev_row_sig[(idx + 1) as usize] as u32) << 16);
            let ns =
                sigprop_cleanup_sig16(coefficients, coefficient_stride, width, height, x, y + 4)
                    | (sigprop_cleanup_sig16(
                        coefficients,
                        coefficient_stride,
                        width,
                        height,
                        x + 4,
                        y + 4,
                    ) << 16);
            let mut u = (ps & 0x8888_8888) >> 3;
            u |= (ns & 0x1111_1111) << 3;

            let cs = sigprop_cleanup_sig16(coefficients, coefficient_stride, width, height, x, y)
                | (sigprop_cleanup_sig16(
                    coefficients,
                    coefficient_stride,
                    width,
                    height,
                    x + 4,
                    y,
                ) << 16);
            let mut mbr = cs;
            mbr |= (cs & 0x7777_7777) << 1;
            mbr |= (cs & 0xeeee_eeee) >> 1;
            mbr |= u;
            let t_mbr = mbr;
            mbr |= t_mbr << 4;
            mbr |= t_mbr >> 4;
            mbr |= prev >> 12;
            mbr &= col_pattern;
            mbr &= !cs;

            let mut new_sig = 0;
            let target_sig =
                sigprop_target_sig16(coefficients, coefficient_stride, width, height, x, y)
                    & col_pattern;
            if mbr != 0 {
                let mut candidates = mbr;
                let mut processed = 0;
                let inv_sig = !cs & col_pattern;
                while candidates != 0 {
                    let bit = trailing_zeros32(candidates);
                    let sample_mask = 1 << bit;
                    candidates &= !sample_mask;
                    processed |= sample_mask;
                    let desired = if (target_sig & sample_mask) != 0 {
                        1
                    } else {
                        0
                    };
                    sigprop_write_bit(&mut writer, out, desired);
                    if writer.failed != 0 {
                        return 0;
                    }
                    if desired != 0 {
                        new_sig |= sample_mask;
                        candidates |= sigprop_spread_mask(bit) & inv_sig & !processed;
                    }
                }

                if new_sig != 0 {
                    let mut sign_bits = new_sig;
                    while sign_bits != 0 {
                        let bit = trailing_zeros32(sign_bits);
                        let sample_mask = 1 << bit;
                        sign_bits &= !sample_mask;
                        sigprop_write_bit(
                            &mut writer,
                            out,
                            sigprop_coefficient_sign(coefficients, coefficient_stride, x, y, bit),
                        );
                        if writer.failed != 0 {
                            return 0;
                        }
                    }
                }
            }

            if (target_sig & !new_sig) != 0 {
                return 0;
            }

            let combined_sig = new_sig | cs;
            prev_row_sig[idx as usize] = (combined_sig & 0xffff) as u16;
            prev_row_sig[(idx + 1) as usize] = ((combined_sig >> 16) & 0xffff) as u16;
            let t = combined_sig;
            let mut next_prev = combined_sig;
            next_prev |= (t & 0x7777) << 1;
            next_prev |= (t & 0xeeee) >> 1;
            prev = (next_prev | u) & 0xf000;

            x += 4;
        }
        y += 4;
    }

    sigprop_finish(&mut writer, out);
    if writer.failed != 0 {
        return 0;
    }
    *bytes_written = writer.pos;
    1
}

fn write_magref_segment(
    coefficients: *const i32,
    coefficient_stride: u32,
    width: u32,
    height: u32,
    out: *mut u8,
    magref_len: u32,
    expected_bits: u32,
) -> u32 {
    if magref_len == 0 {
        return if expected_bits == 0 { 1 } else { 0 };
    }
    let mut idx = 0;
    while idx < magref_len {
        store_u8(out, idx, 0);
        idx += 1;
    }

    let mut bit_idx = 0;
    let mut byte_from_end = 0;
    let mut used_bits = 0;
    let mut unstuff = 1;
    let mut current = 0u8;
    let mut y = 0;
    while y < height {
        let mut x_base = 0;
        while x_base < width {
            let mut col = 0;
            while col < 8 {
                let x = x_base + col;
                if x < width {
                    let mut row = 0;
                    while row < 4 {
                        let yy = y + row;
                        if yy < height {
                            let magnitude = unsigned_magnitude(load_i32(
                                coefficients,
                                yy * coefficient_stride + x,
                            ));
                            if magnitude >= 5 && (magnitude & 1) != 0 {
                                current |= (((magnitude >> 1) & 1) << used_bits) as u8;
                                used_bits += 1;
                                bit_idx += 1;
                                let stuffed =
                                    unstuff != 0 && used_bits == 7 && (current & 0x7f) == 0x7f;
                                if stuffed || used_bits == 8 {
                                    if byte_from_end >= magref_len {
                                        return 0;
                                    }
                                    store_u8(out, magref_len - 1 - byte_from_end, current);
                                    byte_from_end += 1;
                                    unstuff = if current > 0x8f { 1 } else { 0 };
                                    current = 0;
                                    used_bits = 0;
                                }
                            }
                        }
                        row += 1;
                    }
                }
                col += 1;
            }
            x_base += 8;
        }
        y += 4;
    }

    if used_bits != 0 {
        if byte_from_end >= magref_len {
            return 0;
        }
        store_u8(out, magref_len - 1 - byte_from_end, current);
        byte_from_end += 1;
    }
    if bit_idx != expected_bits || byte_from_end > magref_len {
        return 0;
    }
    1
}

#[inline(always)]
fn mel_init(mel: &mut MelEncoder) {
    mel.pos = 0;
    mel.remaining_bits = 8;
    mel.tmp = 0;
    mel.run = 0;
    mel.k = 0;
    mel.threshold = 1;
    mel.failed = 0;
}

#[inline(always)]
fn vlc_init(vlc: &mut VlcEncoder, out: *mut u8) {
    vlc.pos = 1;
    vlc.used_bits = 4;
    vlc.tmp = 0x0f;
    vlc.last_greater_than_8f = 1;
    vlc.failed = 0;
    store_u8(out, HT_VLC_OFFSET + HT_VLC_SIZE - 1, 0xff);
}

#[inline(always)]
fn ms_init(ms: &mut MagSgnEncoder) {
    ms.pos = 0;
    ms.max_bits = 8;
    ms.used_bits = 0;
    ms.tmp = 0;
    ms.failed = 0;
}

#[inline(always)]
fn cleanup_scratch_entries(width: u32) -> u32 {
    min_u32(((width + 1) >> 1) + 2, HT_SIGPROP_SCRATCH as u32)
}

#[inline(always)]
fn mel_emit_bit(mel: &mut MelEncoder, out: *mut u8, bit: bool) {
    mel.tmp = (u32::from(mel.tmp) << 1 | u32::from(bit)) as u8;
    mel.remaining_bits -= 1;
    if mel.remaining_bits == 0 {
        if mel.pos >= HT_MEL_SIZE {
            mel.failed = 1;
            return;
        }
        store_u8(out, HT_MEL_OFFSET + mel.pos, mel.tmp);
        mel.pos += 1;
        mel.remaining_bits = if mel.tmp == 0xff { 7 } else { 8 };
        mel.tmp = 0;
    }
}

#[inline(always)]
fn mel_encode(mel: &mut MelEncoder, out: *mut u8, bit: bool) {
    if !bit {
        mel.run += 1;
        if mel.run >= mel.threshold {
            mel_emit_bit(mel, out, true);
            mel.run = 0;
            mel.k = min_u32(mel.k + 1, 12);
            mel.threshold = 1 << mel_exp(mel.k);
        }
    } else {
        mel_emit_bit(mel, out, false);
        let mut t = mel_exp(mel.k);
        while t > 0 {
            t -= 1;
            mel_emit_bit(mel, out, ((mel.run >> t) & 1) != 0);
        }
        mel.run = 0;
        mel.k = mel.k.saturating_sub(1);
        mel.threshold = 1 << mel_exp(mel.k);
    }
}

#[inline(always)]
fn vlc_encode(vlc: &mut VlcEncoder, out: *mut u8, mut codeword: u32, mut codeword_len: u32) {
    while codeword_len > 0 {
        if vlc.pos >= HT_VLC_SIZE {
            vlc.failed = 1;
            return;
        }
        let mut available_bits = 8 - vlc.last_greater_than_8f - vlc.used_bits;
        let take = min_u32(available_bits, codeword_len);
        let mask = if take == 32 {
            u32::MAX
        } else {
            (1 << take) - 1
        };
        vlc.tmp |= ((codeword & mask) << vlc.used_bits) as u8;
        vlc.used_bits += take;
        available_bits -= take;
        codeword_len -= take;
        codeword >>= take;
        if available_bits == 0 {
            if vlc.last_greater_than_8f != 0 && vlc.tmp != 0x7f {
                vlc.last_greater_than_8f = 0;
                continue;
            }
            let write_index = HT_VLC_SIZE - 1 - vlc.pos;
            store_u8(out, HT_VLC_OFFSET + write_index, vlc.tmp);
            vlc.pos += 1;
            vlc.last_greater_than_8f = if vlc.tmp > 0x8f { 1 } else { 0 };
            vlc.tmp = 0;
            vlc.used_bits = 0;
        }
    }
}

#[inline(always)]
fn ms_encode(ms: &mut MagSgnEncoder, out: *mut u8, mut codeword: u32, mut codeword_len: u32) {
    while codeword_len > 0 {
        if ms.pos >= HT_MS_SIZE {
            ms.failed = 1;
            return;
        }
        let take = min_u32(ms.max_bits - ms.used_bits, codeword_len);
        let mask = if take == 32 {
            u32::MAX
        } else {
            (1 << take) - 1
        };
        ms.tmp |= (codeword & mask) << ms.used_bits;
        ms.used_bits += take;
        codeword >>= take;
        codeword_len -= take;
        if ms.used_bits >= ms.max_bits {
            store_u8(out, ms.pos, ms.tmp as u8);
            ms.pos += 1;
            ms.max_bits = if ms.tmp == 0xff { 7 } else { 8 };
            ms.tmp = 0;
            ms.used_bits = 0;
        }
    }
}

#[inline(always)]
fn ms_terminate(ms: &mut MagSgnEncoder, out: *mut u8) {
    if ms.used_bits > 0 {
        let unused = ms.max_bits - ms.used_bits;
        ms.tmp |= (0xff & ((1 << unused) - 1)) << ms.used_bits;
        ms.used_bits += unused;
        if ms.tmp != 0xff {
            if ms.pos >= HT_MS_SIZE {
                ms.failed = 1;
                return;
            }
            store_u8(out, ms.pos, ms.tmp as u8);
            ms.pos += 1;
        }
    } else if ms.max_bits == 7 {
        ms.pos = ms.pos.saturating_sub(1);
    }
}

#[inline(always)]
fn process_sample(
    slot: u32,
    value: u32,
    p: u32,
    rho_acc: &mut i32,
    e_q: &mut [i32; 8],
    e_qmax: &mut i32,
    s: &mut [u32; 8],
) {
    let mut val = value.wrapping_add(value);
    val >>= p;
    val &= !1;
    if val != 0 {
        *rho_acc |= (1 << (slot & 3)) as i32;
        val -= 1;
        e_q[slot as usize] = (32 - leading_zeros32(val)) as i32;
        *e_qmax = max_i32(*e_qmax, e_q[slot as usize]);
        val -= 1;
        s[slot as usize] = val + (value >> 31);
    }
}

#[inline(always)]
fn uvlc_byte(table: *const u8, index: u32, field: u32) -> u8 {
    load_u8(table, index * 6 + field)
}

#[inline(always)]
fn encode_uvlc_pair(
    vlc: &mut VlcEncoder,
    out: *mut u8,
    uvlc_table: *const u8,
    first_index: u32,
    second_index: u32,
) {
    let first_pre = uvlc_byte(uvlc_table, first_index, 0);
    let first_pre_len = uvlc_byte(uvlc_table, first_index, 1);
    let first_suf = uvlc_byte(uvlc_table, first_index, 2);
    let first_suf_len = uvlc_byte(uvlc_table, first_index, 3);
    let second_pre = uvlc_byte(uvlc_table, second_index, 0);
    let second_pre_len = uvlc_byte(uvlc_table, second_index, 1);
    let second_suf = uvlc_byte(uvlc_table, second_index, 2);
    let second_suf_len = uvlc_byte(uvlc_table, second_index, 3);
    vlc_encode(vlc, out, u32::from(first_pre), u32::from(first_pre_len));
    vlc_encode(vlc, out, u32::from(second_pre), u32::from(second_pre_len));
    vlc_encode(vlc, out, u32::from(first_suf), u32::from(first_suf_len));
    vlc_encode(vlc, out, u32::from(second_suf), u32::from(second_suf_len));
}

fn encode_uvlc(vlc: &mut VlcEncoder, out: *mut u8, uvlc_table: *const u8, u_q0: i32, u_q1: i32) {
    if u_q0 > 2 && u_q1 > 2 {
        encode_uvlc_pair(vlc, out, uvlc_table, (u_q0 - 2) as u32, (u_q1 - 2) as u32);
    } else if u_q0 > 2 && u_q1 > 0 {
        let first_index = u_q0 as u32;
        let first_pre = uvlc_byte(uvlc_table, first_index, 0);
        let first_pre_len = uvlc_byte(uvlc_table, first_index, 1);
        let first_suf = uvlc_byte(uvlc_table, first_index, 2);
        let first_suf_len = uvlc_byte(uvlc_table, first_index, 3);
        vlc_encode(vlc, out, u32::from(first_pre), u32::from(first_pre_len));
        vlc_encode(vlc, out, (u_q1 - 1) as u32, 1);
        vlc_encode(vlc, out, u32::from(first_suf), u32::from(first_suf_len));
    } else {
        encode_uvlc_pair(
            vlc,
            out,
            uvlc_table,
            max_i32(u_q0, 0) as u32,
            max_i32(u_q1, 0) as u32,
        );
    }
}

#[inline(always)]
fn encode_uvlc_non_initial(
    vlc: &mut VlcEncoder,
    out: *mut u8,
    uvlc_table: *const u8,
    u_q0: i32,
    u_q1: i32,
) {
    encode_uvlc_pair(
        vlc,
        out,
        uvlc_table,
        max_i32(u_q0, 0) as u32,
        max_i32(u_q1, 0) as u32,
    );
}

fn encode_mag_signs(
    rho: i32,
    u_q: i32,
    tuple: u16,
    s: &[u32; 8],
    offset: u32,
    ms: &mut MagSgnEncoder,
    out: *mut u8,
) {
    let e_k = u32::from(tuple & 0xf);
    let mut bit = 0;
    while bit < 4 {
        let sample_mask = 1 << bit;
        if (rho & sample_mask) != 0 {
            let reduction = (e_k >> bit) & 1;
            let magnitude_bits = (u_q as u32).saturating_sub(reduction);
            let payload = if magnitude_bits == 0 {
                0
            } else {
                s[(offset + bit as u32) as usize] & ((1 << magnitude_bits) - 1)
            };
            ms_encode(ms, out, payload, magnitude_bits);
        }
        bit += 1;
    }
}

fn encode_quad_initial_row(
    offset: u32,
    c_q: u32,
    rho: i32,
    e_qmax: i32,
    e_q: &[i32; 8],
    s: &[u32; 8],
    lep: u32,
    lcxp: u32,
    e_val: &mut [u8; HT_SIGPROP_SCRATCH],
    cx_val: &mut [u8; HT_SIGPROP_SCRATCH],
    mel: &mut MelEncoder,
    vlc: &mut VlcEncoder,
    ms: &mut MagSgnEncoder,
    out: *mut u8,
    vlc_table0: *const u16,
) -> i32 {
    let u_q = max_i32(e_qmax, 1) - 1;
    let mut eps = 0;
    if u_q > 0 {
        eps |= u32::from(e_q[offset as usize] == e_qmax);
        eps |= u32::from(e_q[(offset + 1) as usize] == e_qmax) << 1;
        eps |= u32::from(e_q[(offset + 2) as usize] == e_qmax) << 2;
        eps |= u32::from(e_q[(offset + 3) as usize] == e_qmax) << 3;
    }
    e_val[lep as usize] = max_u8(e_val[lep as usize], e_q[(offset + 1) as usize] as u8);
    e_val[(lep + 1) as usize] = e_q[(offset + 3) as usize] as u8;
    cx_val[lcxp as usize] |= ((rho & 2) >> 1) as u8;
    cx_val[(lcxp + 1) as usize] = ((rho & 8) >> 3) as u8;

    let tuple = load_u16(vlc_table0, (c_q << 8) | ((rho as u32) << 4) | eps);
    vlc_encode(
        vlc,
        out,
        u32::from(tuple >> 8),
        u32::from((tuple >> 4) & 0x7),
    );
    if c_q == 0 {
        mel_encode(mel, out, rho != 0);
    }
    encode_mag_signs(rho, max_i32(e_qmax, 1), tuple, s, offset, ms, out);
    u_q
}

fn encode_quad_non_initial_row(
    offset: u32,
    c_q: u32,
    rho: i32,
    e_qmax: i32,
    max_e: i32,
    e_q: &[i32; 8],
    s: &[u32; 8],
    mel: &mut MelEncoder,
    vlc: &mut VlcEncoder,
    ms: &mut MagSgnEncoder,
    out: *mut u8,
    vlc_table1: *const u16,
) -> i32 {
    let kappa = if (rho & (rho - 1)) != 0 {
        max_i32(max_e, 1)
    } else {
        1
    };
    let u_q = max_i32(e_qmax, kappa) - kappa;
    let mut eps = 0;
    if u_q > 0 {
        eps |= u32::from(e_q[offset as usize] == e_qmax);
        eps |= u32::from(e_q[(offset + 1) as usize] == e_qmax) << 1;
        eps |= u32::from(e_q[(offset + 2) as usize] == e_qmax) << 2;
        eps |= u32::from(e_q[(offset + 3) as usize] == e_qmax) << 3;
    }

    let tuple = load_u16(vlc_table1, (c_q << 8) | ((rho as u32) << 4) | eps);
    vlc_encode(
        vlc,
        out,
        u32::from(tuple >> 8),
        u32::from((tuple >> 4) & 0x7),
    );
    if c_q == 0 {
        mel_encode(mel, out, rho != 0);
    }
    encode_mag_signs(rho, max_i32(e_qmax, kappa), tuple, s, offset, ms, out);
    u_q
}

#[inline(always)]
fn clear_quad_state(
    rho: &mut [i32; 2],
    e_q: &mut [i32; 8],
    e_qmax: &mut [i32; 2],
    s: &mut [u32; 8],
) {
    rho[0] = 0;
    rho[1] = 0;
    e_qmax[0] = 0;
    e_qmax[1] = 0;
    let mut idx = 0;
    while idx < 8 {
        e_q[idx] = 0;
        s[idx] = 0;
        idx += 1;
    }
}

fn encode_first_quad_pair(
    coefficients: *const i32,
    source_stride: u32,
    width: u32,
    height: u32,
    total_bitplanes: u32,
    p: u32,
    sp: &mut u32,
    x: u32,
    e_val: &mut [u8; HT_SIGPROP_SCRATCH],
    cx_val: &mut [u8; HT_SIGPROP_SCRATCH],
    c_q0: &mut u32,
    rho: &mut [i32; 2],
    e_q: &mut [i32; 8],
    e_qmax: &mut [i32; 2],
    s: &mut [u32; 8],
    mel: &mut MelEncoder,
    vlc: &mut VlcEncoder,
    ms: &mut MagSgnEncoder,
    out: *mut u8,
    vlc_table0: *const u16,
    uvlc_table: *const u8,
    fixed_64: bool,
) {
    let lep = x / 2;
    let lcxp = x / 2;
    process_sample(
        0,
        aligned_sign_magnitude(load_i32(coefficients, *sp), total_bitplanes),
        p,
        &mut rho[0],
        e_q,
        &mut e_qmax[0],
        s,
    );
    process_sample(
        1,
        if fixed_64 || height > 1 {
            aligned_sign_magnitude(load_i32(coefficients, *sp + source_stride), total_bitplanes)
        } else {
            0
        },
        p,
        &mut rho[0],
        e_q,
        &mut e_qmax[0],
        s,
    );
    *sp += 1;

    if fixed_64 || x + 1 < width {
        process_sample(
            2,
            aligned_sign_magnitude(load_i32(coefficients, *sp), total_bitplanes),
            p,
            &mut rho[0],
            e_q,
            &mut e_qmax[0],
            s,
        );
        process_sample(
            3,
            if fixed_64 || height > 1 {
                aligned_sign_magnitude(load_i32(coefficients, *sp + source_stride), total_bitplanes)
            } else {
                0
            },
            p,
            &mut rho[0],
            e_q,
            &mut e_qmax[0],
            s,
        );
        *sp += 1;
    }

    let u_q0 = encode_quad_initial_row(
        0, *c_q0, rho[0], e_qmax[0], e_q, s, lep, lcxp, e_val, cx_val, mel, vlc, ms, out,
        vlc_table0,
    );

    if fixed_64 || x + 2 < width {
        process_sample(
            4,
            aligned_sign_magnitude(load_i32(coefficients, *sp), total_bitplanes),
            p,
            &mut rho[1],
            e_q,
            &mut e_qmax[1],
            s,
        );
        process_sample(
            5,
            if fixed_64 || height > 1 {
                aligned_sign_magnitude(load_i32(coefficients, *sp + source_stride), total_bitplanes)
            } else {
                0
            },
            p,
            &mut rho[1],
            e_q,
            &mut e_qmax[1],
            s,
        );
        *sp += 1;
        if fixed_64 || x + 3 < width {
            process_sample(
                6,
                aligned_sign_magnitude(load_i32(coefficients, *sp), total_bitplanes),
                p,
                &mut rho[1],
                e_q,
                &mut e_qmax[1],
                s,
            );
            process_sample(
                7,
                if fixed_64 || height > 1 {
                    aligned_sign_magnitude(
                        load_i32(coefficients, *sp + source_stride),
                        total_bitplanes,
                    )
                } else {
                    0
                },
                p,
                &mut rho[1],
                e_q,
                &mut e_qmax[1],
                s,
            );
            *sp += 1;
        }
        let c_q1 = ((rho[0] >> 1) | (rho[0] & 1)) as u32;
        let u_q1 = encode_quad_initial_row(
            4,
            c_q1,
            rho[1],
            e_qmax[1],
            e_q,
            s,
            lep + 1,
            lcxp + 1,
            e_val,
            cx_val,
            mel,
            vlc,
            ms,
            out,
            vlc_table0,
        );
        if u_q0 > 0 && u_q1 > 0 {
            mel_encode(mel, out, min_u32(u_q0 as u32, u_q1 as u32) > 2);
        }
        encode_uvlc(vlc, out, uvlc_table, u_q0, u_q1);
        *c_q0 = ((rho[1] >> 1) | (rho[1] & 1)) as u32;
    } else {
        encode_uvlc(vlc, out, uvlc_table, u_q0, 0);
        *c_q0 = 0;
    }

    clear_quad_state(rho, e_q, e_qmax, s);
}

fn encode_non_initial_quad_pair(
    coefficients: *const i32,
    stride: u32,
    width: u32,
    height: u32,
    y: u32,
    total_bitplanes: u32,
    p: u32,
    sp: &mut u32,
    x: u32,
    e_val: &mut [u8; HT_SIGPROP_SCRATCH],
    cx_val: &mut [u8; HT_SIGPROP_SCRATCH],
    lep: &mut u32,
    lcxp: &mut u32,
    max_e: &mut i32,
    c_q0: &mut u32,
    rho: &mut [i32; 2],
    e_q: &mut [i32; 8],
    e_qmax: &mut [i32; 2],
    s: &mut [u32; 8],
    mel: &mut MelEncoder,
    vlc: &mut VlcEncoder,
    ms: &mut MagSgnEncoder,
    out: *mut u8,
    vlc_table1: *const u16,
    uvlc_table: *const u8,
    fixed_64: bool,
) {
    process_sample(
        0,
        aligned_sign_magnitude(load_i32(coefficients, *sp), total_bitplanes),
        p,
        &mut rho[0],
        e_q,
        &mut e_qmax[0],
        s,
    );
    process_sample(
        1,
        if fixed_64 || y + 1 < height {
            aligned_sign_magnitude(load_i32(coefficients, *sp + stride), total_bitplanes)
        } else {
            0
        },
        p,
        &mut rho[0],
        e_q,
        &mut e_qmax[0],
        s,
    );
    *sp += 1;

    if fixed_64 || x + 1 < width {
        process_sample(
            2,
            aligned_sign_magnitude(load_i32(coefficients, *sp), total_bitplanes),
            p,
            &mut rho[0],
            e_q,
            &mut e_qmax[0],
            s,
        );
        process_sample(
            3,
            if fixed_64 || y + 1 < height {
                aligned_sign_magnitude(load_i32(coefficients, *sp + stride), total_bitplanes)
            } else {
                0
            },
            p,
            &mut rho[0],
            e_q,
            &mut e_qmax[0],
            s,
        );
        *sp += 1;
    }

    let prev_max = *max_e;
    let u_q0 = encode_quad_non_initial_row(
        0, *c_q0, rho[0], e_qmax[0], prev_max, e_q, s, mel, vlc, ms, out, vlc_table1,
    );
    e_val[*lep as usize] = max_u8(e_val[*lep as usize], e_q[1] as u8);
    *lep += 1;
    *max_e = i32::from(max_u8(e_val[*lep as usize], e_val[(*lep + 1) as usize])) - 1;
    e_val[*lep as usize] = e_q[3] as u8;
    cx_val[*lcxp as usize] |= ((rho[0] & 2) >> 1) as u8;
    *lcxp += 1;
    let mut c_q1 =
        u32::from(cx_val[*lcxp as usize]) + (u32::from(cx_val[(*lcxp + 1) as usize]) << 2);
    cx_val[*lcxp as usize] = ((rho[0] & 8) >> 3) as u8;

    let mut u_q1 = 0;
    if fixed_64 || x + 2 < width {
        process_sample(
            4,
            aligned_sign_magnitude(load_i32(coefficients, *sp), total_bitplanes),
            p,
            &mut rho[1],
            e_q,
            &mut e_qmax[1],
            s,
        );
        process_sample(
            5,
            if fixed_64 || y + 1 < height {
                aligned_sign_magnitude(load_i32(coefficients, *sp + stride), total_bitplanes)
            } else {
                0
            },
            p,
            &mut rho[1],
            e_q,
            &mut e_qmax[1],
            s,
        );
        *sp += 1;
        if fixed_64 || x + 3 < width {
            process_sample(
                6,
                aligned_sign_magnitude(load_i32(coefficients, *sp), total_bitplanes),
                p,
                &mut rho[1],
                e_q,
                &mut e_qmax[1],
                s,
            );
            process_sample(
                7,
                if fixed_64 || y + 1 < height {
                    aligned_sign_magnitude(load_i32(coefficients, *sp + stride), total_bitplanes)
                } else {
                    0
                },
                p,
                &mut rho[1],
                e_q,
                &mut e_qmax[1],
                s,
            );
            *sp += 1;
        }
        c_q1 |= ((rho[0] & 4) >> 1) as u32;
        c_q1 |= ((rho[0] & 8) >> 2) as u32;
        u_q1 = encode_quad_non_initial_row(
            4, c_q1, rho[1], e_qmax[1], *max_e, e_q, s, mel, vlc, ms, out, vlc_table1,
        );
        e_val[*lep as usize] = max_u8(e_val[*lep as usize], e_q[5] as u8);
        *lep += 1;
        *max_e = i32::from(max_u8(e_val[*lep as usize], e_val[(*lep + 1) as usize])) - 1;
        e_val[*lep as usize] = e_q[7] as u8;
        cx_val[*lcxp as usize] |= ((rho[1] & 2) >> 1) as u8;
        *lcxp += 1;
        *c_q0 = u32::from(cx_val[*lcxp as usize]) + (u32::from(cx_val[(*lcxp + 1) as usize]) << 2);
        cx_val[*lcxp as usize] = ((rho[1] & 8) >> 3) as u8;
        *c_q0 |= ((rho[1] & 4) >> 1) as u32;
        *c_q0 |= ((rho[1] & 8) >> 2) as u32;
    } else {
        *c_q0 = 0;
    }

    encode_uvlc_non_initial(vlc, out, uvlc_table, u_q0, u_q1);
    clear_quad_state(rho, e_q, e_qmax, s);
}

fn terminate_mel_vlc(mel: &mut MelEncoder, vlc: &mut VlcEncoder, out: *mut u8) {
    if mel.run > 0 {
        mel_emit_bit(mel, out, true);
    }
    mel.tmp = (u32::from(mel.tmp) << mel.remaining_bits) as u8;
    let mel_mask = ((0xffu32 << mel.remaining_bits) & 0xff) as u8;
    let vlc_mask = if vlc.used_bits == 0 {
        0
    } else {
        ((1 << vlc.used_bits) - 1) as u8
    };
    if (mel_mask | vlc_mask) == 0 {
        return;
    }
    let fused = mel.tmp | vlc.tmp;
    let fused_ok =
        ((((fused ^ mel.tmp) & mel_mask) | ((fused ^ vlc.tmp) & vlc_mask)) == 0) && fused != 0xff;
    if fused_ok && vlc.pos > 1 {
        if mel.pos >= HT_MEL_SIZE {
            mel.failed = 1;
            return;
        }
        store_u8(out, HT_MEL_OFFSET + mel.pos, fused);
        mel.pos += 1;
    } else {
        if mel.pos >= HT_MEL_SIZE || vlc.pos >= HT_VLC_SIZE {
            mel.failed = 1;
            vlc.failed = 1;
            return;
        }
        store_u8(out, HT_MEL_OFFSET + mel.pos, mel.tmp);
        mel.pos += 1;
        let write_index = HT_VLC_SIZE - 1 - vlc.pos;
        store_u8(out, HT_VLC_OFFSET + write_index, vlc.tmp);
        vlc.pos += 1;
    }
}

fn encode_ht_code_block_impl_with_max_and_assembly(
    coefficients: *const i32,
    out: *mut u8,
    params: J2kHtEncodeParams,
    vlc_table0: *const u16,
    vlc_table1: *const u16,
    uvlc_table: *const u8,
    status: *mut J2kHtEncodeStatus,
    max_magnitude: u32,
    cleanup_only: bool,
    assemble_final: bool,
    fixed_64: bool,
) {
    set_status(status, ENCODE_STATUS_FAIL, 0, 0, 0, 0);
    if params.width == 0
        || params.height == 0
        || params.coefficient_stride < params.width
        || params.width > HT_MAX_CODEBLOCK_WIDTH
        || params.height > HT_MAX_CODEBLOCK_SAMPLES / params.width
        || params.total_bitplanes == 0
        || params.total_bitplanes > HT_MAX_BITPLANES
        || params.output_capacity < HT_MS_SIZE + HT_MEL_SIZE + HT_VLC_SIZE
    {
        set_status(status, ENCODE_STATUS_UNSUPPORTED, 1, 0, 0, 0);
        return;
    }
    if fixed_64 && (params.width != 64 || params.height != 64 || params.coefficient_stride != 64) {
        set_status(status, ENCODE_STATUS_UNSUPPORTED, 1, 0, 0, 0);
        return;
    }
    if cleanup_only {
        if params.target_coding_passes != 1 {
            set_status(status, ENCODE_STATUS_UNSUPPORTED, 5, 0, 0, 0);
            return;
        }
    } else if params.target_coding_passes == 0 || params.target_coding_passes > 3 {
        set_status(status, ENCODE_STATUS_UNSUPPORTED, 5, 0, 0, 0);
        return;
    }

    if max_magnitude == 0 {
        set_status(status, ENCODE_STATUS_OK, 0, 0, 0, params.total_bitplanes);
        return;
    }
    let block_bitplanes = 32 - leading_zeros32(max_magnitude);
    if block_bitplanes > params.total_bitplanes {
        set_status(status, ENCODE_STATUS_FAIL, 2, 0, 0, 0);
        return;
    }

    let mut significant_count = 0;
    if !cleanup_only
        && params.target_coding_passes > 1
        && params.total_bitplanes < params.target_coding_passes
    {
        set_status(status, ENCODE_STATUS_UNSUPPORTED, 5, 0, 0, 0);
        return;
    }
    if !cleanup_only && params.target_coding_passes == 2 {
        let mut y = 0;
        while y < params.height {
            let mut x = 0;
            while x < params.width {
                let magnitude =
                    unsigned_magnitude(load_i32(coefficients, y * params.coefficient_stride + x));
                if magnitude != 0 && (magnitude < 3 || (magnitude & 1) == 0) {
                    set_status(status, ENCODE_STATUS_UNSUPPORTED, 6, 0, 0, 0);
                    return;
                }
                x += 1;
            }
            y += 1;
        }
    } else if !cleanup_only && params.target_coding_passes == 3 {
        let mut y = 0;
        while y < params.height {
            let mut x = 0;
            while x < params.width {
                let magnitude =
                    unsigned_magnitude(load_i32(coefficients, y * params.coefficient_stride + x));
                if magnitude != 0 && magnitude != 3 {
                    significant_count += 1;
                    if magnitude < 5 || (magnitude & 1) == 0 {
                        set_status(status, ENCODE_STATUS_UNSUPPORTED, 6, 0, 0, 0);
                        return;
                    }
                }
                x += 1;
            }
            y += 1;
        }
    }

    let width = if fixed_64 { 64 } else { params.width };
    let height = if fixed_64 { 64 } else { params.height };
    let coefficient_stride = if fixed_64 {
        64
    } else {
        params.coefficient_stride
    };
    let pass_span = if cleanup_only {
        1
    } else {
        params.target_coding_passes
    };
    let missing_msbs = params.total_bitplanes - pass_span;
    let p = 30 - missing_msbs;

    let mut mel = MelEncoder {
        pos: 0,
        remaining_bits: 8,
        tmp: 0,
        run: 0,
        k: 0,
        threshold: 1,
        failed: 0,
    };
    let mut vlc = VlcEncoder {
        pos: 1,
        used_bits: 4,
        tmp: 0x0f,
        last_greater_than_8f: 1,
        failed: 0,
    };
    let mut ms = MagSgnEncoder {
        pos: 0,
        max_bits: 8,
        used_bits: 0,
        tmp: 0,
        failed: 0,
    };
    mel_init(&mut mel);
    vlc_init(&mut vlc, out);
    ms_init(&mut ms);

    let mut e_val = [0u8; HT_SIGPROP_SCRATCH];
    let mut cx_val = [0u8; HT_SIGPROP_SCRATCH];
    let cleanup_entries = if fixed_64 {
        34
    } else {
        cleanup_scratch_entries(width)
    };
    let mut clear = 0;
    while clear < cleanup_entries {
        e_val[clear as usize] = 0;
        cx_val[clear as usize] = 0;
        clear += 1;
    }

    let mut e_qmax = [0i32; 2];
    let mut e_q = [0i32; 8];
    let mut rho = [0i32; 2];
    let mut s = [0u32; 8];
    clear_quad_state(&mut rho, &mut e_q, &mut e_qmax, &mut s);

    let mut c_q0 = 0;
    let mut sp = 0;
    let mut x = 0;
    while x < width {
        encode_first_quad_pair(
            coefficients,
            coefficient_stride,
            width,
            height,
            params.total_bitplanes,
            p,
            &mut sp,
            x,
            &mut e_val,
            &mut cx_val,
            &mut c_q0,
            &mut rho,
            &mut e_q,
            &mut e_qmax,
            &mut s,
            &mut mel,
            &mut vlc,
            &mut ms,
            out,
            vlc_table0,
            uvlc_table,
            fixed_64,
        );
        x += 4;
    }

    let e_val_sentinel = if fixed_64 { 33 } else { (width + 1) / 2 + 1 };
    if e_val_sentinel < HT_SIGPROP_SCRATCH as u32 {
        e_val[e_val_sentinel as usize] = 0;
    }

    let mut y = 2;
    while y < height {
        let mut lep = 0;
        let mut max_e = i32::from(max_u8(e_val[lep as usize], e_val[(lep + 1) as usize])) - 1;
        e_val[lep as usize] = 0;
        let mut lcxp = 0;
        c_q0 = u32::from(cx_val[lcxp as usize]) + (u32::from(cx_val[(lcxp + 1) as usize]) << 2);
        cx_val[lcxp as usize] = 0;
        sp = y * coefficient_stride;
        x = 0;
        while x < width {
            encode_non_initial_quad_pair(
                coefficients,
                coefficient_stride,
                width,
                height,
                y,
                params.total_bitplanes,
                p,
                &mut sp,
                x,
                &mut e_val,
                &mut cx_val,
                &mut lep,
                &mut lcxp,
                &mut max_e,
                &mut c_q0,
                &mut rho,
                &mut e_q,
                &mut e_qmax,
                &mut s,
                &mut mel,
                &mut vlc,
                &mut ms,
                out,
                vlc_table1,
                uvlc_table,
                fixed_64,
            );
            x += 4;
        }
        y += 2;
    }

    terminate_mel_vlc(&mut mel, &mut vlc, out);
    ms_terminate(&mut ms, out);
    if mel.failed != 0 || vlc.failed != 0 || ms.failed != 0 {
        set_status(status, ENCODE_STATUS_FAIL, 3, 0, 0, 0);
        return;
    }

    let ms_len = ms.pos;
    let mel_len = mel.pos;
    let vlc_len = vlc.pos;
    let cleanup_len = ms_len + mel_len + vlc_len;
    let mut sigprop_len = 0;
    let mut magref_len = 0;
    let mut refinement_len = 0;
    if !cleanup_only && params.target_coding_passes == 2 {
        refinement_len = 1;
    } else if !cleanup_only && params.target_coding_passes == 3 {
        let sample_count = width * height;
        let mut actual_sigprop_len = 0;
        if write_sigprop_segment(
            coefficients,
            coefficient_stride,
            width,
            height,
            core::ptr::null_mut(),
            u32::MAX,
            &mut actual_sigprop_len,
        ) == 0
        {
            set_status(status, ENCODE_STATUS_UNSUPPORTED, 6, 0, 0, 0);
            return;
        }
        sigprop_len = max_u32((sample_count + 7) >> 3, actual_sigprop_len);
        magref_len = (significant_count + 6) / 7;
        refinement_len = sigprop_len + magref_len;
    }

    let total_len = cleanup_len + refinement_len;
    if cleanup_len < 2 || total_len > params.output_capacity {
        set_status(status, ENCODE_STATUS_FAIL, 4, 0, 0, 0);
        return;
    }

    if assemble_final {
        let mut idx = 0;
        while idx < mel_len {
            let value = load_u8(out, HT_MEL_OFFSET + idx);
            store_u8(out, ms_len + idx, value);
            idx += 1;
        }
        let vlc_start = HT_VLC_SIZE - vlc_len;
        idx = 0;
        while idx < vlc_len {
            let value = load_u8(out, HT_VLC_OFFSET + vlc_start + idx);
            store_u8(out, ms_len + mel_len + idx, value);
            idx += 1;
        }
        let locator_bytes = mel_len + vlc_len;
        let cleanup_last = cleanup_len - 1;
        let cleanup_prev = cleanup_len - 2;
        store_u8(out, cleanup_last, (locator_bytes >> 4) as u8);
        let prev = load_u8(out, cleanup_prev);
        store_u8(
            out,
            cleanup_prev,
            (u32::from(prev) & 0xf0 | (locator_bytes & 0x0f)) as u8,
        );
        if !cleanup_only && refinement_len != 0 {
            idx = 0;
            while idx < refinement_len {
                store_u8(out, cleanup_len + idx, 0);
                idx += 1;
            }
        }
        if !cleanup_only && params.target_coding_passes == 3 {
            let mut actual_sigprop_len = 0;
            if write_sigprop_segment(
                coefficients,
                coefficient_stride,
                width,
                height,
                unsafe { out.add(cleanup_len as usize) },
                sigprop_len,
                &mut actual_sigprop_len,
            ) == 0
            {
                set_status(status, ENCODE_STATUS_UNSUPPORTED, 6, 0, 0, 0);
                return;
            }
        }
        if !cleanup_only
            && params.target_coding_passes == 3
            && write_magref_segment(
                coefficients,
                coefficient_stride,
                width,
                height,
                unsafe { out.add((cleanup_len + sigprop_len) as usize) },
                magref_len,
                significant_count,
            ) == 0
        {
            set_status(status, ENCODE_STATUS_UNSUPPORTED, 7, 0, 0, 0);
            return;
        }
    }

    set_status_with_segments(
        status,
        ENCODE_STATUS_OK,
        0,
        total_len,
        pass_span,
        missing_msbs,
        cleanup_len,
        refinement_len,
        if assemble_final {
            0
        } else {
            pack_compact_assembly_lengths(mel_len, vlc_len)
        },
    );
}

fn max_magnitude_serial(
    coefficients: *const i32,
    width: u32,
    height: u32,
    coefficient_stride: u32,
) -> u32 {
    let mut max_magnitude = 0;
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            max_magnitude = max_u32(
                max_magnitude,
                unsigned_magnitude(load_i32(coefficients, y * coefficient_stride + x)),
            );
            x += 1;
        }
        y += 1;
    }
    max_magnitude
}

#[inline(always)]
fn params_from_job(job: J2kHtEncodeJob) -> J2kHtEncodeParams {
    J2kHtEncodeParams {
        width: job.width,
        height: job.height,
        coefficient_stride: job.coefficient_stride,
        total_bitplanes: job.total_bitplanes,
        output_capacity: job.output_capacity,
        target_coding_passes: job.target_coding_passes,
    }
}

#[inline(always)]
fn params_from_multi_job(job: J2kHtEncodeMultiInputJob) -> J2kHtEncodeParams {
    J2kHtEncodeParams {
        width: job.width,
        height: job.height,
        coefficient_stride: job.coefficient_stride,
        total_bitplanes: job.total_bitplanes,
        output_capacity: job.output_capacity,
        target_coding_passes: job.target_coding_passes,
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn j2k_htj2k_encode_codeblocks(
        coefficients: *const i32,
        out: *mut u8,
        jobs: *const J2kHtEncodeJob,
        vlc_table0: *const u16,
        vlc_table1: *const u16,
        uvlc_table: *const u8,
        statuses: *mut J2kHtEncodeStatus,
        job_count: u64,
    ) {
        let job_idx = thread::blockIdx_x();
        if u64::from(job_idx) >= job_count || thread::threadIdx_x() != 0 {
            return;
        }
        let job = load_job(jobs, job_idx);
        let params = params_from_job(job);
        let codeblock_coefficients = coefficients.add(job.coefficient_offset as usize);
        let max_magnitude = max_magnitude_serial(
            codeblock_coefficients,
            params.width,
            params.height,
            params.coefficient_stride,
        );
        encode_ht_code_block_impl_with_max_and_assembly(
            codeblock_coefficients,
            out.add(job.output_offset as usize),
            params,
            vlc_table0,
            vlc_table1,
            uvlc_table,
            statuses.add(job_idx as usize),
            max_magnitude,
            false,
            params.target_coding_passes != 1,
            false,
        );
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_encode_codeblocks_multi_input(
        out: *mut u8,
        jobs: *const J2kHtEncodeMultiInputJob,
        vlc_table0: *const u16,
        vlc_table1: *const u16,
        uvlc_table: *const u8,
        statuses: *mut J2kHtEncodeStatus,
        job_count: u64,
    ) {
        let job_idx = thread::blockIdx_x();
        if u64::from(job_idx) >= job_count || thread::threadIdx_x() != 0 {
            return;
        }
        let job = load_job(jobs, job_idx);
        let params = params_from_multi_job(job);
        let coefficients = job.coefficient_ptr as usize as *const i32;
        let codeblock_coefficients = coefficients.add(job.coefficient_offset as usize);
        let max_magnitude = max_magnitude_serial(
            codeblock_coefficients,
            params.width,
            params.height,
            params.coefficient_stride,
        );
        encode_ht_code_block_impl_with_max_and_assembly(
            codeblock_coefficients,
            out.add(job.output_offset as usize),
            params,
            vlc_table0,
            vlc_table1,
            uvlc_table,
            statuses.add(job_idx as usize),
            max_magnitude,
            false,
            params.target_coding_passes != 1,
            false,
        );
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_encode_codeblocks_multi_input_cleanup(
        out: *mut u8,
        jobs: *const J2kHtEncodeMultiInputJob,
        vlc_table0: *const u16,
        vlc_table1: *const u16,
        uvlc_table: *const u8,
        statuses: *mut J2kHtEncodeStatus,
        job_count: u64,
    ) {
        let job_idx = thread::blockIdx_x();
        if u64::from(job_idx) >= job_count || thread::threadIdx_x() != 0 {
            return;
        }
        let job = load_job(jobs, job_idx);
        let params = params_from_multi_job(job);
        let coefficients = job.coefficient_ptr as usize as *const i32;
        let codeblock_coefficients = coefficients.add(job.coefficient_offset as usize);
        let fixed_64 = params.width == 64 && params.height == 64 && params.coefficient_stride == 64;
        let max_magnitude = if fixed_64 {
            max_magnitude_serial(codeblock_coefficients, 64, 64, 64)
        } else {
            max_magnitude_serial(
                codeblock_coefficients,
                params.width,
                params.height,
                params.coefficient_stride,
            )
        };
        encode_ht_code_block_impl_with_max_and_assembly(
            codeblock_coefficients,
            out.add(job.output_offset as usize),
            params,
            vlc_table0,
            vlc_table1,
            uvlc_table,
            statuses.add(job_idx as usize),
            max_magnitude,
            true,
            false,
            fixed_64,
        );
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_encode_codeblocks_multi_input_cleanup_64(
        out: *mut u8,
        jobs: *const J2kHtEncodeMultiInputJob,
        vlc_table0: *const u16,
        vlc_table1: *const u16,
        uvlc_table: *const u8,
        statuses: *mut J2kHtEncodeStatus,
        job_count: u64,
    ) {
        let job_idx = thread::blockIdx_x();
        if u64::from(job_idx) >= job_count || thread::threadIdx_x() != 0 {
            return;
        }
        let job = load_job(jobs, job_idx);
        let params = params_from_multi_job(job);
        let coefficients = job.coefficient_ptr as usize as *const i32;
        let codeblock_coefficients = coefficients.add(job.coefficient_offset as usize);
        let max_magnitude = max_magnitude_serial(codeblock_coefficients, 64, 64, 64);
        encode_ht_code_block_impl_with_max_and_assembly(
            codeblock_coefficients,
            out.add(job.output_offset as usize),
            params,
            vlc_table0,
            vlc_table1,
            uvlc_table,
            statuses.add(job_idx as usize),
            max_magnitude,
            true,
            false,
            true,
        );
    }
}

fn main() {}
