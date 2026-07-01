use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

const JPEG_STATUS_OK: u32 = 0;
const JPEG_STATUS_TRUNCATED: u32 = 1;
const JPEG_STATUS_HUFFMAN: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kJpegHuffmanTable {
    max_code: [i32; 17],
    val_offset: [i32; 17],
    values: [u8; 256],
    values_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct J2kJpeg420Params {
    width: u32,
    height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    entropy_len: u32,
    checkpoint_count: u32,
    out_stride: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kJpegEntropyCheckpoint {
    mcu_index: u32,
    entropy_pos: u32,
    bit_acc: u64,
    bit_count: u32,
    y_prev_dc: i32,
    cb_prev_dc: i32,
    cr_prev_dc: i32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kJpegDecodeStatus {
    code: u32,
    detail: u32,
    position: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct J2kJpegEntropyChunkParams {
    entropy_len: u32,
    entropy_bits: u32,
    subsequence_bits: u32,
    subsequence_count: u32,
    sequence_len: u32,
    max_overflow_subsequences: u32,
    reserved0: u32,
    reserved1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kJpegEntropySyncState {
    code: u32,
    start_bit: u32,
    end_bit: u32,
    bit_pos: u32,
    symbol_count: u32,
    block_phase: u32,
    zigzag_index: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kJpegEntropyOverflowState {
    code: u32,
    from_subsequence: u32,
    to_subsequence: u32,
    overflow_bits: u32,
    synchronized: u32,
    reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kJpegBitReader {
    pos: u32,
    acc: u64,
    bits: u32,
}

const J2K_JPEG_ZIGZAG: [u8; 64] = j2k_codec_math::jpeg::ZIGZAG;

const JPEG_CONST_BITS: i32 = j2k_codec_math::jpeg::idct::CONST_BITS as i32;
const JPEG_PASS1_BITS: i32 = j2k_codec_math::jpeg::idct::PASS1_BITS as i32;
const JPEG_FIX_0_298631336: i32 = j2k_codec_math::jpeg::idct::FIX_0_298631336;
const JPEG_FIX_0_390180644: i32 = j2k_codec_math::jpeg::idct::FIX_0_390180644;
const JPEG_FIX_0_541196100: i32 = j2k_codec_math::jpeg::idct::FIX_0_541196100;
const JPEG_FIX_0_765366865: i32 = j2k_codec_math::jpeg::idct::FIX_0_765366865;
const JPEG_FIX_0_899976223: i32 = j2k_codec_math::jpeg::idct::FIX_0_899976223;
const JPEG_FIX_1_175875602: i32 = j2k_codec_math::jpeg::idct::FIX_1_175875602;
const JPEG_FIX_1_501321110: i32 = j2k_codec_math::jpeg::idct::FIX_1_501321110;
const JPEG_FIX_1_847759065: i32 = j2k_codec_math::jpeg::idct::FIX_1_847759065;
const JPEG_FIX_1_961570560: i32 = j2k_codec_math::jpeg::idct::FIX_1_961570560;
const JPEG_FIX_2_053119869: i32 = j2k_codec_math::jpeg::idct::FIX_2_053119869;
const JPEG_FIX_2_562915447: i32 = j2k_codec_math::jpeg::idct::FIX_2_562915447;
const JPEG_FIX_3_072711026: i32 = j2k_codec_math::jpeg::idct::FIX_3_072711026;

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
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn load_u16(ptr: *const u16, index: u32) -> u16 {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn store_u8(ptr: *mut u8, index: u32, value: u8) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}

#[inline(always)]
fn load_checkpoint(
    ptr: *const J2kJpegEntropyCheckpoint,
    index: u32,
) -> J2kJpegEntropyCheckpoint {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn load_state(ptr: *const J2kJpegEntropySyncState, index: u32) -> J2kJpegEntropySyncState {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn store_state(
    ptr: *mut J2kJpegEntropySyncState,
    index: u32,
    value: J2kJpegEntropySyncState,
) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}

#[inline(always)]
fn store_overflow(
    ptr: *mut J2kJpegEntropyOverflowState,
    index: u32,
    value: J2kJpegEntropyOverflowState,
) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}

#[inline(always)]
fn store_decode_status(ptr: *mut J2kJpegDecodeStatus, index: u32, value: J2kJpegDecodeStatus) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}

#[inline(always)]
fn set_error(status: &mut J2kJpegDecodeStatus, code: u32, detail: u32, position: u32) {
    status.code = code;
    status.detail = detail;
    status.position = position;
}

#[inline(always)]
fn refill_one(reader: &mut J2kJpegBitReader, entropy: *const u8, entropy_len: u32) -> bool {
    if reader.pos >= entropy_len {
        return false;
    }
    let shift = 64 - 8 - reader.bits;
    reader.acc |= (load_u8(entropy, reader.pos) as u64) << shift;
    reader.pos += 1;
    reader.bits += 8;
    true
}

#[inline(always)]
fn ensure_bits(
    reader: &mut J2kJpegBitReader,
    entropy: *const u8,
    entropy_len: u32,
    wanted: u32,
) -> bool {
    while reader.bits < wanted {
        if !refill_one(reader, entropy, entropy_len) {
            return false;
        }
    }
    true
}

#[inline(always)]
fn ensure_bits_padded(
    reader: &mut J2kJpegBitReader,
    entropy: *const u8,
    entropy_len: u32,
    wanted: u32,
) {
    while reader.bits < wanted {
        if !refill_one(reader, entropy, entropy_len) {
            reader.acc |= 1u64 << (63 - reader.bits);
            reader.bits += 1;
        }
    }
}

#[inline(always)]
fn peek_bits(reader: J2kJpegBitReader, count: u32) -> u32 {
    if count == 0 {
        0
    } else {
        (reader.acc >> (64 - count)) as u32
    }
}

#[inline(always)]
fn consume_bits(reader: &mut J2kJpegBitReader, count: u32) {
    reader.acc <<= count;
    reader.bits -= count;
}

#[inline(always)]
fn bit_reader_at_bit(entropy: *const u8, entropy_len: u32, bit_pos: u32) -> J2kJpegBitReader {
    let mut reader = J2kJpegBitReader {
        pos: bit_pos / 8,
        acc: 0,
        bits: 0,
    };
    let skip = bit_pos & 7;
    if skip != 0 && reader.pos < entropy_len {
        reader.acc = (load_u8(entropy, reader.pos) as u64) << 56;
        reader.pos += 1;
        reader.bits = 8;
        consume_bits(&mut reader, skip);
    }
    reader
}

#[inline(always)]
fn zigzag(k: u32) -> u32 {
    J2K_JPEG_ZIGZAG[k as usize] as u32
}

#[inline(always)]
fn real_bits_consumed(
    reader: J2kJpegBitReader,
    before_pos: u32,
    before_bits: u32,
    consumed: &mut u32,
) -> bool {
    let loaded_bits = (reader.pos - before_pos) * 8 + before_bits;
    if reader.bits >= loaded_bits {
        *consumed = 0;
        return false;
    }
    *consumed = loaded_bits - reader.bits;
    true
}

#[inline(always)]
fn receive_extend(
    reader: &mut J2kJpegBitReader,
    entropy: *const u8,
    entropy_len: u32,
    ssss: u32,
    status: &mut J2kJpegDecodeStatus,
    out: &mut i32,
) -> bool {
    if ssss == 0 {
        *out = 0;
        return true;
    }
    if !ensure_bits(reader, entropy, entropy_len, ssss) {
        set_error(status, JPEG_STATUS_TRUNCATED, ssss, reader.pos);
        return false;
    }
    let value = peek_bits(*reader, ssss) as i32;
    consume_bits(reader, ssss);
    let threshold = 1i32 << (ssss - 1);
    *out = if value < threshold {
        value + ((-1i32) << ssss) + 1
    } else {
        value
    };
    true
}

#[inline(always)]
fn decode_symbol(
    reader: &mut J2kJpegBitReader,
    entropy: *const u8,
    entropy_len: u32,
    table: *const J2kJpegHuffmanTable,
    status: &mut J2kJpegDecodeStatus,
    symbol: &mut u8,
) -> bool {
    ensure_bits_padded(reader, entropy, entropy_len, 16);
    let code16 = peek_bits(*reader, 16) as i32;
    let mut len = 1;
    while len <= 16 {
        let max_code = unsafe { (*table).max_code[len as usize] };
        if max_code >= 0 {
            let code = code16 >> (16 - len);
            if code <= max_code {
                let val_offset = unsafe { (*table).val_offset[len as usize] };
                let idx = code + val_offset;
                let values_len = unsafe { (*table).values_len };
                if idx < 0 || idx as u32 >= values_len {
                    set_error(status, JPEG_STATUS_HUFFMAN, len, reader.pos);
                    return false;
                }
                consume_bits(reader, len);
                *symbol = unsafe { (*table).values[idx as usize] };
                return true;
            }
        }
        len += 1;
    }
    set_error(status, JPEG_STATUS_HUFFMAN, 16, reader.pos);
    false
}

#[inline(always)]
fn decode_symbol_real(
    reader: &mut J2kJpegBitReader,
    entropy: *const u8,
    entropy_len: u32,
    table: *const J2kJpegHuffmanTable,
    status: &mut J2kJpegDecodeStatus,
    symbol: &mut u8,
) -> bool {
    let mut len = 1;
    while len <= 16 {
        if !ensure_bits(reader, entropy, entropy_len, len) {
            set_error(status, JPEG_STATUS_TRUNCATED, len, reader.pos);
            return false;
        }
        let max_code = unsafe { (*table).max_code[len as usize] };
        if max_code >= 0 {
            let code = peek_bits(*reader, len) as i32;
            if code <= max_code {
                let val_offset = unsafe { (*table).val_offset[len as usize] };
                let idx = code + val_offset;
                let values_len = unsafe { (*table).values_len };
                if idx < 0 || idx as u32 >= values_len {
                    set_error(status, JPEG_STATUS_HUFFMAN, len, reader.pos);
                    return false;
                }
                consume_bits(reader, len);
                *symbol = unsafe { (*table).values[idx as usize] };
                return true;
            }
        }
        len += 1;
    }
    set_error(status, JPEG_STATUS_HUFFMAN, 16, reader.pos);
    false
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn decode_block(
    reader: &mut J2kJpegBitReader,
    entropy: *const u8,
    entropy_len: u32,
    dc_table: *const J2kJpegHuffmanTable,
    ac_table: *const J2kJpegHuffmanTable,
    quant: *const u16,
    prev_dc: &mut i32,
    status: &mut J2kJpegDecodeStatus,
    coeffs: &mut [i32; 64],
) -> bool {
    let mut i = 0;
    while i < 64 {
        coeffs[i as usize] = 0;
        i += 1;
    }

    let mut ssss = 0;
    if !decode_symbol(reader, entropy, entropy_len, dc_table, status, &mut ssss) {
        return false;
    }
    if ssss > 15 {
        set_error(status, JPEG_STATUS_HUFFMAN, ssss as u32, reader.pos);
        return false;
    }
    let mut diff = 0;
    if !receive_extend(
        reader,
        entropy,
        entropy_len,
        ssss as u32,
        status,
        &mut diff,
    ) {
        return false;
    }
    *prev_dc += diff;
    coeffs[0] = *prev_dc * load_u16(quant, 0) as i32;

    let mut k = 1;
    while k < 64 {
        let mut packed = 0;
        if !decode_symbol(reader, entropy, entropy_len, ac_table, status, &mut packed) {
            return false;
        }
        let run = (packed >> 4) as u32;
        ssss = packed & 0x0f;
        if ssss == 0 {
            if run == 15 {
                k += 16;
                continue;
            }
            break;
        }
        k += run;
        if k >= 64 {
            set_error(status, JPEG_STATUS_HUFFMAN, k, reader.pos);
            return false;
        }
        let mut value = 0;
        if !receive_extend(
            reader,
            entropy,
            entropy_len,
            ssss as u32,
            status,
            &mut value,
        ) {
            return false;
        }
        coeffs[zigzag(k) as usize] = value * load_u16(quant, k) as i32;
        k += 1;
    }
    true
}

#[inline(always)]
fn clamp_i32(value: i32) -> u8 {
    if value < 0 {
        0
    } else if value > 255 {
        255
    } else {
        value as u8
    }
}

#[inline(always)]
fn descale(value: i32, shift: i32) -> i32 {
    value >> shift
}

#[inline(always)]
fn descale_and_clamp(value: i32, shift: i32) -> u8 {
    clamp_i32((value >> shift) + 128)
}

#[inline(always)]
fn idct_column(input: &[i32; 64], work: &mut [i32; 64], col: u32) {
    let p0 = input[col as usize];
    let p1 = input[(col + 8) as usize];
    let p2 = input[(col + 16) as usize];
    let p3 = input[(col + 24) as usize];
    let p4 = input[(col + 32) as usize];
    let p5 = input[(col + 40) as usize];
    let p6 = input[(col + 48) as usize];
    let p7 = input[(col + 56) as usize];

    if p1 == 0 && p2 == 0 && p3 == 0 && p4 == 0 && p5 == 0 && p6 == 0 && p7 == 0 {
        let dc = p0 << JPEG_PASS1_BITS;
        work[col as usize] = dc;
        work[(col + 8) as usize] = dc;
        work[(col + 16) as usize] = dc;
        work[(col + 24) as usize] = dc;
        work[(col + 32) as usize] = dc;
        work[(col + 40) as usize] = dc;
        work[(col + 48) as usize] = dc;
        work[(col + 56) as usize] = dc;
        return;
    }

    let mut z2 = p2;
    let mut z3 = p6;
    let mut z1 = (z2 + z3) * JPEG_FIX_0_541196100;
    let tmp2 = z1 - z3 * JPEG_FIX_1_847759065;
    let tmp3 = z1 + z2 * JPEG_FIX_0_765366865;

    z2 = p0;
    z3 = p4;
    let tmp0 = (z2 + z3) << JPEG_CONST_BITS;
    let tmp1 = (z2 - z3) << JPEG_CONST_BITS;

    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    let mut tmp0 = p7;
    let mut tmp1 = p5;
    let mut tmp2 = p3;
    let mut tmp3 = p1;

    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    let mut z4 = tmp1 + tmp3;
    let z5 = (z3 + z4) * JPEG_FIX_1_175875602;

    tmp0 *= JPEG_FIX_0_298631336;
    tmp1 *= JPEG_FIX_2_053119869;
    tmp2 *= JPEG_FIX_3_072711026;
    tmp3 *= JPEG_FIX_1_501321110;
    z1 *= -JPEG_FIX_0_899976223;
    z2 *= -JPEG_FIX_2_562915447;
    z3 *= -JPEG_FIX_1_961570560;
    z4 *= -JPEG_FIX_0_390180644;

    z3 += z5;
    z4 += z5;

    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;

    let shift = JPEG_CONST_BITS - JPEG_PASS1_BITS;
    let rounding = 1 << (shift - 1);
    work[col as usize] = descale(tmp10 + tmp3 + rounding, shift);
    work[(col + 56) as usize] = descale(tmp10 - tmp3 + rounding, shift);
    work[(col + 8) as usize] = descale(tmp11 + tmp2 + rounding, shift);
    work[(col + 48) as usize] = descale(tmp11 - tmp2 + rounding, shift);
    work[(col + 16) as usize] = descale(tmp12 + tmp1 + rounding, shift);
    work[(col + 40) as usize] = descale(tmp12 - tmp1 + rounding, shift);
    work[(col + 24) as usize] = descale(tmp13 + tmp0 + rounding, shift);
    work[(col + 32) as usize] = descale(tmp13 - tmp0 + rounding, shift);
}

#[inline(always)]
fn idct_row(work: &[i32; 64], pixels: &mut [u8; 64], row: u32) {
    let base = row * 8;
    let p0 = work[base as usize];
    let p1 = work[(base + 1) as usize];
    let p2 = work[(base + 2) as usize];
    let p3 = work[(base + 3) as usize];
    let p4 = work[(base + 4) as usize];
    let p5 = work[(base + 5) as usize];
    let p6 = work[(base + 6) as usize];
    let p7 = work[(base + 7) as usize];

    let shift = JPEG_CONST_BITS + JPEG_PASS1_BITS + 3;
    let rounding = 1 << (shift - 1);

    if p1 == 0 && p2 == 0 && p3 == 0 && p4 == 0 && p5 == 0 && p6 == 0 && p7 == 0 {
        let dc_shift = JPEG_PASS1_BITS + 3;
        let rounding_dc = 1 << (dc_shift - 1);
        let pixel = descale_and_clamp(p0 + rounding_dc, dc_shift);
        let mut i = 0;
        while i < 8 {
            pixels[(base + i) as usize] = pixel;
            i += 1;
        }
        return;
    }

    let mut z2 = p2;
    let mut z3 = p6;
    let mut z1 = (z2 + z3) * JPEG_FIX_0_541196100;
    let tmp2 = z1 - z3 * JPEG_FIX_1_847759065;
    let tmp3 = z1 + z2 * JPEG_FIX_0_765366865;

    let tmp0 = (p0 + p4) << JPEG_CONST_BITS;
    let tmp1 = (p0 - p4) << JPEG_CONST_BITS;

    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    let mut tmp0 = p7;
    let mut tmp1 = p5;
    let mut tmp2 = p3;
    let mut tmp3 = p1;

    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    let mut z4 = tmp1 + tmp3;
    let z5 = (z3 + z4) * JPEG_FIX_1_175875602;

    tmp0 *= JPEG_FIX_0_298631336;
    tmp1 *= JPEG_FIX_2_053119869;
    tmp2 *= JPEG_FIX_3_072711026;
    tmp3 *= JPEG_FIX_1_501321110;
    z1 *= -JPEG_FIX_0_899976223;
    z2 *= -JPEG_FIX_2_562915447;
    z3 *= -JPEG_FIX_1_961570560;
    z4 *= -JPEG_FIX_0_390180644;

    z3 += z5;
    z4 += z5;

    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;

    pixels[base as usize] = descale_and_clamp(tmp10 + tmp3 + rounding, shift);
    pixels[(base + 7) as usize] = descale_and_clamp(tmp10 - tmp3 + rounding, shift);
    pixels[(base + 1) as usize] = descale_and_clamp(tmp11 + tmp2 + rounding, shift);
    pixels[(base + 6) as usize] = descale_and_clamp(tmp11 - tmp2 + rounding, shift);
    pixels[(base + 2) as usize] = descale_and_clamp(tmp12 + tmp1 + rounding, shift);
    pixels[(base + 5) as usize] = descale_and_clamp(tmp12 - tmp1 + rounding, shift);
    pixels[(base + 3) as usize] = descale_and_clamp(tmp13 + tmp0 + rounding, shift);
    pixels[(base + 4) as usize] = descale_and_clamp(tmp13 - tmp0 + rounding, shift);
}

#[inline(always)]
fn idct_islow(coeffs: &[i32; 64], pixels: &mut [u8; 64]) {
    let mut work = [0i32; 64];
    let mut col = 0;
    while col < 8 {
        idct_column(coeffs, &mut work, col);
        col += 1;
    }
    let mut row = 0;
    while row < 8 {
        idct_row(&work, pixels, row);
        row += 1;
    }
}

#[inline(always)]
fn h2v2_sample(
    block: &[u8; 64],
    chroma_cols: u32,
    chroma_rows: u32,
    output_x: u32,
    chroma_y: u32,
    bottom: bool,
) -> u8 {
    let n = if chroma_cols == 0 { 1 } else { chroma_cols };
    let curr_y = if chroma_y < chroma_rows {
        chroma_y
    } else {
        chroma_rows - 1
    };
    let near_y = if bottom {
        if curr_y + 1 < chroma_rows {
            curr_y + 1
        } else {
            chroma_rows - 1
        }
    } else if curr_y == 0 {
        0
    } else {
        curr_y - 1
    };
    let sample = min_u32(output_x / 2, n - 1);
    let curr = block[(curr_y * 8 + sample) as usize] as u32;
    let near = block[(near_y * 8 + sample) as usize] as u32;
    let this_sum = 3 * curr + near;
    if n == 1 {
        return ((4 * this_sum + 8) >> 4) as u8;
    }
    if output_x == 0 {
        return ((this_sum * 4 + 8) >> 4) as u8;
    }
    if output_x == n * 2 - 1 {
        return ((this_sum * 4 + 7) >> 4) as u8;
    }
    if (output_x & 1) == 0 {
        let last_curr = block[(curr_y * 8 + sample - 1) as usize] as u32;
        let last_near = block[(near_y * 8 + sample - 1) as usize] as u32;
        let last_sum = 3 * last_curr + last_near;
        return ((this_sum * 3 + last_sum + 8) >> 4) as u8;
    }
    let next_sample = min_u32(sample + 1, n - 1);
    let next_curr = block[(curr_y * 8 + next_sample) as usize] as u32;
    let next_near = block[(near_y * 8 + next_sample) as usize] as u32;
    let next_sum = 3 * next_curr + next_near;
    ((this_sum * 3 + next_sum + 7) >> 4) as u8
}

#[inline(always)]
fn h2v1_sample(block: &[u8; 64], chroma_cols: u32, output_x: u32, chroma_y: u32) -> u8 {
    let n = if chroma_cols == 0 { 1 } else { chroma_cols };
    let row = min_u32(chroma_y, 7);
    let base = row * 8;
    if n == 1 {
        return block[base as usize];
    }
    let sample = min_u32(output_x / 2, n - 1);
    if output_x == 0 {
        return block[base as usize];
    }
    if output_x == n * 2 - 1 {
        return block[(base + n - 1) as usize];
    }
    let curr = block[(base + sample) as usize] as u32;
    if (output_x & 1) == 0 {
        let prev = block[(base + sample - 1) as usize] as u32;
        return ((3 * curr + prev + 2) >> 2) as u8;
    }
    let next = block[(base + sample + 1) as usize] as u32;
    ((3 * curr + next + 2) >> 2) as u8
}

#[inline(always)]
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8, r: &mut u8, g: &mut u8, b: &mut u8) {
    let yy = y as i32;
    let cb_centered = cb as i32 - 128;
    let cr_centered = cr as i32 - 128;
    *r = clamp_i32(yy + ((91881 * cr_centered + (1 << 15)) >> 16));
    *g = clamp_i32(yy - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    *b = clamp_i32(yy + ((116130 * cb_centered + (1 << 15)) >> 16));
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn store_rgb420_mcu(
    out: *mut u8,
    params: J2kJpeg420Params,
    mx: u32,
    my: u32,
    y0: &[u8; 64],
    y1: &[u8; 64],
    y2: &[u8; 64],
    y3: &[u8; 64],
    cb: &[u8; 64],
    cr: &[u8; 64],
) {
    let base_x = mx * 16;
    let base_y = my * 16;
    let remaining_x = if params.width > base_x {
        params.width - base_x
    } else {
        0
    };
    let remaining_y = if params.height > base_y {
        params.height - base_y
    } else {
        0
    };
    let chroma_cols = min_u32(8, (remaining_x + 1) / 2);
    let chroma_rows = min_u32(8, (remaining_y + 1) / 2);
    let mut yy = 0;
    while yy < 16 {
        let py = base_y + yy;
        if py < params.height {
            let mut xx = 0;
            while xx < 16 {
                let px = base_x + xx;
                if px < params.width {
                    let yb = if yy < 8 {
                        if xx < 8 {
                            y0
                        } else {
                            y1
                        }
                    } else if xx < 8 {
                        y2
                    } else {
                        y3
                    };
                    let y_idx = (yy & 7) * 8 + (xx & 7);
                    let chroma_y = min_u32(yy / 2, chroma_rows - 1);
                    let bottom = (yy & 1) != 0;
                    let cbv = h2v2_sample(cb, chroma_cols, chroma_rows, xx, chroma_y, bottom);
                    let crv = h2v2_sample(cr, chroma_cols, chroma_rows, xx, chroma_y, bottom);
                    let dst = py * params.out_stride + px * 3;
                    let mut r = 0;
                    let mut g = 0;
                    let mut b = 0;
                    ycbcr_to_rgb(yb[y_idx as usize], cbv, crv, &mut r, &mut g, &mut b);
                    store_u8(out, dst, r);
                    store_u8(out, dst + 1, g);
                    store_u8(out, dst + 2, b);
                }
                xx += 1;
            }
        }
        yy += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn store_rgb422_mcu(
    out: *mut u8,
    params: J2kJpeg420Params,
    mx: u32,
    my: u32,
    y0: &[u8; 64],
    y1: &[u8; 64],
    cb: &[u8; 64],
    cr: &[u8; 64],
) {
    let base_x = mx * 16;
    let base_y = my * 8;
    let remaining_x = if params.width > base_x {
        params.width - base_x
    } else {
        0
    };
    let remaining_y = if params.height > base_y {
        params.height - base_y
    } else {
        0
    };
    let chroma_cols = min_u32(8, (remaining_x + 1) / 2);
    let chroma_rows = min_u32(8, remaining_y);
    let mut yy = 0;
    while yy < 8 {
        let py = base_y + yy;
        if py < params.height {
            let chroma_y = min_u32(yy, chroma_rows - 1);
            let mut xx = 0;
            while xx < 16 {
                let px = base_x + xx;
                if px < params.width {
                    let yb = if xx < 8 { y0 } else { y1 };
                    let y_idx = yy * 8 + (xx & 7);
                    let cbv = h2v1_sample(cb, chroma_cols, xx, chroma_y);
                    let crv = h2v1_sample(cr, chroma_cols, xx, chroma_y);
                    let dst = py * params.out_stride + px * 3;
                    let mut r = 0;
                    let mut g = 0;
                    let mut b = 0;
                    ycbcr_to_rgb(yb[y_idx as usize], cbv, crv, &mut r, &mut g, &mut b);
                    store_u8(out, dst, r);
                    store_u8(out, dst + 1, g);
                    store_u8(out, dst + 2, b);
                }
                xx += 1;
            }
        }
        yy += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn store_rgb444_mcu(
    out: *mut u8,
    params: J2kJpeg420Params,
    mx: u32,
    my: u32,
    y: &[u8; 64],
    cb: &[u8; 64],
    cr: &[u8; 64],
) {
    let base_x = mx * 8;
    let base_y = my * 8;
    let mut yy = 0;
    while yy < 8 {
        let py = base_y + yy;
        if py < params.height {
            let mut xx = 0;
            while xx < 8 {
                let px = base_x + xx;
                if px < params.width {
                    let idx = yy * 8 + xx;
                    let dst = py * params.out_stride + px * 3;
                    let mut r = 0;
                    let mut g = 0;
                    let mut b = 0;
                    ycbcr_to_rgb(
                        y[idx as usize],
                        cb[idx as usize],
                        cr[idx as usize],
                        &mut r,
                        &mut g,
                        &mut b,
                    );
                    store_u8(out, dst, r);
                    store_u8(out, dst + 1, g);
                    store_u8(out, dst + 2, b);
                }
                xx += 1;
            }
        }
        yy += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn entropy_scan_one_symbol420(
    entropy: *const u8,
    params: J2kJpegEntropyChunkParams,
    y_dc: *const J2kJpegHuffmanTable,
    y_ac: *const J2kJpegHuffmanTable,
    cb_dc: *const J2kJpegHuffmanTable,
    cb_ac: *const J2kJpegHuffmanTable,
    cr_dc: *const J2kJpegHuffmanTable,
    cr_ac: *const J2kJpegHuffmanTable,
    state: &mut J2kJpegEntropySyncState,
    reader: &mut J2kJpegBitReader,
    status: &mut J2kJpegDecodeStatus,
) -> bool {
    let dc = state.zigzag_index == 0;
    let table = if state.block_phase < 4 {
        if dc {
            y_dc
        } else {
            y_ac
        }
    } else if state.block_phase == 4 {
        if dc {
            cb_dc
        } else {
            cb_ac
        }
    } else if dc {
        cr_dc
    } else {
        cr_ac
    };
    let mut symbol = 0;
    let before_pos = reader.pos;
    let before_bits = reader.bits;
    if !decode_symbol_real(
        reader,
        entropy,
        params.entropy_len,
        table,
        status,
        &mut symbol,
    ) {
        if status.code == JPEG_STATUS_HUFFMAN {
            if !ensure_bits(reader, entropy, params.entropy_len, 1) {
                state.bit_pos = params.entropy_bits;
                status.code = JPEG_STATUS_OK;
                return true;
            }
            consume_bits(reader, 1);
            state.bit_pos += 1;
            status.code = JPEG_STATUS_OK;
            status.detail = 0;
            status.position = 0;
            return true;
        }
        if status.code == JPEG_STATUS_TRUNCATED {
            state.bit_pos = params.entropy_bits;
            status.code = JPEG_STATUS_OK;
            return true;
        }
        return false;
    }

    let run = (symbol >> 4) as u32;
    let ssss = (symbol & 0x0f) as u32;
    let coeff_bits = if dc { symbol as u32 } else { ssss };
    if coeff_bits > 15 {
        set_error(status, JPEG_STATUS_HUFFMAN, coeff_bits, reader.pos);
        return false;
    }
    if !ensure_bits(reader, entropy, params.entropy_len, coeff_bits) {
        state.bit_pos = params.entropy_bits;
        return true;
    }
    consume_bits(reader, coeff_bits);
    let mut consumed = 0;
    if !real_bits_consumed(*reader, before_pos, before_bits, &mut consumed) {
        set_error(status, JPEG_STATUS_TRUNCATED, 0, reader.pos);
        return false;
    }
    state.bit_pos += consumed;
    if dc {
        state.zigzag_index = 1;
        state.symbol_count += 1;
        return true;
    }
    if ssss == 0 && run != 15 {
        state.symbol_count += 64 - state.zigzag_index;
        state.zigzag_index = 0;
        state.block_phase = (state.block_phase + 1) % 6;
        return true;
    }
    state.zigzag_index += run + 1;
    state.symbol_count += run + 1;
    if state.zigzag_index >= 64 {
        state.zigzag_index = 0;
        state.block_phase = (state.block_phase + 1) % 6;
    }
    true
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn j2k_jpeg_entropy_sync420(
        entropy: *const u8,
        params: J2kJpegEntropyChunkParams,
        y_dc: *const J2kJpegHuffmanTable,
        y_ac: *const J2kJpegHuffmanTable,
        cb_dc: *const J2kJpegHuffmanTable,
        cb_ac: *const J2kJpegHuffmanTable,
        cr_dc: *const J2kJpegHuffmanTable,
        cr_ac: *const J2kJpegHuffmanTable,
        states: *mut J2kJpegEntropySyncState,
    ) {
        let gid = thread::index_1d().get() as u32;
        if gid >= params.subsequence_count {
            return;
        }

        let start_bit = gid * params.subsequence_bits;
        let end_bit = if start_bit >= params.entropy_bits {
            params.entropy_bits
        } else {
            let remaining_bits = params.entropy_bits - start_bit;
            start_bit + min_u32(params.subsequence_bits, remaining_bits)
        };
        let mut state = J2kJpegEntropySyncState {
            code: JPEG_STATUS_OK,
            start_bit,
            end_bit,
            bit_pos: start_bit,
            symbol_count: 0,
            block_phase: 0,
            zigzag_index: 0,
            reserved: 0,
        };
        let mut reader = bit_reader_at_bit(entropy, params.entropy_len, start_bit);
        let mut status = J2kJpegDecodeStatus {
            code: JPEG_STATUS_OK,
            detail: 0,
            position: 0,
            reserved: 0,
        };

        while state.bit_pos < state.end_bit && status.code == JPEG_STATUS_OK {
            if !entropy_scan_one_symbol420(
                entropy,
                params,
                y_dc,
                y_ac,
                cb_dc,
                cb_ac,
                cr_dc,
                cr_ac,
                &mut state,
                &mut reader,
                &mut status,
            ) {
                break;
            }
        }
        state.code = status.code;
        store_state(states, gid, state);
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn j2k_jpeg_entropy_overflow420(
        entropy: *const u8,
        params: J2kJpegEntropyChunkParams,
        y_dc: *const J2kJpegHuffmanTable,
        y_ac: *const J2kJpegHuffmanTable,
        cb_dc: *const J2kJpegHuffmanTable,
        cb_ac: *const J2kJpegHuffmanTable,
        cr_dc: *const J2kJpegHuffmanTable,
        cr_ac: *const J2kJpegHuffmanTable,
        states: *const J2kJpegEntropySyncState,
        overflows: *mut J2kJpegEntropyOverflowState,
    ) {
        let gid = thread::index_1d().get() as u32;
        if params.subsequence_count <= 1 {
            return;
        }
        let overflow_count = params.subsequence_count - 1;
        if gid >= overflow_count {
            return;
        }

        let mut out = J2kJpegEntropyOverflowState {
            code: JPEG_STATUS_OK,
            from_subsequence: gid,
            to_subsequence: gid + 1,
            overflow_bits: 0,
            synchronized: 0,
            reserved: [0; 3],
        };
        let source = load_state(states, gid);
        let target = load_state(states, gid + 1);
        if source.code != JPEG_STATUS_OK || target.code != JPEG_STATUS_OK {
            out.code = if source.code != JPEG_STATUS_OK {
                source.code
            } else {
                target.code
            };
            store_overflow(overflows, gid, out);
            return;
        }

        let mut state = source;
        let mut reader = bit_reader_at_bit(entropy, params.entropy_len, state.bit_pos);
        let mut status = J2kJpegDecodeStatus {
            code: JPEG_STATUS_OK,
            detail: 0,
            position: 0,
            reserved: 0,
        };

        let mut stop_bit = state.bit_pos;
        if params.max_overflow_subsequences != 0
            && params.subsequence_bits != 0
            && state.bit_pos < params.entropy_bits
        {
            let remaining_bits = params.entropy_bits - state.bit_pos;
            let mut overflow_limit = remaining_bits;
            if params.max_overflow_subsequences <= remaining_bits / params.subsequence_bits {
                overflow_limit = params.max_overflow_subsequences * params.subsequence_bits;
            }
            stop_bit = state.bit_pos + min_u32(overflow_limit, remaining_bits);
        }

        if state.bit_pos == target.bit_pos
            && state.block_phase == target.block_phase
            && state.zigzag_index == target.zigzag_index
        {
            out.synchronized = 1;
            out.overflow_bits = if state.bit_pos > target.start_bit {
                state.bit_pos - target.start_bit
            } else {
                0
            };
        } else {
            while state.bit_pos < stop_bit && status.code == JPEG_STATUS_OK {
                if !entropy_scan_one_symbol420(
                    entropy,
                    params,
                    y_dc,
                    y_ac,
                    cb_dc,
                    cb_ac,
                    cr_dc,
                    cr_ac,
                    &mut state,
                    &mut reader,
                    &mut status,
                ) {
                    break;
                }
                if state.bit_pos == target.bit_pos
                    && state.block_phase == target.block_phase
                    && state.zigzag_index == target.zigzag_index
                {
                    out.synchronized = 1;
                    out.overflow_bits = if state.bit_pos > target.start_bit {
                        state.bit_pos - target.start_bit
                    } else {
                        0
                    };
                    break;
                }
            }
        }

        if status.code != JPEG_STATUS_OK && out.synchronized == 0 {
            out.code = status.code;
        }
        store_overflow(overflows, gid, out);
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn j2k_jpeg_decode_fast420_rgb8(
        entropy: *const u8,
        out: *mut u8,
        params: J2kJpeg420Params,
        y_quant: *const u16,
        cb_quant: *const u16,
        cr_quant: *const u16,
        y_dc: *const J2kJpegHuffmanTable,
        y_ac: *const J2kJpegHuffmanTable,
        cb_dc: *const J2kJpegHuffmanTable,
        cb_ac: *const J2kJpegHuffmanTable,
        cr_dc: *const J2kJpegHuffmanTable,
        cr_ac: *const J2kJpegHuffmanTable,
        checkpoints: *const J2kJpegEntropyCheckpoint,
        status: *mut J2kJpegDecodeStatus,
    ) {
        let gid = thread::index_1d().get() as u32;
        if gid >= params.checkpoint_count {
            return;
        }
        let mut thread_status = J2kJpegDecodeStatus {
            code: JPEG_STATUS_OK,
            detail: 0,
            position: 0,
            reserved: 0,
        };
        store_decode_status(status, gid, thread_status);

        let total_mcus = params.mcus_per_row * params.mcu_rows;
        let checkpoint = load_checkpoint(checkpoints, gid);
        let start_mcu = checkpoint.mcu_index;
        if start_mcu >= total_mcus {
            return;
        }
        let mut end_mcu = total_mcus;
        if gid + 1 < params.checkpoint_count {
            end_mcu = load_checkpoint(checkpoints, gid + 1).mcu_index;
            if end_mcu > total_mcus {
                end_mcu = total_mcus;
            }
        }
        if end_mcu <= start_mcu {
            return;
        }

        let mut reader = J2kJpegBitReader {
            pos: checkpoint.entropy_pos,
            acc: checkpoint.bit_acc,
            bits: checkpoint.bit_count,
        };
        let mut y_prev_dc = checkpoint.y_prev_dc;
        let mut cb_prev_dc = checkpoint.cb_prev_dc;
        let mut cr_prev_dc = checkpoint.cr_prev_dc;

        let mut coeffs = [0i32; 64];
        let mut y0 = [0u8; 64];
        let mut y1 = [0u8; 64];
        let mut y2 = [0u8; 64];
        let mut y3 = [0u8; 64];
        let mut cb = [0u8; 64];
        let mut cr = [0u8; 64];

        let mut mcu = start_mcu;
        while mcu < end_mcu {
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                y_dc,
                y_ac,
                y_quant,
                &mut y_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut y0);
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                y_dc,
                y_ac,
                y_quant,
                &mut y_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut y1);
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                y_dc,
                y_ac,
                y_quant,
                &mut y_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut y2);
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                y_dc,
                y_ac,
                y_quant,
                &mut y_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut y3);
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                cb_dc,
                cb_ac,
                cb_quant,
                &mut cb_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut cb);
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                cr_dc,
                cr_ac,
                cr_quant,
                &mut cr_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut cr);
            let mx = mcu - (mcu / params.mcus_per_row) * params.mcus_per_row;
            let my = mcu / params.mcus_per_row;
            store_rgb420_mcu(out, params, mx, my, &y0, &y1, &y2, &y3, &cb, &cr);
            mcu += 1;
        }
        store_decode_status(status, gid, thread_status);
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn j2k_jpeg_decode_fast422_rgb8(
        entropy: *const u8,
        out: *mut u8,
        params: J2kJpeg420Params,
        y_quant: *const u16,
        cb_quant: *const u16,
        cr_quant: *const u16,
        y_dc: *const J2kJpegHuffmanTable,
        y_ac: *const J2kJpegHuffmanTable,
        cb_dc: *const J2kJpegHuffmanTable,
        cb_ac: *const J2kJpegHuffmanTable,
        cr_dc: *const J2kJpegHuffmanTable,
        cr_ac: *const J2kJpegHuffmanTable,
        checkpoints: *const J2kJpegEntropyCheckpoint,
        status: *mut J2kJpegDecodeStatus,
    ) {
        let gid = thread::index_1d().get() as u32;
        if gid >= params.checkpoint_count {
            return;
        }
        let mut thread_status = J2kJpegDecodeStatus {
            code: JPEG_STATUS_OK,
            detail: 0,
            position: 0,
            reserved: 0,
        };
        store_decode_status(status, gid, thread_status);

        let total_mcus = params.mcus_per_row * params.mcu_rows;
        let checkpoint = load_checkpoint(checkpoints, gid);
        let start_mcu = checkpoint.mcu_index;
        if start_mcu >= total_mcus {
            return;
        }
        let mut end_mcu = total_mcus;
        if gid + 1 < params.checkpoint_count {
            end_mcu = load_checkpoint(checkpoints, gid + 1).mcu_index;
            if end_mcu > total_mcus {
                end_mcu = total_mcus;
            }
        }
        if end_mcu <= start_mcu {
            return;
        }

        let mut reader = J2kJpegBitReader {
            pos: checkpoint.entropy_pos,
            acc: checkpoint.bit_acc,
            bits: checkpoint.bit_count,
        };
        let mut y_prev_dc = checkpoint.y_prev_dc;
        let mut cb_prev_dc = checkpoint.cb_prev_dc;
        let mut cr_prev_dc = checkpoint.cr_prev_dc;

        let mut coeffs = [0i32; 64];
        let mut y0 = [0u8; 64];
        let mut y1 = [0u8; 64];
        let mut cb = [0u8; 64];
        let mut cr = [0u8; 64];

        let mut mcu = start_mcu;
        while mcu < end_mcu {
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                y_dc,
                y_ac,
                y_quant,
                &mut y_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut y0);
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                y_dc,
                y_ac,
                y_quant,
                &mut y_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut y1);
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                cb_dc,
                cb_ac,
                cb_quant,
                &mut cb_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut cb);
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                cr_dc,
                cr_ac,
                cr_quant,
                &mut cr_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut cr);
            let mx = mcu - (mcu / params.mcus_per_row) * params.mcus_per_row;
            let my = mcu / params.mcus_per_row;
            store_rgb422_mcu(out, params, mx, my, &y0, &y1, &cb, &cr);
            mcu += 1;
        }
        store_decode_status(status, gid, thread_status);
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn j2k_jpeg_decode_fast444_rgb8(
        entropy: *const u8,
        out: *mut u8,
        params: J2kJpeg420Params,
        y_quant: *const u16,
        cb_quant: *const u16,
        cr_quant: *const u16,
        y_dc: *const J2kJpegHuffmanTable,
        y_ac: *const J2kJpegHuffmanTable,
        cb_dc: *const J2kJpegHuffmanTable,
        cb_ac: *const J2kJpegHuffmanTable,
        cr_dc: *const J2kJpegHuffmanTable,
        cr_ac: *const J2kJpegHuffmanTable,
        checkpoints: *const J2kJpegEntropyCheckpoint,
        status: *mut J2kJpegDecodeStatus,
    ) {
        let gid = thread::index_1d().get() as u32;
        if gid >= params.checkpoint_count {
            return;
        }
        let mut thread_status = J2kJpegDecodeStatus {
            code: JPEG_STATUS_OK,
            detail: 0,
            position: 0,
            reserved: 0,
        };
        store_decode_status(status, gid, thread_status);

        let total_mcus = params.mcus_per_row * params.mcu_rows;
        let checkpoint = load_checkpoint(checkpoints, gid);
        let start_mcu = checkpoint.mcu_index;
        if start_mcu >= total_mcus {
            return;
        }
        let mut end_mcu = total_mcus;
        if gid + 1 < params.checkpoint_count {
            end_mcu = load_checkpoint(checkpoints, gid + 1).mcu_index;
            if end_mcu > total_mcus {
                end_mcu = total_mcus;
            }
        }
        if end_mcu <= start_mcu {
            return;
        }

        let mut reader = J2kJpegBitReader {
            pos: checkpoint.entropy_pos,
            acc: checkpoint.bit_acc,
            bits: checkpoint.bit_count,
        };
        let mut y_prev_dc = checkpoint.y_prev_dc;
        let mut cb_prev_dc = checkpoint.cb_prev_dc;
        let mut cr_prev_dc = checkpoint.cr_prev_dc;

        let mut coeffs = [0i32; 64];
        let mut y = [0u8; 64];
        let mut cb = [0u8; 64];
        let mut cr = [0u8; 64];

        let mut mcu = start_mcu;
        while mcu < end_mcu {
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                y_dc,
                y_ac,
                y_quant,
                &mut y_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut y);
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                cb_dc,
                cb_ac,
                cb_quant,
                &mut cb_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut cb);
            if !decode_block(
                &mut reader,
                entropy,
                params.entropy_len,
                cr_dc,
                cr_ac,
                cr_quant,
                &mut cr_prev_dc,
                &mut thread_status,
                &mut coeffs,
            ) {
                store_decode_status(status, gid, thread_status);
                return;
            }
            idct_islow(&coeffs, &mut cr);
            let mx = mcu - (mcu / params.mcus_per_row) * params.mcus_per_row;
            let my = mcu / params.mcus_per_row;
            store_rgb444_mcu(out, params, mx, my, &y, &cb, &cr);
            mcu += 1;
        }
        store_decode_status(status, gid, thread_status);
    }
}

fn main() {}
