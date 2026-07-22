use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

include!("../../../cuda_oxide_simt_prelude.rs");

mod component_planes;

const JPEG_STATUS_OK: u32 = 0;
const JPEG_STATUS_TRUNCATED: u32 = 1;
const JPEG_STATUS_HUFFMAN: u32 = 2;
const JPEG_STATUS_INVALID: u32 = 3;

const JPEG_INVALID_DIMENSIONS: u32 = 1;
const JPEG_INVALID_MCU_GRID: u32 = 2;
const JPEG_INVALID_OUTPUT_LAYOUT: u32 = 3;
const JPEG_INVALID_CHECKPOINT_RANGE: u32 = 4;
const JPEG_INVALID_CHECKPOINT_BITS: u32 = 5;
const JPEG_HUFFMAN_VALUE_CAPACITY: u32 = 256;

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
    reserved_tail: u32,
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

#[derive(Clone, Copy)]
struct Rgb420McuBlocks<'a> {
    y0: &'a [u8; 64],
    y1: &'a [u8; 64],
    y2: &'a [u8; 64],
    y3: &'a [u8; 64],
    cb: &'a [u8; 64],
    cr: &'a [u8; 64],
}

#[derive(Clone, Copy)]
struct Rgb422McuBlocks<'a> {
    y0: &'a [u8; 64],
    y1: &'a [u8; 64],
    cb: &'a [u8; 64],
    cr: &'a [u8; 64],
}

#[derive(Clone, Copy)]
struct Jpeg420EntropyTables {
    y_dc: *const J2kJpegHuffmanTable,
    y_ac: *const J2kJpegHuffmanTable,
    cb_dc: *const J2kJpegHuffmanTable,
    cb_ac: *const J2kJpegHuffmanTable,
    cr_dc: *const J2kJpegHuffmanTable,
    cr_ac: *const J2kJpegHuffmanTable,
}

impl Jpeg420EntropyTables {
    #[inline(always)]
    fn table_for(self, block_phase: u32, dc: bool) -> *const J2kJpegHuffmanTable {
        if block_phase < 4 {
            if dc {
                self.y_dc
            } else {
                self.y_ac
            }
        } else if block_phase == 4 {
            if dc {
                self.cb_dc
            } else {
                self.cb_ac
            }
        } else if dc {
            self.cr_dc
        } else {
            self.cr_ac
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct JpegDecodeQuantPtrs {
    y: *const u16,
    cb: *const u16,
    cr: *const u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct JpegDecodeHuffmanPtrs {
    y_dc: *const J2kJpegHuffmanTable,
    y_ac: *const J2kJpegHuffmanTable,
    cb_dc: *const J2kJpegHuffmanTable,
    cb_ac: *const J2kJpegHuffmanTable,
    cr_dc: *const J2kJpegHuffmanTable,
    cr_ac: *const J2kJpegHuffmanTable,
}

impl From<JpegDecodeHuffmanPtrs> for Jpeg420EntropyTables {
    #[inline(always)]
    fn from(ptrs: JpegDecodeHuffmanPtrs) -> Self {
        Self {
            y_dc: ptrs.y_dc,
            y_ac: ptrs.y_ac,
            cb_dc: ptrs.cb_dc,
            cb_ac: ptrs.cb_ac,
            cr_dc: ptrs.cr_dc,
            cr_ac: ptrs.cr_ac,
        }
    }
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
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn load_u16(ptr: *const u16, index: u32) -> u16 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn store_u8(ptr: *mut u8, index: u32, value: u8) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn load_checkpoint(ptr: *const J2kJpegEntropyCheckpoint, index: u32) -> J2kJpegEntropyCheckpoint {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn load_state(ptr: *const J2kJpegEntropySyncState, index: u32) -> J2kJpegEntropySyncState {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn store_state(ptr: *mut J2kJpegEntropySyncState, index: u32, value: J2kJpegEntropySyncState) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn store_overflow(
    ptr: *mut J2kJpegEntropyOverflowState,
    index: u32,
    value: J2kJpegEntropyOverflowState,
) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn store_decode_status(ptr: *mut J2kJpegDecodeStatus, index: u32, value: J2kJpegDecodeStatus) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn set_error(status: &mut J2kJpegDecodeStatus, code: u32, detail: u32, position: u32) {
    status.code = code;
    status.detail = detail;
    status.position = position;
}

#[inline(always)]
fn refill_one(reader: &mut J2kJpegBitReader, entropy: *const u8, entropy_len: u32) -> bool {
    if reader.pos >= entropy_len || reader.bits > 56 {
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
    if wanted > 64 || reader.bits > 64 {
        return false;
    }
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
) -> bool {
    if wanted > 64 || reader.bits > 64 {
        return false;
    }
    while reader.bits < wanted {
        if !refill_one(reader, entropy, entropy_len) {
            if reader.bits >= 64 {
                return false;
            }
            reader.acc |= 1u64 << (63 - reader.bits);
            reader.bits += 1;
        }
    }
    true
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
    if ssss > 15 {
        set_error(status, JPEG_STATUS_HUFFMAN, ssss, reader.pos);
        return false;
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
fn checked_huffman_value_index(code: i32, val_offset: i32, values_len: u32) -> Option<usize> {
    if values_len > JPEG_HUFFMAN_VALUE_CAPACITY {
        return None;
    }
    let index = match code.checked_add(val_offset) {
        Some(index) if index >= 0 => index as u32,
        _ => return None,
    };
    if index >= values_len || index >= JPEG_HUFFMAN_VALUE_CAPACITY {
        return None;
    }
    Some(index as usize)
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
    let values_len = unsafe { (*table).values_len };
    if values_len > JPEG_HUFFMAN_VALUE_CAPACITY
        || !ensure_bits_padded(reader, entropy, entropy_len, 16)
    {
        set_error(status, JPEG_STATUS_HUFFMAN, values_len, reader.pos);
        return false;
    }
    let code16 = peek_bits(*reader, 16) as i32;
    let mut len = 1;
    while len <= 16 {
        let max_code = unsafe { (*table).max_code[len as usize] };
        if max_code >= 0 {
            let code = code16 >> (16 - len);
            if code <= max_code {
                let val_offset = unsafe { (*table).val_offset[len as usize] };
                let idx = match checked_huffman_value_index(code, val_offset, values_len) {
                    Some(index) => index,
                    None => {
                        set_error(status, JPEG_STATUS_HUFFMAN, len, reader.pos);
                        return false;
                    }
                };
                consume_bits(reader, len);
                *symbol = unsafe { (*table).values[idx] };
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
    let values_len = unsafe { (*table).values_len };
    if values_len > JPEG_HUFFMAN_VALUE_CAPACITY || reader.bits > 64 {
        set_error(status, JPEG_STATUS_HUFFMAN, values_len, reader.pos);
        return false;
    }
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
                let idx = match checked_huffman_value_index(code, val_offset, values_len) {
                    Some(index) => index,
                    None => {
                        set_error(status, JPEG_STATUS_HUFFMAN, len, reader.pos);
                        return false;
                    }
                };
                consume_bits(reader, len);
                *symbol = unsafe { (*table).values[idx] };
                return true;
            }
        }
        len += 1;
    }
    set_error(status, JPEG_STATUS_HUFFMAN, 16, reader.pos);
    false
}

#[derive(Clone, Copy)]
struct JpegDecodeBlockContext {
    entropy: *const u8,
    entropy_len: u32,
    dc_table: *const J2kJpegHuffmanTable,
    ac_table: *const J2kJpegHuffmanTable,
    quant: *const u16,
}

impl JpegDecodeBlockContext {
    #[inline(always)]
    fn new(
        entropy: *const u8,
        entropy_len: u32,
        dc_table: *const J2kJpegHuffmanTable,
        ac_table: *const J2kJpegHuffmanTable,
        quant: *const u16,
    ) -> Self {
        Self {
            entropy,
            entropy_len,
            dc_table,
            ac_table,
            quant,
        }
    }
}

#[inline(always)]
fn decode_block(
    reader: &mut J2kJpegBitReader,
    context: JpegDecodeBlockContext,
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
    if !decode_symbol(
        reader,
        context.entropy,
        context.entropy_len,
        context.dc_table,
        status,
        &mut ssss,
    ) {
        return false;
    }
    if ssss > 11 {
        set_error(status, JPEG_STATUS_HUFFMAN, ssss as u32, reader.pos);
        return false;
    }
    let mut diff = 0;
    if !receive_extend(
        reader,
        context.entropy,
        context.entropy_len,
        ssss as u32,
        status,
        &mut diff,
    ) {
        return false;
    }
    *prev_dc = match (*prev_dc).checked_add(diff) {
        Some(value) => value,
        None => {
            set_error(status, JPEG_STATUS_HUFFMAN, ssss as u32, reader.pos);
            return false;
        }
    };
    coeffs[0] = match (*prev_dc).checked_mul(load_u16(context.quant, 0) as i32) {
        Some(value) => value,
        None => {
            set_error(status, JPEG_STATUS_HUFFMAN, 0, reader.pos);
            return false;
        }
    };

    let mut k = 1;
    while k < 64 {
        let mut packed = 0;
        if !decode_symbol(
            reader,
            context.entropy,
            context.entropy_len,
            context.ac_table,
            status,
            &mut packed,
        ) {
            return false;
        }
        let run = (packed >> 4) as u32;
        ssss = packed & 0x0f;
        if ssss == 0 {
            if run == 15 {
                if k > 48 {
                    set_error(status, JPEG_STATUS_HUFFMAN, k, reader.pos);
                    return false;
                }
                k += 16;
                continue;
            }
            if run == 0 {
                break;
            }
            set_error(status, JPEG_STATUS_HUFFMAN, packed as u32, reader.pos);
            return false;
        }
        if ssss > 10 {
            set_error(status, JPEG_STATUS_HUFFMAN, ssss as u32, reader.pos);
            return false;
        }
        k += run;
        if k >= 64 {
            set_error(status, JPEG_STATUS_HUFFMAN, k, reader.pos);
            return false;
        }
        let mut value = 0;
        if !receive_extend(
            reader,
            context.entropy,
            context.entropy_len,
            ssss as u32,
            status,
            &mut value,
        ) {
            return false;
        }
        coeffs[zigzag(k) as usize] = match value.checked_mul(load_u16(context.quant, k) as i32) {
            Some(value) => value,
            None => {
                set_error(status, JPEG_STATUS_HUFFMAN, k, reader.pos);
                return false;
            }
        };
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
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8, r: &mut u8, g: &mut u8, b: &mut u8) {
    let yy = y as i32;
    let cb_centered = cb as i32 - 128;
    let cr_centered = cr as i32 - 128;
    *r = clamp_i32(yy + ((91881 * cr_centered + (1 << 15)) >> 16));
    *g = clamp_i32(yy - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    *b = clamp_i32(yy + ((116130 * cb_centered + (1 << 15)) >> 16));
}

#[inline(always)]
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

#[derive(Clone, Copy)]
struct JpegDecodeThreadRange {
    checkpoint: J2kJpegEntropyCheckpoint,
    start_mcu: u32,
    end_mcu: u32,
}

#[inline(always)]
fn checkpoint_bit_position(checkpoint: J2kJpegEntropyCheckpoint, entropy_len: u32) -> Option<u64> {
    if checkpoint.reserved != 0 || checkpoint.entropy_pos > entropy_len || checkpoint.bit_count > 63
    {
        return None;
    }

    let loaded_bits = (checkpoint.entropy_pos as u64) * 8;
    if loaded_bits < checkpoint.bit_count as u64 {
        return None;
    }

    if checkpoint.bit_count == 0 {
        if checkpoint.bit_acc != 0 {
            return None;
        }
    } else if checkpoint.bit_count < 64 {
        let unused_bits = 64 - checkpoint.bit_count;
        let unused_mask = (1u64 << unused_bits) - 1;
        if checkpoint.bit_acc & unused_mask != 0 {
            return None;
        }
    }

    Some(loaded_bits - checkpoint.bit_count as u64)
}

#[inline(always)]
fn validate_decode_thread_range(
    params: J2kJpeg420Params,
    checkpoints: *const J2kJpegEntropyCheckpoint,
    gid: u32,
    mcu_width: u32,
    mcu_height: u32,
    status: &mut J2kJpegDecodeStatus,
) -> Option<JpegDecodeThreadRange> {
    if params.width == 0 || params.height == 0 || params.reserved != 0 {
        set_error(status, JPEG_STATUS_INVALID, JPEG_INVALID_DIMENSIONS, gid);
        return None;
    }

    if params.width > u32::MAX / 3 {
        set_error(status, JPEG_STATUS_INVALID, JPEG_INVALID_OUTPUT_LAYOUT, gid);
        return None;
    }
    let row_bytes = params.width * 3;
    let last_output_offset = (params.height as u64 - 1)
        .checked_mul(params.out_stride as u64)
        .and_then(|row_offset| row_offset.checked_add(row_bytes as u64 - 1));
    if params.out_stride < row_bytes
        || !matches!(last_output_offset, Some(offset) if offset <= u32::MAX as u64)
    {
        set_error(status, JPEG_STATUS_INVALID, JPEG_INVALID_OUTPUT_LAYOUT, gid);
        return None;
    }

    let expected_mcus_per_row = (params.width - 1) / mcu_width + 1;
    let expected_mcu_rows = (params.height - 1) / mcu_height + 1;
    if params.mcus_per_row != expected_mcus_per_row
        || params.mcu_rows != expected_mcu_rows
        || params.mcus_per_row > u32::MAX / params.mcu_rows
    {
        set_error(status, JPEG_STATUS_INVALID, JPEG_INVALID_MCU_GRID, gid);
        return None;
    }
    let total_mcus = params.mcus_per_row * params.mcu_rows;

    let first_checkpoint = load_checkpoint(checkpoints, 0);
    if first_checkpoint.mcu_index != 0 {
        set_error(
            status,
            JPEG_STATUS_INVALID,
            JPEG_INVALID_CHECKPOINT_RANGE,
            gid,
        );
        return None;
    }

    let checkpoint = load_checkpoint(checkpoints, gid);
    let current_bit_position = match checkpoint_bit_position(checkpoint, params.entropy_len) {
        Some(position) => position,
        None => {
            set_error(
                status,
                JPEG_STATUS_INVALID,
                JPEG_INVALID_CHECKPOINT_BITS,
                checkpoint.entropy_pos,
            );
            return None;
        }
    };
    let start_mcu = checkpoint.mcu_index;
    if start_mcu >= total_mcus {
        set_error(
            status,
            JPEG_STATUS_INVALID,
            JPEG_INVALID_CHECKPOINT_RANGE,
            start_mcu,
        );
        return None;
    }

    if gid == 0 {
        if current_bit_position != 0
            || checkpoint.y_prev_dc != 0
            || checkpoint.cb_prev_dc != 0
            || checkpoint.cr_prev_dc != 0
        {
            set_error(
                status,
                JPEG_STATUS_INVALID,
                JPEG_INVALID_CHECKPOINT_BITS,
                checkpoint.entropy_pos,
            );
            return None;
        }
    } else {
        let previous = load_checkpoint(checkpoints, gid - 1);
        let previous_bit_position = match checkpoint_bit_position(previous, params.entropy_len) {
            Some(position) => position,
            None => {
                set_error(
                    status,
                    JPEG_STATUS_INVALID,
                    JPEG_INVALID_CHECKPOINT_BITS,
                    previous.entropy_pos,
                );
                return None;
            }
        };
        if previous.mcu_index >= start_mcu || previous_bit_position >= current_bit_position {
            set_error(
                status,
                JPEG_STATUS_INVALID,
                JPEG_INVALID_CHECKPOINT_RANGE,
                start_mcu,
            );
            return None;
        }
    }

    let end_mcu = if gid + 1 < params.checkpoint_count {
        let next = load_checkpoint(checkpoints, gid + 1);
        let next_bit_position = match checkpoint_bit_position(next, params.entropy_len) {
            Some(position) => position,
            None => {
                set_error(
                    status,
                    JPEG_STATUS_INVALID,
                    JPEG_INVALID_CHECKPOINT_BITS,
                    next.entropy_pos,
                );
                return None;
            }
        };
        if next.mcu_index <= start_mcu
            || next.mcu_index > total_mcus
            || next_bit_position <= current_bit_position
        {
            set_error(
                status,
                JPEG_STATUS_INVALID,
                JPEG_INVALID_CHECKPOINT_RANGE,
                next.mcu_index,
            );
            return None;
        }
        next.mcu_index
    } else {
        total_mcus
    };

    Some(JpegDecodeThreadRange {
        checkpoint,
        start_mcu,
        end_mcu,
    })
}

#[inline(always)]
fn entropy_scan_one_symbol420(
    entropy: *const u8,
    params: J2kJpegEntropyChunkParams,
    tables: Jpeg420EntropyTables,
    state: &mut J2kJpegEntropySyncState,
    reader: &mut J2kJpegBitReader,
    status: &mut J2kJpegDecodeStatus,
) -> bool {
    let dc = state.zigzag_index == 0;
    let table = tables.table_for(state.block_phase, dc);
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
    pub unsafe fn j2k_jpeg_entropy_sync420(
        entropy: *const u8,
        params: J2kJpegEntropyChunkParams,
        huffman: JpegDecodeHuffmanPtrs,
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
        let tables = Jpeg420EntropyTables::from(huffman);

        while state.bit_pos < state.end_bit && status.code == JPEG_STATUS_OK {
            if !entropy_scan_one_symbol420(
                entropy,
                params,
                tables,
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
    pub unsafe fn j2k_jpeg_entropy_overflow420(
        entropy: *const u8,
        params: J2kJpegEntropyChunkParams,
        huffman: JpegDecodeHuffmanPtrs,
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
        let tables = Jpeg420EntropyTables::from(huffman);

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
                    tables,
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
    pub unsafe fn j2k_jpeg_decode_fast420_rgb8(
        entropy: *const u8,
        out: *mut u8,
        params: J2kJpeg420Params,
        quant: JpegDecodeQuantPtrs,
        huffman: JpegDecodeHuffmanPtrs,
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

        let thread_range = match validate_decode_thread_range(
            params,
            checkpoints,
            gid,
            16,
            16,
            &mut thread_status,
        ) {
            Some(range) => range,
            None => {
                store_decode_status(status, gid, thread_status);
                return;
            }
        };
        let checkpoint = thread_range.checkpoint;
        let start_mcu = thread_range.start_mcu;
        let end_mcu = thread_range.end_mcu;

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

        let y_decode = JpegDecodeBlockContext::new(
            entropy,
            params.entropy_len,
            huffman.y_dc,
            huffman.y_ac,
            quant.y,
        );
        let cb_decode = JpegDecodeBlockContext::new(
            entropy,
            params.entropy_len,
            huffman.cb_dc,
            huffman.cb_ac,
            quant.cb,
        );
        let cr_decode = JpegDecodeBlockContext::new(
            entropy,
            params.entropy_len,
            huffman.cr_dc,
            huffman.cr_ac,
            quant.cr,
        );

        let mut mcu = start_mcu;
        while mcu < end_mcu {
            if !decode_block(
                &mut reader,
                y_decode,
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
                y_decode,
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
                y_decode,
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
                y_decode,
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
                cb_decode,
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
                cr_decode,
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
            component_planes::store_420_mcu(
                out,
                params,
                mx,
                my,
                Rgb420McuBlocks {
                    y0: &y0,
                    y1: &y1,
                    y2: &y2,
                    y3: &y3,
                    cb: &cb,
                    cr: &cr,
                },
            );
            mcu += 1;
        }
        store_decode_status(status, gid, thread_status);
    }

    #[kernel]
    pub unsafe fn j2k_jpeg_decode_fast422_rgb8(
        entropy: *const u8,
        out: *mut u8,
        params: J2kJpeg420Params,
        quant: JpegDecodeQuantPtrs,
        huffman: JpegDecodeHuffmanPtrs,
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

        let thread_range =
            match validate_decode_thread_range(params, checkpoints, gid, 16, 8, &mut thread_status)
            {
                Some(range) => range,
                None => {
                    store_decode_status(status, gid, thread_status);
                    return;
                }
            };
        let checkpoint = thread_range.checkpoint;
        let start_mcu = thread_range.start_mcu;
        let end_mcu = thread_range.end_mcu;

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

        let y_decode = JpegDecodeBlockContext::new(
            entropy,
            params.entropy_len,
            huffman.y_dc,
            huffman.y_ac,
            quant.y,
        );
        let cb_decode = JpegDecodeBlockContext::new(
            entropy,
            params.entropy_len,
            huffman.cb_dc,
            huffman.cb_ac,
            quant.cb,
        );
        let cr_decode = JpegDecodeBlockContext::new(
            entropy,
            params.entropy_len,
            huffman.cr_dc,
            huffman.cr_ac,
            quant.cr,
        );

        let mut mcu = start_mcu;
        while mcu < end_mcu {
            if !decode_block(
                &mut reader,
                y_decode,
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
                y_decode,
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
                cb_decode,
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
                cr_decode,
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
            component_planes::store_422_mcu(
                out,
                params,
                mx,
                my,
                Rgb422McuBlocks {
                    y0: &y0,
                    y1: &y1,
                    cb: &cb,
                    cr: &cr,
                },
            );
            mcu += 1;
        }
        store_decode_status(status, gid, thread_status);
    }

    #[kernel]
    pub unsafe fn j2k_jpeg_subsampled_planes_to_rgb8(
        planes: *const u8,
        out: *mut u8,
        params: J2kJpeg420Params,
        sampling: u32,
    ) {
        let pixel = thread::index_1d().get() as u32;
        let pixel_count = params.width * params.height;
        if pixel >= pixel_count {
            return;
        }
        component_planes::convert_pixel(planes, out, params, sampling, pixel);
    }

    #[kernel]
    pub unsafe fn j2k_jpeg_decode_fast444_rgb8(
        entropy: *const u8,
        out: *mut u8,
        params: J2kJpeg420Params,
        quant: JpegDecodeQuantPtrs,
        huffman: JpegDecodeHuffmanPtrs,
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

        let thread_range = match validate_decode_thread_range(
            params,
            checkpoints,
            gid,
            8,
            8,
            &mut thread_status,
        ) {
            Some(range) => range,
            None => {
                store_decode_status(status, gid, thread_status);
                return;
            }
        };
        let checkpoint = thread_range.checkpoint;
        let start_mcu = thread_range.start_mcu;
        let end_mcu = thread_range.end_mcu;

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

        let y_decode = JpegDecodeBlockContext::new(
            entropy,
            params.entropy_len,
            huffman.y_dc,
            huffman.y_ac,
            quant.y,
        );
        let cb_decode = JpegDecodeBlockContext::new(
            entropy,
            params.entropy_len,
            huffman.cb_dc,
            huffman.cb_ac,
            quant.cb,
        );
        let cr_decode = JpegDecodeBlockContext::new(
            entropy,
            params.entropy_len,
            huffman.cr_dc,
            huffman.cr_ac,
            quant.cr,
        );

        let mut mcu = start_mcu;
        while mcu < end_mcu {
            if !decode_block(
                &mut reader,
                y_decode,
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
                cb_decode,
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
                cr_decode,
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
