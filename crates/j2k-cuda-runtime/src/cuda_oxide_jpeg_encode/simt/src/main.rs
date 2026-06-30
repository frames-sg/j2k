#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::manual_div_ceil,
    clippy::many_single_char_names,
    clippy::too_many_arguments
)]

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

const JPEG_BASELINE_ENCODE_FORMAT_GRAY8: u32 = 0;
const JPEG_BASELINE_ENCODE_FORMAT_RGB8: u32 = 1;
const JPEG_BASELINE_ENCODE_STATUS_OK: u32 = 0;
const JPEG_BASELINE_ENCODE_STATUS_OVERFLOW: u32 = 1;
const JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN: u32 = 2;
const JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS: u32 = 3;

const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

const COS_TABLE: [[f32; 8]; 8] = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [
        0.98078525,
        0.8314696,
        0.55557024,
        0.19509032,
        -0.19509032,
        -0.55557024,
        -0.8314696,
        -0.98078525,
    ],
    [
        0.9238795,
        0.38268343,
        -0.38268343,
        -0.9238795,
        -0.9238795,
        -0.38268343,
        0.38268343,
        0.9238795,
    ],
    [
        0.8314696,
        -0.19509032,
        -0.98078525,
        -0.55557024,
        0.55557024,
        0.98078525,
        0.19509032,
        -0.8314696,
    ],
    [
        0.70710677,
        -0.70710677,
        -0.70710677,
        0.70710677,
        0.70710677,
        -0.70710677,
        -0.70710677,
        0.70710677,
    ],
    [
        0.55557024,
        -0.98078525,
        0.19509032,
        0.8314696,
        -0.8314696,
        -0.19509032,
        0.98078525,
        -0.55557024,
    ],
    [
        0.38268343,
        -0.9238795,
        0.9238795,
        -0.38268343,
        -0.38268343,
        0.9238795,
        -0.9238795,
        0.38268343,
    ],
    [
        0.19509032,
        -0.55557024,
        0.8314696,
        -0.98078525,
        0.98078525,
        -0.8314696,
        0.55557024,
        -0.19509032,
    ],
];

#[repr(C)]
#[derive(Clone, Copy)]
pub struct J2kJpegBaselineEncodeParams {
    input_offset_bytes: u32,
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    pitch_bytes: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    format: u32,
    components: u32,
    max_h: u32,
    max_v: u32,
    h0: u32,
    v0: u32,
    h1: u32,
    v1: u32,
    h2: u32,
    v2: u32,
    entropy_offset_bytes: u32,
    entropy_capacity: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct J2kJpegBaselineEncodeHuffmanTable {
    codes: [u16; 256],
    lens: [u8; 256],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct J2kJpegBaselineEncodeStatus {
    code: u32,
    entropy_len: u32,
    detail: u32,
    reserved: u32,
}

#[derive(Clone, Copy)]
struct JpegBaselineBitWriter {
    pos: u32,
    current: u8,
    used: u32,
    overflow: bool,
}

#[inline(always)]
fn min_u32(a: u32, b: u32) -> u32 {
    if a < b { a } else { b }
}

#[inline(always)]
fn clamp_u8(value: i32) -> u8 {
    if value < 0 {
        0
    } else if value > 255 {
        255
    } else {
        value as u8
    }
}

#[inline(always)]
fn round_to_i32(value: f32) -> i32 {
    if value >= 0.0 {
        (value + 0.5) as i32
    } else {
        (value - 0.5) as i32
    }
}

#[inline(always)]
fn load_u8(ptr: *const u8, index: u32) -> u8 {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn store_u8(ptr: *mut u8, index: u32, value: u8) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}

#[inline(always)]
fn load_params(ptr: *const J2kJpegBaselineEncodeParams, index: u32) -> J2kJpegBaselineEncodeParams {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn load_huff_symbol(table: *const J2kJpegBaselineEncodeHuffmanTable, symbol: u32) -> (u16, u8) {
    unsafe {
        (
            (*table).codes[symbol as usize],
            (*table).lens[symbol as usize],
        )
    }
}

#[inline(always)]
fn set_status(
    status: *mut J2kJpegBaselineEncodeStatus,
    code: u32,
    entropy_len: u32,
    detail: u32,
) {
    unsafe {
        (*status).code = code;
        (*status).entropy_len = entropy_len;
        (*status).detail = detail;
        (*status).reserved = 0;
    }
}

#[inline(always)]
fn component_h(params: J2kJpegBaselineEncodeParams, component: u32) -> u32 {
    if component == 0 {
        params.h0
    } else if component == 1 {
        params.h1
    } else {
        params.h2
    }
}

#[inline(always)]
fn component_v(params: J2kJpegBaselineEncodeParams, component: u32) -> u32 {
    if component == 0 {
        params.v0
    } else if component == 1 {
        params.v1
    } else {
        params.v2
    }
}

#[inline(always)]
fn rgb_to_ycbcr_component(r: u8, g: u8, b: u8, component: u32) -> u8 {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    if component == 0 {
        clamp_u8((19_595 * r + 38_470 * g + 7_471 * b + 32_768) >> 16)
    } else if component == 1 {
        clamp_u8((-11_059 * r - 21_709 * g + 32_768 * b + 8_421_376) >> 16)
    } else {
        clamp_u8((32_768 * r - 27_439 * g - 5_329 * b + 8_421_376) >> 16)
    }
}

fn jpeg_encode_sample_component(
    input: *const u8,
    params: J2kJpegBaselineEncodeParams,
    component: u32,
    x: u32,
    y: u32,
) -> u8 {
    if x >= params.input_width || y >= params.input_height {
        return 0;
    }
    let row = y * params.pitch_bytes;
    if params.format == JPEG_BASELINE_ENCODE_FORMAT_GRAY8 {
        return load_u8(input, row + x);
    }
    let offset = row + x * 3;
    rgb_to_ycbcr_component(
        load_u8(input, offset),
        load_u8(input, offset + 1),
        load_u8(input, offset + 2),
        component,
    )
}

fn jpeg_encode_sample_block(
    input: *const u8,
    params: J2kJpegBaselineEncodeParams,
    component: u32,
    mcu_x: u32,
    mcu_y: u32,
    block_x: u32,
    block_y: u32,
    block: &mut [u8; 64],
) {
    let comp_h = component_h(params, component);
    let comp_v = component_v(params, component);
    let x_scale = params.max_h / comp_h;
    let y_scale = params.max_v / comp_v;
    let mcu_origin_x = mcu_x * params.max_h * 8;
    let mcu_origin_y = mcu_y * params.max_v * 8;

    let mut y = 0;
    while y < 8 {
        let mut x = 0;
        while x < 8 {
            let value = if component == 0 || params.components == 1 {
                let sx = min_u32(mcu_origin_x + block_x * 8 + x, params.output_width - 1);
                let sy = min_u32(mcu_origin_y + block_y * 8 + y, params.output_height - 1);
                jpeg_encode_sample_component(input, params, component, sx, sy)
            } else {
                let mut sum = 0u32;
                let mut dy = 0;
                while dy < y_scale {
                    let mut dx = 0;
                    while dx < x_scale {
                        let sx = min_u32(
                            mcu_origin_x + (block_x * 8 + x) * x_scale + dx,
                            params.output_width - 1,
                        );
                        let sy = min_u32(
                            mcu_origin_y + (block_y * 8 + y) * y_scale + dy,
                            params.output_height - 1,
                        );
                        sum += jpeg_encode_sample_component(input, params, component, sx, sy)
                            as u32;
                        dx += 1;
                    }
                    dy += 1;
                }
                (sum / (x_scale * y_scale)) as u8
            };
            block[(y * 8 + x) as usize] = value;
            x += 1;
        }
        y += 1;
    }
}

fn jpeg_encode_fdct_quantize(block: &[u8; 64], quant: *const u8, coeffs: &mut [i32; 64]) {
    const INV_SQRT_2: f32 = 0.70710677;
    let mut v = 0;
    while v < 8 {
        let mut u = 0;
        while u < 8 {
            let mut sum = 0.0f32;
            let mut y = 0;
            while y < 8 {
                let mut x = 0;
                while x < 8 {
                    let sample = block[(y * 8 + x) as usize] as f32 - 128.0;
                    sum += sample * COS_TABLE[u as usize][x as usize] * COS_TABLE[v as usize][y as usize];
                    x += 1;
                }
                y += 1;
            }
            let cu = if u == 0 { INV_SQRT_2 } else { 1.0 };
            let cv = if v == 0 { INV_SQRT_2 } else { 1.0 };
            let natural = v * 8 + u;
            let transformed = 0.25 * cu * cv * sum;
            let divisor = load_u8(quant, natural) as f32;
            coeffs[natural as usize] = round_to_i32(transformed / divisor);
            u += 1;
        }
        v += 1;
    }
}

fn jpeg_encode_push_raw_byte(
    entropy: *mut u8,
    capacity: u32,
    writer: &mut JpegBaselineBitWriter,
    byte: u8,
) {
    if writer.pos >= capacity {
        writer.overflow = true;
        return;
    }
    store_u8(entropy, writer.pos, byte);
    writer.pos += 1;
}

fn jpeg_encode_push_data_byte(
    entropy: *mut u8,
    capacity: u32,
    writer: &mut JpegBaselineBitWriter,
    byte: u8,
) {
    jpeg_encode_push_raw_byte(entropy, capacity, writer, byte);
    if !writer.overflow && byte == 0xff {
        jpeg_encode_push_raw_byte(entropy, capacity, writer, 0);
    }
}

fn jpeg_encode_write_bits(
    entropy: *mut u8,
    capacity: u32,
    writer: &mut JpegBaselineBitWriter,
    code: u16,
    len: u32,
) {
    let mut bit = len as i32 - 1;
    while bit >= 0 {
        let value = ((code >> (bit as u32)) & 1) as u8;
        writer.current = (writer.current << 1) | value;
        writer.used += 1;
        if writer.used == 8 {
            jpeg_encode_push_data_byte(entropy, capacity, writer, writer.current);
            writer.current = 0;
            writer.used = 0;
            if writer.overflow {
                return;
            }
        }
        bit -= 1;
    }
}

fn jpeg_encode_align_with_ones(
    entropy: *mut u8,
    capacity: u32,
    writer: &mut JpegBaselineBitWriter,
) {
    if writer.used == 0 {
        return;
    }
    let remaining = 8 - writer.used;
    writer.current = (writer.current << remaining) | ((1u32 << remaining) - 1) as u8;
    jpeg_encode_push_data_byte(entropy, capacity, writer, writer.current);
    writer.current = 0;
    writer.used = 0;
}

fn jpeg_encode_push_restart_marker(
    entropy: *mut u8,
    capacity: u32,
    writer: &mut JpegBaselineBitWriter,
    rst: u32,
) {
    jpeg_encode_align_with_ones(entropy, capacity, writer);
    if writer.overflow {
        return;
    }
    jpeg_encode_push_raw_byte(entropy, capacity, writer, 0xff);
    jpeg_encode_push_raw_byte(entropy, capacity, writer, 0xd0 + (rst & 0x07) as u8);
}

fn jpeg_encode_magnitude_category(value: i32) -> u32 {
    if value == 0 {
        return 0;
    }
    let mut abs_value = if value < 0 {
        (!(value as u32)).wrapping_add(1)
    } else {
        value as u32
    };
    let mut size = 0;
    while abs_value > 0 {
        size += 1;
        abs_value >>= 1;
    }
    size
}

fn jpeg_encode_magnitude_bits(value: i32, size: u32) -> u16 {
    if size == 0 {
        return 0;
    }
    if value >= 0 {
        value as u16
    } else {
        (value + ((1i32 << size) - 1)) as u16
    }
}

fn jpeg_encode_write_symbol(
    entropy: *mut u8,
    capacity: u32,
    writer: &mut JpegBaselineBitWriter,
    table: *const J2kJpegBaselineEncodeHuffmanTable,
    symbol: u32,
    status: *mut J2kJpegBaselineEncodeStatus,
) -> bool {
    let (code, len) = load_huff_symbol(table, symbol);
    if len == 0 {
        set_status(
            status,
            JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN,
            writer.pos,
            symbol,
        );
        return false;
    }
    jpeg_encode_write_bits(entropy, capacity, writer, code, len as u32);
    if writer.overflow {
        set_status(status, JPEG_BASELINE_ENCODE_STATUS_OVERFLOW, writer.pos, 0);
        return false;
    }
    true
}

fn jpeg_encode_block(
    coeffs: &[i32; 64],
    prev_dc: &mut i32,
    dc_table: *const J2kJpegBaselineEncodeHuffmanTable,
    ac_table: *const J2kJpegBaselineEncodeHuffmanTable,
    entropy: *mut u8,
    capacity: u32,
    writer: &mut JpegBaselineBitWriter,
    status: *mut J2kJpegBaselineEncodeStatus,
) -> bool {
    let diff = coeffs[0] - *prev_dc;
    *prev_dc = coeffs[0];
    let dc_size = jpeg_encode_magnitude_category(diff);
    if !jpeg_encode_write_symbol(entropy, capacity, writer, dc_table, dc_size, status) {
        return false;
    }
    if dc_size > 0 {
        jpeg_encode_write_bits(
            entropy,
            capacity,
            writer,
            jpeg_encode_magnitude_bits(diff, dc_size),
            dc_size,
        );
        if writer.overflow {
            set_status(status, JPEG_BASELINE_ENCODE_STATUS_OVERFLOW, writer.pos, 0);
            return false;
        }
    }

    let mut zero_run = 0;
    let mut k = 1;
    while k < 64 {
        let coeff = coeffs[ZIGZAG[k as usize] as usize];
        if coeff == 0 {
            zero_run += 1;
            k += 1;
            continue;
        }
        while zero_run >= 16 {
            if !jpeg_encode_write_symbol(entropy, capacity, writer, ac_table, 0xf0, status) {
                return false;
            }
            zero_run -= 16;
        }
        let size = jpeg_encode_magnitude_category(coeff);
        let symbol = (zero_run << 4) | size;
        if !jpeg_encode_write_symbol(entropy, capacity, writer, ac_table, symbol, status) {
            return false;
        }
        jpeg_encode_write_bits(
            entropy,
            capacity,
            writer,
            jpeg_encode_magnitude_bits(coeff, size),
            size,
        );
        if writer.overflow {
            set_status(status, JPEG_BASELINE_ENCODE_STATUS_OVERFLOW, writer.pos, 0);
            return false;
        }
        zero_run = 0;
        k += 1;
    }
    zero_run == 0 || jpeg_encode_write_symbol(entropy, capacity, writer, ac_table, 0, status)
}

#[allow(clippy::too_many_arguments)]
fn jpeg_encode_baseline_entropy_one(
    input: *const u8,
    entropy: *mut u8,
    status: *mut J2kJpegBaselineEncodeStatus,
    params: J2kJpegBaselineEncodeParams,
    q_luma: *const u8,
    q_chroma: *const u8,
    dc_luma: *const J2kJpegBaselineEncodeHuffmanTable,
    ac_luma: *const J2kJpegBaselineEncodeHuffmanTable,
    dc_chroma: *const J2kJpegBaselineEncodeHuffmanTable,
    ac_chroma: *const J2kJpegBaselineEncodeHuffmanTable,
) {
    set_status(status, JPEG_BASELINE_ENCODE_STATUS_OK, 0, 0);
    if params.input_width == 0
        || params.input_height == 0
        || params.output_width == 0
        || params.output_height == 0
        || params.mcus_per_row == 0
        || params.mcu_rows == 0
        || params.max_h == 0
        || params.max_v == 0
        || params.h0 == 0
        || params.v0 == 0
        || !(params.format == JPEG_BASELINE_ENCODE_FORMAT_GRAY8
            || params.format == JPEG_BASELINE_ENCODE_FORMAT_RGB8)
    {
        set_status(status, JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS, 0, 0);
        return;
    }

    let mut writer = JpegBaselineBitWriter {
        pos: 0,
        current: 0,
        used: 0,
        overflow: false,
    };
    let mut prev_dc = [0i32; 3];
    let mut mcus_since_restart = 0;
    let mut rst = 0;
    let mut mcu_y = 0;
    while mcu_y < params.mcu_rows {
        let mut mcu_x = 0;
        while mcu_x < params.mcus_per_row {
            if params.restart_interval_mcus != 0
                && mcus_since_restart == params.restart_interval_mcus
            {
                jpeg_encode_push_restart_marker(entropy, params.entropy_capacity, &mut writer, rst);
                if writer.overflow {
                    set_status(status, JPEG_BASELINE_ENCODE_STATUS_OVERFLOW, writer.pos, 0);
                    return;
                }
                rst = (rst + 1) & 7;
                prev_dc = [0, 0, 0];
                mcus_since_restart = 0;
            }

            let mut component = 0;
            while component < params.components {
                let h = component_h(params, component);
                let v = component_v(params, component);
                if h == 0 || v == 0 {
                    set_status(status, JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS, writer.pos, component);
                    return;
                }
                let mut block_y = 0;
                while block_y < v {
                    let mut block_x = 0;
                    while block_x < h {
                        let mut block = [0u8; 64];
                        let mut coeffs = [0i32; 64];
                        jpeg_encode_sample_block(
                            input, params, component, mcu_x, mcu_y, block_x, block_y, &mut block,
                        );
                        let ok = if component == 0 {
                            jpeg_encode_fdct_quantize(&block, q_luma, &mut coeffs);
                            jpeg_encode_block(
                                &coeffs,
                                &mut prev_dc[component as usize],
                                dc_luma,
                                ac_luma,
                                entropy,
                                params.entropy_capacity,
                                &mut writer,
                                status,
                            )
                        } else {
                            jpeg_encode_fdct_quantize(&block, q_chroma, &mut coeffs);
                            jpeg_encode_block(
                                &coeffs,
                                &mut prev_dc[component as usize],
                                dc_chroma,
                                ac_chroma,
                                entropy,
                                params.entropy_capacity,
                                &mut writer,
                                status,
                            )
                        };
                        if !ok {
                            return;
                        }
                        block_x += 1;
                    }
                    block_y += 1;
                }
                component += 1;
            }
            mcus_since_restart += 1;
            mcu_x += 1;
        }
        mcu_y += 1;
    }
    jpeg_encode_align_with_ones(entropy, params.entropy_capacity, &mut writer);
    if writer.overflow {
        set_status(status, JPEG_BASELINE_ENCODE_STATUS_OVERFLOW, writer.pos, 0);
        return;
    }
    set_status(status, JPEG_BASELINE_ENCODE_STATUS_OK, writer.pos, 0);
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn j2k_jpeg_encode_baseline_entropy(
        input: *const u8,
        entropy: *mut u8,
        status: *mut J2kJpegBaselineEncodeStatus,
        params: J2kJpegBaselineEncodeParams,
        q_luma: *const u8,
        q_chroma: *const u8,
        dc_luma: *const J2kJpegBaselineEncodeHuffmanTable,
        ac_luma: *const J2kJpegBaselineEncodeHuffmanTable,
        dc_chroma: *const J2kJpegBaselineEncodeHuffmanTable,
        ac_chroma: *const J2kJpegBaselineEncodeHuffmanTable,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        jpeg_encode_baseline_entropy_one(
            input, entropy, status, params, q_luma, q_chroma, dc_luma, ac_luma, dc_chroma,
            ac_chroma,
        );
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn j2k_jpeg_encode_baseline_entropy_batch(
        input: *const u8,
        entropy: *mut u8,
        status: *mut J2kJpegBaselineEncodeStatus,
        params: *const J2kJpegBaselineEncodeParams,
        q_luma: *const u8,
        q_chroma: *const u8,
        dc_luma: *const J2kJpegBaselineEncodeHuffmanTable,
        ac_luma: *const J2kJpegBaselineEncodeHuffmanTable,
        dc_chroma: *const J2kJpegBaselineEncodeHuffmanTable,
        ac_chroma: *const J2kJpegBaselineEncodeHuffmanTable,
        tile_count: u32,
    ) {
        let gid = thread::index_1d().get() as u32;
        if gid >= tile_count {
            return;
        }
        let tile_params = load_params(params, gid);
        jpeg_encode_baseline_entropy_one(
            input.add(tile_params.input_offset_bytes as usize),
            entropy.add(tile_params.entropy_offset_bytes as usize),
            status.add(gid as usize),
            tile_params,
            q_luma,
            q_chroma,
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
        );
    }
}

fn main() {}
