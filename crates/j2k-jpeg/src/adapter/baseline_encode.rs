// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::encoder::{
    EncodedJpeg, JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSubsampling,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Baseline JPEG component sampling parameters.
pub struct JpegBaselineSampling {
    /// Number of encoded components.
    pub components: u8,
    /// Horizontal sampling factor per component.
    pub h: [u8; 3],
    /// Vertical sampling factor per component.
    pub v: [u8; 3],
    /// Maximum horizontal sampling factor across components.
    pub max_h: u8,
    /// Maximum vertical sampling factor across components.
    pub max_v: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Canonical Huffman lookup table for encoding.
pub struct JpegBaselineHuffmanTable {
    /// Huffman code value by symbol.
    pub codes: [u16; 256],
    /// Huffman code length by symbol.
    pub lens: [u8; 256],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Tables needed to assemble and entropy-code a baseline JPEG frame.
pub struct JpegBaselineEncodeTables {
    /// Component sampling metadata.
    pub sampling: JpegBaselineSampling,
    /// Luma quantization table in natural order.
    pub q_luma: [u8; 64],
    /// Chroma quantization table in natural order.
    pub q_chroma: [u8; 64],
    /// Luma DC Huffman table.
    pub huff_dc_luma: JpegBaselineHuffmanTable,
    /// Luma AC Huffman table.
    pub huff_ac_luma: JpegBaselineHuffmanTable,
    /// Chroma DC Huffman table.
    pub huff_dc_chroma: JpegBaselineHuffmanTable,
    /// Chroma AC Huffman table.
    pub huff_ac_chroma: JpegBaselineHuffmanTable,
}

/// Build quantization, sampling, and Huffman tables for baseline encoding.
pub fn baseline_encode_tables(
    options: JpegEncodeOptions,
) -> Result<JpegBaselineEncodeTables, JpegEncodeError> {
    validate_jpeg_baseline_restart_interval(options.restart_interval)?;
    Ok(JpegBaselineEncodeTables {
        sampling: jpeg_baseline_sampling_for(options.subsampling),
        q_luma: scaled_quant_table(&STD_LUMA_Q, options.quality),
        q_chroma: scaled_quant_table(&STD_CHROMA_Q, options.quality),
        huff_dc_luma: encode_huffman_table(&STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES)?,
        huff_ac_luma: encode_huffman_table(&STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES)?,
        huff_dc_chroma: encode_huffman_table(&STD_CHROMA_DC_BITS, &STD_CHROMA_DC_VALUES)?,
        huff_ac_chroma: encode_huffman_table(&STD_CHROMA_AC_BITS, &STD_CHROMA_AC_VALUES)?,
    })
}

/// Validate that dimensions can be represented in baseline JPEG markers.
pub fn validate_jpeg_baseline_dimensions(width: u32, height: u32) -> Result<(), JpegEncodeError> {
    if width == 0 || height == 0 {
        return Err(JpegEncodeError::EmptyDimensions);
    }
    if width > u32::from(u16::MAX) || height > u32::from(u16::MAX) {
        return Err(JpegEncodeError::DimensionsTooLarge { width, height });
    }
    Ok(())
}

/// Validate a user-provided restart interval.
pub fn validate_jpeg_baseline_restart_interval(
    restart_interval: Option<u16>,
) -> Result<(), JpegEncodeError> {
    if restart_interval == Some(0) {
        return Err(JpegEncodeError::InvalidRestartInterval);
    }
    Ok(())
}

/// Return JPEG component sampling factors for a public subsampling mode.
pub fn jpeg_baseline_sampling_for(subsampling: JpegSubsampling) -> JpegBaselineSampling {
    match subsampling {
        JpegSubsampling::Gray => JpegBaselineSampling {
            components: 1,
            h: [1, 0, 0],
            v: [1, 0, 0],
            max_h: 1,
            max_v: 1,
        },
        JpegSubsampling::Ybr444 => JpegBaselineSampling {
            components: 3,
            h: [1, 1, 1],
            v: [1, 1, 1],
            max_h: 1,
            max_v: 1,
        },
        JpegSubsampling::Ybr422 => JpegBaselineSampling {
            components: 3,
            h: [2, 1, 1],
            v: [1, 1, 1],
            max_h: 2,
            max_v: 1,
        },
        JpegSubsampling::Ybr420 => JpegBaselineSampling {
            components: 3,
            h: [2, 1, 1],
            v: [2, 1, 1],
            max_h: 2,
            max_v: 2,
        },
    }
}

/// Conservative upper bound for entropy bytes produced by the CPU encoder.
pub fn jpeg_baseline_entropy_capacity_bytes(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    restart_interval: Option<u16>,
) -> Result<usize, JpegEncodeError> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = u64::from(width.div_ceil(mcu_width));
    let mcu_rows = u64::from(height.div_ceil(mcu_height));
    let total_mcus = mcus_per_row
        .checked_mul(mcu_rows)
        .ok_or_else(|| JpegEncodeError::Internal("JPEG MCU count overflow".into()))?;
    let blocks_per_mcu = u64::from(
        sampling.h[0] * sampling.v[0]
            + sampling.h[1] * sampling.v[1]
            + sampling.h[2] * sampling.v[2],
    );
    let restart_markers = restart_interval.map_or(0, |interval| {
        total_mcus.saturating_sub(1) / u64::from(interval)
    });
    let capacity = total_mcus
        .checked_mul(blocks_per_mcu)
        .and_then(|blocks| blocks.checked_mul(512))
        .and_then(|bytes| bytes.checked_add(restart_markers.saturating_mul(2)))
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or_else(|| JpegEncodeError::Internal("JPEG entropy capacity overflow".into()))?;
    usize::try_from(capacity)
        .map_err(|_| JpegEncodeError::Internal("JPEG entropy capacity exceeds usize".into()))
}

/// Assemble a complete baseline JPEG codestream from entropy bytes and tables.
pub fn assemble_jpeg_baseline_frame(
    entropy: &[u8],
    width: u32,
    height: u32,
    tables: &JpegBaselineEncodeTables,
    options: JpegEncodeOptions,
    backend: JpegBackend,
) -> Result<EncodedJpeg, JpegEncodeError> {
    validate_jpeg_baseline_dimensions(width, height)?;
    validate_jpeg_baseline_restart_interval(options.restart_interval)?;

    let mut out = Vec::with_capacity(768usize.saturating_add(entropy.len()));
    write_marker(&mut out, 0xD8);
    write_dqt(&mut out, 0, &tables.q_luma)?;
    if tables.sampling.components == 3 {
        write_dqt(&mut out, 1, &tables.q_chroma)?;
    }
    if let Some(restart_interval) = options.restart_interval {
        write_dri(&mut out, restart_interval)?;
    }
    write_sof0(&mut out, width, height, tables.sampling)?;
    write_dht(&mut out, 0, 0, &STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES)?;
    write_dht(&mut out, 1, 0, &STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES)?;
    if tables.sampling.components == 3 {
        write_dht(&mut out, 0, 1, &STD_CHROMA_DC_BITS, &STD_CHROMA_DC_VALUES)?;
        write_dht(&mut out, 1, 1, &STD_CHROMA_AC_BITS, &STD_CHROMA_AC_VALUES)?;
    }
    write_sos(&mut out, tables.sampling.components)?;
    out.extend_from_slice(entropy);
    write_marker(&mut out, 0xD9);

    Ok(EncodedJpeg { data: out, backend })
}

pub(crate) fn assemble_jpeg_baseline_frame_with_quant_tables(
    entropy: &[u8],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: Option<&[u8; 64]>,
    backend: JpegBackend,
) -> Result<EncodedJpeg, JpegEncodeError> {
    validate_jpeg_baseline_dimensions(width, height)?;

    let mut out = Vec::with_capacity(768usize.saturating_add(entropy.len()));
    write_marker(&mut out, 0xD8);
    write_dqt(&mut out, 0, q_luma)?;
    if sampling.components == 3 {
        let q_chroma = q_chroma.ok_or_else(|| {
            JpegEncodeError::Internal("three-component DCT JPEG requires chroma quant table".into())
        })?;
        write_dqt(&mut out, 1, q_chroma)?;
    }
    write_sof0(&mut out, width, height, sampling)?;
    write_dht(&mut out, 0, 0, &STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES)?;
    write_dht(&mut out, 1, 0, &STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES)?;
    if sampling.components == 3 {
        write_dht(&mut out, 0, 1, &STD_CHROMA_DC_BITS, &STD_CHROMA_DC_VALUES)?;
        write_dht(&mut out, 1, 1, &STD_CHROMA_AC_BITS, &STD_CHROMA_AC_VALUES)?;
    }
    write_sos(&mut out, sampling.components)?;
    out.extend_from_slice(entropy);
    write_marker(&mut out, 0xD9);

    Ok(EncodedJpeg { data: out, backend })
}

/// JPEG zigzag coefficient order used by baseline entropy coding.
pub const JPEG_BASELINE_ZIGZAG: [u8; 64] = j2k_codec_math::jpeg::ZIGZAG;

fn encode_huffman_table(
    bits: &[u8; 16],
    values: &[u8],
) -> Result<JpegBaselineHuffmanTable, JpegEncodeError> {
    let mut table = JpegBaselineHuffmanTable {
        codes: [0; 256],
        lens: [0; 256],
    };
    let mut code = 0u16;
    let mut idx = 0usize;
    for (len_minus_1, count) in bits.iter().copied().enumerate() {
        let len = u8::try_from(len_minus_1 + 1).expect("JPEG Huffman code length is bounded by 16");
        for _ in 0..count {
            let symbol = *values.get(idx).ok_or_else(|| {
                JpegEncodeError::Internal("Huffman table count exceeds values".into())
            })?;
            table.codes[symbol as usize] = code;
            table.lens[symbol as usize] = len;
            code = code
                .checked_add(1)
                .ok_or_else(|| JpegEncodeError::Internal("Huffman code overflow".into()))?;
            idx += 1;
        }
        code <<= 1;
    }
    if idx != values.len() {
        return Err(JpegEncodeError::Internal(
            "Huffman values exceed table counts".into(),
        ));
    }
    Ok(table)
}

fn scaled_quant_table(base: &[u8; 64], quality: u8) -> [u8; 64] {
    let quality = quality.clamp(1, 100);
    let scale = if quality < 50 {
        5000 / u32::from(quality)
    } else {
        200 - u32::from(quality) * 2
    };
    let mut out = [0u8; 64];
    for (idx, value) in base.iter().copied().enumerate() {
        let scaled = (u32::from(value) * scale + 50) / 100;
        out[idx] = scaled.clamp(1, 255) as u8;
    }
    out
}

fn write_marker(out: &mut Vec<u8>, marker: u8) {
    out.push(0xFF);
    out.push(marker);
}

fn write_segment(
    out: &mut Vec<u8>,
    marker: u8,
    payload: &[u8],
    name: &'static str,
) -> Result<(), JpegEncodeError> {
    let len = payload
        .len()
        .checked_add(2)
        .and_then(|value| u16::try_from(value).ok())
        .ok_or(JpegEncodeError::SegmentTooLarge { name })?;
    write_marker(out, marker);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
    Ok(())
}

fn write_dqt(out: &mut Vec<u8>, table_id: u8, quant: &[u8; 64]) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(65);
    payload.push(table_id);
    for &natural_idx in &JPEG_BASELINE_ZIGZAG {
        payload.push(quant[natural_idx as usize]);
    }
    write_segment(out, 0xDB, &payload, "DQT")
}

fn write_dri(out: &mut Vec<u8>, restart_interval: u16) -> Result<(), JpegEncodeError> {
    write_segment(out, 0xDD, &restart_interval.to_be_bytes(), "DRI")
}

fn write_sof0(
    out: &mut Vec<u8>,
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(6 + sampling.components as usize * 3);
    payload.push(8);
    payload.extend_from_slice(
        &u16::try_from(height)
            .expect("JPEG SOF0 height validated as u16")
            .to_be_bytes(),
    );
    payload.extend_from_slice(
        &u16::try_from(width)
            .expect("JPEG SOF0 width validated as u16")
            .to_be_bytes(),
    );
    payload.push(sampling.components);
    for component in 0..sampling.components as usize {
        payload.push(u8::try_from(component + 1).expect("JPEG component id fits in u8"));
        payload.push((sampling.h[component] << 4) | sampling.v[component]);
        payload.push(u8::from(component != 0));
    }
    write_segment(out, 0xC0, &payload, "SOF0")
}

fn write_dht(
    out: &mut Vec<u8>,
    class: u8,
    table_id: u8,
    bits: &[u8; 16],
    values: &[u8],
) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(17 + values.len());
    payload.push((class << 4) | table_id);
    payload.extend_from_slice(bits);
    payload.extend_from_slice(values);
    write_segment(out, 0xC4, &payload, "DHT")
}

fn write_sos(out: &mut Vec<u8>, components: u8) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(4 + components as usize * 2);
    payload.push(components);
    for component in 0..components {
        payload.push(component + 1);
        payload.push(if component == 0 { 0x00 } else { 0x11 });
    }
    payload.push(0);
    payload.push(63);
    payload.push(0);
    write_segment(out, 0xDA, &payload, "SOS")
}

const STD_LUMA_Q: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

const STD_CHROMA_Q: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

const STD_LUMA_DC_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const STD_LUMA_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const STD_CHROMA_DC_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const STD_CHROMA_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const STD_LUMA_AC_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
const STD_LUMA_AC_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

const STD_CHROMA_AC_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const STD_CHROMA_AC_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];
