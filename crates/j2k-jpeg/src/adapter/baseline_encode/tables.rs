// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sampling, quality scaling, quantization, and Huffman table planning.

use super::types::{JpegBaselineEncodeTables, JpegBaselineHuffmanTable, JpegBaselineSampling};
use super::validation::validate_jpeg_baseline_restart_interval;
use crate::encoder::{JpegEncodeError, JpegEncodeOptions, JpegSubsampling};

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

/// Return JPEG component sampling factors for a public subsampling mode.
pub(super) fn jpeg_baseline_sampling_for(subsampling: JpegSubsampling) -> JpegBaselineSampling {
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
        let len =
            u8::try_from(len_minus_1 + 1).map_err(|_| JpegEncodeError::InternalInvariant {
                reason: "Huffman code length exceeds u8",
            })?;
        for _ in 0..count {
            let symbol = *values.get(idx).ok_or(JpegEncodeError::InternalInvariant {
                reason: "Huffman table count exceeds values",
            })?;
            table.codes[symbol as usize] = code;
            table.lens[symbol as usize] = len;
            code = code
                .checked_add(1)
                .ok_or(JpegEncodeError::InternalInvariant {
                    reason: "Huffman code overflow",
                })?;
            idx += 1;
        }
        code <<= 1;
    }
    if idx != values.len() {
        return Err(JpegEncodeError::InternalInvariant {
            reason: "Huffman values exceed table counts",
        });
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

pub(super) const STD_LUMA_DC_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
pub(super) const STD_LUMA_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
pub(super) const STD_CHROMA_DC_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
pub(super) const STD_CHROMA_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

pub(super) const STD_LUMA_AC_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
pub(super) const STD_LUMA_AC_VALUES: [u8; 162] = [
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

pub(super) const STD_CHROMA_AC_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
pub(super) const STD_CHROMA_AC_VALUES: [u8; 162] = [
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
