// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sampling, quality scaling, quantization, and Huffman table planning.

use super::types::{JpegBaselineEncodeTables, JpegBaselineSampling};
use super::validation::validate_jpeg_baseline_restart_interval;
use crate::encoder::{JpegEncodeError, JpegEncodeOptions, JpegSubsampling};
pub(super) use crate::standard_huffman::{
    STD_CHROMA_AC_BITS, STD_CHROMA_AC_VALUES, STD_CHROMA_DC_BITS, STD_CHROMA_DC_VALUES,
    STD_LUMA_AC_BITS, STD_LUMA_AC_VALUES, STD_LUMA_DC_BITS, STD_LUMA_DC_VALUES,
};
use crate::standard_huffman::{
    STD_CHROMA_AC_CANONICAL, STD_CHROMA_DC_CANONICAL, STD_LUMA_AC_CANONICAL, STD_LUMA_DC_CANONICAL,
};

/// Build quantization, sampling, and Huffman tables for baseline encoding.
pub fn baseline_encode_tables(
    options: JpegEncodeOptions,
) -> Result<JpegBaselineEncodeTables, JpegEncodeError> {
    validate_jpeg_baseline_restart_interval(options.restart_interval)?;
    Ok(JpegBaselineEncodeTables {
        sampling: jpeg_baseline_sampling_for(options.subsampling),
        q_luma: scaled_quant_table(&STD_LUMA_Q, options.quality),
        q_chroma: scaled_quant_table(&STD_CHROMA_Q, options.quality),
        huff_dc_luma: STD_LUMA_DC_CANONICAL,
        huff_ac_luma: STD_LUMA_AC_CANONICAL,
        huff_dc_chroma: STD_CHROMA_DC_CANONICAL,
        huff_ac_chroma: STD_CHROMA_AC_CANONICAL,
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
