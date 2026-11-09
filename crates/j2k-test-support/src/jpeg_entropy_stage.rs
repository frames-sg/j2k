// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed coefficients for the baseline entropy-stage tests.
//!
//! A GPU backend's entropy stage codes these coefficients from a buffer,
//! skipping its colour conversion and forward DCT; the CPU encoder's frame of
//! the same samples codes exactly the same coefficients, because they are the
//! shared integer transform of `j2k_codec_math::jpeg::fdct` applied to the
//! CPU's own component samples. So the stage's bytes must equal the entropy
//! segment of the CPU's frame.
//!
//! A test crate that cannot depend on j2k-jpeg, such as the CUDA engine's,
//! takes the tables from [`jpeg_standard_tables`] and checks its segments
//! against [`JPEG_ENTROPY_STAGE_SHA256`]. j2k-jpeg's tests tie both to the
//! CPU encoder.

use j2k_codec_math::jpeg::{derive_canonical_huffman, fdct::fdct_quantize, ycbcr};
use sha2::{Digest, Sha256};

use crate::{JpegEncodeMatrixCase, JpegEncodeMatrixSampling};

/// Framed SHA-256 of the CPU encoder's entropy segments of
/// [`jpeg_entropy_stage_cases`], in order.
///
/// See [`jpeg_entropy_stage_digest`].
pub const JPEG_ENTROPY_STAGE_SHA256: &str =
    "9b23e7ce60e3eafa3eeee4370b8075e7e35e28ebbdfe93fe10dd9f6f196bedc8";

const DIGEST_DOMAIN: &[u8] = b"J2K-JPEG-BASELINE-ENTROPY-STAGE\0";

/// Frame size of the entropy-stage cases: five MCUs by three, ending inside
/// an MCU in both directions.
pub const JPEG_ENTROPY_STAGE_SIZE: (u32, u32) = (37, 21);

/// The entropy-stage cases: gray and 4:4:4 at [`JPEG_ENTROPY_STAGE_SIZE`].
///
/// Each runs at quality 20 and 100, without a restart interval and with
/// intervals of one and three MCUs, so markers fall inside and at the ends of
/// MCU rows.
pub fn jpeg_entropy_stage_cases() -> Vec<JpegEncodeMatrixCase> {
    let mut cases = Vec::new();
    for sampling in [
        JpegEncodeMatrixSampling::Gray,
        JpegEncodeMatrixSampling::Ybr444,
    ] {
        for quality in [20, 100] {
            for restart_interval in [None, Some(1), Some(3)] {
                cases.push(JpegEncodeMatrixCase {
                    sampling,
                    width: JPEG_ENTROPY_STAGE_SIZE.0,
                    height: JPEG_ENTROPY_STAGE_SIZE.1,
                    quality,
                    restart_interval,
                });
            }
        }
    }
    cases
}

/// The framed SHA-256 of entropy segments: each segment's length, then its
/// bytes.
///
/// # Panics
///
/// Panics if a segment's length does not fit in a `u64`.
pub fn jpeg_entropy_stage_digest<'a>(segments: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut digest = Sha256::new();
    digest.update(DIGEST_DOMAIN);
    for segment in segments {
        digest.update(
            u64::try_from(segment.len())
                .expect("segment length fits u64")
                .to_le_bytes(),
        );
        digest.update(segment);
    }
    format!("{:x}", digest.finalize())
}

/// Huffman codes and lengths by symbol, as a baseline encoder looks them up.
pub type JpegHuffmanBySymbol = ([u16; 256], [u8; 256]);

/// The tables of a standard baseline encode at one quality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JpegStandardTables {
    /// Luma quantizers in natural order.
    pub q_luma: [u8; 64],
    /// Chroma quantizers in natural order.
    pub q_chroma: [u8; 64],
    /// Huffman tables: luma DC, luma AC, chroma DC and chroma AC.
    pub huffman: [JpegHuffmanBySymbol; 4],
}

/// The Annex K quantization tables scaled to `quality` as IJG libjpeg scales
/// them, and the Annex K Huffman tables.
///
/// These are the tables of `j2k_jpeg::adapter::baseline_encode_tables`, for
/// test crates that cannot depend on j2k-jpeg; j2k-jpeg's tests check that the
/// two agree.
///
/// # Panics
///
/// Never: the Annex K Huffman tables are valid canonical tables.
pub fn jpeg_standard_tables(quality: u8) -> JpegStandardTables {
    JpegStandardTables {
        q_luma: scaled_quantizers(&ANNEX_K_LUMA_Q, quality),
        q_chroma: scaled_quantizers(&ANNEX_K_CHROMA_Q, quality),
        huffman: [
            huffman_by_symbol(&LUMA_DC_BITS, &DC_VALUES),
            huffman_by_symbol(&LUMA_AC_BITS, &LUMA_AC_VALUES),
            huffman_by_symbol(&CHROMA_DC_BITS, &DC_VALUES),
            huffman_by_symbol(&CHROMA_AC_BITS, &CHROMA_AC_VALUES),
        ],
    }
}

fn scaled_quantizers(base: &[u8; 64], quality: u8) -> [u8; 64] {
    let quality = u32::from(quality.clamp(1, 100));
    let scale = if quality < 50 {
        5000 / quality
    } else {
        200 - quality * 2
    };
    base.map(|value| {
        u8::try_from(((u32::from(value) * scale + 50) / 100).clamp(1, 255))
            .expect("clamped quantizer fits u8")
    })
}

fn huffman_by_symbol(bits: &[u8; 16], values: &[u8]) -> JpegHuffmanBySymbol {
    let derived = derive_canonical_huffman(bits, values.len()).expect("Annex K table derives");
    let mut codes = [0u16; 256];
    let mut lengths = [0u8; 256];
    for (index, &symbol) in values.iter().enumerate() {
        codes[usize::from(symbol)] = derived.huffcode[index];
        lengths[usize::from(symbol)] = derived.huffsize[index];
    }
    (codes, lengths)
}

/// T.81 Annex K.1 luminance quantizers, natural order.
const ANNEX_K_LUMA_Q: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// T.81 Annex K.1 chrominance quantizers, natural order.
const ANNEX_K_CHROMA_Q: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// T.81 Annex K.3 Huffman tables: code counts by length, then symbols.
const LUMA_DC_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const CHROMA_DC_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const LUMA_AC_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
const LUMA_AC_VALUES: [u8; 162] = [
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
const CHROMA_AC_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const CHROMA_AC_VALUES: [u8; 162] = [
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

/// Natural-order quantized coefficients of every block of an unsubsampled
/// (grayscale or 4:4:4) frame, in MCU order with one block per component per
/// MCU, as the GPU entropy stages read them.
///
/// `samples` are tightly packed interleaved Gray8 (`channels == 1`) or Rgb8
/// (`channels == 3`) pixels. RGB converts with the forward transform of
/// `j2k_codec_math::jpeg::ycbcr`, and the samples beyond the frame repeat its
/// last column and row, as the CPU encoder codes them.
///
/// # Panics
///
/// Panics if `channels` is not 1 or 3, or `samples` does not hold a
/// non-empty `width` x `height` frame.
pub fn jpeg_entropy_stage_coefficients(
    samples: &[u8],
    (width, height): (u32, u32),
    channels: usize,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
) -> Vec<[i32; 64]> {
    let (width, height) = (width as usize, height as usize);
    assert!(channels == 1 || channels == 3, "grayscale or RGB samples");
    assert!(width > 0 && height > 0 && samples.len() == width * height * channels);
    let clamp = |value: i32| u8::try_from(value.clamp(0, 255)).expect("clamped sample fits u8");
    let component_sample = |pixel: usize, component: usize| -> u8 {
        if channels == 1 {
            return samples[pixel];
        }
        let rgb = &samples[pixel * 3..pixel * 3 + 3];
        let (r, g, b) = (i32::from(rgb[0]), i32::from(rgb[1]), i32::from(rgb[2]));
        clamp(match component {
            0 => ycbcr::rgb_to_y(r, g, b),
            1 => ycbcr::rgb_to_cb(r, g, b, 128),
            _ => ycbcr::rgb_to_cr(r, g, b, 128),
        })
    };
    let mut blocks = Vec::new();
    for mcu_y in 0..height.div_ceil(8) {
        for mcu_x in 0..width.div_ceil(8) {
            for component in 0..channels {
                let block: [u8; 64] = core::array::from_fn(|index| {
                    let x = (mcu_x * 8 + index % 8).min(width - 1);
                    let y = (mcu_y * 8 + index / 8).min(height - 1);
                    component_sample(y * width + x, component)
                });
                let quant = if component == 0 { q_luma } else { q_chroma };
                blocks.push(fdct_quantize(&block, quant).map(i32::from));
            }
        }
    }
    blocks
}

/// The entropy-coded segment of a baseline JPEG frame: the bytes between its
/// SOS header and EOI.
///
/// # Panics
///
/// Panics if `frame` is not a sequence of marker segments from SOI to an SOS
/// segment.
pub fn jpeg_entropy_segment(frame: &[u8]) -> &[u8] {
    assert_eq!(frame[..2], [0xff, 0xd8], "frame starts with SOI");
    // Walk the segments rather than search for 0xFFDA, which a table's
    // payload may contain.
    let mut position = 2;
    loop {
        assert_eq!(frame[position], 0xff, "marker at {position}");
        let length = usize::from(u16::from_be_bytes([
            frame[position + 2],
            frame[position + 3],
        ]));
        if frame[position + 1] == 0xda {
            return &frame[position + 2 + length..frame.len() - 2];
        }
        position += 2 + length;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_grey_blocks_quantize_to_a_lone_dc() {
        let samples = [200u8; 9 * 3];
        let blocks = jpeg_entropy_stage_coefficients(&samples, (3, 3), 3, &[1; 64], &[2; 64]);
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0][0], 8 * (200 - 128));
        assert_eq!(blocks[1][0], 0);
        assert!(blocks
            .iter()
            .all(|block| block[1..].iter().all(|&ac| ac == 0)));
    }

    #[test]
    fn standard_tables_scale_annex_k_and_derive_its_codes() {
        assert_eq!(jpeg_standard_tables(50).q_luma, ANNEX_K_LUMA_Q);
        assert_eq!(jpeg_standard_tables(100).q_chroma, [1; 64]);
        let tables = jpeg_standard_tables(20);
        // Scale 250: (16 * 250 + 50) / 100.
        assert_eq!(tables.q_luma[0], 40);
        // Annex K.3: luma DC category 0 is `00`, category 11 is `111111110`.
        let (codes, lengths) = &tables.huffman[0];
        assert_eq!((codes[0], lengths[0]), (0b00, 2));
        assert_eq!((codes[11], lengths[11]), (0b1_1111_1110, 9));
        // Luma AC end of block is `1010`.
        let (codes, lengths) = &tables.huffman[1];
        assert_eq!((codes[0x00], lengths[0x00]), (0b1010, 4));
    }

    #[test]
    fn stage_cases_cover_both_samplings_qualities_and_restarts() {
        let cases = jpeg_entropy_stage_cases();
        assert_eq!(cases.len(), 12);
        assert!(cases
            .iter()
            .all(|case| (case.width, case.height) == JPEG_ENTROPY_STAGE_SIZE));
    }

    #[test]
    fn entropy_segment_spans_sos_header_to_eoi() {
        let frame = [
            0xff, 0xd8, 0xff, 0xda, 0x00, 0x03, 0x01, 0xab, 0xcd, 0xff, 0xd9,
        ];
        assert_eq!(jpeg_entropy_segment(&frame), [0xab, 0xcd]);
        // A table whose payload holds the SOS marker's bytes.
        let frame = [
            0xff, 0xd8, 0xff, 0xdb, 0x00, 0x04, 0xff, 0xda, 0xff, 0xda, 0x00, 0x03, 0x01, 0xab,
            0xcd, 0xff, 0xd9,
        ];
        assert_eq!(jpeg_entropy_segment(&frame), [0xab, 0xcd]);
    }
}
