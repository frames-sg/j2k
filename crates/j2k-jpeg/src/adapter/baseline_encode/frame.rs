// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact baseline JPEG marker, table-segment, and frame assembly.

use super::tables::{
    STD_CHROMA_AC_BITS, STD_CHROMA_AC_VALUES, STD_CHROMA_DC_BITS, STD_CHROMA_DC_VALUES,
    STD_LUMA_AC_BITS, STD_LUMA_AC_VALUES, STD_LUMA_DC_BITS, STD_LUMA_DC_VALUES,
};
use super::types::{JpegBaselineEncodeTables, JpegBaselineSampling};
use super::validation::{
    validate_jpeg_baseline_dimensions, validate_jpeg_baseline_restart_interval,
};
use crate::encoded_frame::{DctFrame, FrameKind, FrameWriter, HuffmanTable};
use crate::encoder::{EncodedJpeg, JpegBackend, JpegEncodeError, JpegEncodeOptions};

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

    assemble_standard_frame(
        entropy,
        (width, height),
        tables.sampling,
        &tables.q_luma,
        Some(&tables.q_chroma),
        options.restart_interval,
        backend,
    )
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

    assemble_standard_frame(
        entropy,
        (width, height),
        sampling,
        q_luma,
        q_chroma,
        None,
        backend,
    )
}

fn assemble_standard_frame(
    entropy: &[u8],
    dimensions: (u32, u32),
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: Option<&[u8; 64]>,
    restart_interval: Option<u16>,
    backend: JpegBackend,
) -> Result<EncodedJpeg, JpegEncodeError> {
    let frame = FrameWriter::baseline(entropy.len())?;
    let mut quant = [q_luma.map(u16::from), [0; 64]];
    if sampling.components == 3 {
        quant[1] = q_chroma
            .ok_or(JpegEncodeError::InternalInvariant {
                reason: "three-component DCT JPEG requires chroma quant table",
            })?
            .map(u16::from);
    }
    frame.assemble_dct(
        DctFrame {
            kind: FrameKind::Baseline,
            precision: 8,
            dimensions,
            sampling,
            quant: &quant,
            huffman: [
                Some([
                    HuffmanTable {
                        bits: &STD_LUMA_DC_BITS,
                        values: &STD_LUMA_DC_VALUES,
                    },
                    HuffmanTable {
                        bits: &STD_LUMA_AC_BITS,
                        values: &STD_LUMA_AC_VALUES,
                    },
                ]),
                (sampling.components == 3).then_some([
                    HuffmanTable {
                        bits: &STD_CHROMA_DC_BITS,
                        values: &STD_CHROMA_DC_VALUES,
                    },
                    HuffmanTable {
                        bits: &STD_CHROMA_AC_BITS,
                        values: &STD_CHROMA_AC_VALUES,
                    },
                ]),
            ],
            restart_interval,
        },
        entropy,
        backend,
    )
}
