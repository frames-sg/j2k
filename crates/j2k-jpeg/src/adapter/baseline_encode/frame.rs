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
use crate::dct_contract::JpegDctImageError;
use crate::encoded_frame::{
    DctFrame, FrameComponent, FrameKind, FrameWriter, HuffmanTable, ScanComponent, ScanParameters,
};
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

#[derive(Clone, Copy)]
pub(crate) struct JpegBaselineFrameMetadata<'a> {
    pub(crate) sampling: JpegBaselineSampling,
    pub(crate) component_ids: &'a [u8],
    pub(crate) component_quant: &'a [[u8; 64]],
    pub(crate) app_segments: &'a [u8],
}

/// Assemble a baseline frame that keeps a source's component identifiers,
/// per-component quantization tables and color `APPn` segments.
pub(crate) fn assemble_jpeg_baseline_frame_with_metadata(
    entropy: &[u8],
    width: u32,
    height: u32,
    metadata: JpegBaselineFrameMetadata<'_>,
    backend: JpegBackend,
) -> Result<EncodedJpeg, JpegEncodeError> {
    let JpegBaselineFrameMetadata {
        sampling,
        component_ids,
        component_quant,
        app_segments,
    } = metadata;
    validate_jpeg_baseline_dimensions(width, height)?;
    validate_app_segments(app_segments)?;
    let count = usize::from(sampling.components);
    validate_frame_components(component_ids, component_quant, count)?;

    let extra = app_segments
        .len()
        .checked_add(component_quant.len().saturating_mul(65))
        .ok_or(JpegEncodeError::HostAllocationFailed { bytes: usize::MAX })?;
    let mut frame = FrameWriter::baseline(
        entropy
            .len()
            .checked_add(extra)
            .ok_or(JpegEncodeError::HostAllocationFailed { bytes: usize::MAX })?,
    )?;
    frame.raw_segments(app_segments)?;

    // Components sharing a table share its DQT segment.
    let mut unique_quant = [[0_u8; 64]; 3];
    let mut unique_quant_count = 0usize;
    let mut components = [FrameComponent {
        id: 0,
        h: 1,
        v: 1,
        quant_table: 0,
    }; 3];
    let mut scan = [ScanComponent {
        id: 0,
        dc_table: 0,
        ac_table: 0,
    }; 3];
    for (index, (table, &id)) in component_quant.iter().zip(component_ids).enumerate() {
        let slot = unique_quant[..unique_quant_count]
            .iter()
            .position(|existing| existing == table)
            .unwrap_or_else(|| {
                unique_quant[unique_quant_count] = *table;
                unique_quant_count += 1;
                unique_quant_count - 1
            });
        let huffman = u8::from(index != 0);
        components[index] = FrameComponent {
            id,
            h: sampling.h[index],
            v: sampling.v[index],
            quant_table: u8::try_from(slot).map_err(|_| JpegEncodeError::InternalInvariant {
                reason: "DCT quantization table identifier exceeds u8",
            })?,
        };
        scan[index] = ScanComponent {
            id,
            dc_table: huffman,
            ac_table: huffman,
        };
    }
    for (id, table) in (0_u8..).zip(&unique_quant[..unique_quant_count]) {
        frame.dqt(id, &table.map(u16::from))?;
    }
    frame.sof(
        FrameKind::Baseline,
        8,
        (width, height),
        &components[..count],
    )?;
    for (table_id, tables) in (0_u8..).zip(standard_huffman_tables(count == 3)) {
        if let Some([dc, ac]) = tables {
            frame.dht(0, table_id, dc)?;
            frame.dht(1, table_id, ac)?;
        }
    }
    frame.sos(
        &scan[..count],
        ScanParameters {
            ss: 0,
            se: 63,
            ah: 0,
            al: 0,
        },
    )?;
    frame.finish(entropy, backend)
}

/// The component count matches and component identifiers are unique.
fn validate_frame_components(
    component_ids: &[u8],
    component_quant: &[[u8; 64]],
    count: usize,
) -> Result<(), JpegEncodeError> {
    let invalid = |reason| JpegEncodeError::InvalidDctImage {
        reason: JpegDctImageError::InvalidMetadata { reason },
    };
    if component_ids.len() != count || component_quant.len() != count {
        return Err(invalid("DCT frame metadata does not match component count"));
    }
    if (1..component_ids.len()).any(|index| component_ids[..index].contains(&component_ids[index]))
    {
        return Err(invalid("DCT frame component identifiers must be unique"));
    }
    Ok(())
}

/// The standard luma tables, and the chroma tables when `chroma`.
fn standard_huffman_tables(chroma: bool) -> [Option<[HuffmanTable<'static>; 2]>; 2] {
    [
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
        chroma.then_some([
            HuffmanTable {
                bits: &STD_CHROMA_DC_BITS,
                values: &STD_CHROMA_DC_VALUES,
            },
            HuffmanTable {
                bits: &STD_CHROMA_AC_BITS,
                values: &STD_CHROMA_AC_VALUES,
            },
        ]),
    ]
}

/// Accept only complete `APPn` marker segments.
fn validate_app_segments(segments: &[u8]) -> Result<(), JpegEncodeError> {
    let mut offset = 0usize;
    while offset < segments.len() {
        let marker = *segments.get(offset).unwrap_or(&0);
        let kind = *segments.get(offset + 1).unwrap_or(&0);
        let length = segments
            .get(offset + 2..offset + 4)
            .and_then(|bytes| <[u8; 2]>::try_from(bytes).ok())
            .map(u16::from_be_bytes)
            .map(usize::from);
        let Some(end) = length
            .filter(|length| *length >= 2)
            .and_then(|length| offset.checked_add(length + 2))
        else {
            return Err(JpegEncodeError::InvalidDctImage {
                reason: JpegDctImageError::InvalidMetadata {
                    reason: "DCT frame APP marker is truncated",
                },
            });
        };
        if marker != 0xff || !(0xe0..=0xef).contains(&kind) || end > segments.len() {
            return Err(JpegEncodeError::InvalidDctImage {
                reason: JpegDctImageError::InvalidMetadata {
                    reason: "DCT frame metadata must contain complete APP markers only",
                },
            });
        }
        offset = end;
    }
    Ok(())
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
            huffman: standard_huffman_tables(sampling.components == 3),
            restart_interval,
        },
        entropy,
        backend,
    )
}
