// SPDX-License-Identifier: Apache-2.0

use crate::error::{JpegError, MarkerKind};
use crate::info::{ColorSpace, SamplingFactors, SofKind};
use crate::parse::header::parse_header;
use crate::parse::tables::RawHuffmanTable;
use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalFast420PacketError {
    Decode(JpegError),
    UnsupportedSof(SofKind),
    UnsupportedColorSpace(ColorSpace),
    UnsupportedSampling,
    RestartIntervalUnsupported,
    UnsupportedComponentOrder,
    MissingScan,
    MissingQuantTable { slot: u8 },
    MissingHuffmanTable { kind: TableKind, slot: u8 },
    EntropyMarkerUnsupported { marker: u8 },
    TruncatedEntropy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKind {
    Dc,
    Ac,
}

impl From<JpegError> for MetalFast420PacketError {
    fn from(value: JpegError) -> Self {
        Self::Decode(value)
    }
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalHuffmanTable {
    pub bits: [u8; 16],
    pub values_len: u16,
    pub values: [u8; 256],
}

impl MetalHuffmanTable {
    fn from_raw(raw: &RawHuffmanTable) -> Self {
        let mut values = [0u8; 256];
        let slice = raw.values.as_slice();
        values[..slice.len()].copy_from_slice(slice);
        Self {
            bits: raw.bits,
            values_len: slice.len() as u16,
            values,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegMetalFast420PacketV1 {
    pub dimensions: (u32, u32),
    pub mcus_per_row: u32,
    pub mcu_rows: u32,
    pub y_quant: [u16; 64],
    pub cb_quant: [u16; 64],
    pub cr_quant: [u16; 64],
    pub y_dc_table: MetalHuffmanTable,
    pub y_ac_table: MetalHuffmanTable,
    pub cb_dc_table: MetalHuffmanTable,
    pub cb_ac_table: MetalHuffmanTable,
    pub cr_dc_table: MetalHuffmanTable,
    pub cr_ac_table: MetalHuffmanTable,
    pub entropy_bytes: Vec<u8>,
}

pub fn build_metal_fast420_packet(
    bytes: &[u8],
) -> Result<JpegMetalFast420PacketV1, MetalFast420PacketError> {
    let header = parse_header(bytes)?;
    if !matches!(header.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return Err(MetalFast420PacketError::UnsupportedSof(header.sof_kind));
    }
    if header.bit_depth != 8 {
        return Err(MetalFast420PacketError::Decode(
            JpegError::UnsupportedBitDepth {
                depth: header.bit_depth,
            },
        ));
    }
    if header.color_space() != ColorSpace::YCbCr {
        return Err(MetalFast420PacketError::UnsupportedColorSpace(
            header.color_space(),
        ));
    }
    if header.restart_interval.is_some() {
        return Err(MetalFast420PacketError::RestartIntervalUnsupported);
    }
    if header.sampling != SamplingFactors::from_components(&[(2, 2), (1, 1), (1, 1)]) {
        return Err(MetalFast420PacketError::UnsupportedSampling);
    }
    let scan = header
        .scan
        .as_ref()
        .ok_or(MetalFast420PacketError::MissingScan)?;
    if header.component_ids.as_slice() != [1, 2, 3] || scan.components.len() != 3 {
        return Err(MetalFast420PacketError::UnsupportedComponentOrder);
    }
    for (expected_id, component) in [1u8, 2, 3].into_iter().zip(scan.components.iter()) {
        if component.id != expected_id {
            return Err(MetalFast420PacketError::UnsupportedComponentOrder);
        }
    }

    let y_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 0)?;
    let cb_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 1)?;
    let cr_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 2)?;
    let y_dc_table = huffman_table(
        &header.huffman_tables.dc,
        TableKind::Dc,
        scan.components[0].dc_table,
    )?;
    let y_ac_table = huffman_table(
        &header.huffman_tables.ac,
        TableKind::Ac,
        scan.components[0].ac_table,
    )?;
    let cb_dc_table = huffman_table(
        &header.huffman_tables.dc,
        TableKind::Dc,
        scan.components[1].dc_table,
    )?;
    let cb_ac_table = huffman_table(
        &header.huffman_tables.ac,
        TableKind::Ac,
        scan.components[1].ac_table,
    )?;
    let cr_dc_table = huffman_table(
        &header.huffman_tables.dc,
        TableKind::Dc,
        scan.components[2].dc_table,
    )?;
    let cr_ac_table = huffman_table(
        &header.huffman_tables.ac,
        TableKind::Ac,
        scan.components[2].ac_table,
    )?;

    let entropy_offset = header
        .sos_offset
        .ok_or(MetalFast420PacketError::MissingScan)?;
    let entropy_bytes = extract_entropy_bytes(&bytes[entropy_offset..])?;
    let (width, height) = header.dimensions;

    Ok(JpegMetalFast420PacketV1 {
        dimensions: header.dimensions,
        mcus_per_row: width.div_ceil(16),
        mcu_rows: height.div_ceil(16),
        y_quant,
        cb_quant,
        cr_quant,
        y_dc_table,
        y_ac_table,
        cb_dc_table,
        cb_ac_table,
        cr_dc_table,
        cr_ac_table,
        entropy_bytes,
    })
}

pub fn build_metal_fast420_packet_for_decoder(
    decoder: &crate::decoder::Decoder<'_>,
) -> Result<JpegMetalFast420PacketV1, MetalFast420PacketError> {
    if !decoder.plan.matches_fast_tile_shape() {
        return Err(MetalFast420PacketError::UnsupportedSampling);
    }
    build_metal_fast420_packet(decoder.bytes)
}

fn quant_for_component(
    quant_table_ids: &[u8],
    tables: &[Option<[u16; 64]>; 4],
    component_idx: usize,
) -> Result<[u16; 64], MetalFast420PacketError> {
    let slot = *quant_table_ids
        .get(component_idx)
        .ok_or(MetalFast420PacketError::UnsupportedComponentOrder)?;
    tables[slot as usize].ok_or(MetalFast420PacketError::MissingQuantTable { slot })
}

fn huffman_table(
    tables: &[Option<RawHuffmanTable>; 4],
    kind: TableKind,
    slot: u8,
) -> Result<MetalHuffmanTable, MetalFast420PacketError> {
    let raw = tables[slot as usize]
        .as_ref()
        .ok_or(MetalFast420PacketError::MissingHuffmanTable { kind, slot })?;
    Ok(MetalHuffmanTable::from_raw(raw))
}

fn extract_entropy_bytes(bytes: &[u8]) -> Result<Vec<u8>, MetalFast420PacketError> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut pos = 0usize;
    while pos < bytes.len() {
        let byte = bytes[pos];
        if byte != 0xFF {
            out.push(byte);
            pos += 1;
            continue;
        }
        let next = *bytes
            .get(pos + 1)
            .ok_or(MetalFast420PacketError::TruncatedEntropy)?;
        match next {
            0x00 => {
                out.push(0xFF);
                pos += 2;
            }
            0xD9 => return Ok(out),
            marker => {
                return Err(MetalFast420PacketError::EntropyMarkerUnsupported { marker });
            }
        }
    }
    Err(MetalFast420PacketError::Decode(JpegError::MissingMarker {
        marker: MarkerKind::Eoi,
    }))
}
