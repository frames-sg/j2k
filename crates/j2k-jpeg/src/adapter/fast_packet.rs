// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::decoder::Decoder;
use crate::entropy::huffman::{derive_canonical_huffman, CanonicalHuffmanDerivation};
use crate::error::{HuffmanFailure, JpegError, MarkerKind};
use crate::info::{ColorSpace, SamplingFactors, SofKind};
use crate::internal::checkpoint::{build_checkpoint_plan, DeviceCheckpoint};
use crate::parse::header::parse_header;
use crate::parse::scan::ScanComponent;
use crate::parse::tables::{HuffmanValues, RawHuffmanTable};
use alloc::vec::Vec;

const MAX_NONRESTART_ENTROPY_CHECKPOINTS: u32 = 2048;

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
/// Error while building a backend fast-path JPEG packet.
pub enum FastPacketError {
    /// Header or entropy decode failed.
    Decode(JpegError),
    /// JPEG SOF kind is not supported by the fast path.
    UnsupportedSof(SofKind),
    /// JPEG color space is not supported by the selected fast path.
    UnsupportedColorSpace(ColorSpace),
    /// JPEG component sampling does not match the selected fast path.
    UnsupportedSampling,
    /// Scan component order does not match SOF component order.
    UnsupportedComponentOrder,
    /// Stream does not contain a scan payload.
    MissingScan,
    /// Referenced quantization table is absent.
    MissingQuantTable {
        /// Quantization table slot.
        slot: u8,
    },
    /// Referenced Huffman table is absent.
    MissingHuffmanTable {
        /// Huffman table class.
        kind: TableKind,
        /// Huffman table slot.
        slot: u8,
    },
    /// Entropy payload contains a marker unsupported by the fast path.
    EntropyMarkerUnsupported {
        /// Raw marker byte following `0xff`.
        marker: u8,
    },
    /// Entropy payload ended before the packet could be built.
    TruncatedEntropy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
/// Huffman table class used by fast-path packet builders.
pub enum TableKind {
    /// DC Huffman table.
    Dc,
    /// AC Huffman table.
    Ac,
}

impl From<JpegError> for FastPacketError {
    fn from(value: JpegError) -> Self {
        Self::Decode(value)
    }
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
/// Huffman table payload copied into backend-compatible packet structs.
pub struct JpegHuffmanTable {
    /// JPEG BITS counts for code lengths 1 through 16.
    pub bits: [u8; 16],
    /// Number of populated entries in `values`.
    pub values_len: u16,
    /// JPEG HUFFVAL symbols padded to fixed capacity.
    pub values: [u8; 256],
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
/// Canonical Annex C Huffman table derivation for backend packet tables.
pub struct JpegCanonicalHuffmanTable {
    /// Smallest canonical code for each code length 1 through 16.
    pub min_code: [i32; 17],
    /// Largest canonical code for each code length 1 through 16.
    pub max_code: [i32; 17],
    /// Symbol-value offset for each code length 1 through 16.
    pub val_offset: [i32; 17],
    /// Canonical Huffman code for each populated symbol index.
    pub huffcode: [u16; 256],
    /// Canonical Huffman code length for each populated symbol index.
    pub huffsize: [u8; 256],
    /// Number of populated canonical entries.
    pub huffsize_len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
/// Entropy decoder resume point for fast-path packet decoding.
pub struct JpegEntropyCheckpointV1 {
    /// MCU index for this checkpoint.
    pub mcu_index: u32,
    /// Byte offset into the entropy payload.
    pub entropy_pos: u32,
    /// Buffered entropy bits.
    pub bit_acc: u64,
    /// Number of valid bits in `bit_acc`.
    pub bit_count: u32,
    /// Previous Y DC predictor.
    pub y_prev_dc: i32,
    /// Previous Cb DC predictor.
    pub cb_prev_dc: i32,
    /// Previous Cr DC predictor.
    pub cr_prev_dc: i32,
    /// Reserved for ABI-compatible future expansion.
    pub reserved: u32,
}

impl JpegHuffmanTable {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "canonical JPEG Huffman tables contain at most 256 symbol positions"
    )]
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

    /// Derive the JPEG Annex C canonical code tables for this packet table.
    #[doc(hidden)]
    pub fn derive_canonical(&self) -> Result<JpegCanonicalHuffmanTable, JpegError> {
        let values_len = usize::from(self.values_len);
        if values_len > self.values.len() {
            return Err(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::CodeOverflow,
            });
        }
        let raw = RawHuffmanTable {
            bits: self.bits,
            values: HuffmanValues::from_slice(&self.values[..values_len]),
        };
        let derivation = derive_canonical_huffman(&raw)?;
        Ok(JpegCanonicalHuffmanTable::from_canonical_derivation(
            &derivation,
        ))
    }
}

impl JpegCanonicalHuffmanTable {
    fn from_canonical_derivation(value: &CanonicalHuffmanDerivation) -> Self {
        Self {
            min_code: value.min_code,
            max_code: value.max_code,
            val_offset: value.val_offset,
            huffcode: value.huffcode,
            huffsize: value.huffsize,
            huffsize_len: value.huffsize_len,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
/// Backend fast-path packet for 8-bit 4:2:0 JPEG tiles.
pub struct JpegFast420PacketV1 {
    /// Image dimensions as `(width, height)` in pixels.
    pub dimensions: (u32, u32),
    /// Number of MCUs per row.
    pub mcus_per_row: u32,
    /// Number of MCU rows.
    pub mcu_rows: u32,
    /// Restart interval in MCUs, or zero when absent.
    pub restart_interval_mcus: u32,
    /// Byte offsets of restart-addressable entropy segments.
    pub restart_offsets: Vec<u32>,
    /// Entropy resume checkpoints.
    pub entropy_checkpoints: Vec<JpegEntropyCheckpointV1>,
    /// Y quantization table in natural order.
    pub y_quant: [u16; 64],
    /// Cb quantization table in natural order.
    pub cb_quant: [u16; 64],
    /// Cr quantization table in natural order.
    pub cr_quant: [u16; 64],
    /// Y DC Huffman table.
    pub y_dc_table: JpegHuffmanTable,
    /// Y AC Huffman table.
    pub y_ac_table: JpegHuffmanTable,
    /// Cb DC Huffman table.
    pub cb_dc_table: JpegHuffmanTable,
    /// Cb AC Huffman table.
    pub cb_ac_table: JpegHuffmanTable,
    /// Cr DC Huffman table.
    pub cr_dc_table: JpegHuffmanTable,
    /// Cr AC Huffman table.
    pub cr_ac_table: JpegHuffmanTable,
    /// Entropy-coded scan bytes.
    pub entropy_bytes: Vec<u8>,
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
/// Backend fast-path packet for 8-bit 4:2:2 JPEG tiles.
pub struct JpegFast422PacketV1 {
    /// Image dimensions as `(width, height)` in pixels.
    pub dimensions: (u32, u32),
    /// Number of MCUs per row.
    pub mcus_per_row: u32,
    /// Number of MCU rows.
    pub mcu_rows: u32,
    /// Restart interval in MCUs, or zero when absent.
    pub restart_interval_mcus: u32,
    /// Byte offsets of restart-addressable entropy segments.
    pub restart_offsets: Vec<u32>,
    /// Entropy resume checkpoints.
    pub entropy_checkpoints: Vec<JpegEntropyCheckpointV1>,
    /// Y quantization table in natural order.
    pub y_quant: [u16; 64],
    /// Cb quantization table in natural order.
    pub cb_quant: [u16; 64],
    /// Cr quantization table in natural order.
    pub cr_quant: [u16; 64],
    /// Y DC Huffman table.
    pub y_dc_table: JpegHuffmanTable,
    /// Y AC Huffman table.
    pub y_ac_table: JpegHuffmanTable,
    /// Cb DC Huffman table.
    pub cb_dc_table: JpegHuffmanTable,
    /// Cb AC Huffman table.
    pub cb_ac_table: JpegHuffmanTable,
    /// Cr DC Huffman table.
    pub cr_dc_table: JpegHuffmanTable,
    /// Cr AC Huffman table.
    pub cr_ac_table: JpegHuffmanTable,
    /// Entropy-coded scan bytes.
    pub entropy_bytes: Vec<u8>,
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
/// Backend fast-path packet for 8-bit 4:4:4 JPEG tiles.
pub struct JpegFast444PacketV1 {
    /// Image dimensions as `(width, height)` in pixels.
    pub dimensions: (u32, u32),
    /// Number of MCUs per row.
    pub mcus_per_row: u32,
    /// Number of MCU rows.
    pub mcu_rows: u32,
    /// Restart interval in MCUs, or zero when absent.
    pub restart_interval_mcus: u32,
    /// Byte offsets of restart-addressable entropy segments.
    pub restart_offsets: Vec<u32>,
    /// Entropy resume checkpoints.
    pub entropy_checkpoints: Vec<JpegEntropyCheckpointV1>,
    /// Y quantization table in natural order.
    pub y_quant: [u16; 64],
    /// Cb quantization table in natural order.
    pub cb_quant: [u16; 64],
    /// Cr quantization table in natural order.
    pub cr_quant: [u16; 64],
    /// Y DC Huffman table.
    pub y_dc_table: JpegHuffmanTable,
    /// Y AC Huffman table.
    pub y_ac_table: JpegHuffmanTable,
    /// Cb DC Huffman table.
    pub cb_dc_table: JpegHuffmanTable,
    /// Cb AC Huffman table.
    pub cb_ac_table: JpegHuffmanTable,
    /// Cr DC Huffman table.
    pub cr_dc_table: JpegHuffmanTable,
    /// Cr AC Huffman table.
    pub cr_ac_table: JpegHuffmanTable,
    /// Entropy-coded scan bytes.
    pub entropy_bytes: Vec<u8>,
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
/// Backend fast-path packet for 8-bit grayscale JPEG tiles.
pub struct JpegGrayPacketV1 {
    /// Image dimensions as `(width, height)` in pixels.
    pub dimensions: (u32, u32),
    /// Number of MCUs per row.
    pub mcus_per_row: u32,
    /// Number of MCU rows.
    pub mcu_rows: u32,
    /// Restart interval in MCUs, or zero when absent.
    pub restart_interval_mcus: u32,
    /// Byte offsets of restart-addressable entropy segments.
    pub restart_offsets: Vec<u32>,
    /// Y quantization table in natural order.
    pub y_quant: [u16; 64],
    /// Y DC Huffman table.
    pub y_dc_table: JpegHuffmanTable,
    /// Y AC Huffman table.
    pub y_ac_table: JpegHuffmanTable,
    /// Entropy-coded scan bytes.
    pub entropy_bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct FastLayout {
    sampling: &'static [(u8, u8)],
    allow_rgb: bool,
    mcu_width: u32,
    mcu_height: u32,
}

const FAST420_LAYOUT: FastLayout = FastLayout {
    sampling: &[(2, 2), (1, 1), (1, 1)],
    allow_rgb: false,
    mcu_width: 16,
    mcu_height: 16,
};

const FAST422_LAYOUT: FastLayout = FastLayout {
    sampling: &[(2, 1), (1, 1), (1, 1)],
    allow_rgb: false,
    mcu_width: 16,
    mcu_height: 8,
};

const FAST444_LAYOUT: FastLayout = FastLayout {
    sampling: &[(1, 1), (1, 1), (1, 1)],
    allow_rgb: true,
    mcu_width: 8,
    mcu_height: 8,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct ColorFastPacketParts {
    dimensions: (u32, u32),
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offsets: Vec<u32>,
    entropy_checkpoints: Vec<JpegEntropyCheckpointV1>,
    y_quant: [u16; 64],
    cb_quant: [u16; 64],
    cr_quant: [u16; 64],
    y_dc_table: JpegHuffmanTable,
    y_ac_table: JpegHuffmanTable,
    cb_dc_table: JpegHuffmanTable,
    cb_ac_table: JpegHuffmanTable,
    cr_dc_table: JpegHuffmanTable,
    cr_ac_table: JpegHuffmanTable,
    entropy_bytes: Vec<u8>,
}

macro_rules! impl_from_color_fast_packet_parts {
    ($packet:ty) => {
        impl From<ColorFastPacketParts> for $packet {
            fn from(parts: ColorFastPacketParts) -> Self {
                Self {
                    dimensions: parts.dimensions,
                    mcus_per_row: parts.mcus_per_row,
                    mcu_rows: parts.mcu_rows,
                    restart_interval_mcus: parts.restart_interval_mcus,
                    restart_offsets: parts.restart_offsets,
                    entropy_checkpoints: parts.entropy_checkpoints,
                    y_quant: parts.y_quant,
                    cb_quant: parts.cb_quant,
                    cr_quant: parts.cr_quant,
                    y_dc_table: parts.y_dc_table,
                    y_ac_table: parts.y_ac_table,
                    cb_dc_table: parts.cb_dc_table,
                    cb_ac_table: parts.cb_ac_table,
                    cr_dc_table: parts.cr_dc_table,
                    cr_ac_table: parts.cr_ac_table,
                    entropy_bytes: parts.entropy_bytes,
                }
            }
        }
    };
}

impl_from_color_fast_packet_parts!(JpegFast420PacketV1);
impl_from_color_fast_packet_parts!(JpegFast422PacketV1);
impl_from_color_fast_packet_parts!(JpegFast444PacketV1);

#[derive(Debug, Clone, PartialEq, Eq)]
struct EntropySegments {
    entropy_bytes: Vec<u8>,
    restart_offsets: Vec<u32>,
}

/// Build a 4:2:0 fast-path packet from JPEG bytes.
#[doc(hidden)]
pub fn build_fast420_packet(bytes: &[u8]) -> Result<JpegFast420PacketV1, FastPacketError> {
    build_color_fast_packet(bytes, FAST420_LAYOUT).map(Into::into)
}

/// Build a 4:4:4 fast-path packet from JPEG bytes.
#[doc(hidden)]
pub fn build_fast444_packet(bytes: &[u8]) -> Result<JpegFast444PacketV1, FastPacketError> {
    build_color_fast_packet(bytes, FAST444_LAYOUT).map(Into::into)
}

/// Build a 4:2:2 fast-path packet from JPEG bytes.
#[doc(hidden)]
pub fn build_fast422_packet(bytes: &[u8]) -> Result<JpegFast422PacketV1, FastPacketError> {
    build_color_fast_packet(bytes, FAST422_LAYOUT).map(Into::into)
}

fn build_color_fast_packet(
    bytes: &[u8],
    layout: FastLayout,
) -> Result<ColorFastPacketParts, FastPacketError> {
    let decoder = Decoder::new(bytes)?;
    let header = parse_header(bytes)?;
    if !matches!(header.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return Err(FastPacketError::UnsupportedSof(header.sof_kind));
    }
    if header.bit_depth != 8 {
        return Err(FastPacketError::Decode(JpegError::UnsupportedBitDepth {
            depth: header.bit_depth,
        }));
    }
    let color_space = header.color_space();
    if color_space != ColorSpace::YCbCr && !(layout.allow_rgb && color_space == ColorSpace::Rgb) {
        return Err(FastPacketError::UnsupportedColorSpace(header.color_space()));
    }
    if header.sampling != SamplingFactors::from_validated_components(layout.sampling) {
        return Err(FastPacketError::UnsupportedSampling);
    }
    let scan = header.scan.as_ref().ok_or(FastPacketError::MissingScan)?;
    let [y_scan, cb_scan, cr_scan] = ordered_scan_triplet(&header.component_ids, &scan.components)?;

    let y_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 0)?;
    let cb_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 1)?;
    let cr_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 2)?;
    let y_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, y_scan.dc_table)?;
    let y_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, y_scan.ac_table)?;
    let cb_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, cb_scan.dc_table)?;
    let cb_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, cb_scan.ac_table)?;
    let cr_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, cr_scan.dc_table)?;
    let cr_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, cr_scan.ac_table)?;

    let entropy_offset = header.sos_offset.ok_or(FastPacketError::MissingScan)?;
    let restart_interval_mcus = u32::from(header.restart_interval.unwrap_or(0));
    let EntropySegments {
        entropy_bytes,
        restart_offsets,
    } = extract_entropy_segments(&bytes[entropy_offset..], header.restart_interval)?;
    let (width, height) = header.dimensions;
    let mcus_per_row = width.div_ceil(layout.mcu_width);
    let mcu_rows = height.div_ceil(layout.mcu_height);
    let total_mcus = mcus_per_row
        .checked_mul(mcu_rows)
        .ok_or(FastPacketError::Decode(JpegError::DimensionOverflow {
            width,
            height,
        }))?;
    let entropy_checkpoints =
        build_fast_entropy_checkpoints(&decoder, &bytes[entropy_offset..], total_mcus)?;

    Ok(ColorFastPacketParts {
        dimensions: header.dimensions,
        mcus_per_row,
        mcu_rows,
        restart_interval_mcus,
        restart_offsets,
        entropy_checkpoints,
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

/// Build a grayscale fast-path packet from JPEG bytes.
#[doc(hidden)]
pub fn build_gray_packet(bytes: &[u8]) -> Result<JpegGrayPacketV1, FastPacketError> {
    let header = parse_header(bytes)?;
    if !matches!(header.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return Err(FastPacketError::UnsupportedSof(header.sof_kind));
    }
    if header.bit_depth != 8 {
        return Err(FastPacketError::Decode(JpegError::UnsupportedBitDepth {
            depth: header.bit_depth,
        }));
    }
    if header.color_space() != ColorSpace::Grayscale {
        return Err(FastPacketError::UnsupportedColorSpace(header.color_space()));
    }
    if header.sampling != SamplingFactors::from_validated_components(&[(1, 1)]) {
        return Err(FastPacketError::UnsupportedSampling);
    }

    let scan = header.scan.as_ref().ok_or(FastPacketError::MissingScan)?;
    if header.component_ids.len() != 1 || scan.components.len() != 1 {
        return Err(FastPacketError::UnsupportedComponentOrder);
    }
    if scan.components[0].id != header.component_ids[0] {
        return Err(FastPacketError::UnsupportedComponentOrder);
    }

    let y_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 0)?;
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

    let entropy_offset = header.sos_offset.ok_or(FastPacketError::MissingScan)?;
    let restart_interval_mcus = u32::from(header.restart_interval.unwrap_or(0));
    let EntropySegments {
        entropy_bytes,
        restart_offsets,
    } = extract_entropy_segments(&bytes[entropy_offset..], header.restart_interval)?;
    let (width, height) = header.dimensions;

    Ok(JpegGrayPacketV1 {
        dimensions: header.dimensions,
        mcus_per_row: width.div_ceil(8),
        mcu_rows: height.div_ceil(8),
        restart_interval_mcus,
        restart_offsets,
        y_quant,
        y_dc_table,
        y_ac_table,
        entropy_bytes,
    })
}

fn quant_for_component(
    quant_table_ids: &[u8],
    tables: &[Option<[u16; 64]>; 4],
    component_idx: usize,
) -> Result<[u16; 64], FastPacketError> {
    let slot = *quant_table_ids
        .get(component_idx)
        .ok_or(FastPacketError::UnsupportedComponentOrder)?;
    tables[slot as usize].ok_or(FastPacketError::MissingQuantTable { slot })
}

fn ordered_scan_triplet(
    component_ids: &[u8],
    scan_components: &[ScanComponent],
) -> Result<[ScanComponent; 3], FastPacketError> {
    if component_ids.len() != 3 || scan_components.len() != 3 {
        return Err(FastPacketError::UnsupportedComponentOrder);
    }

    let mut ordered = [None; 3];
    for (index, &component_id) in component_ids.iter().enumerate() {
        let Some(component) = scan_components
            .iter()
            .copied()
            .find(|component| component.id == component_id)
        else {
            return Err(FastPacketError::UnsupportedComponentOrder);
        };
        ordered[index] = Some(component);
    }

    match ordered {
        [Some(first), Some(second), Some(third)] => Ok([first, second, third]),
        _ => Err(FastPacketError::UnsupportedComponentOrder),
    }
}

fn huffman_table(
    tables: &[Option<RawHuffmanTable>; 4],
    kind: TableKind,
    slot: u8,
) -> Result<JpegHuffmanTable, FastPacketError> {
    let raw = tables[slot as usize]
        .as_ref()
        .ok_or(FastPacketError::MissingHuffmanTable { kind, slot })?;
    Ok(JpegHuffmanTable::from_raw(raw))
}

fn nonrestart_entropy_chunk_mcus(total_mcus: u32) -> u32 {
    total_mcus
        .div_ceil(MAX_NONRESTART_ENTROPY_CHECKPOINTS)
        .max(1)
}

fn build_fast_entropy_checkpoints(
    decoder: &Decoder<'_>,
    scan_bytes: &[u8],
    total_mcus: u32,
) -> Result<Vec<JpegEntropyCheckpointV1>, FastPacketError> {
    let device_checkpoints = build_checkpoint_plan(
        &decoder.plan,
        scan_bytes,
        nonrestart_entropy_chunk_mcus(total_mcus),
    )?;
    device_checkpoints
        .iter()
        .map(|checkpoint| packet_checkpoint_from_device(checkpoint, scan_bytes))
        .collect()
}

fn packet_checkpoint_from_device(
    checkpoint: &DeviceCheckpoint,
    scan_bytes: &[u8],
) -> Result<JpegEntropyCheckpointV1, FastPacketError> {
    Ok(JpegEntropyCheckpointV1 {
        mcu_index: checkpoint.mcu_index,
        entropy_pos: destuffed_entropy_offset(scan_bytes, checkpoint.scan_offset)?,
        bit_acc: checkpoint.bit_accumulator,
        bit_count: u32::from(checkpoint.bits_buffered),
        y_prev_dc: checkpoint.prev_dc[0],
        cb_prev_dc: checkpoint.prev_dc[1],
        cr_prev_dc: checkpoint.prev_dc[2],
        reserved: 0,
    })
}

fn destuffed_entropy_offset(scan_bytes: &[u8], target: usize) -> Result<u32, FastPacketError> {
    if target > scan_bytes.len() {
        return Err(FastPacketError::TruncatedEntropy);
    }

    let mut pos = 0usize;
    let mut destuffed = 0usize;
    while pos < target {
        if scan_bytes[pos] != 0xff {
            pos += 1;
            destuffed += 1;
            continue;
        }

        let marker = *scan_bytes
            .get(pos + 1)
            .ok_or(FastPacketError::TruncatedEntropy)?;
        if pos + 2 > target {
            return Err(FastPacketError::TruncatedEntropy);
        }
        match marker {
            0x00 => {
                pos += 2;
                destuffed += 1;
            }
            0xd0..=0xd7 | 0xd9 => {
                pos += 2;
            }
            marker => return Err(FastPacketError::EntropyMarkerUnsupported { marker }),
        }
    }

    if pos != target {
        return Err(FastPacketError::TruncatedEntropy);
    }
    u32::try_from(destuffed).map_err(|_| FastPacketError::TruncatedEntropy)
}

fn extract_entropy_segments(
    bytes: &[u8],
    restart_interval: Option<u16>,
) -> Result<EntropySegments, FastPacketError> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut restart_offsets = vec![0u32];
    let mut pos = 0usize;
    let mut expected_rst = 0xD0u8;
    while pos < bytes.len() {
        let byte = bytes[pos];
        if byte != 0xFF {
            out.push(byte);
            pos += 1;
            continue;
        }
        let next = *bytes
            .get(pos + 1)
            .ok_or(FastPacketError::TruncatedEntropy)?;
        match next {
            0x00 => {
                out.push(0xFF);
                pos += 2;
            }
            0xD9 => {
                return Ok(EntropySegments {
                    entropy_bytes: out,
                    restart_offsets,
                });
            }
            0xD0..=0xD7 if restart_interval.unwrap_or(0) != 0 => {
                if next != expected_rst {
                    return Err(FastPacketError::EntropyMarkerUnsupported { marker: next });
                }
                restart_offsets
                    .push(u32::try_from(out.len()).map_err(|_| FastPacketError::TruncatedEntropy)?);
                expected_rst = if expected_rst == 0xD7 {
                    0xD0
                } else {
                    expected_rst + 1
                };
                pos += 2;
            }
            marker => {
                return Err(FastPacketError::EntropyMarkerUnsupported { marker });
            }
        }
    }
    Err(FastPacketError::Decode(JpegError::MissingMarker {
        marker: MarkerKind::Eoi,
    }))
}
