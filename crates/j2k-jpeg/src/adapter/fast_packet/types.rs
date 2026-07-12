// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::entropy::huffman::{derive_canonical_huffman, CanonicalHuffmanDerivation};
use crate::error::{HuffmanFailure, JpegError};
use crate::parse::tables::{HuffmanValues, RawHuffmanTable};
use alloc::vec::Vec;

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
    pub(super) fn from_raw(raw: &RawHuffmanTable) -> Self {
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
#[derive(Debug, PartialEq, Eq)]
#[doc(hidden)]
/// Backend fast-path packet for 8-bit 4:2:0 JPEG tiles.
///
/// Large retained packet payloads are intentionally move-only. Backends that
/// share a packet across submissions should retain it behind `Arc`.
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
#[derive(Debug, PartialEq, Eq)]
#[doc(hidden)]
/// Backend fast-path packet for 8-bit 4:2:2 JPEG tiles.
///
/// Large retained packet payloads are intentionally move-only. Backends that
/// share a packet across submissions should retain it behind `Arc`.
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
#[derive(Debug, PartialEq, Eq)]
#[doc(hidden)]
/// Backend fast-path packet for 8-bit 4:4:4 JPEG tiles.
///
/// Large retained packet payloads are intentionally move-only. Backends that
/// share a packet across submissions should retain it behind `Arc`.
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
#[derive(Debug, PartialEq, Eq)]
#[doc(hidden)]
/// Backend fast-path packet for 8-bit grayscale JPEG tiles.
///
/// Large retained packet payloads are intentionally move-only. Backends that
/// share a packet across submissions should retain it behind `Arc`.
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
