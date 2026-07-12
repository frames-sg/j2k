// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared baseline JPEG entropy-writing primitives.

use alloc::vec::Vec;

use crate::adapter::{JpegBaselineHuffmanTable, JPEG_BASELINE_ZIGZAG};
use crate::baseline_encode_contract::JpegEncodeError;
use crate::encoded_output::CappedBytes;

pub(crate) struct BitWriter {
    bytes: CappedBytes,
    current: u8,
    used: u8,
}

impl BitWriter {
    pub(crate) fn try_with_max_bytes(max_bytes: usize) -> Result<Self, JpegEncodeError> {
        Ok(Self {
            bytes: CappedBytes::try_with_capacity(max_bytes, max_bytes)?,
            current: 0,
            used: 0,
        })
    }

    fn write_bits(&mut self, code: u32, len: u8) -> Result<(), JpegEncodeError> {
        for bit_idx in (0..len).rev() {
            let bit = u8::from(((code >> bit_idx) & 1) != 0);
            self.current = (self.current << 1) | bit;
            self.used += 1;
            if self.used == 8 {
                self.push_byte(self.current)?;
                self.current = 0;
                self.used = 0;
            }
        }
        Ok(())
    }

    fn align_with_ones(&mut self) -> Result<(), JpegEncodeError> {
        if self.used == 0 {
            return Ok(());
        }
        let remaining = 8 - self.used;
        self.current <<= remaining;
        self.current |= (1u8 << remaining) - 1;
        self.push_byte(self.current)?;
        self.current = 0;
        self.used = 0;
        Ok(())
    }

    pub(crate) fn into_bytes(mut self) -> Result<Vec<u8>, JpegEncodeError> {
        self.align_with_ones()?;
        Ok(self.bytes.into_vec())
    }

    pub(crate) fn capacity_bytes(&self) -> usize {
        self.bytes.capacity()
    }

    pub(crate) fn write_restart_marker(&mut self, marker: u8) -> Result<(), JpegEncodeError> {
        self.align_with_ones()?;
        self.bytes.push(0xFF)?;
        self.bytes.push(marker)
    }

    fn push_byte(&mut self, byte: u8) -> Result<(), JpegEncodeError> {
        self.bytes.push(byte)?;
        if byte == 0xFF {
            self.bytes.push(0x00)?;
        }
        Ok(())
    }
}

pub(crate) fn encode_block(
    coeffs: &[i32; 64],
    prev_dc: &mut i32,
    dc_table: &JpegBaselineHuffmanTable,
    ac_table: &JpegBaselineHuffmanTable,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    let diff = coeffs[0] - *prev_dc;
    *prev_dc = coeffs[0];
    let (dc_size, dc_bits) = magnitude(diff);
    write_huffman_symbol(dc_table, dc_size, writer)?;
    if dc_size > 0 {
        writer.write_bits(dc_bits, dc_size)?;
    }

    let mut zero_run = 0u8;
    for k in 1..64 {
        let coeff = coeffs[JPEG_BASELINE_ZIGZAG[k] as usize];
        if coeff == 0 {
            zero_run = zero_run.saturating_add(1);
            continue;
        }
        while zero_run >= 16 {
            write_huffman_symbol(ac_table, 0xF0, writer)?;
            zero_run -= 16;
        }
        let (size, bits) = magnitude(coeff);
        let symbol = (zero_run << 4) | size;
        write_huffman_symbol(ac_table, symbol, writer)?;
        writer.write_bits(bits, size)?;
        zero_run = 0;
    }
    if zero_run > 0 {
        write_huffman_symbol(ac_table, 0, writer)?;
    }
    Ok(())
}

fn write_huffman_symbol(
    table: &JpegBaselineHuffmanTable,
    symbol: u8,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    let len = table.lens[symbol as usize];
    if len == 0 {
        return Err(JpegEncodeError::MissingHuffmanCode { symbol });
    }
    writer.write_bits(u32::from(table.codes[symbol as usize]), len)
}

pub(crate) fn magnitude(value: i32) -> (u8, u32) {
    if value == 0 {
        return (0, 0);
    }
    let magnitude = value.unsigned_abs();
    let mut remaining = magnitude;
    let mut size = 0u8;
    while remaining > 0 {
        size += 1;
        remaining >>= 1;
    }
    let category_mask = u32::MAX >> (u32::BITS - u32::from(size));
    let bits = if value >= 0 {
        magnitude
    } else {
        category_mask - magnitude
    };
    (size, bits)
}
