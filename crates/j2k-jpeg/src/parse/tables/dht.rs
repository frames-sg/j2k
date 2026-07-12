// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::JpegError;
use crate::parse::allocation::ParsedMetadataBudget;

use super::{HuffmanTables, HuffmanValues, RawHuffmanTable};

#[expect(
    clippy::cast_possible_truncation,
    reason = "DHT segment lengths are bounded by the JPEG 16-bit marker grammar"
)]
pub(crate) fn parse_dht(
    payload: &[u8],
    payload_offset: usize,
    tables: &mut HuffmanTables,
    budget: &mut ParsedMetadataBudget,
) -> Result<(), JpegError> {
    let mut i = 0;
    while i < payload.len() {
        if i + 17 > payload.len() {
            return Err(JpegError::Truncated {
                offset: payload_offset + i + 17,
                expected: (i + 17) - payload.len(),
            });
        }
        let class = payload[i] >> 4;
        let slot = usize::from(payload[i] & 0x0f);
        if slot > 3 || class > 1 {
            return Err(JpegError::InvalidSegmentLength {
                offset: payload_offset + i,
                marker: 0xc4,
                length: (payload.len() + 2) as u16,
            });
        }
        let mut bits = [0u8; 16];
        bits.copy_from_slice(&payload[i + 1..i + 17]);
        let total_values: usize = bits.iter().map(|&count| usize::from(count)).sum();
        if total_values > 256 {
            return Err(JpegError::InvalidSegmentLength {
                offset: payload_offset + i + 1,
                marker: 0xc4,
                length: (payload.len() + 2) as u16,
            });
        }
        if i + 17 + total_values > payload.len() {
            return Err(JpegError::Truncated {
                offset: payload_offset + i + 17 + total_values,
                expected: (i + 17 + total_values) - payload.len(),
            });
        }
        let values = HuffmanValues::from_slice(&payload[i + 17..i + 17 + total_values]);
        tables.define(class, slot, RawHuffmanTable { bits, values }, budget)?;
        i += 17 + total_values;
    }
    Ok(())
}
