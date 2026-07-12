// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::JpegError;
use crate::parse::allocation::ParsedMetadataBudget;

use super::QuantTables;

#[expect(
    clippy::cast_possible_truncation,
    reason = "DQT segment lengths are bounded by the JPEG 16-bit marker grammar"
)]
pub(crate) fn parse_dqt(
    payload: &[u8],
    payload_offset: usize,
    tables: &mut QuantTables,
    budget: &mut ParsedMetadataBudget,
) -> Result<(), JpegError> {
    let mut i = 0;
    while i < payload.len() {
        let pq = payload[i] >> 4;
        let tq = usize::from(payload[i] & 0x0f);
        if tq > 3 {
            return Err(JpegError::InvalidSegmentLength {
                offset: payload_offset + i,
                marker: 0xdb,
                length: (payload.len() + 2) as u16,
            });
        }
        let entry_bytes = match pq {
            0 => 1,
            1 => 2,
            _ => return Err(JpegError::UnsupportedBitDepth { depth: pq }),
        };
        let needed = 1 + 64 * entry_bytes;
        if i + needed > payload.len() {
            return Err(JpegError::Truncated {
                offset: payload_offset + i + needed,
                expected: (i + needed) - payload.len(),
            });
        }
        let mut entries = [0u16; 64];
        if pq == 0 {
            for k in 0..64 {
                entries[k] = u16::from(payload[i + 1 + k]);
                if entries[k] == 0 {
                    return Err(JpegError::InvalidQuantizationValue {
                        offset: payload_offset + i + 1 + k,
                        table: tq as u8,
                        coefficient: k as u8,
                    });
                }
            }
        } else if pq == 1 {
            for k in 0..64 {
                entries[k] =
                    u16::from_be_bytes([payload[i + 1 + k * 2], payload[i + 1 + k * 2 + 1]]);
                if entries[k] == 0 {
                    return Err(JpegError::InvalidQuantizationValue {
                        offset: payload_offset + i + 1 + k * 2,
                        table: tq as u8,
                        coefficient: k as u8,
                    });
                }
            }
        }
        tables.define(tq, entries, budget)?;
        i += needed;
    }
    Ok(())
}
