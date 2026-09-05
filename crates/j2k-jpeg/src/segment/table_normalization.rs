// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free DQT/DHT definition parsing and TIFF table normalization.

use super::{
    is_sof_marker, iter_segments, parse_sof_info_allowing_zero_dimensions, DuplicateTablePolicy,
    JpegSofInfo,
};
use crate::error::{JpegError, TableKind};
use crate::info::SofKind;
use crate::parse::scan::{parse_scan_header, ParsedScan};
use alloc::vec::Vec;

const JPEG_TABLE_SLOTS: usize = 4;
const JPEG_HUFFMAN_CLASSES: usize = 2;
const MAX_TABLE_ID: u8 = 3;
const MAX_HUFFMAN_CLASS: u8 = 1;
const MAX_DISTINCT_TABLES: usize = JPEG_TABLE_SLOTS + JPEG_TABLE_SLOTS * JPEG_HUFFMAN_CLASSES;

#[derive(Clone, Copy, Debug)]
pub(super) enum NormalizedSegment<'a> {
    Borrowed(&'a [u8]),
    FilteredTable {
        marker: u8,
        definitions: [&'a [u8]; MAX_DISTINCT_TABLES],
        definition_count: usize,
        payload_len: usize,
    },
}

impl NormalizedSegment<'_> {
    pub(super) fn byte_len(self) -> usize {
        match self {
            Self::Borrowed(bytes) => bytes.len(),
            Self::FilteredTable { payload_len, .. } => payload_len + 4,
        }
    }

    pub(super) fn append_to(self, output: &mut Vec<u8>) -> Result<(), JpegError> {
        match self {
            Self::Borrowed(bytes) => output.extend_from_slice(bytes),
            Self::FilteredTable {
                marker,
                definitions,
                definition_count,
                payload_len,
            } => {
                let segment_len =
                    u16::try_from(payload_len + 2).map_err(|_| JpegError::InternalInvariant {
                        reason: "normalized JPEG table segment exceeded the marker length ABI",
                    })?;
                output.extend_from_slice(&[0xff, marker]);
                output.extend_from_slice(&segment_len.to_be_bytes());
                for definition in definitions.iter().take(definition_count) {
                    output.extend_from_slice(definition);
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TableKey {
    Quant(u8),
    HuffmanDc(u8),
    HuffmanAc(u8),
}

impl TableKey {
    fn table_kind(self) -> TableKind {
        match self {
            Self::Quant(_) => TableKind::Quant,
            Self::HuffmanDc(_) => TableKind::HuffmanDc,
            Self::HuffmanAc(_) => TableKind::HuffmanAc,
        }
    }

    fn id(self) -> u8 {
        match self {
            Self::Quant(id) | Self::HuffmanDc(id) | Self::HuffmanAc(id) => id,
        }
    }
}

#[derive(Clone, Copy)]
struct TableDefinition<'a> {
    key: TableKey,
    offset: usize,
    bytes: &'a [u8],
}

#[derive(Default)]
struct NormalizationState<'a> {
    quant: [Option<&'a [u8]>; JPEG_TABLE_SLOTS],
    huffman_dc: [Option<&'a [u8]>; JPEG_TABLE_SLOTS],
    huffman_ac: [Option<&'a [u8]>; JPEG_TABLE_SLOTS],
}

impl<'a> NormalizationState<'a> {
    fn table_slot_mut(&mut self, key: TableKey) -> &mut Option<&'a [u8]> {
        match key {
            TableKey::Quant(id) => &mut self.quant[usize::from(id)],
            TableKey::HuffmanDc(id) => &mut self.huffman_dc[usize::from(id)],
            TableKey::HuffmanAc(id) => &mut self.huffman_ac[usize::from(id)],
        }
    }
}

pub(super) fn for_each_normalized_segment<'a>(
    input: &'a [u8],
    policy: DuplicateTablePolicy,
    mut visit: impl FnMut(NormalizedSegment<'a>) -> Result<(), JpegError>,
) -> Result<(), JpegError> {
    let mut state = NormalizationState::default();
    let mut saw_soi = false;
    let mut parsed_through = 0usize;
    for segment in iter_segments(input) {
        let segment = match segment {
            Ok(segment) => segment,
            // TIFF already supplies an exact byte boundary for JPEGTables.
            // Accept a missing terminal EOI only after a valid SOI and wholly
            // parsed marker segments; partial marker bytes remain errors.
            Err(JpegError::Truncated {
                offset,
                expected: 2,
            }) if saw_soi && offset == input.len() && parsed_through == input.len() => break,
            Err(error) => return Err(error),
        };
        saw_soi |= segment.marker == 0xd8;
        parsed_through = segment
            .payload_offset
            .checked_add(segment.payload.len())
            .ok_or(JpegError::InternalInvariant {
                reason: "parsed JPEGTables segment boundary overflowed",
            })?;
        if matches!(segment.marker, 0xd8 | 0xd9) {
            continue;
        }
        let total_end = segment
            .payload_offset
            .checked_add(segment.payload.len())
            .ok_or(JpegError::InternalInvariant {
                reason: "parsed JPEG segment range overflowed",
            })?;
        let segment_bytes =
            input
                .get(segment.marker_offset..total_end)
                .ok_or(JpegError::InternalInvariant {
                    reason: "parsed JPEG segment range escaped its input",
                })?;
        match segment.marker {
            0xdb | 0xc4 => normalize_table_segment(
                &mut state,
                segment.marker,
                segment.payload,
                segment.payload_offset,
                segment_bytes,
                policy,
                &mut visit,
            )?,
            // TIFF JPEGTables supplies shared coding tables, not per-tile
            // restart or arithmetic-conditioning state. Importing DRI or DAC
            // here would silently change the tile's entropy interpretation.
            0xcc | 0xdd => {}
            _ => visit(NormalizedSegment::Borrowed(segment_bytes))?,
        }
    }
    Ok(())
}

fn normalize_table_segment<'a>(
    state: &mut NormalizationState<'a>,
    marker: u8,
    payload: &'a [u8],
    payload_offset: usize,
    segment_bytes: &'a [u8],
    policy: DuplicateTablePolicy,
    visit: &mut impl FnMut(NormalizedSegment<'a>) -> Result<(), JpegError>,
) -> Result<(), JpegError> {
    let mut definitions = [&[][..]; MAX_DISTINCT_TABLES];
    let mut definition_count = 0usize;
    let mut payload_len = 0usize;
    let mut filtered_duplicate = false;

    for_each_table_definition(marker, payload, payload_offset, |definition| {
        let slot = state.table_slot_mut(definition.key);
        if let Some(existing) = *slot {
            if existing != definition.bytes {
                return Err(conflicting_duplicate_error(definition));
            }
            if policy == DuplicateTablePolicy::AllowIdentical {
                filtered_duplicate = true;
                return Ok(());
            }
        } else {
            *slot = Some(definition.bytes);
        }

        if policy == DuplicateTablePolicy::AllowIdentical {
            let Some(output_slot) = definitions.get_mut(definition_count) else {
                return Err(JpegError::InternalInvariant {
                    reason: "normalized JPEG marker exceeded distinct DQT/DHT table capacity",
                });
            };
            *output_slot = definition.bytes;
            definition_count += 1;
            payload_len += definition.bytes.len();
        }
        Ok(())
    })?;

    if policy == DuplicateTablePolicy::RejectConflicting || !filtered_duplicate {
        return visit(NormalizedSegment::Borrowed(segment_bytes));
    }
    if definition_count == 0 {
        return Ok(());
    }
    visit(NormalizedSegment::FilteredTable {
        marker,
        definitions,
        definition_count,
        payload_len,
    })
}

fn for_each_table_definition<'a>(
    marker: u8,
    payload: &'a [u8],
    payload_offset: usize,
    mut visit: impl FnMut(TableDefinition<'a>) -> Result<(), JpegError>,
) -> Result<(), JpegError> {
    if payload.is_empty() {
        return Err(invalid_table_segment(payload_offset, marker, payload.len()));
    }
    match marker {
        0xdb => for_each_dqt_definition(payload, payload_offset, &mut visit),
        0xc4 => for_each_dht_definition(payload, payload_offset, &mut visit),
        _ => Err(JpegError::InternalInvariant {
            reason: "non-table marker reached JPEG table definition parser",
        }),
    }
}

#[derive(Default)]
struct CodingTablePresence {
    quant: [bool; JPEG_TABLE_SLOTS],
    huffman_dc: [bool; JPEG_TABLE_SLOTS],
    huffman_ac: [bool; JPEG_TABLE_SLOTS],
}

impl CodingTablePresence {
    fn observe(&mut self, definition: TableDefinition<'_>) {
        let slot = usize::from(definition.key.id());
        match definition.key {
            TableKey::Quant(_) => self.quant[slot] = true,
            TableKey::HuffmanDc(_) => self.huffman_dc[slot] = true,
            TableKey::HuffmanAc(_) => self.huffman_ac[slot] = true,
        }
    }

    fn satisfies_scan(&self, sof: &JpegSofInfo, scan: &ParsedScan) -> bool {
        scan.components.iter().all(|component| {
            let Some(component_index) = sof.component_ids.iter().position(|&id| id == component.id)
            else {
                // External coding tables cannot repair an unknown component;
                // leave that structural error to normal tile validation.
                return true;
            };
            let quant_present = sof.sof_kind == SofKind::Lossless
                || sof
                    .quant_table_ids
                    .get(component_index)
                    .and_then(|&id| self.quant.get(usize::from(id)))
                    .copied()
                    .unwrap_or(false);
            let dc_present = self
                .huffman_dc
                .get(usize::from(component.dc_table))
                .copied()
                .unwrap_or(false);
            let ac_present = self
                .huffman_ac
                .get(usize::from(component.ac_table))
                .copied()
                .unwrap_or(false);
            match sof.sof_kind {
                SofKind::Lossless => dc_present,
                SofKind::Progressive8 | SofKind::Progressive12 if scan.ss == 0 => {
                    quant_present && dc_present
                }
                SofKind::Progressive8 | SofKind::Progressive12 => quant_present && ac_present,
                SofKind::Baseline8 | SofKind::Extended8 | SofKind::Extended12 => {
                    quant_present && dc_present && ac_present
                }
            }
        })
    }
}

/// Report whether the initial scan can resolve its referenced coding tables
/// from the tile itself. A false result identifies a TIFF abbreviated stream
/// that needs shared `JPEGTables` even when it retains SOI and EOI markers.
pub(super) fn stream_has_complete_coding_tables(input: &[u8]) -> Result<bool, JpegError> {
    let mut tables = CodingTablePresence::default();
    let mut sof = None;
    for segment in iter_segments(input) {
        let segment = segment?;
        match segment.marker {
            0xdb | 0xc4 => for_each_table_definition(
                segment.marker,
                segment.payload,
                segment.payload_offset,
                |definition| {
                    tables.observe(definition);
                    Ok(())
                },
            )?,
            marker if is_sof_marker(marker) => {
                sof = Some(parse_sof_info_allowing_zero_dimensions(
                    marker,
                    segment.payload,
                    segment.payload_offset,
                )?);
            }
            0xda => {
                let Some(sof) = sof.as_ref() else {
                    // Supplying coding tables cannot repair a missing SOF.
                    return Ok(true);
                };
                let scan = parse_scan_header(segment.payload, segment.payload_offset)?;
                // The initial scan is the table-state boundary supplied by
                // TIFF JPEGTables. Later progressive scans may legitimately
                // redefine their own tables inside the tile.
                return Ok(tables.satisfies_scan(sof, &scan));
            }
            _ => {}
        }
    }
    Ok(true)
}

fn for_each_dqt_definition<'a>(
    payload: &'a [u8],
    payload_offset: usize,
    visit: &mut impl FnMut(TableDefinition<'a>) -> Result<(), JpegError>,
) -> Result<(), JpegError> {
    let mut position = 0usize;
    while position < payload.len() {
        let selector = payload[position];
        let precision = selector >> 4;
        let id = selector & 0x0f;
        if id > MAX_TABLE_ID {
            return Err(invalid_table_segment(
                payload_offset + position,
                0xdb,
                payload.len(),
            ));
        }
        let value_bytes = match precision {
            0 => 1usize,
            1 => 2,
            _ => return Err(JpegError::UnsupportedBitDepth { depth: precision }),
        };
        let definition_len = 1 + 64 * value_bytes;
        let end = position + definition_len;
        if end > payload.len() {
            return Err(JpegError::Truncated {
                offset: payload_offset + end,
                expected: end - payload.len(),
            });
        }
        visit(TableDefinition {
            key: TableKey::Quant(id),
            offset: payload_offset + position,
            bytes: &payload[position..end],
        })?;
        position = end;
    }
    Ok(())
}

fn for_each_dht_definition<'a>(
    payload: &'a [u8],
    payload_offset: usize,
    visit: &mut impl FnMut(TableDefinition<'a>) -> Result<(), JpegError>,
) -> Result<(), JpegError> {
    let mut position = 0usize;
    while position < payload.len() {
        let counts_end = position + 17;
        if counts_end > payload.len() {
            return Err(JpegError::Truncated {
                offset: payload_offset + counts_end,
                expected: counts_end - payload.len(),
            });
        }
        let selector = payload[position];
        let class = selector >> 4;
        let id = selector & 0x0f;
        if class > MAX_HUFFMAN_CLASS || id > MAX_TABLE_ID {
            return Err(invalid_table_segment(
                payload_offset + position,
                0xc4,
                payload.len(),
            ));
        }
        let value_count = payload[position + 1..counts_end]
            .iter()
            .map(|&count| usize::from(count))
            .sum::<usize>();
        if value_count > 256 {
            return Err(invalid_table_segment(
                payload_offset + position + 1,
                0xc4,
                payload.len(),
            ));
        }
        let end = counts_end + value_count;
        if end > payload.len() {
            return Err(JpegError::Truncated {
                offset: payload_offset + end,
                expected: end - payload.len(),
            });
        }
        let key = if class == 0 {
            TableKey::HuffmanDc(id)
        } else {
            TableKey::HuffmanAc(id)
        };
        visit(TableDefinition {
            key,
            offset: payload_offset + position,
            bytes: &payload[position..end],
        })?;
        position = end;
    }
    Ok(())
}

fn invalid_table_segment(offset: usize, marker: u8, payload_len: usize) -> JpegError {
    JpegError::InvalidSegmentLength {
        offset,
        marker,
        length: u16::try_from(payload_len + 2).unwrap_or(u16::MAX),
    }
}

fn conflicting_duplicate_error(definition: TableDefinition<'_>) -> JpegError {
    JpegError::ConflictingDuplicateTable {
        offset: definition.offset,
        table: definition.key.table_kind(),
        id: definition.key.id(),
    }
}

#[cfg(test)]
mod tests;
