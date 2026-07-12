// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public JPEG marker-level utilities.

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, try_vec_with_capacity,
};
use crate::error::{JpegError, MarkerKind, UnsupportedReason};
use crate::info::{SamplingFactors, SofKind};
use crate::parse::markers::next_marker_after_entropy;
use alloc::vec::Vec;
use core::ops::Range;
use memchr::memchr;

mod table_normalization;

use table_normalization::for_each_normalized_segment;

/// One marker segment in a JPEG byte stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegSegment<'a> {
    /// Raw marker byte after the `0xff` prefix.
    pub marker: u8,
    /// Offset of the marker prefix byte.
    pub marker_offset: usize,
    /// Offset of the segment payload. Standalone markers use the byte after the marker.
    pub payload_offset: usize,
    /// Segment payload excluding marker and length bytes.
    pub payload: &'a [u8],
}

/// Iterator over marker segments in a JPEG byte stream.
#[derive(Debug)]
pub struct JpegSegmentIter<'a> {
    input: &'a [u8],
    pos: usize,
    started: bool,
    finished: bool,
    scan_entropy: bool,
}

/// Parsed Start-of-Frame facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegSofInfo {
    /// Raw SOF marker byte.
    pub marker: u8,
    /// Supported SOF kind classification.
    pub sof_kind: SofKind,
    /// Component sample precision.
    pub bit_depth: u8,
    /// Width and height from the SOF payload.
    pub dimensions: (u16, u16),
    /// Component identifiers in declaration order.
    pub component_ids: Vec<u8>,
    /// Component sampling factors in declaration order.
    pub sampling: SamplingFactors,
    /// Quantization-table selectors in declaration order.
    pub quant_table_ids: Vec<u8>,
}

/// Byte ranges around the first Start-of-Scan marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegScanRanges {
    /// Offset of the SOS marker prefix byte.
    pub sos_marker_offset: usize,
    /// Payload range of the SOS segment, excluding marker and length bytes.
    pub sos_payload_range: Range<usize>,
    /// Entropy-coded scan data range after SOS and before EOI or the next marker.
    pub entropy_range: Range<usize>,
    /// Offset of EOI when present.
    pub eoi_marker_offset: Option<usize>,
}

/// Options for preparing TIFF/WSI JPEG tile payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegTilePrepareOptions {
    /// Container-derived expected tile dimensions.
    pub expected_dimensions: Option<(u16, u16)>,
    /// Duplicate DQT/DHT handling policy within supplied `JPEGTables` bytes.
    pub duplicate_table_policy: DuplicateTablePolicy,
    /// Repair zero SOF dimensions using `expected_dimensions`.
    pub repair_zero_sof_dimensions: bool,
    /// Validate restart marker order in scan data.
    pub validate_restart_markers: bool,
}

impl Default for JpegTilePrepareOptions {
    fn default() -> Self {
        Self {
            expected_dimensions: None,
            duplicate_table_policy: DuplicateTablePolicy::RejectConflicting,
            repair_zero_sof_dimensions: false,
            validate_restart_markers: false,
        }
    }
}

/// Duplicate JPEG table handling policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateTablePolicy {
    /// Coalesce byte-identical duplicate definitions while rejecting conflicts.
    AllowIdentical,
    /// Preserve byte-identical redefinitions but reject conflicting definitions.
    ///
    /// This mode retains the source marker bytes when no conflict is present.
    RejectConflicting,
}

/// Prepared JPEG bytes, borrowed when unchanged and owned when normalized.
#[derive(Debug, PartialEq, Eq)]
pub enum PreparedJpeg<'a> {
    /// Original tile bytes can be decoded directly.
    Borrowed(&'a [u8]),
    /// Preparation changed the byte stream.
    Owned(Vec<u8>),
}

impl PreparedJpeg<'_> {
    /// Return decode-ready JPEG bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes,
        }
    }

    /// Fallibly duplicate this prepared JPEG.
    ///
    /// Borrowed inputs remain borrowed without allocation. Owned normalized
    /// payloads are copied through the same typed, capped allocation contract
    /// used by dimension rewrite and TIFF assembly.
    ///
    /// # Errors
    ///
    /// Returns [`JpegError::MemoryCapExceeded`] when the owned payload exceeds
    /// the JPEG host cap, or [`JpegError::HostAllocationFailed`] when a
    /// cap-valid copy cannot be reserved.
    pub fn try_clone(&self) -> Result<Self, JpegError> {
        match self {
            Self::Borrowed(bytes) => Ok(Self::Borrowed(bytes)),
            Self::Owned(bytes) => Ok(Self::Owned(try_copy_bytes(bytes)?)),
        }
    }
}

impl AsRef<[u8]> for PreparedJpeg<'_> {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

/// Iterate over JPEG marker segments.
#[must_use]
pub fn iter_segments(input: &[u8]) -> JpegSegmentIter<'_> {
    JpegSegmentIter {
        input,
        pos: 0,
        started: false,
        finished: false,
        scan_entropy: false,
    }
}

/// Return true for JPEG SOF marker classes.
#[must_use]
pub const fn is_sof_marker(marker: u8) -> bool {
    matches!(
        marker,
        0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf
    )
}

/// Parse a Start-of-Frame payload.
///
/// # Errors
///
/// Returns an error when the marker or SOF payload is malformed.
pub fn parse_sof_info(marker: u8, payload: &[u8]) -> Result<JpegSofInfo, JpegError> {
    parse_sof_info_at(marker, payload, 0, false)
}

pub(crate) fn parse_sof_info_allowing_zero_dimensions(
    marker: u8,
    payload: &[u8],
    payload_offset: usize,
) -> Result<JpegSofInfo, JpegError> {
    parse_sof_info_at(marker, payload, payload_offset, true)
}

/// Parse a Define Restart Interval payload.
///
/// # Errors
///
/// Returns an error when the DRI payload length is not exactly two bytes.
pub fn parse_dri(payload: &[u8]) -> Result<Option<u16>, JpegError> {
    if payload.len() != 2 {
        return Err(JpegError::InvalidSegmentLength {
            offset: 0,
            marker: 0xdd,
            length: u16::try_from(payload.len() + 2).unwrap_or(u16::MAX),
        });
    }
    let interval = u16::from_be_bytes([payload[0], payload[1]]);
    Ok((interval > 0).then_some(interval))
}

/// Find the first SOS header and scan data ranges.
///
/// # Errors
///
/// Returns an error when segment syntax is malformed or no SOS marker exists.
pub fn find_scan_ranges(input: &[u8]) -> Result<JpegScanRanges, JpegError> {
    for segment in iter_segments(input) {
        let segment = segment?;
        if segment.marker == 0xda {
            let entropy_start = segment.payload_offset + segment.payload.len();
            let next_marker = next_marker_after_entropy(input, entropy_start)?;
            let (entropy_end, eoi_marker_offset) = match next_marker {
                Some((marker_offset, 0xd9)) => (marker_offset, Some(marker_offset)),
                Some((marker_offset, _)) => (marker_offset, None),
                None => (input.len(), None),
            };
            return Ok(JpegScanRanges {
                sos_marker_offset: segment.marker_offset,
                sos_payload_range: segment.payload_offset..entropy_start,
                entropy_range: entropy_start..entropy_end,
                eoi_marker_offset,
            });
        }
    }
    Err(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })
}

/// Return a copy of `input` with the first SOF dimensions rewritten.
///
/// # Errors
///
/// Returns an error for zero dimensions, malformed segment syntax, or a
/// missing SOF marker.
pub fn rewrite_sof_dimensions(input: &[u8], dimensions: (u16, u16)) -> Result<Vec<u8>, JpegError> {
    if dimensions.0 == 0 || dimensions.1 == 0 {
        return Err(JpegError::ZeroDimension {
            width: dimensions.0,
            height: dimensions.1,
        });
    }
    for segment in iter_segments(input) {
        let segment = segment?;
        if is_sof_marker(segment.marker) {
            if segment.payload.len() < 5 {
                return Err(JpegError::Truncated {
                    offset: segment.payload_offset + segment.payload.len(),
                    expected: 5 - segment.payload.len(),
                });
            }
            let mut out = try_copy_bytes(input)?;
            let width = dimensions.0.to_be_bytes();
            let height = dimensions.1.to_be_bytes();
            out[segment.payload_offset + 1] = height[0];
            out[segment.payload_offset + 2] = height[1];
            out[segment.payload_offset + 3] = width[0];
            out[segment.payload_offset + 4] = width[1];
            return Ok(out);
        }
    }
    Err(JpegError::MissingMarker {
        marker: MarkerKind::Sof,
    })
}

fn try_copy_bytes(input: &[u8]) -> Result<Vec<u8>, JpegError> {
    checked_allocation_bytes::<u8>(input.len())?;
    let mut out = try_vec_with_capacity(input.len())?;
    out.extend_from_slice(input);
    Ok(out)
}

/// Prepare a TIFF/WSI JPEG tile for decode.
///
/// # Errors
///
/// Returns an error when a complete tile is invalid or an abbreviated tile
/// cannot be assembled safely with the supplied tables and options.
pub fn prepare_tiff_jpeg_tile<'a>(
    tile: &'a [u8],
    tables: Option<&'a [u8]>,
    opts: JpegTilePrepareOptions,
) -> Result<PreparedJpeg<'a>, JpegError> {
    if is_complete_jpeg(tile) {
        validate_complete_tile(tile, opts)
    } else {
        assemble_abbreviated_tile(tile, tables, opts)
    }
}

fn is_complete_jpeg(input: &[u8]) -> bool {
    input.len() >= 4
        && input[0] == 0xff
        && input[1] == 0xd8
        && input[input.len() - 2] == 0xff
        && input[input.len() - 1] == 0xd9
}

fn validate_complete_tile(
    tile: &[u8],
    opts: JpegTilePrepareOptions,
) -> Result<PreparedJpeg<'_>, JpegError> {
    if let Some(repaired) = finalize_prepared_bytes(tile, opts)? {
        return Ok(PreparedJpeg::Owned(repaired));
    }
    Ok(PreparedJpeg::Borrowed(tile))
}

fn finalize_prepared_bytes(
    bytes: &[u8],
    opts: JpegTilePrepareOptions,
) -> Result<Option<Vec<u8>>, JpegError> {
    let repaired = repair_or_validate_dimensions(bytes, opts)?;
    let validation_input = repaired.as_deref().unwrap_or(bytes);
    let _ = find_scan_ranges(validation_input)?;
    if opts.validate_restart_markers {
        validate_restart_markers(validation_input)?;
    }
    Ok(repaired)
}

fn finalize_owned_prepared_bytes(
    bytes: &mut [u8],
    opts: JpegTilePrepareOptions,
) -> Result<(), JpegError> {
    if let Some(repair) = planned_dimension_repair(bytes, opts)? {
        apply_dimension_repair(bytes, repair);
        validate_nonzero_sof_dimensions(bytes, opts)?;
    }
    let _ = find_scan_ranges(bytes)?;
    if opts.validate_restart_markers {
        validate_restart_markers(bytes)?;
    }
    Ok(())
}

fn repair_or_validate_dimensions(
    bytes: &[u8],
    opts: JpegTilePrepareOptions,
) -> Result<Option<Vec<u8>>, JpegError> {
    let Some(repair) = planned_dimension_repair(bytes, opts)? else {
        return Ok(None);
    };
    let mut repaired = try_copy_bytes(bytes)?;
    apply_dimension_repair(&mut repaired, repair);
    validate_nonzero_sof_dimensions(&repaired, opts)?;
    Ok(Some(repaired))
}

#[derive(Clone, Copy)]
struct SofDimensionRepair {
    payload_offset: usize,
    dimensions: (u16, u16),
}

fn planned_dimension_repair(
    bytes: &[u8],
    opts: JpegTilePrepareOptions,
) -> Result<Option<SofDimensionRepair>, JpegError> {
    let mut saw_sof = false;
    for segment in iter_segments(bytes) {
        let segment = segment?;
        if segment.marker == 0xda {
            break;
        }
        if is_sof_marker(segment.marker) {
            saw_sof = true;
            let sof = parse_sof_info_allowing_zero_dimensions(
                segment.marker,
                segment.payload,
                segment.payload_offset,
            )?;
            if sof.dimensions.0 == 0 || sof.dimensions.1 == 0 {
                let Some(expected) = opts.expected_dimensions else {
                    return Err(JpegError::ExpectedDimensionsRequired {
                        offset: segment.marker_offset,
                    });
                };
                if !opts.repair_zero_sof_dimensions {
                    return Err(JpegError::ZeroDimension {
                        width: sof.dimensions.0,
                        height: sof.dimensions.1,
                    });
                }
                if expected.0 == 0 || expected.1 == 0 {
                    return Err(JpegError::ZeroDimension {
                        width: expected.0,
                        height: expected.1,
                    });
                }
                return Ok(Some(SofDimensionRepair {
                    payload_offset: segment.payload_offset,
                    dimensions: expected,
                }));
            }
            if let Some(expected) = opts.expected_dimensions {
                if expected != sof.dimensions {
                    return Err(JpegError::ConflictingExpectedDimensions {
                        offset: segment.marker_offset,
                        expected,
                        actual: sof.dimensions,
                    });
                }
            }
        }
    }
    if !saw_sof {
        return Err(JpegError::MissingMarker {
            marker: MarkerKind::Sof,
        });
    }
    Ok(None)
}

fn apply_dimension_repair(bytes: &mut [u8], repair: SofDimensionRepair) {
    let width = repair.dimensions.0.to_be_bytes();
    let height = repair.dimensions.1.to_be_bytes();
    bytes[repair.payload_offset + 1] = height[0];
    bytes[repair.payload_offset + 2] = height[1];
    bytes[repair.payload_offset + 3] = width[0];
    bytes[repair.payload_offset + 4] = width[1];
}

fn validate_nonzero_sof_dimensions(
    bytes: &[u8],
    opts: JpegTilePrepareOptions,
) -> Result<(), JpegError> {
    let mut saw_sof = false;
    for segment in iter_segments(bytes) {
        let segment = segment?;
        if is_sof_marker(segment.marker) {
            saw_sof = true;
            let sof = parse_sof_info(segment.marker, segment.payload)?;
            if let Some(expected) = opts.expected_dimensions {
                if expected != sof.dimensions {
                    return Err(JpegError::ConflictingExpectedDimensions {
                        offset: segment.marker_offset,
                        expected,
                        actual: sof.dimensions,
                    });
                }
            }
        }
    }
    if !saw_sof {
        return Err(JpegError::MissingMarker {
            marker: MarkerKind::Sof,
        });
    }
    Ok(())
}

fn validate_restart_markers(bytes: &[u8]) -> Result<(), JpegError> {
    let ranges = find_scan_ranges(bytes)?;
    let mut expected = 0u8;
    let mut pos = ranges.entropy_range.start;
    while pos < ranges.entropy_range.end {
        let Some(relative) = memchr(0xff, &bytes[pos..ranges.entropy_range.end]) else {
            break;
        };
        let prefix = pos + relative;
        let mut marker_pos = prefix + 1;
        while marker_pos < ranges.entropy_range.end && bytes[marker_pos] == 0xff {
            marker_pos += 1;
        }
        if marker_pos >= ranges.entropy_range.end {
            return Err(JpegError::Truncated {
                offset: prefix,
                expected: 1,
            });
        }
        let marker = bytes[marker_pos];
        match marker {
            0x00 => pos = marker_pos + 1,
            0xd0..=0xd7 => {
                let found = marker & 0x07;
                if found != expected {
                    return Err(JpegError::RestartMismatch {
                        offset: marker_pos - 1,
                        expected,
                        found: marker,
                    });
                }
                expected = (expected + 1) & 0x07;
                pos = marker_pos + 1;
            }
            0xd9 => break,
            _ => {
                return Err(JpegError::UnexpectedMarker {
                    offset: marker_pos - 1,
                    expected: MarkerKind::Eoi,
                    found: marker,
                });
            }
        }
    }
    Ok(())
}

fn assemble_abbreviated_tile<'a>(
    tile: &'a [u8],
    tables: Option<&'a [u8]>,
    opts: JpegTilePrepareOptions,
) -> Result<PreparedJpeg<'a>, JpegError> {
    let Some(tables) = tables else {
        return Err(JpegError::InvalidJpegAssembly {
            offset: 0,
            reason: "abbreviated JPEG tile requires JPEGTables",
        });
    };
    let tile_body = normalized_abbreviated_tile_body(tile)?;
    let output_len = abbreviated_output_len(tables, tile_body.len(), opts.duplicate_table_policy)?;
    let mut out = try_vec_with_capacity(output_len)?;
    out.extend_from_slice(&[0xff, 0xd8]);
    for_each_normalized_segment(tables, opts.duplicate_table_policy, |segment| {
        segment.append_to(&mut out)
    })?;
    out.extend_from_slice(tile_body);
    out.extend_from_slice(&[0xff, 0xd9]);
    if out.len() != output_len {
        return Err(JpegError::InternalInvariant {
            reason: "normalized TIFF JPEG output length diverged from its preflight plan",
        });
    }
    finalize_owned_prepared_bytes(&mut out, opts)?;
    Ok(PreparedJpeg::Owned(out))
}

fn abbreviated_output_len(
    tables: &[u8],
    tile_body_len: usize,
    policy: DuplicateTablePolicy,
) -> Result<usize, JpegError> {
    let mut table_bytes = 0usize;
    for_each_normalized_segment(tables, policy, |segment| {
        table_bytes = checked_add_allocation_bytes(table_bytes, segment.byte_len())?;
        Ok(())
    })?;
    checked_abbreviated_output_len(table_bytes, tile_body_len)
}

fn checked_abbreviated_output_len(
    normalized_table_bytes: usize,
    tile_body_len: usize,
) -> Result<usize, JpegError> {
    let mut output_len = checked_allocation_bytes::<u8>(2)?;
    output_len = checked_add_allocation_bytes(output_len, normalized_table_bytes)?;
    output_len = checked_add_allocation_bytes(output_len, tile_body_len)?;
    checked_add_allocation_bytes(output_len, 2)
}

fn normalized_abbreviated_tile_body(tile: &[u8]) -> Result<&[u8], JpegError> {
    let start = if tile.starts_with(&[0xff, 0xd8]) {
        2
    } else {
        0
    };
    let end = if tile.len() >= start + 2 && tile[tile.len() - 2..] == [0xff, 0xd9] {
        tile.len() - 2
    } else {
        tile.len()
    };
    if start >= end {
        return Err(JpegError::InvalidJpegAssembly {
            offset: 0,
            reason: "abbreviated JPEG tile is empty",
        });
    }
    Ok(&tile[start..end])
}

impl<'a> Iterator for JpegSegmentIter<'a> {
    type Item = Result<JpegSegment<'a>, JpegError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if !self.started {
            self.started = true;
            return Some(self.read_soi());
        }
        if self.scan_entropy {
            match next_marker_after_entropy(self.input, self.pos) {
                Ok(Some((marker_offset, marker))) => {
                    self.pos = marker_offset;
                    self.scan_entropy = false;
                    if marker == 0xd9 {
                        return Some(self.read_standalone_marker());
                    }
                }
                Ok(None) => {
                    self.finished = true;
                    return None;
                }
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            }
        }
        Some(self.read_segment())
    }
}

impl<'a> JpegSegmentIter<'a> {
    fn read_soi(&mut self) -> Result<JpegSegment<'a>, JpegError> {
        if self.input.len() < 2 {
            return Err(JpegError::Truncated {
                offset: 0,
                expected: 2 - self.input.len(),
            });
        }
        if self.input[0] != 0xff || self.input[1] != 0xd8 {
            return Err(JpegError::UnexpectedMarker {
                offset: 0,
                expected: MarkerKind::Soi,
                found: self.input.get(1).copied().unwrap_or(0),
            });
        }
        self.pos = 2;
        Ok(JpegSegment {
            marker: 0xd8,
            marker_offset: 0,
            payload_offset: 2,
            payload: &[],
        })
    }

    fn read_segment(&mut self) -> Result<JpegSegment<'a>, JpegError> {
        if self.pos >= self.input.len() {
            return Err(JpegError::Truncated {
                offset: self.pos,
                expected: 2,
            });
        }
        if self.input[self.pos] != 0xff {
            return Err(JpegError::InvalidMarker {
                offset: self.pos,
                marker: self.input[self.pos],
            });
        }
        while self.pos < self.input.len() && self.input[self.pos] == 0xff {
            self.pos += 1;
        }
        if self.pos >= self.input.len() {
            return Err(JpegError::Truncated {
                offset: self.pos,
                expected: 1,
            });
        }
        let marker = self.input[self.pos];
        let marker_offset = self.pos - 1;
        self.pos += 1;

        match marker {
            0x01 | 0xd0..=0xd9 => {
                if marker == 0xd9 {
                    self.finished = true;
                }
                Ok(JpegSegment {
                    marker,
                    marker_offset,
                    payload_offset: self.pos,
                    payload: &[],
                })
            }
            0x00 => Err(JpegError::InvalidMarker {
                offset: marker_offset,
                marker,
            }),
            _ => {
                if self.pos + 2 > self.input.len() {
                    return Err(JpegError::Truncated {
                        offset: self.pos,
                        expected: self.pos + 2 - self.input.len(),
                    });
                }
                let length = u16::from_be_bytes([self.input[self.pos], self.input[self.pos + 1]]);
                if length < 2 {
                    return Err(JpegError::InvalidSegmentLength {
                        offset: self.pos,
                        marker,
                        length,
                    });
                }
                let payload_offset = self.pos + 2;
                let payload_end = self.pos.checked_add(usize::from(length)).ok_or(
                    JpegError::InvalidSegmentLength {
                        offset: self.pos,
                        marker,
                        length,
                    },
                )?;
                if payload_end > self.input.len() {
                    return Err(JpegError::Truncated {
                        offset: payload_offset,
                        expected: payload_end - self.input.len(),
                    });
                }
                self.pos = payload_end;
                if marker == 0xda {
                    self.scan_entropy = true;
                }
                Ok(JpegSegment {
                    marker,
                    marker_offset,
                    payload_offset,
                    payload: &self.input[payload_offset..payload_end],
                })
            }
        }
    }

    fn read_standalone_marker(&mut self) -> Result<JpegSegment<'a>, JpegError> {
        let marker_offset = self.pos;
        if self.pos + 1 >= self.input.len() {
            return Err(JpegError::Truncated {
                offset: self.pos,
                expected: self.pos + 2 - self.input.len(),
            });
        }
        let marker = self.input[self.pos + 1];
        self.pos += 2;
        if marker == 0xd9 {
            self.finished = true;
        }
        Ok(JpegSegment {
            marker,
            marker_offset,
            payload_offset: self.pos,
            payload: &[],
        })
    }
}

fn parse_sof_info_at(
    marker: u8,
    payload: &[u8],
    payload_offset: usize,
    allow_zero_dimensions: bool,
) -> Result<JpegSofInfo, JpegError> {
    if payload.len() < 8 {
        return Err(JpegError::Truncated {
            offset: payload_offset + payload.len(),
            expected: 8 - payload.len(),
        });
    }

    let bit_depth = payload[0];
    let height = u16::from_be_bytes([payload[1], payload[2]]);
    let width = u16::from_be_bytes([payload[3], payload[4]]);
    let component_count = payload[5];
    let expected_len = 6 + usize::from(component_count) * 3;
    if payload.len() < expected_len {
        return Err(JpegError::Truncated {
            offset: payload_offset + payload.len(),
            expected: expected_len - payload.len(),
        });
    }

    let sof_kind = match (marker, bit_depth) {
        (0xc0, 8) => SofKind::Baseline8,
        (0xc1, 8) => SofKind::Extended8,
        (0xc1, 12) => SofKind::Extended12,
        (0xc2, 8) => SofKind::Progressive8,
        (0xc2, 12) => SofKind::Progressive12,
        (0xc3, 2..=16) => SofKind::Lossless,
        (0xc5, _) => {
            return Err(JpegError::UnsupportedSof {
                marker,
                reason: UnsupportedReason::DifferentialBaseline,
            });
        }
        (0xc6 | 0xc7, _) => {
            return Err(JpegError::UnsupportedSof {
                marker,
                reason: UnsupportedReason::Hierarchical,
            });
        }
        (0xc9..=0xcb, _) => {
            return Err(JpegError::UnsupportedSof {
                marker,
                reason: UnsupportedReason::ArithmeticCoding,
            });
        }
        (0xcd..=0xcf, _) => {
            return Err(JpegError::UnsupportedSof {
                marker,
                reason: UnsupportedReason::ArithmeticAndHierarchical,
            });
        }
        (_, bad_precision) => {
            return Err(JpegError::UnsupportedBitDepth {
                depth: bad_precision,
            })
        }
    };

    if !allow_zero_dimensions && (width == 0 || height == 0) {
        return Err(JpegError::ZeroDimension { width, height });
    }
    if width > 65_500 || height > 65_500 {
        return Err(JpegError::DimensionOverflow {
            width: u32::from(width),
            height: u32::from(height),
        });
    }
    if !matches!(component_count, 1 | 3 | 4) {
        return Err(JpegError::UnsupportedComponentCount {
            count: component_count,
        });
    }

    let components = parse_sof_components(payload, component_count)?;

    Ok(JpegSofInfo {
        marker,
        sof_kind,
        bit_depth,
        dimensions: (width, height),
        component_ids: components.ids,
        sampling: SamplingFactors::from_validated_components(
            &components.sampling[..usize::from(component_count)],
        ),
        quant_table_ids: components.quant_table_ids,
    })
}

struct ParsedSofComponents {
    ids: Vec<u8>,
    sampling: [(u8, u8); 4],
    quant_table_ids: Vec<u8>,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "validated SOF component indices fit the JPEG component-count byte"
)]
fn parse_sof_components(
    payload: &[u8],
    component_count: u8,
) -> Result<ParsedSofComponents, JpegError> {
    let count = usize::from(component_count);
    let mut sampling = [(0u8, 0u8); 4];
    let mut ids = try_vec_with_capacity(count)?;
    let mut quant_table_ids = try_vec_with_capacity(count)?;
    for (i, sampling_slot) in sampling.iter_mut().take(count).enumerate() {
        let base = 6 + i * 3;
        let sampling_byte = payload[base + 1];
        let (h, v) = (sampling_byte >> 4, sampling_byte & 0x0f);
        if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
            return Err(JpegError::InvalidSampling {
                component: i as u8,
                h,
                v,
            });
        }
        ids.push(payload[base]);
        *sampling_slot = (h, v);
        quant_table_ids.push(payload[base + 2]);
    }
    Ok(ParsedSofComponents {
        ids,
        sampling,
        quant_table_ids,
    })
}

#[cfg(test)]
#[path = "segment/allocation_tests.rs"]
mod allocation_tests;
