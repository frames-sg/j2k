// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lightweight JPEG 2000 codestream header inspection.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

use crate::{MAX_J2K_IMAGE_DIMENSION, MAX_J2K_SPEC_COMPONENTS, MAX_J2K_TILE_COUNT};

const MARKER_SOC: u8 = 0x4F;
const MARKER_CAP: u8 = 0x50;
const MARKER_SIZ: u8 = 0x51;
const MARKER_COD: u8 = 0x52;
const MARKER_SOT: u8 = 0x90;
const MARKER_SOD: u8 = 0x93;
const MARKER_EOC: u8 = 0xD9;

/// Parsed JPEG 2000 codestream metadata from the main header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kCodestreamHeaderMetadata {
    /// Reference-grid image dimensions derived from SIZ.
    pub dimensions: (u32, u32),
    /// Number of codestream components.
    pub components: u16,
    /// Maximum component precision in bits.
    pub bit_depth: u8,
    /// Reference tile width and height.
    pub tile_size: (u32, u32),
    /// Number of reference tiles horizontally and vertically.
    pub tile_count: (u32, u32),
    /// Per-component SIZ precision and sampling metadata.
    pub component_info: Vec<J2kCodestreamComponentHeader>,
    /// Number of resolution levels from COD.
    pub resolution_levels: u8,
    /// Whether COD enables a multi-component transform.
    pub has_mct: bool,
    /// Whether COD selects the reversible 5/3 transform.
    pub reversible: bool,
    /// Whether the codestream advertises high-throughput block coding.
    pub high_throughput: bool,
}

/// Parsed SIZ component metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kCodestreamComponentHeader {
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
}

/// Error returned by [`inspect_j2k_codestream_header`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum J2kCodestreamHeaderError {
    /// Input was shorter than the required prefix.
    TooShort {
        /// Required byte count.
        need: usize,
        /// Available byte count.
        have: usize,
    },
    /// Input ended while reading a marker or marker segment.
    TruncatedAt {
        /// Byte offset where the truncated segment begins.
        offset: usize,
        /// Segment being read.
        segment: &'static str,
    },
    /// A codestream marker did not start with `0xFF`.
    InvalidMarker {
        /// Byte offset of the invalid marker.
        offset: usize,
        /// Byte found where the marker code was expected.
        marker: u8,
    },
    /// A required codestream marker was absent.
    MissingRequiredMarker {
        /// Missing marker name.
        marker: &'static str,
    },
    /// A generic marker segment was malformed.
    InvalidSegment {
        /// Byte offset of the segment length.
        offset: usize,
        /// Description of the invalid segment.
        what: &'static str,
    },
    /// The SIZ marker segment was malformed or unsupported.
    InvalidSiz {
        /// Description of the invalid SIZ segment.
        what: &'static str,
    },
    /// The COD marker segment was malformed or unsupported.
    InvalidCod {
        /// Description of the invalid COD segment.
        what: &'static str,
    },
    /// The header is valid, but outside the public inspection contract.
    Unsupported {
        /// Description of the unsupported feature.
        what: &'static str,
    },
}

impl fmt::Display for J2kCodestreamHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { need, have } => {
                write!(f, "input too short: need {need} bytes, have {have}")
            }
            Self::TruncatedAt { offset, segment } => {
                write!(f, "truncated {segment} at offset {offset}")
            }
            Self::InvalidMarker { offset, marker } => {
                write!(
                    f,
                    "invalid codestream marker FF{marker:02X} at offset {offset}"
                )
            }
            Self::MissingRequiredMarker { marker } => {
                write!(f, "missing required codestream marker {marker}")
            }
            Self::InvalidSegment { what, .. } => write!(f, "invalid marker segment: {what}"),
            Self::InvalidSiz { what } => write!(f, "invalid SIZ segment: {what}"),
            Self::InvalidCod { what } => write!(f, "invalid COD segment: {what}"),
            Self::Unsupported { what } => write!(f, "unsupported codestream header: {what}"),
        }
    }
}

/// Inspect a raw JPEG 2000 codestream main header without decoding tile data.
///
/// This helper reads SIZ/COD metadata and stops at SOT/SOD/EOC. It intentionally
/// does not require full decode headers such as QCD, so callers can inspect the
/// same lightweight codestreams that later decode construction may reject.
pub fn inspect_j2k_codestream_header(
    input: &[u8],
) -> Result<J2kCodestreamHeaderMetadata, J2kCodestreamHeaderError> {
    if input.len() < 2 {
        return Err(J2kCodestreamHeaderError::TooShort {
            need: 2,
            have: input.len(),
        });
    }
    if !looks_like_j2k_codestream(input) {
        return Err(J2kCodestreamHeaderError::InvalidMarker {
            offset: 0,
            marker: input[1],
        });
    }

    let mut offset = 2usize;
    let mut siz = None;
    let mut cod = None;
    let mut high_throughput_cap = false;
    let mut terminated = false;

    while offset < input.len() {
        let marker = read_marker(input, &mut offset)?;
        match marker {
            MARKER_SOT | MARKER_SOD | MARKER_EOC => {
                terminated = true;
                break;
            }
            MARKER_SIZ => {
                let payload = read_segment_payload(input, &mut offset, "SIZ")?;
                siz = Some(parse_siz(payload)?);
            }
            MARKER_COD => {
                let payload = read_segment_payload(input, &mut offset, "COD")?;
                cod = Some(parse_cod(payload)?);
            }
            MARKER_CAP => {
                let _ = read_segment_payload(input, &mut offset, "CAP")?;
                high_throughput_cap = true;
            }
            _ => {
                let _ = read_segment_payload(input, &mut offset, "segment")?;
            }
        }
    }

    if !terminated {
        return Err(J2kCodestreamHeaderError::TruncatedAt {
            offset,
            segment: "main header terminator",
        });
    }

    let siz = siz.ok_or(J2kCodestreamHeaderError::MissingRequiredMarker { marker: "SIZ" })?;
    let cod = cod
        .ok_or(J2kCodestreamHeaderError::MissingRequiredMarker { marker: "COD" })?
        .with_high_throughput_cap(high_throughput_cap);

    Ok(J2kCodestreamHeaderMetadata {
        dimensions: siz.dimensions,
        components: siz.components,
        bit_depth: siz.bit_depth,
        tile_size: siz.tile_size,
        tile_count: siz.tile_count,
        component_info: siz.component_info,
        resolution_levels: cod.resolution_levels,
        has_mct: cod.has_mct,
        reversible: cod.reversible,
        high_throughput: cod.high_throughput,
    })
}

/// Return whether bytes start with the raw JPEG 2000 SOC marker.
#[must_use]
pub fn looks_like_j2k_codestream(input: &[u8]) -> bool {
    input.len() >= 2 && input[0] == 0xFF && input[1] == MARKER_SOC
}

#[derive(Debug, Clone)]
struct ParsedSiz {
    dimensions: (u32, u32),
    components: u16,
    bit_depth: u8,
    tile_size: (u32, u32),
    tile_count: (u32, u32),
    component_info: Vec<J2kCodestreamComponentHeader>,
}

#[derive(Debug, Clone, Copy)]
struct ParsedCod {
    resolution_levels: u8,
    has_mct: bool,
    reversible: bool,
    high_throughput: bool,
}

impl ParsedCod {
    const fn with_high_throughput_cap(mut self, high_throughput_cap: bool) -> Self {
        self.high_throughput |= high_throughput_cap;
        self
    }
}

fn read_marker(input: &[u8], offset: &mut usize) -> Result<u8, J2kCodestreamHeaderError> {
    if *offset + 2 > input.len() {
        return Err(J2kCodestreamHeaderError::TruncatedAt {
            offset: *offset,
            segment: "marker",
        });
    }
    if input[*offset] != 0xFF {
        return Err(J2kCodestreamHeaderError::InvalidMarker {
            offset: *offset,
            marker: input[*offset],
        });
    }
    let marker = input[*offset + 1];
    *offset += 2;
    Ok(marker)
}

fn read_segment_payload<'a>(
    input: &'a [u8],
    offset: &mut usize,
    segment: &'static str,
) -> Result<&'a [u8], J2kCodestreamHeaderError> {
    if *offset + 2 > input.len() {
        return Err(J2kCodestreamHeaderError::TruncatedAt {
            offset: *offset,
            segment,
        });
    }
    let length = u16::from_be_bytes([input[*offset], input[*offset + 1]]) as usize;
    if length < 2 {
        return Err(J2kCodestreamHeaderError::InvalidSegment {
            offset: *offset,
            what: "segment length smaller than header",
        });
    }
    let start = *offset + 2;
    let end = *offset + length;
    if end > input.len() {
        return Err(J2kCodestreamHeaderError::TruncatedAt {
            offset: *offset,
            segment,
        });
    }
    *offset = end;
    Ok(&input[start..end])
}

#[allow(clippy::similar_names)]
fn parse_siz(payload: &[u8]) -> Result<ParsedSiz, J2kCodestreamHeaderError> {
    if payload.len() < 36 {
        return Err(J2kCodestreamHeaderError::InvalidSiz {
            what: "payload shorter than fixed SIZ header",
        });
    }
    let x_size = read_u32(payload, 2);
    let y_size = read_u32(payload, 6);
    let x_origin = read_u32(payload, 10);
    let y_origin = read_u32(payload, 14);
    let tile_width = read_u32(payload, 18);
    let tile_height = read_u32(payload, 22);
    let tile_x_origin = read_u32(payload, 26);
    let tile_y_origin = read_u32(payload, 30);
    let component_count = read_u16(payload, 34);

    let component_bytes = usize::from(component_count) * 3;
    if payload.len() < 36 + component_bytes {
        return Err(J2kCodestreamHeaderError::InvalidSiz {
            what: "component descriptors truncated",
        });
    }
    if component_count == 0 {
        return Err(J2kCodestreamHeaderError::InvalidSiz {
            what: "component count must be non-zero",
        });
    }
    if component_count > MAX_J2K_SPEC_COMPONENTS {
        return Err(J2kCodestreamHeaderError::InvalidSiz {
            what: "component count exceeds JPEG 2000 limit",
        });
    }
    if x_size <= x_origin || y_size <= y_origin {
        return Err(J2kCodestreamHeaderError::InvalidSiz {
            what: "image origin must be smaller than image size",
        });
    }
    if tile_width == 0 || tile_height == 0 {
        return Err(J2kCodestreamHeaderError::InvalidSiz {
            what: "tile size must be non-zero",
        });
    }
    if tile_x_origin >= x_size || tile_y_origin >= y_size {
        return Err(J2kCodestreamHeaderError::InvalidSiz {
            what: "tile origin must be within image bounds",
        });
    }
    if tile_x_origin > x_origin || tile_y_origin > y_origin {
        return Err(J2kCodestreamHeaderError::InvalidSiz {
            what: "tile origin must not exceed image origin",
        });
    }
    if tile_x_origin
        .checked_add(tile_width)
        .ok_or(J2kCodestreamHeaderError::InvalidSiz {
            what: "tile extent overflows",
        })?
        <= x_origin
        || tile_y_origin
            .checked_add(tile_height)
            .ok_or(J2kCodestreamHeaderError::InvalidSiz {
                what: "tile extent overflows",
            })?
            <= y_origin
    {
        return Err(J2kCodestreamHeaderError::InvalidSiz {
            what: "first tile must overlap image area",
        });
    }

    let width = x_size - x_origin;
    let height = y_size - y_origin;
    if width > MAX_J2K_IMAGE_DIMENSION || height > MAX_J2K_IMAGE_DIMENSION {
        return Err(J2kCodestreamHeaderError::InvalidSiz {
            what: "image dimensions exceed JPEG 2000 inspect limit",
        });
    }
    let tiles_x = (x_size - tile_x_origin).div_ceil(tile_width);
    let tiles_y = (y_size - tile_y_origin).div_ceil(tile_height);
    let tile_count = u64::from(tiles_x) * u64::from(tiles_y);
    if tile_count > MAX_J2K_TILE_COUNT {
        return Err(J2kCodestreamHeaderError::InvalidSiz {
            what: "image has too many tiles",
        });
    }
    let mut bit_depth = 0u8;
    let mut component_info = Vec::with_capacity(usize::from(component_count));
    for idx in 0..usize::from(component_count) {
        let ssiz = payload[36 + idx * 3];
        let precision = (ssiz & 0x7F) + 1;
        let x_rsiz = payload[36 + idx * 3 + 1];
        let y_rsiz = payload[36 + idx * 3 + 2];
        if x_rsiz == 0 || y_rsiz == 0 {
            return Err(J2kCodestreamHeaderError::InvalidSiz {
                what: "component sampling factors must be non-zero",
            });
        }
        bit_depth = bit_depth.max(precision);
        component_info.push(J2kCodestreamComponentHeader {
            bit_depth: precision,
            signed: ssiz & 0x80 != 0,
            x_rsiz,
            y_rsiz,
        });
    }

    Ok(ParsedSiz {
        dimensions: (width, height),
        components: component_count,
        bit_depth,
        tile_size: (tile_width, tile_height),
        tile_count: (tiles_x, tiles_y),
        component_info,
    })
}

fn parse_cod(payload: &[u8]) -> Result<ParsedCod, J2kCodestreamHeaderError> {
    if payload.len() < 10 {
        return Err(J2kCodestreamHeaderError::InvalidCod {
            what: "payload shorter than fixed COD header",
        });
    }
    Ok(ParsedCod {
        resolution_levels: payload[5].saturating_add(1),
        has_mct: payload[4] != 0,
        reversible: payload[9] == 1,
        high_throughput: payload[8] & 0x40 != 0,
    })
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::{inspect_j2k_codestream_header, J2kCodestreamHeaderError};
    use alloc::{vec, vec::Vec};

    #[test]
    fn inspect_j2k_codestream_header_accepts_minimal_main_header() {
        let header = inspect_j2k_codestream_header(&minimal_codestream()).expect("header");

        assert_eq!(header.dimensions, (128, 64));
        assert_eq!(header.components, 3);
        assert_eq!(header.bit_depth, 8);
        assert_eq!(header.tile_size, (64, 64));
        assert_eq!(header.tile_count, (2, 1));
        assert_eq!(header.resolution_levels, 6);
        assert!(header.reversible);
    }

    #[test]
    fn inspect_rejects_zero_component_sampling() {
        let mut bytes = minimal_codestream();
        rewrite_component_sampling(&mut bytes, 0, 0, 1);

        let err = inspect_j2k_codestream_header(&bytes).expect_err("zero sampling must reject");

        assert!(matches!(err, J2kCodestreamHeaderError::InvalidSiz { .. }));
    }

    #[test]
    fn inspect_rejects_oversized_dimensions() {
        let mut bytes = minimal_codestream();
        rewrite_siz_u32(&mut bytes, 2, 60_001);

        let err = inspect_j2k_codestream_header(&bytes).expect_err("oversized width must reject");

        assert!(matches!(err, J2kCodestreamHeaderError::InvalidSiz { .. }));
    }

    #[test]
    fn inspect_rejects_tile_origin_after_image_origin() {
        let mut bytes = minimal_codestream();
        rewrite_siz_u32(&mut bytes, 26, 1);

        let err = inspect_j2k_codestream_header(&bytes).expect_err("bad tile origin must reject");

        assert!(matches!(err, J2kCodestreamHeaderError::InvalidSiz { .. }));
    }

    #[test]
    fn inspect_rejects_tile_extent_overflow() {
        let mut bytes = minimal_codestream();
        rewrite_siz_u32(&mut bytes, 2, u32::MAX);
        rewrite_siz_u32(&mut bytes, 10, u32::MAX - 1);
        rewrite_siz_u32(&mut bytes, 18, 10);
        rewrite_siz_u32(&mut bytes, 26, u32::MAX - 2);

        let err = inspect_j2k_codestream_header(&bytes).expect_err("overflow must reject");

        assert!(matches!(err, J2kCodestreamHeaderError::InvalidSiz { .. }));
    }

    #[test]
    fn inspect_rejects_excessive_tile_count() {
        let mut bytes = minimal_codestream();
        rewrite_siz_u32(&mut bytes, 2, 257);
        rewrite_siz_u32(&mut bytes, 6, 257);
        rewrite_siz_u32(&mut bytes, 18, 1);
        rewrite_siz_u32(&mut bytes, 22, 1);

        let err = inspect_j2k_codestream_header(&bytes).expect_err("tile count must reject");

        assert!(matches!(err, J2kCodestreamHeaderError::InvalidSiz { .. }));
    }

    fn minimal_codestream() -> Vec<u8> {
        let mut bytes = vec![0xFF, 0x4F];
        let mut siz = Vec::new();
        push_u16(&mut siz, 0);
        push_u32(&mut siz, 128);
        push_u32(&mut siz, 64);
        push_u32(&mut siz, 0);
        push_u32(&mut siz, 0);
        push_u32(&mut siz, 64);
        push_u32(&mut siz, 64);
        push_u32(&mut siz, 0);
        push_u32(&mut siz, 0);
        push_u16(&mut siz, 3);
        for _ in 0..3 {
            siz.extend_from_slice(&[0x07, 0x01, 0x01]);
        }
        bytes.extend_from_slice(&[0xFF, 0x51]);
        push_u16(&mut bytes, (siz.len() + 2) as u16);
        bytes.extend_from_slice(&siz);

        let cod = [0x00, 0x00, 0x00, 0x01, 0x01, 0x05, 0x04, 0x04, 0x00, 0x01];
        bytes.extend_from_slice(&[0xFF, 0x52]);
        push_u16(&mut bytes, (cod.len() + 2) as u16);
        bytes.extend_from_slice(&cod);
        bytes.extend_from_slice(&[0xFF, 0x90, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        bytes
    }

    fn push_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_be_bytes());
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_be_bytes());
    }

    fn rewrite_siz_u32(bytes: &mut [u8], payload_offset: usize, value: u32) {
        let siz = bytes
            .windows(2)
            .position(|marker| marker == [0xFF, 0x51])
            .expect("SIZ marker");
        let offset = siz + 4 + payload_offset;
        bytes[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
    }

    fn rewrite_component_sampling(bytes: &mut [u8], component: usize, x_rsiz: u8, y_rsiz: u8) {
        let siz = bytes
            .windows(2)
            .position(|marker| marker == [0xFF, 0x51])
            .expect("SIZ marker");
        let component_offset = siz + 40 + component * 3;
        bytes[component_offset + 1] = x_rsiz;
        bytes[component_offset + 2] = y_rsiz;
    }
}
