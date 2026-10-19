// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free public header inspection walk.

use crate::error::{JpegError, MarkerKind};
use crate::info::{Info, McuGeometry, SofKind};
use crate::parse::adobe_app14::{is_jfif_app0, parse_adobe_app14};
use crate::parse::markers::MarkerWalker;
use crate::parse::scan::parse_scan_header;
use crate::parse::sof::{parse_sof, ParsedSof};

use super::markers::count_scan_markers;
use super::types::color_space_for_frame;
use super::validation::{
    is_split_frame_scan, normalize_restart_interval, validate_scan_parameters,
    validate_sequential_scan_components, validate_split_scan,
};

#[expect(
    clippy::cast_possible_truncation,
    reason = "reported JPEG segment lengths are bounded by the 16-bit marker grammar"
)]
pub(crate) fn parse_info(bytes: &[u8]) -> Result<Info, JpegError> {
    let mut walker = MarkerWalker::new(bytes);
    walker.read_soi()?;

    let mut sof = None;
    let mut restart_interval = None;
    let mut adobe = None;
    let mut jfif = false;
    let mut scan_count = 0u16;
    while let Some(marker) = walker.next_marker()? {
        match marker.code {
            0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => {
                if sof.is_some() {
                    return Err(JpegError::DuplicateMarker {
                        offset: marker.offset,
                        marker: MarkerKind::Sof,
                    });
                }
                sof = Some(parse_sof(marker.code, marker.payload, marker.offset + 4)?);
            }
            0xdd => {
                if marker.payload.len() != 2 {
                    return Err(JpegError::InvalidSegmentLength {
                        offset: marker.offset,
                        marker: 0xdd,
                        length: (marker.payload.len() + 2) as u16,
                    });
                }
                restart_interval = normalize_restart_interval(u16::from_be_bytes([
                    marker.payload[0],
                    marker.payload[1],
                ]));
            }
            0xda => {
                let scan = parse_scan_header(marker.payload, marker.offset + 4)?;
                if let Some(frame) = sof.as_ref() {
                    validate_scan_parameters(frame.sof_kind, &scan, marker.offset + 4)?;
                    let split = is_split_frame_scan(frame, &scan);
                    if split {
                        validate_split_scan(frame, &scan, marker.offset + 4)?;
                    } else {
                        validate_sequential_scan_components(frame, &scan, marker.offset + 4)?;
                    }
                    scan_count = if split
                        || matches!(
                            frame.sof_kind,
                            SofKind::Progressive8 | SofKind::Progressive12
                        ) {
                        count_scan_markers(bytes, walker.position())
                    } else {
                        1
                    };
                }
                break;
            }
            0xee => adobe = parse_adobe_app14(marker.payload).or(adobe),
            0xe0 => jfif |= is_jfif_app0(marker.payload),
            0xdb | 0xc4 | 0xe1..=0xef | 0xfe => {}
            _ => {
                return Err(JpegError::InvalidMarker {
                    offset: marker.offset,
                    marker: marker.code,
                });
            }
        }
    }

    let sof: ParsedSof = sof.ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sof,
    })?;
    let dimensions = (u32::from(sof.width), u32::from(sof.height));
    Ok(Info {
        dimensions,
        color_space: color_space_for_frame(
            sof.component_ids.as_slice(),
            sof.sof_kind == SofKind::Lossless,
            adobe,
            jfif,
        ),
        sampling: sof.sampling,
        sof_kind: sof.sof_kind,
        bit_depth: sof.bit_depth,
        restart_interval,
        mcu_geometry: McuGeometry::from_sampling(dimensions, sof.sampling),
        scan_count,
    })
}
