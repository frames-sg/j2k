// SPDX-License-Identifier: MIT OR Apache-2.0

//! Post-entropy marker payload and scan-count helpers.

use crate::error::JpegError;
use memchr::memchr;

#[expect(
    clippy::cast_possible_truncation,
    reason = "marker payload lengths are bounded by the JPEG 16-bit segment-length field"
)]
pub(super) fn marker_payload(
    bytes: &[u8],
    marker_offset: usize,
    marker: u8,
) -> Result<(&[u8], usize), JpegError> {
    let length_offset = marker_offset + 2;
    if length_offset + 1 >= bytes.len() {
        return Err(JpegError::Truncated {
            offset: length_offset,
            expected: length_offset + 2 - bytes.len(),
        });
    }
    let length = usize::from(u16::from_be_bytes([
        bytes[length_offset],
        bytes[length_offset + 1],
    ]));
    if length < 2 {
        return Err(JpegError::InvalidSegmentLength {
            offset: length_offset,
            marker,
            length: length as u16,
        });
    }
    let payload_start = length_offset + 2;
    let payload_end = length_offset
        .checked_add(length)
        .ok_or(JpegError::InvalidSegmentLength {
            offset: length_offset,
            marker,
            length: length as u16,
        })?;
    if payload_end > bytes.len() {
        return Err(JpegError::Truncated {
            offset: payload_start,
            expected: payload_end - bytes.len(),
        });
    }
    Ok((&bytes[payload_start..payload_end], payload_end))
}

pub(super) fn count_scan_markers(bytes: &[u8], mut position: usize) -> u16 {
    let mut count = 1u16;
    while position < bytes.len() {
        let Some(relative) = memchr(0xff, &bytes[position..]) else {
            break;
        };
        let marker_offset = position + relative;
        let mut code_offset = marker_offset + 1;
        while code_offset < bytes.len() && bytes[code_offset] == 0xff {
            code_offset += 1;
        }
        if code_offset >= bytes.len() {
            break;
        }
        position = code_offset + 1;
        match bytes[code_offset] {
            0x00 | 0xd0..=0xd7 => {}
            0xd9 => break,
            0xda => {
                count = count.saturating_add(1);
                let Some(next) = skip_marker_segment(bytes, marker_offset) else {
                    break;
                };
                position = next;
            }
            0x01 | 0xd8 => position = marker_offset + 2,
            _ => {
                let Some(next) = skip_marker_segment(bytes, marker_offset) else {
                    break;
                };
                position = next;
            }
        }
    }
    count
}

fn skip_marker_segment(bytes: &[u8], marker_offset: usize) -> Option<usize> {
    let length_offset = marker_offset + 2;
    if length_offset + 1 >= bytes.len() {
        return None;
    }
    let length = usize::from(u16::from_be_bytes([
        bytes[length_offset],
        bytes[length_offset + 1],
    ]));
    if length < 2 {
        return None;
    }
    let next = length_offset.checked_add(length)?;
    (next <= bytes.len()).then_some(next)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "validated JPEG segment lengths fit the 16-bit marker grammar"
)]
pub(super) fn segment_length(payload: &[u8]) -> u16 {
    (payload.len() + 2) as u16
}
