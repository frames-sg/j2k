// SPDX-License-Identifier: MIT OR Apache-2.0

//! Post-scan APP/COM marker warning classification.

use super::super::markers::marker_payload;
use crate::error::{JpegError, Warning};
use crate::parse::adobe_app14::{parse_adobe_app14, AdobeTransform};

pub(super) fn application_warning(
    bytes: &[u8],
    marker_offset: usize,
    code: u8,
) -> Result<(Option<Warning>, usize), JpegError> {
    let (payload, next) = marker_payload(bytes, marker_offset, code)?;
    let warning = match code {
        0xe0 | 0xfe => None,
        0xe2 => Some(Warning::IccProfileIgnored {
            size: payload.len(),
        }),
        0xee => match parse_adobe_app14(payload) {
            Some(AdobeTransform::Unknown) if payload.len() >= 12 && payload[11] > 2 => {
                Some(Warning::AdobeApp14Ambiguous {
                    raw_transform: payload[11],
                })
            }
            Some(_) => None,
            None => Some(Warning::UnknownAppMarker {
                marker: 0xee,
                size: payload.len(),
            }),
        },
        marker => Some(Warning::UnknownAppMarker {
            marker,
            size: payload.len(),
        }),
    };
    Ok((warning, next))
}
