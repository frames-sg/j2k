// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{ColorSpace, Decoder, Info, SofKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
/// Color fast-packet family selected from validated JPEG metadata.
pub enum JpegFastPacketFamily {
    /// 8-bit YCbCr 4:2:0 packet.
    Fast420,
    /// 8-bit YCbCr 4:2:2 packet.
    Fast422,
    /// 8-bit YCbCr or RGB 4:4:4 packet.
    Fast444,
}

/// Select the only color fast-packet family that can match a parsed decoder.
///
/// Packet construction still performs the complete scan-order, table, entropy,
/// allocation, and invariant validation for the selected family.
#[doc(hidden)]
#[must_use]
pub fn classify_color_fast_packet_family(decoder: &Decoder<'_>) -> Option<JpegFastPacketFamily> {
    classify_color_fast_packet_info(decoder.info())
}

pub(super) fn classify_color_fast_packet_info(info: &Info) -> Option<JpegFastPacketFamily> {
    if info.bit_depth != 8 || !matches!(info.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return None;
    }

    match (info.color_space, info.sampling.components()) {
        (ColorSpace::YCbCr, [(2, 2), (1, 1), (1, 1)]) => Some(JpegFastPacketFamily::Fast420),
        (ColorSpace::YCbCr, [(2, 1), (1, 1), (1, 1)]) => Some(JpegFastPacketFamily::Fast422),
        (ColorSpace::YCbCr | ColorSpace::Rgb, [(1, 1), (1, 1), (1, 1)]) => {
            Some(JpegFastPacketFamily::Fast444)
        }
        _ => None,
    }
}
