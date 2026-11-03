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
    if info.bit_depth != 8
        || !matches!(info.sof_kind, SofKind::Baseline8 | SofKind::Extended8)
        || info.scan_count != 1
    {
        return None;
    }
    fast_packet_family_for_shape(
        info.color_space,
        info.sampling.components(),
        info.dimensions.0,
    )
}

/// The color fast-packet family a coded color space and per-component
/// sampling select for an image `width` pixels wide, before scan, table and
/// component-order validation. The one shape rule shared by packet
/// classification, the device batch summary and the fast CPU routes.
pub(crate) fn fast_packet_family_for_shape(
    color_space: ColorSpace,
    sampling: &[(u8, u8)],
    width: u32,
) -> Option<JpegFastPacketFamily> {
    // libjpeg-turbo replicates full-size 2:1 chroma at most two samples wide
    // (images at most four pixels wide) instead of smoothing it; the fused
    // kernels only smooth, so such images take the generic CPU decoder.
    let smoothed_chroma = width > 4;
    match (color_space, sampling) {
        (ColorSpace::YCbCr, [(2, 2), (1, 1), (1, 1)]) if smoothed_chroma => {
            Some(JpegFastPacketFamily::Fast420)
        }
        (ColorSpace::YCbCr, [(2, 1), (1, 1), (1, 1)]) if smoothed_chroma => {
            Some(JpegFastPacketFamily::Fast422)
        }
        (ColorSpace::YCbCr | ColorSpace::Rgb, [(1, 1), (1, 1), (1, 1)]) => {
            Some(JpegFastPacketFamily::Fast444)
        }
        _ => None,
    }
}
