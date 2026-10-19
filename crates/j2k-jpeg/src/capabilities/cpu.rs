// SPDX-License-Identifier: MIT OR Apache-2.0

//! Correctness eligibility for portable CPU decode.

use super::{JpegBackendEligibility, JpegCapabilityRequest};
use crate::{ColorSpace, Info, SofKind};
use j2k_core::PixelFormat;

pub(super) fn cpu_eligibility(
    info: &Info,
    request: JpegCapabilityRequest,
) -> JpegBackendEligibility {
    match info.sof_kind {
        SofKind::Extended12 if is_twelve_bit_output_request(request.fmt) => {
            return twelve_bit_eligibility(info, request.fmt, TwelveBitSof::Extended);
        }
        SofKind::Progressive12 if is_twelve_bit_output_request(request.fmt) => {
            return twelve_bit_eligibility(info, request.fmt, TwelveBitSof::Progressive);
        }
        SofKind::Extended12 | SofKind::Progressive12 => {
            return JpegBackendEligibility::rejected(
                "JPEG CPU decode does not yet support this 12-bit JPEG output",
            );
        }
        SofKind::Lossless
            if matches!(
                request.fmt,
                PixelFormat::Gray8
                    | PixelFormat::Gray16
                    | PixelFormat::Rgb8
                    | PixelFormat::Rgba8
                    | PixelFormat::Rgb16
                    | PixelFormat::Rgba16
            ) =>
        {
            // Samples up to 8 bits decode to 8-bit output and 9- to 16-bit
            // samples to 16-bit output, at their native range.
            let wide = info.bit_depth > 8;
            return match (info.color_space, wide, request.fmt) {
                (ColorSpace::Grayscale, false, PixelFormat::Gray8)
                | (ColorSpace::Grayscale, true, PixelFormat::Gray16) => {
                    JpegBackendEligibility::eligible()
                }
                (ColorSpace::Rgb | ColorSpace::YCbCr, false, PixelFormat::Rgb8 | PixelFormat::Rgba8)
                | (ColorSpace::Rgb | ColorSpace::YCbCr, true, PixelFormat::Rgb16 | PixelFormat::Rgba16)
                    if is_supported_lossless_color_sampling(info) =>
                {
                    JpegBackendEligibility::eligible()
                }
                (ColorSpace::Rgb, false, PixelFormat::Rgb8 | PixelFormat::Rgba8)
                | (ColorSpace::Rgb, true, PixelFormat::Rgb16 | PixelFormat::Rgba16) => {
                    JpegBackendEligibility::rejected(
                        "JPEG CPU lossless SOF3 RGB decode supports sampling whose component factors divide the maximum factors only",
                    )
                }
                (ColorSpace::YCbCr, false, PixelFormat::Rgb8 | PixelFormat::Rgba8)
                | (ColorSpace::YCbCr, true, PixelFormat::Rgb16 | PixelFormat::Rgba16) => {
                    JpegBackendEligibility::rejected(
                        "JPEG CPU lossless SOF3 YCbCr decode supports sampling whose component factors divide the maximum factors only",
                    )
                }
                _ => JpegBackendEligibility::rejected(
                    "JPEG CPU lossless SOF3 decode supports Gray8 (grayscale) or Rgb8/Rgba8 (RGB or YCbCr) output for 2- to 8-bit precision, and Gray16 or Rgb16/Rgba16 output for 9- to 16-bit precision",
                ),
            };
        }
        SofKind::Lossless => {
            return JpegBackendEligibility::rejected(
                "JPEG CPU decode does not yet support lossless SOF3 JPEG",
            );
        }
        SofKind::Baseline8 | SofKind::Extended8 | SofKind::Progressive8 => {}
    }

    match (request.fmt, request.op.scale()) {
        (PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8, _) => {
            JpegBackendEligibility::eligible()
        }
        (PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16, _) => {
            JpegBackendEligibility::rejected("JPEG CPU decode does not support 16-bit output")
        }
        _ => JpegBackendEligibility::rejected("unsupported JPEG CPU output format"),
    }
}

#[derive(Debug, Clone, Copy)]
enum TwelveBitSof {
    Extended,
    Progressive,
}

impl TwelveBitSof {
    const fn ycbcr_sampling_reason(self) -> &'static str {
        match self {
            Self::Extended => "JPEG CPU 12-bit extended YCbCr decode currently supports sampling whose component factors divide the maximum factors only",
            Self::Progressive => "JPEG CPU 12-bit progressive YCbCr decode currently supports sampling whose component factors divide the maximum factors only",
        }
    }

    const fn rgb_sampling_reason(self) -> &'static str {
        match self {
            Self::Extended => "JPEG CPU 12-bit extended RGB decode currently supports sampling whose component factors divide the maximum factors only",
            Self::Progressive => "JPEG CPU 12-bit progressive RGB decode currently supports sampling whose component factors divide the maximum factors only",
        }
    }

    const fn four_component_sampling_reason(self) -> &'static str {
        match self {
            Self::Extended => "JPEG CPU 12-bit extended four-component CMYK/YCCK decode currently supports sampling whose component factors divide the maximum factors only",
            Self::Progressive => "JPEG CPU 12-bit progressive four-component CMYK/YCCK decode currently supports sampling whose component factors divide the maximum factors only",
        }
    }

    const fn output_reason(self) -> &'static str {
        match self {
            Self::Extended => "JPEG CPU 12-bit extended decode currently supports grayscale Gray16/Rgb16/Rgba16, APP14 RGB or YCbCr Rgb16/Rgba16 whose component factors divide the maximum factors, or CMYK/YCCK Rgb16/Rgba16 whose component factors divide the maximum factors only",
            Self::Progressive => "JPEG CPU 12-bit progressive decode currently supports grayscale Gray16/Rgb16/Rgba16, APP14 RGB or YCbCr Rgb16/Rgba16 whose component factors divide the maximum factors, or CMYK/YCCK Rgb16/Rgba16 whose component factors divide the maximum factors only",
        }
    }
}

fn is_twelve_bit_output_request(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Gray16 | PixelFormat::Rgb16 | PixelFormat::Rgba16
    )
}

fn twelve_bit_eligibility(
    info: &Info,
    fmt: PixelFormat,
    sof: TwelveBitSof,
) -> JpegBackendEligibility {
    match (info.color_space, fmt) {
        (ColorSpace::Grayscale, PixelFormat::Gray16 | PixelFormat::Rgb16 | PixelFormat::Rgba16) => {
            JpegBackendEligibility::eligible()
        }
        (ColorSpace::Rgb | ColorSpace::YCbCr, PixelFormat::Rgb16 | PixelFormat::Rgba16)
            if is_supported_12bit_three_component_sampling(info) =>
        {
            JpegBackendEligibility::eligible()
        }
        (ColorSpace::Cmyk | ColorSpace::Ycck, PixelFormat::Rgb16 | PixelFormat::Rgba16)
            if is_supported_extended12_four_component_sampling(info) =>
        {
            JpegBackendEligibility::eligible()
        }
        (ColorSpace::YCbCr, PixelFormat::Rgb16 | PixelFormat::Rgba16) => {
            JpegBackendEligibility::rejected(sof.ycbcr_sampling_reason())
        }
        (ColorSpace::Rgb, PixelFormat::Rgb16 | PixelFormat::Rgba16) => {
            JpegBackendEligibility::rejected(sof.rgb_sampling_reason())
        }
        (ColorSpace::Cmyk | ColorSpace::Ycck, PixelFormat::Rgb16 | PixelFormat::Rgba16) => {
            JpegBackendEligibility::rejected(sof.four_component_sampling_reason())
        }
        _ => JpegBackendEligibility::rejected(sof.output_reason()),
    }
}

/// 4:4:4, 4:2:2 and 4:2:0 use the direct writers; every other whole-ratio
/// layout (YCCK with Y and K at 2x2, ...) renders through component planes.
fn is_supported_extended12_four_component_sampling(info: &Info) -> bool {
    info.sampling.len() == 4 && info.sampling.has_integral_ratios()
}

/// 4:4:4, 4:2:2 and 4:2:0 use the direct writers; every other whole-ratio
/// layout (4:4:0, 4:1:1, 4:1:0, 1x4, ...) renders through component planes.
fn is_supported_12bit_three_component_sampling(info: &Info) -> bool {
    info.sampling.len() == 3 && info.sampling.has_integral_ratios()
}

/// Lossless color reconstructs any precision and size; subsampled
/// components are replicated on output, so any whole-ratio layout works.
fn is_supported_lossless_color_sampling(info: &Info) -> bool {
    info.sampling.len() == 3 && info.sampling.has_integral_ratios()
}
