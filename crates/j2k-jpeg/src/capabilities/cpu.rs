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
            return match (info.color_space, info.bit_depth, request.fmt) {
                (ColorSpace::Grayscale, 8, PixelFormat::Gray8)
                | (ColorSpace::Grayscale, 16, PixelFormat::Gray16) => {
                    JpegBackendEligibility::eligible()
                }
                (
                    ColorSpace::Rgb | ColorSpace::YCbCr,
                    8,
                    PixelFormat::Rgb8 | PixelFormat::Rgba8,
                )
                | (
                    ColorSpace::Rgb | ColorSpace::YCbCr,
                    16,
                    PixelFormat::Rgb16 | PixelFormat::Rgba16,
                ) if is_supported_lossless_color_sampling(info) => {
                    JpegBackendEligibility::eligible()
                }
                (ColorSpace::Rgb, 8, PixelFormat::Rgb8 | PixelFormat::Rgba8)
                | (ColorSpace::Rgb, 16, PixelFormat::Rgb16 | PixelFormat::Rgba16) => {
                    JpegBackendEligibility::rejected(
                        "JPEG CPU lossless SOF3 APP14 RGB decode currently supports 4:4:4 sampling, even-width 8/16-bit 4:2:2 sampling, or even-dimension 8/16-bit 4:2:0 sampling",
                    )
                }
                (ColorSpace::YCbCr, 8, PixelFormat::Rgb8 | PixelFormat::Rgba8)
                | (ColorSpace::YCbCr, 16, PixelFormat::Rgb16 | PixelFormat::Rgba16) => {
                    JpegBackendEligibility::rejected(
                        "JPEG CPU lossless SOF3 YCbCr decode currently supports 4:4:4 sampling, even-width 8/16-bit 4:2:2 sampling, or even-dimension 8/16-bit 4:2:0 sampling",
                    )
                }
                _ => JpegBackendEligibility::rejected(
                    "JPEG CPU lossless SOF3 decode currently supports 8-bit Gray8, 16-bit Gray16, 8-bit YCbCr Rgb8/Rgba8 including even-width 4:2:2 and even-dimension 4:2:0, 16-bit YCbCr Rgb16/Rgba16 including even-width 4:2:2 and even-dimension 4:2:0, 8-bit APP14 RGB Rgb8/Rgba8 including even-width 4:2:2 and even-dimension 4:2:0, or 16-bit APP14 RGB Rgb16/Rgba16 including even-width 4:2:2 and even-dimension 4:2:0 output only",
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
            Self::Extended => "JPEG CPU 12-bit extended YCbCr decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only",
            Self::Progressive => "JPEG CPU 12-bit progressive YCbCr decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only",
        }
    }

    const fn rgb_sampling_reason(self) -> &'static str {
        match self {
            Self::Extended => "JPEG CPU 12-bit extended RGB decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only",
            Self::Progressive => "JPEG CPU 12-bit progressive RGB decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only",
        }
    }

    const fn four_component_sampling_reason(self) -> &'static str {
        match self {
            Self::Extended => "JPEG CPU 12-bit extended four-component CMYK/YCCK decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only",
            Self::Progressive => "JPEG CPU 12-bit progressive four-component CMYK/YCCK decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only",
        }
    }

    const fn output_reason(self) -> &'static str {
        match self {
            Self::Extended => "JPEG CPU 12-bit extended decode currently supports grayscale Gray16/Rgb16/Rgba16, APP14 RGB 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16, YCbCr 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16, or CMYK/YCCK 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16 only",
            Self::Progressive => "JPEG CPU 12-bit progressive decode currently supports grayscale Gray16/Rgb16/Rgba16, APP14 RGB 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16, YCbCr 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16, or CMYK/YCCK 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16 only",
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

fn is_supported_extended12_four_component_sampling(info: &Info) -> bool {
    info.sampling.len() == 4
        && matches!(
            (
                info.sampling.max_h,
                info.sampling.max_v,
                info.sampling.components()
            ),
            (1, 1, [(1, 1), (1, 1), (1, 1), (1, 1)])
                | (2, 1, [(2, 1), (1, 1), (1, 1), (1, 1)])
                | (2, 2, [(2, 2), (1, 1), (1, 1), (1, 1)])
        )
}

fn is_supported_12bit_three_component_sampling(info: &Info) -> bool {
    info.sampling.len() == 3
        && matches!(
            (
                info.sampling.max_h,
                info.sampling.max_v,
                info.sampling.components()
            ),
            (1, 1, [(1, 1), (1, 1), (1, 1)])
                | (2, 1, [(2, 1), (1, 1), (1, 1)])
                | (2, 2, [(2, 2), (1, 1), (1, 1)])
        )
}

fn is_supported_lossless_color_sampling(info: &Info) -> bool {
    info.sampling.len() == 3
        && matches!(
            (
                info.bit_depth,
                info.dimensions.0.is_multiple_of(2),
                info.dimensions.1.is_multiple_of(2),
                info.sampling.max_h,
                info.sampling.max_v,
                info.sampling.components()
            ),
            (_, _, _, 1, 1, [(1, 1), (1, 1), (1, 1)])
                | (8 | 16, true, _, 2, 1, [(2, 1), (1, 1), (1, 1)])
                | (8 | 16, true, true, 2, 2, [(2, 2), (1, 1), (1, 1)])
        )
}
