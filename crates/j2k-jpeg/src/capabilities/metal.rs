// SPDX-License-Identifier: MIT OR Apache-2.0

//! Correctness eligibility for Metal decode surfaces and resident batches.

use super::{JpegBackendEligibility, JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp};
use crate::{ColorSpace, DeviceBatchSummary, Info, SofKind};
use j2k_core::{Downscale, PixelFormat};

pub(super) fn metal_fast_eligibility(
    info: &Info,
    device: DeviceBatchSummary,
    request: JpegCapabilityRequest,
) -> JpegBackendEligibility {
    if !matches!(
        request.fmt,
        PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8
    ) {
        return JpegBackendEligibility::rejected(
            "JPEG Metal fast path supports Gray8, Rgb8, or Rgba8 output formats",
        );
    }
    if !matches!(info.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return JpegBackendEligibility::rejected(
            "JPEG Metal fast path currently supports baseline/extended 8-bit sequential JPEG only",
        );
    }
    if !matches!(
        info.color_space,
        ColorSpace::Grayscale | ColorSpace::YCbCr | ColorSpace::Rgb
    ) {
        return JpegBackendEligibility::rejected(
            "JPEG Metal fast path requires grayscale, YCbCr, or RGB input color",
        );
    }
    if device.matches_fast_420 || device.matches_fast_422 || device.matches_fast_444 {
        JpegBackendEligibility::eligible()
    } else {
        JpegBackendEligibility::rejected(
            "JPEG Metal fast path requires a fast 4:2:0, 4:2:2, or 4:4:4 packet shape",
        )
    }
}

impl JpegCapabilityReport {
    /// Eligibility for explicit reusable RGB8 Metal batch outputs.
    ///
    /// This is narrower than [`Self::metal_fast`]: it describes the current
    /// caller-owned Metal buffer/texture batch APIs, not every Metal-capable
    /// surface decode shape.
    #[must_use]
    pub fn metal_resident_rgb8_batch_output(&self) -> JpegBackendEligibility {
        metal_resident_rgb8_batch_output_eligibility(self.device, self.request)
    }
}

fn metal_resident_rgb8_batch_output_eligibility(
    device: DeviceBatchSummary,
    request: JpegCapabilityRequest,
) -> JpegBackendEligibility {
    if request.fmt != PixelFormat::Rgb8 {
        return JpegBackendEligibility::rejected(
            "JPEG Metal reusable resident batch output currently supports RGB8 output only",
        );
    }
    if !(device.matches_fast_420 || device.matches_fast_422 || device.matches_fast_444) {
        return JpegBackendEligibility::rejected(
            "JPEG Metal reusable resident batch output requires a fast 4:2:0, 4:2:2, or 4:4:4 packet shape",
        );
    }

    match request.op {
        JpegDecodeOp::Full => JpegBackendEligibility::eligible(),
        JpegDecodeOp::Scaled(scale) | JpegDecodeOp::RegionScaled { scale, .. }
            if supports_metal_resident_batch_scale(scale) =>
        {
            JpegBackendEligibility::eligible()
        }
        JpegDecodeOp::Scaled(_) | JpegDecodeOp::RegionScaled { .. } => {
            JpegBackendEligibility::rejected(
                "JPEG Metal reusable resident batch output currently supports half, quarter, or eighth scaling",
            )
        }
        JpegDecodeOp::Region(_) => JpegBackendEligibility::rejected(
            "JPEG Metal reusable resident batch output currently supports full, scaled, or region-scaled decode shapes",
        ),
    }
}

fn supports_metal_resident_batch_scale(scale: Downscale) -> bool {
    matches!(
        scale,
        Downscale::Half | Downscale::Quarter | Downscale::Eighth
    )
}
