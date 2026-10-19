// SPDX-License-Identifier: MIT OR Apache-2.0

//! Correctness eligibility for Metal decode surfaces and resident batches.

use super::{
    cpu::cpu_eligibility, JpegBackendEligibility, JpegCapabilityReport, JpegCapabilityRequest,
    JpegDecodeOp,
};
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
    if !matches!(info.sof_kind, SofKind::Baseline8 | SofKind::Extended8) || info.scan_count != 1 {
        return JpegBackendEligibility::rejected(
            "JPEG Metal fast path currently supports single-scan baseline/extended 8-bit sequential JPEG only",
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

/// Correctness eligibility for hybrid Metal decode.
///
/// Hybrid decode runs the serial entropy stage on the CPU and the IDCT (or
/// lossless predictor reconstruction), upsampling, color conversion, and
/// output packing on Metal. It accepts exactly the requests portable CPU
/// decode accepts and produces identical bytes. Output-format rejections are worded for Metal callers;
/// other rejections carry the CPU decoder's reason.
#[must_use]
pub fn metal_hybrid_eligibility(
    info: &Info,
    request: JpegCapabilityRequest,
) -> JpegBackendEligibility {
    let cpu = cpu_eligibility(info, request);
    if !cpu.eligible {
        return metal_rejection_for(info, request.fmt, cpu);
    }
    JpegBackendEligibility::eligible()
}

/// Word the common output-format rejections for Metal callers; every other
/// rejection keeps the CPU decoder's reason.
fn metal_rejection_for(
    info: &Info,
    fmt: PixelFormat,
    cpu: JpegBackendEligibility,
) -> JpegBackendEligibility {
    let sixteen_bit = matches!(
        fmt,
        PixelFormat::Gray16 | PixelFormat::Rgb16 | PixelFormat::Rgba16
    );
    match info.sof_kind {
        SofKind::Baseline8 | SofKind::Extended8 | SofKind::Progressive8 if sixteen_bit => {
            JpegBackendEligibility::rejected(
                "JPEG Metal decode supports Gray8, Rgb8, or Rgba8 output for 8-bit JPEG, matching CPU decode",
            )
        }
        SofKind::Extended12 | SofKind::Progressive12
            if !sixteen_bit || (fmt == PixelFormat::Gray16 && info.color_space != ColorSpace::Grayscale) =>
        {
            JpegBackendEligibility::rejected(
                "JPEG Metal decode supports Rgb16 or Rgba16 output (and Gray16 for grayscale) for 12-bit JPEG, matching CPU decode",
            )
        }
        _ => cpu,
    }
}

impl JpegCapabilityReport {
    /// Eligibility for hybrid Metal decode; see [`metal_hybrid_eligibility`].
    ///
    /// Explicit Metal requests use the fast-packet kernels when
    /// [`Self::metal_fast`] is eligible and hybrid decode otherwise.
    #[must_use]
    pub fn metal_hybrid(&self) -> JpegBackendEligibility {
        metal_hybrid_eligibility(&self.info, self.request)
    }

    /// Eligibility for explicit reusable RGB8 Metal batch outputs.
    ///
    /// Batches of fast-packet tiles that share a sampling family run on the
    /// batched fast-packet kernels; every other RGB8-decodable input is
    /// hybrid-decoded per tile into the same caller-owned output.
    #[must_use]
    pub fn metal_resident_rgb8_batch_output(&self) -> JpegBackendEligibility {
        metal_resident_rgb8_batch_output_eligibility(&self.info, self.request)
    }
}

fn metal_resident_rgb8_batch_output_eligibility(
    info: &Info,
    request: JpegCapabilityRequest,
) -> JpegBackendEligibility {
    if request.fmt != PixelFormat::Rgb8 {
        return JpegBackendEligibility::rejected(
            "JPEG Metal reusable resident batch output currently supports RGB8 output only",
        );
    }
    let hybrid = metal_hybrid_eligibility(info, request);
    if !hybrid.eligible {
        return hybrid;
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
