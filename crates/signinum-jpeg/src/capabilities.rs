// SPDX-License-Identifier: Apache-2.0

//! Public JPEG capability introspection for backend routing.

use crate::adapter::{summarize_device_batch, DeviceBatchSummary};
use crate::decoder::Decoder;
use crate::error::JpegError;
use crate::info::{ColorSpace, Info, Rect, SofKind};
use signinum_core::{Downscale, PixelFormat};

/// JPEG decode operation shape for capability routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegDecodeOp {
    /// Full-image/tile decode.
    Full,
    /// Source-coordinate region decode.
    Region(Rect),
    /// Full-image/tile decode at reduced resolution.
    Scaled(Downscale),
    /// Source-coordinate region decode at reduced resolution.
    RegionScaled {
        /// Source-coordinate region.
        roi: Rect,
        /// Reduced-resolution factor.
        scale: Downscale,
    },
}

impl JpegDecodeOp {
    fn scale(self) -> Downscale {
        match self {
            Self::Full | Self::Region(_) => Downscale::None,
            Self::Scaled(scale) | Self::RegionScaled { scale, .. } => scale,
        }
    }
}

/// Capability request for a JPEG decode route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegCapabilityRequest {
    /// Decode operation shape.
    pub op: JpegDecodeOp,
    /// Requested output pixel format.
    pub fmt: PixelFormat,
}

/// Backend eligibility result with a stable rejection reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegBackendEligibility {
    /// Whether this backend can handle the requested decode shape.
    pub eligible: bool,
    /// Static rejection reason when `eligible` is false.
    pub reason: Option<&'static str>,
}

impl JpegBackendEligibility {
    const fn eligible() -> Self {
        Self {
            eligible: true,
            reason: None,
        }
    }

    const fn rejected(reason: &'static str) -> Self {
        Self {
            eligible: false,
            reason: Some(reason),
        }
    }
}

/// Parsed JPEG metadata and backend eligibility for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegCapabilityReport {
    /// Original capability request.
    pub request: JpegCapabilityRequest,
    /// Public JPEG metadata.
    pub info: Info,
    /// Device batch summary derived from Signinum's parser/planner.
    pub device: DeviceBatchSummary,
    /// Portable CPU decode eligibility.
    pub cpu: JpegBackendEligibility,
    /// Signinum-owned CUDA-kernel eligibility.
    pub owned_cuda: JpegBackendEligibility,
    /// Metal fast-packet shape eligibility.
    pub metal_fast: JpegBackendEligibility,
}

impl JpegCapabilityReport {
    /// Inspect JPEG bytes and report decode-route eligibility.
    ///
    /// # Errors
    /// Returns [`JpegError`] when JPEG header parsing fails.
    pub fn inspect(input: &[u8], request: JpegCapabilityRequest) -> Result<Self, JpegError> {
        let decoder = Decoder::new(input)?;
        Ok(Self::for_decoder(&decoder, request))
    }

    /// Build a capability report from an already parsed decoder.
    #[must_use]
    pub fn for_decoder(decoder: &Decoder<'_>, request: JpegCapabilityRequest) -> Self {
        let info = decoder.info().clone();
        let device = summarize_device_batch(decoder, 4);
        Self {
            request,
            info: info.clone(),
            device,
            cpu: cpu_eligibility(&info, request),
            owned_cuda: owned_cuda_eligibility(&info, device, request),
            metal_fast: metal_fast_eligibility(device, request),
        }
    }

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

fn cpu_eligibility(info: &Info, request: JpegCapabilityRequest) -> JpegBackendEligibility {
    let _ = info;
    match (request.fmt, request.op.scale()) {
        (PixelFormat::Rgb8 | PixelFormat::Gray8, _) => JpegBackendEligibility::eligible(),
        (PixelFormat::Rgba8, Downscale::None) => JpegBackendEligibility::eligible(),
        (PixelFormat::Rgba8, _) => JpegBackendEligibility::rejected(
            "JPEG CPU decode supports RGBA8 only for unscaled output",
        ),
        (PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16, _) => {
            JpegBackendEligibility::rejected("JPEG CPU decode does not support 16-bit output")
        }
        _ => JpegBackendEligibility::rejected("unsupported JPEG CPU output format"),
    }
}

fn owned_cuda_eligibility(
    info: &Info,
    device: DeviceBatchSummary,
    request: JpegCapabilityRequest,
) -> JpegBackendEligibility {
    if request.op != JpegDecodeOp::Full || request.fmt != PixelFormat::Rgb8 {
        return JpegBackendEligibility::rejected(
            "Signinum-owned CUDA JPEG decode currently supports full-tile RGB8 fast 4:2:0, 4:2:2, or 4:4:4 only",
        );
    }
    if !matches!(info.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return JpegBackendEligibility::rejected(
            "Signinum-owned CUDA JPEG decode supports baseline/extended 8-bit sequential JPEG only",
        );
    }
    if info.color_space != ColorSpace::YCbCr
        || !(device.matches_fast_420 || device.matches_fast_422 || device.matches_fast_444)
    {
        return JpegBackendEligibility::rejected(
            "Signinum-owned CUDA JPEG decode currently requires a YCbCr 4:2:0, 4:2:2, or 4:4:4 fast packet shape",
        );
    }
    JpegBackendEligibility::eligible()
}

fn metal_fast_eligibility(
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
    if device.matches_fast_420 || device.matches_fast_422 || device.matches_fast_444 {
        JpegBackendEligibility::eligible()
    } else {
        JpegBackendEligibility::rejected(
            "JPEG Metal fast path requires a fast 4:2:0, 4:2:2, or 4:4:4 packet shape",
        )
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
