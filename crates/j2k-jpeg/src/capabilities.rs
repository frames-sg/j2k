// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public JPEG capability introspection for backend routing.

use crate::adapter::summarize_device_batch;
use crate::decoder::{Decoder, JpegView};
use crate::error::JpegError;
use crate::info::{ColorSpace, Info, Rect, SofKind};
use crate::DeviceBatchSummary;
use j2k_core::{BackendRequest, Downscale, PixelFormat};

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

/// Complete JPEG decode request used by backend path resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegDecodeRequest {
    /// Requested backend policy.
    pub backend: BackendRequest,
    /// Requested output pixel format.
    pub fmt: PixelFormat,
    /// Decode operation shape.
    pub op: JpegDecodeOp,
}

impl JpegDecodeRequest {
    /// Return the capability-only portion of the request.
    #[must_use]
    pub const fn capability(self) -> JpegCapabilityRequest {
        JpegCapabilityRequest {
            op: self.op,
            fmt: self.fmt,
        }
    }
}

/// Normalized JPEG decode path selected for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegResolvedDecodePath {
    /// Portable CPU host decode.
    CpuHost,
    /// J2K-owned CUDA RGB8 decode path.
    OwnedCudaRgb8,
    /// J2K Metal fast-packet decode path.
    MetalFast,
    /// Request cannot be satisfied by this path resolver.
    Rejected {
        /// Backend requested by the caller.
        backend: BackendRequest,
        /// Stable rejection reason.
        reason: &'static str,
    },
}

/// Parsed JPEG metadata plus the selected backend path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegResolvedDecode {
    /// Original decode request.
    pub request: JpegDecodeRequest,
    /// Capability report used for the decision.
    pub capabilities: JpegCapabilityReport,
    /// Output rectangle after ROI and scale are applied.
    pub output_rect: Rect,
    /// Selected backend path.
    pub path: JpegResolvedDecodePath,
}

impl JpegResolvedDecode {
    /// Inspect JPEG bytes and resolve the requested backend path.
    pub fn inspect(input: &[u8], request: JpegDecodeRequest) -> Result<Self, JpegError> {
        let capabilities = JpegCapabilityReport::inspect(input, request.capability())?;
        Ok(Self::from_capabilities(capabilities, request))
    }

    /// Resolve a path from an existing capability report.
    #[must_use]
    pub fn from_capabilities(
        capabilities: JpegCapabilityReport,
        request: JpegDecodeRequest,
    ) -> Self {
        let output_rect = output_rect_for_request(&capabilities.info, request.op);
        let path = capabilities.resolve_path(request.backend);
        Self {
            request,
            capabilities,
            output_rect,
            path,
        }
    }
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
    /// Device batch summary derived from J2K's parser/planner.
    pub device: DeviceBatchSummary,
    /// Portable CPU decode eligibility.
    pub cpu: JpegBackendEligibility,
    /// J2K-owned CUDA-kernel eligibility.
    pub owned_cuda: JpegBackendEligibility,
    /// Metal fast-packet shape eligibility.
    pub metal_fast: JpegBackendEligibility,
}

impl JpegCapabilityReport {
    /// Inspect JPEG bytes and report decode-route eligibility.
    ///
    /// # Errors
    /// Returns [`JpegError`] when JPEG header parsing fails or planner
    /// validation finds malformed decode-table state. Parseable JPEG classes
    /// that J2K has not implemented yet still return a report with
    /// rejected backend eligibility.
    pub fn inspect(input: &[u8], request: JpegCapabilityRequest) -> Result<Self, JpegError> {
        let view = JpegView::parse(input)?;
        let info = view.info().clone();
        let has_lossless_subsampled_color_capability_shape =
            view.has_lossless_subsampled_color_capability_shape();
        match Decoder::from_view(view) {
            Ok(decoder) => Ok(Self::for_decoder(&decoder, request)),
            Err(err)
                if can_report_from_parsed_info(
                    &err,
                    has_lossless_subsampled_color_capability_shape,
                ) =>
            {
                Ok(Self::for_planner_rejected_info(info, request, &err))
            }
            Err(err) => Err(err),
        }
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
            metal_fast: metal_fast_eligibility(&info, device, request),
        }
    }

    fn for_parsed_info(info: Info, request: JpegCapabilityRequest) -> Self {
        let device = unavailable_device_summary(&info);
        Self {
            request,
            info: info.clone(),
            device,
            cpu: cpu_eligibility(&info, request),
            owned_cuda: owned_cuda_eligibility(&info, device, request),
            metal_fast: metal_fast_eligibility(&info, device, request),
        }
    }

    fn for_planner_rejected_info(
        info: Info,
        request: JpegCapabilityRequest,
        err: &JpegError,
    ) -> Self {
        let mut report = Self::for_parsed_info(info, request);
        if report.cpu.eligible && matches!(err, JpegError::NotImplemented { .. }) {
            report.cpu = JpegBackendEligibility::rejected(
                "JPEG CPU decode planner rejected this stream shape before decode",
            );
        }
        report
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

    /// Resolve a backend request using this report's eligibility results.
    #[must_use]
    pub fn resolve_path(&self, backend: BackendRequest) -> JpegResolvedDecodePath {
        match backend {
            BackendRequest::Cpu => {
                if self.cpu.eligible {
                    JpegResolvedDecodePath::CpuHost
                } else {
                    JpegResolvedDecodePath::Rejected {
                        backend,
                        reason: self
                            .cpu
                            .reason
                            .unwrap_or("JPEG CPU decode rejected this request"),
                    }
                }
            }
            BackendRequest::Auto => JpegResolvedDecodePath::CpuHost,
            BackendRequest::Cuda => {
                if self.owned_cuda.eligible {
                    JpegResolvedDecodePath::OwnedCudaRgb8
                } else {
                    JpegResolvedDecodePath::Rejected {
                        backend,
                        reason: self
                            .owned_cuda
                            .reason
                            .unwrap_or("J2K-owned CUDA JPEG decode rejected this request"),
                    }
                }
            }
            BackendRequest::Metal => {
                if self.metal_fast.eligible {
                    JpegResolvedDecodePath::MetalFast
                } else {
                    JpegResolvedDecodePath::Rejected {
                        backend,
                        reason: self
                            .metal_fast
                            .reason
                            .unwrap_or("JPEG Metal fast path rejected this request"),
                    }
                }
            }
        }
    }
}

fn output_rect_for_request(info: &Info, op: JpegDecodeOp) -> Rect {
    match op {
        JpegDecodeOp::Full => Rect::full(info.dimensions),
        JpegDecodeOp::Region(roi) => roi,
        JpegDecodeOp::Scaled(scale) => scaled_rect(Rect::full(info.dimensions), scale),
        JpegDecodeOp::RegionScaled { roi, scale } => scaled_rect(roi, scale),
    }
}

fn scaled_rect(rect: Rect, scale: Downscale) -> Rect {
    let denom = scale.denominator();
    let x_end = rect.x.saturating_add(rect.w);
    let y_end = rect.y.saturating_add(rect.h);
    let x0 = rect.x / denom;
    let y0 = rect.y / denom;
    let x1 = x_end.div_ceil(denom);
    let y1 = y_end.div_ceil(denom);
    Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
    }
}

fn cpu_eligibility(info: &Info, request: JpegCapabilityRequest) -> JpegBackendEligibility {
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
            )
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
                )
                    if is_supported_lossless_color_sampling(info) =>
                {
                    JpegBackendEligibility::eligible()
                }
                (ColorSpace::Rgb, 8, PixelFormat::Rgb8 | PixelFormat::Rgba8)
                | (ColorSpace::Rgb, 16, PixelFormat::Rgb16 | PixelFormat::Rgba16) => JpegBackendEligibility::rejected(
                    "JPEG CPU lossless SOF3 APP14 RGB decode currently supports 4:4:4 sampling, even-width 8/16-bit 4:2:2 sampling, or even-dimension 8/16-bit 4:2:0 sampling",
                ),
                (ColorSpace::YCbCr, 8, PixelFormat::Rgb8 | PixelFormat::Rgba8)
                | (ColorSpace::YCbCr, 16, PixelFormat::Rgb16 | PixelFormat::Rgba16) => JpegBackendEligibility::rejected(
                    "JPEG CPU lossless SOF3 YCbCr decode currently supports 4:4:4 sampling, even-width 8/16-bit 4:2:2 sampling, or even-dimension 8/16-bit 4:2:0 sampling",
                ),
                _ => JpegBackendEligibility::rejected(
                    "JPEG CPU lossless SOF3 decode currently supports 8-bit Gray8, 16-bit Gray16, 8-bit YCbCr Rgb8/Rgba8 including even-width 4:2:2 and even-dimension 4:2:0, 16-bit YCbCr Rgb16/Rgba16 including even-width 4:2:2 and even-dimension 4:2:0, 8-bit APP14 RGB Rgb8/Rgba8 including even-width 4:2:2 and even-dimension 4:2:0, or 16-bit APP14 RGB Rgb16/Rgba16 including even-width 4:2:2 and even-dimension 4:2:0 output only",
                ),
            };
        }
        SofKind::Lossless => {
            return JpegBackendEligibility::rejected(
                "JPEG CPU decode does not yet support lossless SOF3 JPEG",
            )
        }
        SofKind::Baseline8 | SofKind::Extended8 | SofKind::Progressive8 => {}
    }

    match (request.fmt, request.op.scale()) {
        (PixelFormat::Rgb8 | PixelFormat::Gray8, _) => JpegBackendEligibility::eligible(),
        (PixelFormat::Rgba8, _) => JpegBackendEligibility::eligible(),
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
            Self::Extended => {
                "JPEG CPU 12-bit extended YCbCr decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only"
            }
            Self::Progressive => {
                "JPEG CPU 12-bit progressive YCbCr decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only"
            }
        }
    }

    const fn rgb_sampling_reason(self) -> &'static str {
        match self {
            Self::Extended => {
                "JPEG CPU 12-bit extended RGB decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only"
            }
            Self::Progressive => {
                "JPEG CPU 12-bit progressive RGB decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only"
            }
        }
    }

    const fn four_component_sampling_reason(self) -> &'static str {
        match self {
            Self::Extended => {
                "JPEG CPU 12-bit extended four-component CMYK/YCCK decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only"
            }
            Self::Progressive => {
                "JPEG CPU 12-bit progressive four-component CMYK/YCCK decode currently supports 4:4:4, 4:2:2, or 4:2:0 sampling only"
            }
        }
    }

    const fn output_reason(self) -> &'static str {
        match self {
            Self::Extended => {
                "JPEG CPU 12-bit extended decode currently supports grayscale Gray16/Rgb16/Rgba16, APP14 RGB 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16, YCbCr 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16, or CMYK/YCCK 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16 only"
            }
            Self::Progressive => {
                "JPEG CPU 12-bit progressive decode currently supports grayscale Gray16/Rgb16/Rgba16, APP14 RGB 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16, YCbCr 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16, or CMYK/YCCK 4:4:4/4:2:2/4:2:0 Rgb16/Rgba16 only"
            }
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

fn owned_cuda_eligibility(
    info: &Info,
    device: DeviceBatchSummary,
    request: JpegCapabilityRequest,
) -> JpegBackendEligibility {
    if request.op != JpegDecodeOp::Full || request.fmt != PixelFormat::Rgb8 {
        return JpegBackendEligibility::rejected(
            "J2K-owned CUDA JPEG decode currently supports full-tile RGB8 fast 4:2:0, 4:2:2, or 4:4:4 only",
        );
    }
    if !matches!(info.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return JpegBackendEligibility::rejected(
            "J2K-owned CUDA JPEG decode supports baseline/extended 8-bit sequential JPEG only",
        );
    }
    if info.color_space != ColorSpace::YCbCr
        || !(device.matches_fast_420 || device.matches_fast_422 || device.matches_fast_444)
    {
        return JpegBackendEligibility::rejected(
            "J2K-owned CUDA JPEG decode currently requires a YCbCr 4:2:0, 4:2:2, or 4:4:4 fast packet shape",
        );
    }
    JpegBackendEligibility::eligible()
}

fn metal_fast_eligibility(
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

fn can_report_from_parsed_info(
    err: &JpegError,
    has_lossless_subsampled_color_capability_shape: bool,
) -> bool {
    match err {
        JpegError::UnsupportedColorSpace { .. } => true,
        JpegError::NotImplemented { sof } if *sof != SofKind::Lossless => true,
        JpegError::NotImplemented {
            sof: SofKind::Lossless,
        } => has_lossless_subsampled_color_capability_shape,
        _ => false,
    }
}

fn unavailable_device_summary(info: &Info) -> DeviceBatchSummary {
    DeviceBatchSummary {
        restart_interval: info.restart_interval,
        checkpoint_count: 0,
        matches_fast_420: false,
        matches_fast_422: false,
        matches_fast_444: false,
    }
}
