// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public JPEG capability introspection for backend routing.

mod cpu;
mod cuda;
mod metal;
mod output_geometry;
mod rejection;
mod request;
mod resolve;

pub use rejection::JpegBackendEligibility;
pub use request::{JpegCapabilityRequest, JpegDecodeOp, JpegDecodeRequest};
pub use resolve::{JpegResolvedDecode, JpegResolvedDecodePath};

use self::{
    cpu::cpu_eligibility,
    cuda::owned_cuda_eligibility,
    metal::metal_fast_eligibility,
    rejection::{can_report_from_parsed_info, unavailable_device_summary},
};
use crate::{
    adapter::summarize_device_batch, Decoder, DeviceBatchSummary, Info, JpegError, JpegView,
};

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

    #[expect(
        clippy::needless_pass_by_value,
        reason = "the small Copy capability request is stored by value in the resulting report"
    )]
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
}
