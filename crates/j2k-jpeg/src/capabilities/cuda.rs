// SPDX-License-Identifier: MIT OR Apache-2.0

//! Correctness eligibility for the J2K-owned CUDA JPEG path.

use super::{JpegBackendEligibility, JpegCapabilityRequest, JpegDecodeOp};
use crate::{ColorSpace, DeviceBatchSummary, Info, SofKind};
use j2k_core::PixelFormat;

pub(super) fn owned_cuda_eligibility(
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
    if !owned_cuda_rgb8_output_is_addressable(info.dimensions) {
        return JpegBackendEligibility::rejected(
            "J2K-owned CUDA JPEG decode requires RGB8 output addressable by u32 byte offsets",
        );
    }
    JpegBackendEligibility::eligible()
}

fn owned_cuda_rgb8_output_is_addressable((width, height): (u32, u32)) -> bool {
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(3))
        .is_some_and(|bytes| bytes <= u64::from(u32::MAX) + 1)
}
