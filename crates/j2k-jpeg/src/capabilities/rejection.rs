// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend rejection contracts and planner-rejection classification.

use crate::{DeviceBatchSummary, Info, JpegError, SofKind};

/// Backend eligibility result with a stable rejection reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegBackendEligibility {
    /// Whether this backend can handle the requested decode shape.
    pub eligible: bool,
    /// Static rejection reason when `eligible` is false.
    pub reason: Option<&'static str>,
}

impl JpegBackendEligibility {
    pub(super) const fn eligible() -> Self {
        Self {
            eligible: true,
            reason: None,
        }
    }

    pub(super) const fn rejected(reason: &'static str) -> Self {
        Self {
            eligible: false,
            reason: Some(reason),
        }
    }
}

pub(super) fn can_report_from_parsed_info(
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

pub(super) fn unavailable_device_summary(info: &Info) -> DeviceBatchSummary {
    DeviceBatchSummary {
        restart_interval: info.restart_interval,
        checkpoint_count: 0,
        matches_fast_420: false,
        matches_fast_422: false,
        matches_fast_444: false,
    }
}
