// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal acceleration for coefficient-domain JPEG to HTJ2K transcode stages.
//!
//! The supported targets are direct DCT-grid to one-level 5/3 and 9/7 wavelet
//! projections used by `j2k-transcode`'s HTJ2K paths. CPU scalar code
//! remains the oracle and fallback.
//!
//! Auto routing is intentionally batch-first for the expensive Metal transcode
//! paths: the default single-job reversible 5/3 and 9/7 thresholds are
//! `usize::MAX`, so single-tile requests stay on the CPU unless callers opt in
//! with `with_auto_reversible_min_samples` or `with_auto_dwt97_min_samples`.

#[cfg(target_os = "macos")]
mod metal;

#[doc(hidden)]
pub mod weights;

mod accelerator;
mod error;
mod route;

pub use accelerator::MetalDctToWaveletStageAccelerator;
pub use error::{MetalRuntimeFailure, MetalTranscodeError};
#[cfg(target_os = "macos")]
pub use route::resident_codestream_buffer_from_metal_encoded_j2k;
pub use route::{
    jpeg_to_htj2k_batch_with_metal_route, jpeg_to_htj2k_with_metal_route, MetalEncodedTranscode,
    MetalEncodedTranscodeBatch, MetalTranscodeFallbackReason, MetalTranscodeRouteReport,
};

#[cfg(target_os = "macos")]
pub use metal::MetalTranscodeSession;

/// Stable message returned when Metal is unavailable.
pub const METAL_UNAVAILABLE: &str = "Metal is unavailable on this host";

#[cfg(not(target_os = "macos"))]
#[derive(Clone, Copy, Debug, Default)]
/// Placeholder Metal transcode session for hosts without Metal support.
pub struct MetalTranscodeSession {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
impl MetalTranscodeSession {
    /// Return `MetalUnavailable` on hosts without Metal support.
    pub const fn system_default() -> Result<Self, MetalTranscodeError> {
        Err(MetalTranscodeError::MetalUnavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::{
        ensure_strict_metal_batch_dispatched, ensure_strict_metal_dispatched, route_report,
    };
    use j2k_core::{BackendKind, BackendRequest};
    use j2k_transcode::{BatchTranscodeReport, JpegToHtj2kCoefficientPath, TranscodeTimingReport};

    #[derive(Debug)]
    struct TestRuntimeError;

    impl core::fmt::Display for TestRuntimeError {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            formatter.write_str("driver rejected execution")
        }
    }

    impl std::error::Error for TestRuntimeError {}

    #[cfg(target_os = "macos")]
    #[test]
    fn runtime_failure_retains_operation_detail_and_source() {
        let error = MetalTranscodeError::runtime("Metal test command buffer", TestRuntimeError);

        assert_eq!(
            error.to_string(),
            "Metal test command buffer: driver rejected execution"
        );
        let runtime = std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<MetalRuntimeFailure>())
            .expect("runtime failure wrapper");
        let source = std::error::Error::source(runtime).expect("concrete runtime source");
        assert!(source.downcast_ref::<TestRuntimeError>().is_some());
        assert!(!error.is_recoverable());
    }

    #[test]
    fn allocation_failures_are_hard_in_auto_mode() {
        let error = MetalTranscodeError::HostAllocationTooLarge {
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what: "test output",
        };
        assert!(!error.is_recoverable());
        assert!(MetalTranscodeError::UnsupportedJob("test decline").is_recoverable());
    }

    #[test]
    fn route_report_uses_shared_accelerator_work_classifier() {
        let timings = TranscodeTimingReport {
            dwt97_batch_readback_bytes: 128,
            ..TranscodeTimingReport::default()
        };
        let route = route_report(BackendRequest::Auto, &timings);
        assert_eq!(route.selected_transform_backend, BackendKind::Metal);
        assert_eq!(route.fallback_reason, None);
    }

    #[test]
    fn strict_metal_accepts_shared_accelerator_work_evidence() {
        let timings = TranscodeTimingReport {
            dwt97_batch_pack_upload_transfers: 1,
            ..TranscodeTimingReport::default()
        };
        let batch_report = BatchTranscodeReport {
            tile_count: 1,
            successful_tiles: 1,
            failed_tiles: 0,
            transformed_components: 1,
            reversible_dwt53_batches: 0,
            reversible_dwt53_batch_jobs: 0,
            extract_us: 0,
            transform_us: 0,
            encode_us: 0,
            timings,
            coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear53,
        };

        ensure_strict_metal_dispatched(&timings).expect("shared classifier marks Metal work");
        ensure_strict_metal_batch_dispatched(&batch_report)
            .expect("batch strict route uses shared classifier");
    }
}
