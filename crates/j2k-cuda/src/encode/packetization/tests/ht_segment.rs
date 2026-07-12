// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::packet_math::HtSegmentLengthError;

use super::super::state::cuda_ht_segment_length_error;
use super::super::types::CudaHtj2kPacketizationPlanError;

#[test]
fn ht_segment_validation_errors_keep_invalid_and_overflow_categories() {
    for error in [
        HtSegmentLengthError::ContributionLengthExceedsU32 {
            data_len: usize::MAX,
        },
        HtSegmentLengthError::MultiPassLengthOverflow {
            cleanup_length: u32::MAX,
            refinement_length: 1,
        },
    ] {
        assert!(matches!(
            cuda_ht_segment_length_error(error),
            CudaHtj2kPacketizationPlanError::ArithmeticOverflow(reason)
                if reason == error.reason()
        ));
    }

    for error in [
        HtSegmentLengthError::EmptyContributionHasSegments,
        HtSegmentLengthError::RefinementOnlyLengthMismatch {
            data_len: 2,
            refinement_length: 1,
        },
        HtSegmentLengthError::RefinementLengthOutOfRange {
            refinement_length: u32::MAX,
        },
        HtSegmentLengthError::SinglePassHasRefinement {
            refinement_length: 1,
        },
        HtSegmentLengthError::SinglePassLengthMismatch {
            data_len: 2,
            cleanup_length: 1,
        },
        HtSegmentLengthError::MultiPassRequiresSegments {
            cleanup_length: 0,
            refinement_length: 0,
        },
        HtSegmentLengthError::MultiPassLengthMismatch {
            data_len: 3,
            cleanup_length: 1,
            refinement_length: 1,
        },
        HtSegmentLengthError::CleanupLengthOutOfRange {
            cleanup_length: u32::MAX,
        },
    ] {
        assert!(matches!(
            cuda_ht_segment_length_error(error),
            CudaHtj2kPacketizationPlanError::Invalid(reason) if reason == error.reason()
        ));
    }
}
