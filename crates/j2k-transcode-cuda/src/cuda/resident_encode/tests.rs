// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_transcode::{Htj2k97CodeBlockAxis, Htj2k97CodeBlockOptionsError};

use super::unsupported_htj2k97_codeblock_options;
use crate::CudaTranscodeError;

#[test]
fn every_known_htj2k97_option_error_maps_to_stable_unsupported_job() {
    for error in [
        Htj2k97CodeBlockOptionsError::NumericOptionsOutOfRange,
        Htj2k97CodeBlockOptionsError::QuantizationOptionsOutOfRange,
        Htj2k97CodeBlockOptionsError::DimensionExponentUnsupported {
            axis: Htj2k97CodeBlockAxis::Width,
            exponent_minus_two: u8::MAX,
        },
        Htj2k97CodeBlockOptionsError::DimensionsExceedLimits {
            width: usize::MAX,
            height: usize::MAX,
        },
    ] {
        assert!(matches!(
            unsupported_htj2k97_codeblock_options(error),
            CudaTranscodeError::UnsupportedJob(reason) if reason == error.reason()
        ));
    }
}
