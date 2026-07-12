// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared HTJ2K 9/7 validator-to-CUDA classification policy.

use super::super::{assert_pattern_checks, PatternCheck};
use super::CudaTranscodeSources;

pub(super) fn assert_policy(sources: &CudaTranscodeSources) {
    assert_pattern_checks(&[
        PatternCheck::new("typed CUDA HTJ2K option mapping", &sources.combined())
            .required(&[
                "fn unsupported_htj2k97_codeblock_options(",
                "Htj2k97CodeBlockOptionsError::NumericOptionsOutOfRange",
                "Htj2k97CodeBlockOptionsError::QuantizationOptionsOutOfRange",
                "Htj2k97CodeBlockOptionsError::DimensionExponentUnsupported",
                "Htj2k97CodeBlockOptionsError::DimensionsExceedLimits",
                "CudaTranscodeError::UnsupportedJob",
            ])
            .forbidden(&[
                "unsupported_htj2k97_codeblock_options(error.to_string())",
                "unsupported_htj2k97_codeblock_options(format!",
            ]),
        PatternCheck::new(
            "CUDA HTJ2K option mapping regression",
            &sources.full_combined(),
        )
        .required(&["every_known_htj2k97_option_error_maps_to_stable_unsupported_job"]),
    ]);
}
