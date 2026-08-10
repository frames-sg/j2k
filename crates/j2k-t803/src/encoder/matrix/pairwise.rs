// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{validation, EncoderMatrixError};
use crate::encoder::{
    EncoderBlockCoding, EncoderCase, EncoderMode, EncoderPairwiseScope, EncoderPayload,
    EncoderProgression,
};

pub(super) fn validate_pairwise_coverage(cases: &[EncoderCase]) -> Result<(), EncoderMatrixError> {
    for &(scope, block_codings, payloads) in &[
        (
            EncoderPairwiseScope::Part1,
            &[EncoderBlockCoding::Classic][..],
            &[EncoderPayload::Codestream][..],
        ),
        (
            EncoderPairwiseScope::Part15,
            &[EncoderBlockCoding::HighThroughput][..],
            &[EncoderPayload::Codestream, EncoderPayload::Jph][..],
        ),
    ] {
        validate_scope_coverage(scope, block_codings, payloads, cases)?;
    }
    Ok(())
}

fn validate_scope_coverage(
    scope: EncoderPairwiseScope,
    block_codings: &[EncoderBlockCoding],
    payloads: &[EncoderPayload],
    cases: &[EncoderCase],
) -> Result<(), EncoderMatrixError> {
    let axes = [
        block_codings.iter().map(debug_value).collect::<Vec<_>>(),
        payloads.iter().map(debug_value).collect(),
        [EncoderMode::Lossless, EncoderMode::Lossy]
            .iter()
            .map(debug_value)
            .collect(),
        [[32, 32], [63, 47]]
            .iter()
            .map(|[width, height]| format!("{width}x{height}"))
            .collect(),
        [false, true].iter().map(debug_value).collect(),
        [8_u8, 12].iter().map(debug_value).collect(),
        [1_u16, 3].iter().map(debug_value).collect(),
        [
            EncoderProgression::Lrcp,
            EncoderProgression::Rlcp,
            EncoderProgression::Rpcl,
            EncoderProgression::Pcrl,
            EncoderProgression::Cprl,
        ]
        .iter()
        .map(debug_value)
        .collect(),
    ];
    let rows = cases
        .iter()
        .filter(|case| case.pairwise_scope == Some(scope))
        .map(|case| {
            [
                debug_value(&case.block_coding),
                debug_value(&case.payload),
                debug_value(&case.mode),
                format!("{}x{}", case.width, case.height),
                debug_value(&case.signed),
                debug_value(&case.bit_depth),
                debug_value(&case.components),
                debug_value(&case.progression),
            ]
        })
        .collect::<Vec<_>>();
    for left in 0..axes.len() {
        for right in (left + 1)..axes.len() {
            for left_value in &axes[left] {
                for right_value in &axes[right] {
                    if !rows
                        .iter()
                        .any(|row| row[left] == *left_value && row[right] == *right_value)
                    {
                        return validation(format!(
                            "{scope:?} pairwise rows do not cover axes {left}/{right} values {left_value}/{right_value}"
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn debug_value(value: &impl std::fmt::Debug) -> String {
    format!("{value:?}")
}
