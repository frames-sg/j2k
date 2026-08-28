// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
};
use j2k_core::DeviceSubmission;
use j2k_metal::{
    submit_lossless_batch_to_metal, MetalBackendSession, MetalEncodeInputStaging,
    MetalLosslessBufferEncodeBatchOutcome, MetalLosslessEncodeBatchRequest,
    MetalLosslessEncodeConfig, MetalLosslessEncodeTile,
};

pub(crate) const DIMENSION: u32 = 512;

pub(crate) fn options(block_coding_mode: J2kBlockCodingMode) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(block_coding_mode)
        .with_max_decomposition_levels(Some(3))
        .with_validation(J2kEncodeValidation::External)
}

pub(crate) fn run_device_batch(
    session: &MetalBackendSession,
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    inflight_tiles: Option<usize>,
) -> MetalLosslessBufferEncodeBatchOutcome {
    let outcome = submit_lossless_batch_to_metal(
        MetalLosslessEncodeBatchRequest {
            tiles,
            staging: MetalEncodeInputStaging::AlreadyPaddedContiguous,
            config: MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: inflight_tiles,
                gpu_encode_memory_budget_bytes: None,
            },
        },
        options,
        session,
    )
    .expect("submit resident packetization benchmark batch")
    .wait()
    .expect("complete resident packetization benchmark batch");
    assert_eq!(outcome.outcomes.len(), tiles.len());
    assert!(
        outcome
            .outcomes
            .iter()
            .all(|item| item.resident.packetization_used),
        "benchmark must exercise resident packetization"
    );
    outcome
}
