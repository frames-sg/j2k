// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{
    BlockCodingMode, EncodeOptions, J2kEncodeStageAccelerator, J2kResidentEncodeInput,
    J2kResidentHtj2kTileEncodeJob, ResidentHtj2kEncodeError,
};
use super::super::resident::encode_resident_impl;

#[derive(Default)]
struct NeverCalledAccelerator {
    calls: usize,
}

impl J2kEncodeStageAccelerator for NeverCalledAccelerator {
    fn encode_resident_htj2k_tile(
        &mut self,
        _job: J2kResidentHtj2kTileEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<alloc::vec::Vec<u8>>> {
        self.calls += 1;
        Ok(None)
    }
}

#[test]
fn resident_invalid_options_do_not_masquerade_as_resource_failures() {
    let options = EncodeOptions {
        num_layers: 0,
        use_ht_block_coding: true,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    };
    let input = J2kResidentEncodeInput::new(8, 8, 1, 8, false).expect("resident input");
    let mut accelerator = NeverCalledAccelerator::default();

    let error = encode_resident_impl(
        input,
        &options,
        BlockCodingMode::HighThroughput,
        &mut accelerator,
    )
    .expect_err("zero layers must be rejected as caller input");

    assert_eq!(
        error,
        ResidentHtj2kEncodeError::InvalidInput("quality layer count must be non-zero")
    );
    assert_eq!(accelerator.calls, 0);
}
