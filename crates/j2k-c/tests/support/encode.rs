// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lossless test-image encoding shared by the integration tests.

use j2k::{encode_j2k_lossless, J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLosslessSamples};

pub(crate) fn encode(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u16,
    bits: u8,
    signed: bool,
    ht: bool,
) -> Vec<u8> {
    let input = J2kLosslessSamples::new(data, width, height, channels, bits, signed).unwrap();
    let options = J2kLosslessEncodeOptions::default()
        .with_cpu_only_backend()
        .with_max_decomposition_levels(Some(3))
        .with_block_coding_mode(if ht {
            J2kBlockCodingMode::HighThroughput
        } else {
            J2kBlockCodingMode::Classic
        });
    encode_j2k_lossless(input, &options).unwrap().codestream
}
