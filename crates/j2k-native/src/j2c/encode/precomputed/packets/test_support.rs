// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{NativeEncodeRetainedInput, NativeEncodeSession};
use super::{
    move_preencoded_payloads_into_skeleton, prepared_subband_metadata,
    try_preencoded_owned_skeleton, ConstructionTracker, NativeEncodePipelineError,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PreencodedHtj2k97Subband,
    PreparedCodeBlockCoefficients, PreparedEncodeCodeBlock, PreparedEncodeSubband,
};
use alloc::vec;

pub(in crate::j2c::encode) fn prepared_subband_from_preencoded_owned_for_test(
    subband: PreencodedHtj2k97Subband,
) -> PreparedEncodeSubband {
    let image = PreencodedHtj2k97Image {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        components: vec![PreencodedHtj2k97Component {
            x_rsiz: 1,
            y_rsiz: 1,
            resolutions: vec![crate::PreencodedHtj2k97Resolution {
                subbands: vec![subband],
            }],
        }],
    };
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("test construction session");
    let mut tracker = ConstructionTracker::new(&session, 0);
    let mut prepared =
        try_preencoded_owned_skeleton(&image, &mut tracker).expect("test skeleton construction");
    move_preencoded_payloads_into_skeleton(image, &mut prepared).expect("test payload move");
    prepared
        .pop()
        .and_then(|mut packets| packets.pop())
        .and_then(|mut packet| packet.subbands.pop())
        .expect("test prepared subband")
}

#[test]
fn prepared_subband_dimension_overflow_is_arithmetic() {
    let code_blocks = vec![
        PreparedEncodeCodeBlock {
            coefficients: PreparedCodeBlockCoefficients::Empty,
            width: u32::MAX,
            height: 1,
        },
        PreparedEncodeCodeBlock {
            coefficients: PreparedCodeBlockCoefficients::Empty,
            width: 1,
            height: 1,
        },
    ];

    let error =
        prepared_subband_metadata(2, 1, 1, crate::J2kSubBandType::LowLow, code_blocks, None)
            .err()
            .expect("overflowing subband dimensions must fail");

    assert!(matches!(
        error,
        NativeEncodePipelineError::ArithmeticOverflow("precomputed subband width overflow")
    ));
}
