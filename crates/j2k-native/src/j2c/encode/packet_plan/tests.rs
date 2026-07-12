// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::NativeEncodeRetainedInput;
use super::*;

mod accelerator_ownership;

#[test]
fn packet_descriptor_layer_count_errors_remain_invalid_input() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("packet session");
    let error = packet_descriptors_for_order_for_session(
        &[],
        0,
        2,
        EncodeProgressionOrder::Lrcp,
        &session,
        0,
    )
    .expect_err("multiple packet contribution layers must be rejected")
    .into_encode_error();

    assert!(matches!(error, crate::EncodeError::InvalidInput { .. }));
}
