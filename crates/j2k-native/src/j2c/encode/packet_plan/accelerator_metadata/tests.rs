// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::BlockCodingMode;
use super::*;
use crate::j2c::packet_encode::{CodeBlockPacketData, SubbandPrecinct};
use crate::{EncodeError, NativeEncodeRetainedInput};

fn exact_vec<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .expect("small packet metadata test allocation");
    values
}

fn packet_fixture() -> Vec<ResolutionPacket> {
    let mut data = exact_vec(3);
    data.extend([2, 7]);
    let mut code_blocks = exact_vec(1);
    code_blocks.push(CodeBlockPacketData {
        data,
        ht_cleanup_length: 2,
        ht_refinement_length: 0,
        num_coding_passes: 1,
        classic_segment_lengths: Vec::new(),
        num_zero_bitplanes: 0,
        previously_included: false,
        l_block: 3,
        block_coding_mode: BlockCodingMode::HighThroughput,
    });
    let mut subbands = exact_vec(1);
    subbands.push(SubbandPrecinct {
        code_blocks,
        num_cbs_x: 1,
        num_cbs_y: 1,
    });
    let mut packets = exact_vec(1);
    packets.push(ResolutionPacket { subbands });
    packets
}

#[test]
fn packet_accelerator_metadata_accepts_exact_peak_and_rejects_cap_minus_one() {
    let packets = packet_fixture();
    let owned_packet_bytes =
        crate::j2c::packet_encode::owned_packet_retained_bytes_for_public_descriptors(
            &packets,
            packets.capacity(),
            0,
            0,
        )
        .expect("packet ownership");
    let measurement_session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("measurement session");
    let measured =
        try_public_packetization_resolutions(&packets, &measurement_session, owned_packet_bytes)
            .expect("measure packet accelerator metadata");
    let exact_peak = crate::j2c::packet_encode::packet_metadata_retained_bytes(
        &measured,
        measured.capacity(),
        owned_packet_bytes,
    )
    .expect("metadata peak");
    drop(measured);

    let exact_session =
        NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_peak)
            .expect("exact session");
    let exact = try_public_packetization_resolutions(&packets, &exact_session, owned_packet_bytes)
        .expect("exact packet metadata peak is accepted");
    assert_eq!(exact.len(), 1);
    assert_eq!(exact[0].subbands.len(), 1);
    assert_eq!(exact[0].subbands[0].code_blocks.len(), 1);
    drop(exact);

    let cap = exact_peak - 1;
    let under_session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("cap-minus-one session");
    let error = try_public_packetization_resolutions(&packets, &under_session, owned_packet_bytes)
        .expect_err("cap-minus-one packet metadata must fail")
        .into_encode_error();
    assert!(matches!(
        error,
        EncodeError::AllocationTooLarge {
            requested,
            cap: observed,
            ..
        } if requested == exact_peak && observed == cap
    ));
}
