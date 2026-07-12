// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::*;
use crate::j2c::encode::{
    NativeEncodePipelineError, NativeEncodeRetainedInput, NativeEncodeSession,
};
use crate::EncodeError;

fn exact_vec<T: Copy>(values: &[T]) -> Vec<T> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .expect("small consuming tile-part fixture");
    output.extend_from_slice(values);
    output
}

fn packetized_fixture() -> PacketizedTileData {
    PacketizedTileData {
        data: exact_vec(&[1, 2, 3, 4]),
        packet_lengths: exact_vec(&[2, 2]),
        packet_headers: vec![exact_vec(&[0xA1]), exact_vec(&[0xB1, 0xB2])],
    }
}

fn consume_with_cap(
    cap: usize,
    packet_limit: Option<u16>,
) -> NativeEncodePipelineResult<Vec<EncodedTilePart>> {
    let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("valid consuming tile-part cap");
    consume_packetized_tile_into_tile_parts(3, packetized_fixture(), packet_limit, 5, &session)
}

#[test]
fn unsplit_transition_moves_packetized_owners_without_payload_copy() {
    let packetized = packetized_fixture();
    let data_ptr = packetized.data.as_ptr();
    let header_ptr = packetized.packet_headers[0].as_ptr();
    let source_bytes = packet_encode::packetized_tile_retained_bytes(&packetized)
        .expect("packetized source bytes");
    let session = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
    .expect("discovery session");
    let parts =
        consume_packetized_tile_into_tile_parts(3, packetized, None, 5, &session).expect("move");
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].data.as_ptr(), data_ptr);
    assert_eq!(parts[0].packet_headers[0].as_ptr(), header_ptr);
    let retained =
        encoded_tile_parts_retained_bytes(&parts, parts.capacity()).expect("retained part bytes");
    assert_eq!(
        retained,
        source_bytes + parts.capacity() * core::mem::size_of::<EncodedTilePart>()
    );

    let exact_cap = 5 + retained;
    consume_with_cap(exact_cap, None).expect("exact move-only handoff peak");
    let error = consume_with_cap(exact_cap - 1, None)
        .err()
        .expect("one byte below the move-only handoff peak must fail");
    assert!(matches!(
        error,
        NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge {
            requested,
            cap,
            ..
        }) if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn split_transition_accepts_exact_overlap_peak_and_rejects_one_byte_less() {
    let source = packetized_fixture();
    let source_bytes =
        packet_encode::packetized_tile_retained_bytes(&source).expect("packetized source bytes");
    let discovered =
        consume_with_cap(crate::DEFAULT_MAX_CODEC_BYTES, Some(1)).expect("discover split capacity");
    assert_eq!(discovered.len(), 2);
    assert_eq!(discovered[0].data, [1, 2]);
    assert_eq!(discovered[1].data, [3, 4]);
    assert_eq!(discovered[0].packet_headers, [vec![0xA1]]);
    assert_eq!(discovered[1].packet_headers, [vec![0xB1, 0xB2]]);
    let exact_cap = 5
        + source_bytes
        + encoded_tile_parts_retained_bytes(&discovered, discovered.capacity())
            .expect("split part bytes");
    consume_with_cap(exact_cap, Some(1)).expect("exact split overlap peak");
    let error = consume_with_cap(exact_cap - 1, Some(1))
        .err()
        .expect("one byte below split overlap peak must fail");
    assert!(matches!(
        error,
        NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge {
            requested,
            cap,
            ..
        }) if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn malformed_packet_metadata_is_a_typed_internal_invariant() {
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("metadata validation session");

    let mut mismatched_length = packetized_fixture();
    mismatched_length.packet_lengths[1] = 3;
    let length_error =
        consume_packetized_tile_into_tile_parts(3, mismatched_length, None, 0, &session)
            .err()
            .expect("mismatched packet length must fail before ownership transfer");
    assert!(matches!(
        length_error,
        NativeEncodePipelineError::Typed(EncodeError::InternalInvariant {
            what: "packet lengths do not match tile data length"
        })
    ));

    let mut mismatched_headers = packetized_fixture();
    mismatched_headers.packet_headers.pop();
    let header_error =
        consume_packetized_tile_into_tile_parts(3, mismatched_headers, None, 0, &session)
            .err()
            .expect("mismatched packet header count must fail before ownership transfer");
    assert!(matches!(
        header_error,
        NativeEncodePipelineError::Typed(EncodeError::InternalInvariant {
            what: "packet header count does not match packet length count"
        })
    ));

    let missing_lengths = PacketizedTileData {
        data: exact_vec(&[1, 2]),
        packet_lengths: Vec::new(),
        packet_headers: Vec::new(),
    };
    let split_error =
        consume_packetized_tile_into_tile_parts(3, missing_lengths, Some(1), 0, &session)
            .err()
            .expect("split request without packet lengths must fail");
    assert!(matches!(
        split_error,
        NativeEncodePipelineError::Typed(EncodeError::InternalInvariant {
            what: "tile-part splitting requires packet-length metadata"
        })
    ));
}
