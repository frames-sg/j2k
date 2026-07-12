// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::*;
use crate::{EncodeError, NativeEncodeRetainedInput};

fn part_with_capacities(data: usize, lengths: usize) -> EncodedTilePart {
    let mut payload = Vec::new();
    payload.try_reserve_exact(data).expect("small payload");
    let mut packet_lengths = Vec::new();
    packet_lengths
        .try_reserve_exact(lengths)
        .expect("small packet lengths");
    EncodedTilePart {
        tile_index: 0,
        tile_part_index: 0,
        num_tile_parts: 1,
        data: payload,
        packet_lengths,
        packet_headers: Vec::new(),
    }
}

#[test]
fn initial_part_reservation_accepts_exact_actual_capacity_and_rejects_one_byte_less() {
    let planning_bytes = 7;
    let discovery = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
    .expect("part reservation discovery session");
    let parts = reserve_tile_parts(3, planning_bytes, &discovery)
        .expect("discover actual part-owner capacity");
    let exact_cap = planning_bytes + parts.capacity() * core::mem::size_of::<EncodedTilePart>();

    let exact = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap)
        .expect("exact part reservation session");
    let _exact_parts =
        reserve_tile_parts(3, planning_bytes, &exact).expect("exact actual part-owner capacity");

    let under = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap - 1)
        .expect("part reservation baseline remains below cap");
    let error = reserve_tile_parts(3, planning_bytes, &under)
        .err()
        .expect("one byte below actual part-owner capacity must fail");
    assert!(matches!(
        error,
        super::super::super::NativeEncodePipelineError::Typed(
            EncodeError::AllocationTooLarge { requested, cap, .. }
        ) if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn accumulated_parts_accept_exact_cap_and_reject_one_byte_over() {
    let mut retained = Vec::new();
    retained.try_reserve_exact(1).expect("retained part owner");
    retained.push(part_with_capacities(5, 2));
    let incoming = vec![part_with_capacities(7, 1)];
    let planning_bytes = 3;
    let scratch_bytes = 11;
    let future_len = retained.len() + incoming.len();
    let retained_bytes =
        encoded_tile_parts_retained_bytes(&retained, retained.capacity()).expect("retained bytes");
    let retained_outer_bytes = retained.capacity() * core::mem::size_of::<EncodedTilePart>();
    let incoming_bytes =
        encoded_tile_parts_retained_bytes(&incoming, incoming.capacity()).expect("incoming bytes");
    let requested_peak = planning_bytes
        + retained_bytes
        + incoming_bytes
        + scratch_bytes
        + future_len * core::mem::size_of::<EncodedTilePart>();

    let discovery = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
    .expect("discovery session");
    append_encoded_tile_parts(
        &mut retained,
        incoming,
        planning_bytes,
        scratch_bytes,
        &discovery,
    )
    .expect("discover actual append capacity");
    let actual_reallocation_peak = planning_bytes
        + retained.capacity() * core::mem::size_of::<EncodedTilePart>()
        + (retained_bytes - retained_outer_bytes)
        + incoming_bytes
        + scratch_bytes;
    let exact_cap = requested_peak.max(actual_reallocation_peak);

    let exact = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap)
        .expect("exact append session");
    let mut exact_retained = Vec::new();
    exact_retained
        .try_reserve_exact(1)
        .expect("retained part owner");
    exact_retained.push(part_with_capacities(5, 2));
    append_encoded_tile_parts(
        &mut exact_retained,
        vec![part_with_capacities(7, 1)],
        planning_bytes,
        scratch_bytes,
        &exact,
    )
    .expect("exact append peak");

    let over = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap - 1)
        .expect("baseline below cap");
    let mut over_retained = Vec::new();
    over_retained
        .try_reserve_exact(1)
        .expect("retained part owner");
    over_retained.push(part_with_capacities(5, 2));
    let error = append_encoded_tile_parts(
        &mut over_retained,
        vec![part_with_capacities(7, 1)],
        planning_bytes,
        scratch_bytes,
        &over,
    )
    .expect_err("append peak is one byte over cap");
    assert!(matches!(
        error,
        super::super::super::NativeEncodePipelineError::Typed(
            EncodeError::AllocationTooLarge { .. }
        )
    ));
}

#[test]
fn append_with_spare_capacity_does_not_count_a_fictitious_outer_allocation() {
    let planning_bytes = 3;
    let scratch_bytes = 11;
    let mut retained = Vec::new();
    retained.try_reserve_exact(2).expect("retained part owners");
    retained.push(part_with_capacities(5, 2));
    let incoming = vec![part_with_capacities(7, 1)];
    let exact_cap = planning_bytes
        + encoded_tile_parts_retained_bytes(&retained, retained.capacity())
            .expect("retained bytes")
        + encoded_tile_parts_retained_bytes(&incoming, incoming.capacity())
            .expect("incoming bytes")
        + scratch_bytes;

    let exact = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap)
        .expect("exact append session");
    append_encoded_tile_parts(
        &mut retained,
        incoming,
        planning_bytes,
        scratch_bytes,
        &exact,
    )
    .expect("spare-capacity append at exact cap");

    let over = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap - 1)
        .expect("baseline below cap");
    let mut over_retained = Vec::new();
    over_retained
        .try_reserve_exact(2)
        .expect("retained part owners");
    over_retained.push(part_with_capacities(5, 2));
    let error = append_encoded_tile_parts(
        &mut over_retained,
        vec![part_with_capacities(7, 1)],
        planning_bytes,
        scratch_bytes,
        &over,
    )
    .expect_err("spare-capacity append is one byte over cap");
    assert!(matches!(
        error,
        super::super::super::NativeEncodePipelineError::Typed(
            EncodeError::AllocationTooLarge { .. }
        )
    ));
}
