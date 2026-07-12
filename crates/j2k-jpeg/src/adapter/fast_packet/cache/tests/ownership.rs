// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;
use std::sync::Arc;

use super::super::shared_allocation::{shared_owner_bytes, shared_slice_owner_bytes};
use super::super::{JpegFastPacket, SharedJpegFastPacket, SharedJpegInput};
use crate::adapter::{
    build_fast420_packet, build_fast422_packet, build_fast444_packet, JpegEntropyCheckpointV1,
};
use j2k_test_support::{JPEG_BASELINE_420_16X16, JPEG_BASELINE_422_16X8, JPEG_BASELINE_444_8X8};

#[test]
fn shared_input_is_fallibly_copied_and_exposes_actual_capacity() {
    let source = b"complete JPEG identity";
    let input = SharedJpegInput::try_copy_from_slice(source).expect("copy shared input");
    let cloned = input.clone();

    assert_eq!(input.as_slice(), source);
    assert_eq!(input.as_bytes(), source);
    assert_eq!(input.as_ref(), source);
    assert!(input.data_capacity() >= source.len());
    assert!(input.retained_cache_bytes().unwrap() >= input.data_capacity());
    assert!(SharedJpegInput::ptr_eq(&input, &cloned));
}

#[test]
fn shared_input_limit_accepts_exact_bytes_and_rejects_one_over_before_reserve() {
    let probe = SharedJpegInput::try_copy_from_slice_with_cap(b"four", usize::MAX)
        .expect("probe copied owner");
    let exact_cap = probe.retained_cache_bytes().unwrap();
    let exact = SharedJpegInput::try_copy_from_slice_with_cap(b"four", exact_cap)
        .expect("exact shared-input cap");
    assert_eq!(exact.as_bytes(), b"four");

    let error = SharedJpegInput::try_copy_from_slice_with_cap(b"four", exact_cap - 1)
        .expect_err("one byte over shared-input cap");
    let cloned = error.clone();
    assert!(matches!(
        error,
        super::super::JpegPlanCacheError::Limit {
            what: "shared JPEG copied input owner graph",
            requested,
            cap,
        } if requested == exact_cap && cap == exact_cap - 1
    ));
    assert!(matches!(
        cloned,
        super::super::JpegPlanCacheError::Limit { .. }
    ));
}

#[test]
fn arc_input_moves_without_payload_copy_and_charges_its_fixed_slice_owner() {
    let owner = Arc::<[u8]>::from(b"immutable shared JPEG".as_slice());
    let owner_pointer = owner.as_ptr();
    let input = SharedJpegInput::try_from_arc(owner.clone()).expect("adopt Arc input");
    let cloned = input.clone();

    assert_eq!(input.as_bytes().as_ptr(), owner_pointer);
    assert_eq!(input.data_capacity(), owner.len());
    assert_eq!(
        input.retained_cache_bytes().unwrap(),
        shared_slice_owner_bytes(owner.len()).unwrap()
    );
    assert!(SharedJpegInput::ptr_eq(&input, &cloned));

    let copied = SharedJpegInput::try_copy_from_slice(owner.as_ref()).expect("copy same bytes");
    assert!(!SharedJpegInput::ptr_eq(&input, &copied));
}

#[test]
fn arc_input_limit_accepts_exact_length_and_rejects_one_over() {
    let exact_cap = shared_slice_owner_bytes(4).unwrap();
    let exact = SharedJpegInput::try_from_arc_with_cap(Arc::from(b"four".as_slice()), exact_cap)
        .expect("exact Arc-input cap");
    assert_eq!(exact.as_bytes(), b"four");

    let error =
        SharedJpegInput::try_from_arc_with_cap(Arc::from(b"four".as_slice()), exact_cap - 1)
            .expect_err("one-byte-over Arc-input cap");
    assert!(matches!(
        error,
        super::super::JpegPlanCacheError::Limit {
            what: "shared JPEG Arc input owner graph",
            requested,
            cap,
        } if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn external_live_input_owner_boundaries_are_exact_before_copy_or_adoption() {
    let copied_probe = SharedJpegInput::try_copy_from_slice(b"owner").unwrap();
    let copied_bytes = copied_probe.retained_cache_bytes().unwrap();
    let external = 7;
    let exact = external + copied_bytes;
    SharedJpegInput::try_copy_from_slice_with_external_live_and_cap(b"owner", external, exact)
        .expect("external plus copied owner exact cap");
    assert!(matches!(
        SharedJpegInput::try_copy_from_slice_with_external_live_and_cap(
            b"owner",
            external,
            exact - 1,
        ),
        Err(super::super::JpegPlanCacheError::Limit { requested, cap, .. })
            if requested == exact && cap == exact - 1
    ));

    let arc_bytes = shared_slice_owner_bytes(5).unwrap();
    let exact = external + arc_bytes;
    SharedJpegInput::try_from_arc_with_external_live_and_cap(
        Arc::from(b"owner".as_slice()),
        external,
        exact,
    )
    .expect("external plus Arc owner exact cap");
    assert!(matches!(
        SharedJpegInput::try_from_arc_with_external_live_and_cap(
            Arc::from(b"owner".as_slice()),
            external,
            exact - 1,
        ),
        Err(super::super::JpegPlanCacheError::Limit { requested, cap, .. })
            if requested == exact && cap == exact - 1
    ));
}

#[test]
fn shared_one_family_packets_charge_every_nested_vector_capacity_exactly() {
    let packet420 = build_fast420_packet(JPEG_BASELINE_420_16X16).expect("build 420 packet");
    let expected420 = expected_packet_bytes(
        packet420.restart_offsets.capacity(),
        packet420.entropy_checkpoints.capacity(),
        packet420.entropy_bytes.capacity(),
    );
    let shared420 = SharedJpegFastPacket::try_new(packet420.into()).unwrap();
    assert_eq!(shared420.retained_cache_bytes().unwrap(), expected420);
    assert!(shared420.fast420().is_some());
    assert!(shared420.fast422().is_none());
    assert!(shared420.fast444().is_none());

    let packet422 = build_fast422_packet(JPEG_BASELINE_422_16X8).expect("build 422 packet");
    let expected422 = expected_packet_bytes(
        packet422.restart_offsets.capacity(),
        packet422.entropy_checkpoints.capacity(),
        packet422.entropy_bytes.capacity(),
    );
    let shared422 = SharedJpegFastPacket::try_new(packet422.into()).unwrap();
    assert_eq!(shared422.retained_cache_bytes().unwrap(), expected422);
    assert!(shared422.fast420().is_none());
    assert!(shared422.fast422().is_some());
    assert!(shared422.fast444().is_none());

    let packet444 = build_fast444_packet(JPEG_BASELINE_444_8X8).expect("build 444 packet");
    let expected444 = expected_packet_bytes(
        packet444.restart_offsets.capacity(),
        packet444.entropy_checkpoints.capacity(),
        packet444.entropy_bytes.capacity(),
    );
    let shared444 = SharedJpegFastPacket::try_new(packet444.into()).unwrap();
    assert_eq!(shared444.retained_cache_bytes().unwrap(), expected444);
    assert!(shared444.fast420().is_none());
    assert!(shared444.fast422().is_none());
    assert!(shared444.fast444().is_some());

    let cloned = shared444.clone();
    assert!(SharedJpegFastPacket::ptr_eq(&shared444, &cloned));
}

#[test]
fn shared_packet_arc_boundary_accepts_exact_aggregate_and_rejects_one_over() {
    let packet = build_fast420_packet(JPEG_BASELINE_420_16X16).expect("build probe packet");
    let retained = SharedJpegFastPacket::try_new(packet.into())
        .expect("share probe packet")
        .retained_cache_bytes()
        .expect("probe retained bytes");
    let external = 17;
    let exact_cap = external + retained;

    let exact_packet = build_fast420_packet(JPEG_BASELINE_420_16X16).expect("build exact packet");
    SharedJpegFastPacket::try_new_with_cap_for_test(exact_packet.into(), external, exact_cap)
        .expect("fixed packet owner fits exact cap");

    let rejected_packet =
        build_fast420_packet(JPEG_BASELINE_420_16X16).expect("build rejected packet");
    assert!(matches!(
        SharedJpegFastPacket::try_new_with_cap_for_test(
            rejected_packet.into(),
            external,
            exact_cap - 1,
        ),
        Err(super::super::JpegPlanCacheError::Limit {
            what: "shared JPEG fast-packet owner graph",
            requested,
            cap,
        }) if requested == exact_cap && cap == exact_cap - 1
    ));
}

fn expected_packet_bytes(
    restart_capacity: usize,
    checkpoint_capacity: usize,
    entropy_capacity: usize,
) -> usize {
    let nested = restart_capacity
        .checked_mul(size_of::<u32>())
        .and_then(|bytes| {
            checkpoint_capacity
                .checked_mul(size_of::<JpegEntropyCheckpointV1>())
                .and_then(|checkpoint_bytes| bytes.checked_add(checkpoint_bytes))
        })
        .and_then(|bytes| bytes.checked_add(entropy_capacity))
        .expect("test packet capacity arithmetic");
    shared_owner_bytes::<JpegFastPacket>(nested).expect("test shared-owner arithmetic")
}
