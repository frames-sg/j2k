// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_combined_context_bytes, is_capacity_error, ComponentData, DecoderContext, SimdBuffer,
    TileDecodeContext, CONTEXT_ALLOCATION_WHAT, DEFAULT_MAX_DECODE_BYTES, SIMD_WIDTH,
};
use crate::error::{DecodeError, ValidationError};
use crate::j2c::decode::{DecodeDebugCounters, OutputRegion};
use alloc::{vec, vec::Vec};
use core::mem::size_of;

fn retained_channels() -> (Vec<ComponentData>, usize) {
    let mut samples = Vec::with_capacity(16);
    samples.resize(9, 3.0);
    let mut integers = Vec::with_capacity(12);
    integers.resize(5, 7_i64);
    let component = ComponentData {
        container: SimdBuffer::<SIMD_WIDTH>::new(samples),
        integer_container: Some(integers),
        bit_depth: 29,
        signed: true,
    };
    let nested_bytes = component.container.capacity() * size_of::<f32>()
        + component
            .integer_container
            .as_ref()
            .map_or(0, |values| values.capacity() * size_of::<i64>());
    let mut channels = Vec::with_capacity(3);
    channels.push(component);
    let retained_bytes = channels.capacity() * size_of::<ComponentData>() + nested_bytes;
    (channels, retained_bytes)
}

#[test]
fn channel_release_drops_actual_capacity_without_mutating_other_tile_state() {
    let (channels, expected_retained_bytes) = retained_channels();
    let mut context = TileDecodeContext {
        channel_data: channels,
        output_region: Some(OutputRegion::from_tuple((2, 3, 5, 7))),
        debug_counters: DecodeDebugCounters {
            decoded_code_blocks: 11,
            ..DecodeDebugCounters::default()
        },
        idwt_scratch_buffer: vec![13.0; 4],
        ..TileDecodeContext::default()
    };
    let scratch_ptr = context.idwt_scratch_buffer.as_ptr();
    let scratch_capacity = context.idwt_scratch_buffer.capacity();

    assert_eq!(
        context
            .retained_channel_bytes()
            .expect("retained owner accounting"),
        expected_retained_bytes
    );

    context.release_channel_allocations();

    assert!(context.channel_data.is_empty());
    assert_eq!(context.channel_data.capacity(), 0);
    assert_eq!(
        context
            .retained_channel_bytes()
            .expect("released owner accounting"),
        0
    );
    assert_eq!(context.idwt_scratch_buffer.as_ptr(), scratch_ptr);
    assert_eq!(context.idwt_scratch_buffer.capacity(), scratch_capacity);
    assert_eq!(
        context.output_region,
        Some(OutputRegion::from_tuple((2, 3, 5, 7)))
    );
    assert_eq!(context.debug_counters.decoded_code_blocks, 11);
}

#[test]
fn decoder_discard_releases_only_the_reused_channel_owner() {
    let (channels, expected_retained_bytes) = retained_channels();
    let mut context = DecoderContext::default();
    context.tile_decode_context.channel_data = channels;
    context.tile_decode_context.idwt_scratch_buffer = vec![17.0; 6];
    let scratch_ptr = context.tile_decode_context.idwt_scratch_buffer.as_ptr();
    let scratch_capacity = context.tile_decode_context.idwt_scratch_buffer.capacity();
    assert_eq!(
        context
            .tile_decode_context
            .retained_channel_bytes()
            .expect("retained owner accounting"),
        expected_retained_bytes
    );

    context.discard_reused_channels();

    assert!(context.tile_decode_context.channel_data.is_empty());
    assert_eq!(context.tile_decode_context.channel_data.capacity(), 0);
    assert_eq!(
        context.tile_decode_context.idwt_scratch_buffer.as_ptr(),
        scratch_ptr
    );
    assert_eq!(
        context.tile_decode_context.idwt_scratch_buffer.capacity(),
        scratch_capacity
    );
}

#[test]
fn capacity_classification_includes_overflow_but_excludes_unrelated_failures() {
    let overflow = checked_combined_context_bytes(usize::MAX, 1, usize::MAX)
        .expect_err("combined context size must overflow");
    assert_eq!(
        overflow,
        DecodeError::AllocationTooLarge {
            what: CONTEXT_ALLOCATION_WHAT,
            requested: usize::MAX,
            cap: DEFAULT_MAX_DECODE_BYTES,
        }
    );

    assert!(is_capacity_error(&overflow));
    assert!(is_capacity_error(&DecodeError::Validation(
        ValidationError::ImageTooLarge
    )));
    assert!(!is_capacity_error(&DecodeError::Validation(
        ValidationError::InvalidDimensions
    )));
    assert!(!is_capacity_error(&DecodeError::HostAllocationFailed {
        what: "test owner",
        bytes: 8,
    }));
}
