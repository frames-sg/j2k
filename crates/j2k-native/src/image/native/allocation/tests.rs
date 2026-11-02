// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{bit_capacity_bytes, NativeOutputBudget};
use crate::error::DecodeError;
use crate::j2c::{ComponentData, DecoderContext};
use crate::math::{SimdBuffer, SIMD_WIDTH};
use crate::DEFAULT_MAX_DECODE_BYTES;
use crate::{
    encode, ColorSpace, ComponentPlane, CpuDecodeParallelism, DecodeErrorClass, DecodeSettings,
    DecodedComponents, EncodeOptions, Image, RawBitmap,
};
use alloc::{vec, vec::Vec};
use core::mem::size_of;

#[test]
fn native_output_budget_has_an_exact_shared_cap_boundary() {
    let mut budget = NativeOutputBudget::from_retained_image(DEFAULT_MAX_DECODE_BYTES - 1)
        .expect("one byte of native-output headroom");
    budget.include_elements::<u8>(1).expect("exact cap fits");
    assert!(matches!(
        budget.include_elements::<u8>(1),
        Err(DecodeError::AllocationTooLarge { requested, cap, .. })
            if requested > cap && cap == DEFAULT_MAX_DECODE_BYTES
    ));
}

#[test]
fn borrowed_handoff_counts_actual_backing_and_metadata_capacities() {
    let mut integer_container = Vec::with_capacity(5);
    integer_container.push(7_i64);
    let mut components = Vec::with_capacity(3);
    components.push(ComponentData {
        container: SimdBuffer::<SIMD_WIDTH>::new(vec![1.0]),
        integer_container: Some(integer_container),
        bit_depth: 16,
        signed: false,
    });

    let mut profile = Vec::with_capacity(7);
    profile.push(1);
    let mut planes = Vec::with_capacity(4);
    planes.push(ComponentPlane {
        samples: components[0].container.truncated(),
        dimensions: (1, 1),
        bit_depth: 16,
        signed: false,
        sampling: (1, 1),
    });
    let mut packed = DecodedComponents {
        dimensions: (1, 1),
        color_space: ColorSpace::Icc {
            profile,
            num_channels: 1,
        },
        has_alpha: false,
        planes,
        live_bytes: 0,
    };

    packed.live_bytes =
        NativeOutputBudget::validate_borrowed_pack(0, &components, components.capacity(), &packed)
            .expect("small retained handoff fits");

    let expected = components.capacity() * size_of::<ComponentData>()
        + components[0].container.capacity() * size_of::<f32>()
        + components[0]
            .integer_container
            .as_ref()
            .expect("integer shadow")
            .capacity()
            * size_of::<i64>()
        + packed.planes.capacity() * size_of::<ComponentPlane<'_>>()
        + match &packed.color_space {
            ColorSpace::Icc { profile, .. } => profile.capacity(),
            _ => 0,
        };
    assert_eq!(packed.live_bytes(), expected);
    assert!(components[0].container.capacity() > packed.planes[0].samples.len());
}

#[test]
fn retained_image_metadata_shares_the_output_cap() {
    let components = Vec::<ComponentData>::new();
    let error = NativeOutputBudget::for_decoded_channels(
        DEFAULT_MAX_DECODE_BYTES,
        &components,
        components.capacity(),
    )
    .and_then(|mut budget| budget.include_elements::<u8>(1))
    .expect_err("one output byte must exceed a full retained-image baseline");
    assert!(matches!(
        error,
        DecodeError::AllocationTooLarge { requested, cap, .. }
            if requested > cap && cap == DEFAULT_MAX_DECODE_BYTES
    ));
}

#[test]
fn full_decode_crop_budget_counts_retained_channels_at_exact_boundary() {
    let components = vec![ComponentData {
        container: SimdBuffer::<SIMD_WIDTH>::new(vec![1.0]),
        integer_container: None,
        bit_depth: 8,
        signed: false,
    }];
    let bitmap = RawBitmap {
        data: vec![1],
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        component_signed: vec![false],
        num_components: 1,
        bytes_per_sample: 1,
    };
    let channel_bytes = components.capacity() * size_of::<ComponentData>()
        + components[0].container.capacity() * size_of::<f32>();
    let bitmap_bytes =
        bitmap.data.capacity() + bit_capacity_bytes(bitmap.component_signed.capacity());
    let retained = DEFAULT_MAX_DECODE_BYTES - channel_bytes - bitmap_bytes;

    let exact = NativeOutputBudget::for_raw_bitmap_with_decoded_channels(
        retained,
        &components,
        components.capacity(),
        &bitmap,
    )
    .expect("decoded channels plus full bitmap fit exact cap");
    assert_eq!(exact.allocation.bytes(), DEFAULT_MAX_DECODE_BYTES);
    assert!(matches!(
        NativeOutputBudget::for_raw_bitmap_with_decoded_channels(
            retained + 1,
            &components,
            components.capacity(),
            &bitmap,
        ),
        Err(DecodeError::AllocationTooLarge {
            requested,
            cap: DEFAULT_MAX_DECODE_BYTES,
            ..
        }) if requested == DEFAULT_MAX_DECODE_BYTES + 1
    ));
}

#[test]
fn allocator_capacity_overage_is_applied_before_the_next_output_owner() {
    let mut budget = NativeOutputBudget::from_retained_image(DEFAULT_MAX_DECODE_BYTES - 4)
        .expect("four bytes of native-output headroom");
    let error = budget
        .include_capacity_overage::<u8>(1, 6)
        .expect_err("five overage bytes exceed four remaining bytes");
    assert!(matches!(
        error,
        DecodeError::AllocationTooLarge { requested, cap, .. }
            if requested > cap && cap == DEFAULT_MAX_DECODE_BYTES
    ));
}

#[test]
fn bit_vector_capacity_is_counted_in_storage_bytes() {
    let mut exact = NativeOutputBudget::from_retained_image(DEFAULT_MAX_DECODE_BYTES - 1)
        .expect("one byte of native-output headroom");
    exact
        .include_bit_capacity(8)
        .expect("eight bits need one byte at the exact boundary");

    let mut one_over = NativeOutputBudget::from_retained_image(DEFAULT_MAX_DECODE_BYTES - 1)
        .expect("one byte of native-output headroom");
    assert!(matches!(
        one_over.include_bit_capacity(9),
        Err(DecodeError::AllocationTooLarge { requested, cap, .. })
            if requested > cap && cap == DEFAULT_MAX_DECODE_BYTES
    ));
}

/// The packed planes of a native component decode are its output, like the
/// rows route's planes: the decode budget holds the `f32` planes they are
/// packed from, not the output beside them. An image that does not stream
/// its rows (five components), given a budget with room for its working
/// memory but not for its `f32` planes and output together, decodes to the
/// bytes a decode with the whole budget gives; one with no room for the
/// `f32` planes is still refused.
#[test]
fn native_component_planes_are_output_beside_the_decode_budget() {
    let (width, height, components) = (256_u32, 256_u32, 5_u16);
    let samples: Vec<u8> = (0..width * height * u32::from(components) * 2)
        .map(|index| (index.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    let options = EncodeOptions {
        num_decomposition_levels: 2,
        tile_size: Some((64, 64)),
        ..EncodeOptions::default()
    };
    let bytes = encode(&samples, width, height, components, 12, false, &options).expect("encode");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    assert!(!image.region_decode_streams_rows());
    let planes = |retained_baseline_bytes| {
        let mut context = DecoderContext::default();
        context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
        image
            .decode_native_components_with_context_and_retained_baseline(
                &mut context,
                retained_baseline_bytes,
            )
            .map(|decoded| {
                decoded
                    .planes
                    .into_iter()
                    .map(|plane| (plane.data, plane.bit_depth, plane.bytes_per_sample))
                    .collect::<Vec<_>>()
            })
    };
    let expected = planes(0).expect("a decode with the whole budget");
    let output: usize = expected.iter().map(|(data, ..)| data.len()).sum();
    assert_eq!(output, 256 * 256 * 5 * 2);
    // The `f32` planes without padding or owners: less than the decode holds.
    let f32_planes = 256 * 256 * 5 * size_of::<f32>();
    let decoded = planes(DEFAULT_MAX_DECODE_BYTES - (f32_planes + output - 1))
        .expect("the output is not charged to the decode budget");
    assert!(decoded == expected, "same planes");
    let error = planes(DEFAULT_MAX_DECODE_BYTES - f32_planes / 2)
        .expect_err("the f32 planes pass the budget");
    assert!(
        matches!(error.classify(), DecodeErrorClass::ResourceLimit { .. }),
        "{error:?}"
    );
}
