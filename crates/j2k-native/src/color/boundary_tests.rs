// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::error::DecodeError;
use crate::math::{SimdBuffer, SIMD_WIDTH};
use alloc::vec;

fn component(values: &[f32], bit_depth: u8) -> ComponentData {
    ComponentData {
        container: SimdBuffer::<SIMD_WIDTH>::new(values.to_vec()),
        integer_container: None,
        bit_depth,
        signed: false,
    }
}

#[test]
fn interleaved_output_validation_rejects_empty_and_short_destinations() {
    let boxes = ImageBoxes::default();
    let mut empty = Vec::new();
    let empty_image = DecodedImage {
        decoded_components: &mut empty,
        boxes: &boxes,
    };
    assert_eq!(
        validate_interleaved_output_buffer(&empty_image, &[]),
        Err(DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure))
    );

    let mut components = vec![component(&[1.0, 2.0], 8)];
    let image = DecodedImage {
        decoded_components: &mut components,
        boxes: &boxes,
    };
    assert_eq!(
        validate_interleaved_output_buffer(&image, &[0]),
        Err(DecodeError::Decoding(DecodingError::OutputBufferTooSmall))
    );
    validate_interleaved_output_buffer(&image, &[0, 0]).expect("exact output buffer");
}

#[test]
fn interleaver_covers_two_channel_fast_path_and_mixed_depth_scaling() {
    let boxes = ImageBoxes::default();
    let mut fast_components = vec![
        component(&[1.0, 2.0, 3.0], 8),
        component(&[251.0, 252.0, 253.0], 8),
    ];
    let mut fast_image = DecodedImage {
        decoded_components: &mut fast_components,
        boxes: &boxes,
    };
    let mut fast = vec![0; 6];
    interleave_and_convert(&mut fast_image, &mut fast).expect("two-channel fast interleave");
    assert_eq!(fast, [1, 251, 2, 252, 3, 253]);

    let mut mixed_components = vec![component(&[0.0, 15.0], 4), component(&[128.0, 255.0], 8)];
    let mut mixed_image = DecodedImage {
        decoded_components: &mut mixed_components,
        boxes: &boxes,
    };
    let mut mixed = vec![0; 4];
    interleave_and_convert(&mut mixed_image, &mut mixed).expect("mixed-depth slow interleave");
    assert_eq!(mixed, [0, 128, 255, 255]);
}

#[test]
fn region_interleaver_matches_fast_and_slow_full_projection_crops() {
    let boxes = ImageBoxes::default();
    let roi = (1, 0, 1, 2);
    let mut fast_components = vec![
        component(&[1.0, 2.0, 3.0, 4.0], 8),
        component(&[11.0, 12.0, 13.0, 14.0], 8),
    ];
    let mut fast_image = DecodedImage {
        decoded_components: &mut fast_components,
        boxes: &boxes,
    };
    let mut fast = vec![0; 4];
    interleave_and_convert_region(&mut fast_image, 2, roi, &mut fast);
    assert_eq!(fast, [2, 12, 4, 14]);

    let mut slow_components = vec![
        component(&[0.0, 15.0, 8.0, 4.0], 4),
        component(&[1.0, 2.0, 3.0, 4.0], 8),
    ];
    let mut slow_image = DecodedImage {
        decoded_components: &mut slow_components,
        boxes: &boxes,
    };
    let mut slow = vec![0; 4];
    interleave_and_convert_region(&mut slow_image, 2, roi, &mut slow);
    assert_eq!(slow, [255, 2, 68, 4]);
}

#[test]
fn native_component_dimensions_accept_full_or_sampled_shapes_only() {
    assert_eq!(
        native_component_plane_dimensions((5, 3), (1, 1), 15).expect("full-resolution plane"),
        (5, 3)
    );
    assert_eq!(
        native_component_plane_dimensions((5, 3), (2, 2), 6).expect("subsampled component plane"),
        (3, 2)
    );
    assert_eq!(
        native_component_plane_dimensions((5, 3), (0, 2), 6),
        Err(DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure))
    );
    assert_eq!(
        native_component_plane_dimensions((5, 3), (2, 2), 5),
        Err(DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure))
    );
}

#[test]
fn public_color_and_component_metadata_accessors_preserve_owned_values() {
    assert_eq!(ColorSpace::Unknown { num_channels: 7 }.num_channels(), 7);
    assert_eq!(
        ColorSpace::Icc {
            profile: vec![1, 2, 3],
            num_channels: 5,
        }
        .num_channels(),
        5
    );

    let plane = NativeComponentPlane {
        data: vec![1, 2, 3, 4],
        dimensions: (2, 1),
        bit_depth: 16,
        signed: true,
        sampling: (2, 1),
        bytes_per_sample: 2,
    };
    assert_eq!(plane.data(), [1, 2, 3, 4]);
    assert_eq!(plane.dimensions(), (2, 1));
    assert_eq!(plane.bit_depth(), 16);
    assert!(plane.signed());
    assert_eq!(plane.sampling(), (2, 1));
    assert_eq!(plane.bytes_per_sample(), 2);

    let decoded = DecodedNativeComponents {
        dimensions: (2, 1),
        color_space: ColorSpace::Unknown { num_channels: 1 },
        has_alpha: true,
        planes: vec![plane],
    };
    assert_eq!(decoded.dimensions(), (2, 1));
    assert_eq!(decoded.color_space().num_channels(), 1);
    assert!(decoded.has_alpha());
    assert_eq!(decoded.planes().len(), 1);
}
