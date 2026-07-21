// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{fixture_gray, fixture_ht_gray};
use crate::{
    encode, encode_typed_component_planes_53, DecodeSettings, DecoderContext, EncodeOptions,
    EncodeTypedComponentPlane, Image,
};

#[test]
fn decoder_workspace_reuses_component_owners_across_distinct_input_lifetimes() {
    let mut workspace = crate::DecoderWorkspace::default();
    let first_pixels = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("first image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("first workspace decode");
        let pixels = decoded.data.clone();
        drop(decoded);
        workspace = context.into_workspace();
        pixels
    };

    let second_pixels = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("second image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("second workspace decode");
        let pixels = decoded.data.clone();
        drop(decoded);
        workspace = context.into_workspace();
        pixels
    };

    assert_eq!(second_pixels, first_pixels);
    assert_eq!(workspace.stats().decode_calls(), 2);
    assert_eq!(workspace.stats().component_owner_reuses(), 1);
    assert!(workspace.stats().retained_component_bytes() > 0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScratchCapacitySnapshot {
    classic_coefficients: (*const crate::j2c::bitplane::Coefficient, usize),
    classic_payload: (*const u8, usize),
    ht_coefficients: (*const u32, usize),
    idwt_output: (*const f32, usize),
    idwt_scratch: (*const f32, usize),
    idwt_output_i64: (*const i64, usize),
    idwt_scratch_i64: (*const i64, usize),
}

fn scratch_capacity_snapshot(context: &DecoderContext<'_>) -> ScratchCapacitySnapshot {
    let tile = &context.tile_decode_context;
    ScratchCapacitySnapshot {
        classic_coefficients: (
            tile.bit_plane_decode_context.coefficient_ptr_for_test(),
            tile.bit_plane_decode_context
                .coefficient_capacity_for_test(),
        ),
        classic_payload: tile
            .bit_plane_decode_buffers
            .combined_layers_owner_for_test(),
        ht_coefficients: tile.ht_block_decode_context.coefficient_owner_for_test(),
        idwt_output: (
            tile.idwt_output.coefficients.as_ptr(),
            tile.idwt_output.coefficients.capacity(),
        ),
        idwt_scratch: (
            tile.idwt_scratch_buffer.as_ptr(),
            tile.idwt_scratch_buffer.capacity(),
        ),
        idwt_output_i64: (
            tile.idwt_output.coefficients_i64.as_ptr(),
            tile.idwt_output.coefficients_i64.capacity(),
        ),
        idwt_scratch_i64: (
            tile.idwt_scratch_buffer_i64.as_ptr(),
            tile.idwt_scratch_buffer_i64.capacity(),
        ),
    }
}

#[test]
fn decoder_workspace_reuses_classic_tier1_and_idwt_owners_across_input_lifetimes() {
    let mut workspace = crate::DecoderWorkspace::default();
    let first = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("first image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("first classic decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let second = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("second image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("second classic decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };

    assert!(first.classic_coefficients.1 > 0);
    assert!(first.classic_payload.1 > 0);
    assert!(first.idwt_output.1 > 0);
    assert_eq!(second, first);
    assert_eq!(workspace.stats().tier1_owner_reuses(), 1);
    assert_eq!(workspace.stats().idwt_owner_reuses(), 1);
    assert!(workspace.stats().retained_tier1_bytes() > 0);
    assert!(workspace.stats().retained_idwt_bytes() > 0);
}

#[test]
fn decoder_workspace_reuses_ht_tier1_owner_across_input_lifetimes() {
    let mut workspace = crate::DecoderWorkspace::default();
    let first = {
        let bytes = fixture_ht_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("first HT image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("first HT decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let second = {
        let bytes = fixture_ht_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("second HT image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("second HT decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };

    assert!(first.ht_coefficients.1 > 0);
    assert!(first.idwt_output.1 > 0);
    assert_eq!(second, first);
    assert_eq!(workspace.stats().tier1_owner_reuses(), 1);
    assert_eq!(workspace.stats().idwt_owner_reuses(), 1);
}

#[test]
fn decoder_workspace_reuses_scratch_across_alternating_shapes_and_precision() {
    let large_pixels = (0..16_u16 * 16)
        .map(|sample| (sample & 0xff) as u8)
        .collect::<Vec<_>>();
    let large_bytes = encode(
        &large_pixels,
        16,
        16,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode large classic fixture");
    let exact_samples = [0_u32, 1, (1_u32 << 28) + 7, (1_u32 << 29) - 1];
    let exact_pixels = exact_samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let exact_planes = [EncodeTypedComponentPlane {
        data: &exact_pixels,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 29,
        signed: false,
    }];
    let exact_bytes = encode_typed_component_planes_53(
        &exact_planes,
        2,
        2,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode exact fixture");

    let mut workspace = crate::DecoderWorkspace::default();
    let large_scratch = {
        let image = Image::new(&large_bytes, &DecodeSettings::default()).expect("large image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_native_with_context(&mut context)
            .expect("large decode");
        assert_eq!(decoded.data, large_pixels);
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let exact_scratch = {
        let image = Image::new(&exact_bytes, &DecodeSettings::default()).expect("exact image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_native_with_context(&mut context)
            .expect("exact decode");
        assert_eq!(decoded.data, exact_pixels);
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let final_scratch = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("small image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("small decode");
        assert_eq!(decoded.data, (0_u8..16).collect::<Vec<_>>());
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };

    assert_eq!(
        final_scratch.classic_coefficients,
        large_scratch.classic_coefficients
    );
    assert_eq!(final_scratch.classic_payload, large_scratch.classic_payload);
    assert_eq!(final_scratch.idwt_output, large_scratch.idwt_output);
    assert_eq!(final_scratch.idwt_scratch, large_scratch.idwt_scratch);
    assert!(exact_scratch.idwt_output_i64.1 > 0);
    assert_eq!(final_scratch.idwt_output_i64, exact_scratch.idwt_output_i64);
    assert_eq!(workspace.stats().decode_calls(), 3);
    assert_eq!(workspace.stats().tier1_owner_reuses(), 2);
    assert_eq!(workspace.stats().idwt_owner_reuses(), 2);
    assert_eq!(workspace.stats().scratch_capacity_retries(), 0);
}
