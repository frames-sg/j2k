// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::J2kEncodeStageAccelerator;
use j2k_native::{CpuOnlyJ2kEncodeStageAccelerator, DecodeSettings, EncodeOptions, Image};

#[test]
fn resident_lossy_ht_matches_scalar_codestream_and_pixels() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    for (width, height, components) in [(64, 48, 1), (65, 49, 1), (64, 48, 3), (65, 49, 3)] {
        let pixels = if components == 1 {
            j2k_test_support::patterned_gray8(width, height)
        } else {
            j2k_test_support::patterned_rgb8(width, height)
        };
        let options = EncodeOptions {
            reversible: false,
            num_decomposition_levels: 3,
            guard_bits: 2,
            use_mct: components == 3,
            use_ht_block_coding: true,
            ..EncodeOptions::default()
        };
        let expected = j2k_native::encode_with_accelerator(
            &pixels,
            width,
            height,
            components,
            8,
            false,
            &options,
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        )
        .expect("scalar lossy encode");
        let mut accelerator = crate::MetalEncodeStageAccelerator::default();
        let actual = j2k_native::encode_with_accelerator(
            &pixels,
            width,
            height,
            components,
            8,
            false,
            &options,
            &mut accelerator,
        )
        .expect("resident lossy encode");
        assert!(
            accelerator.ht_tile_required_magnitude_bound().is_some(),
            "resident lossy tile hook must run"
        );
        assert_eq!(actual, expected, "{width}x{height} components {components}");
        let decode = |bytes: &[u8]| {
            Image::new(bytes, &DecodeSettings::default())
                .expect("parse lossy")
                .decode_native()
                .expect("decode lossy")
                .data
        };
        assert_eq!(decode(&actual), decode(&expected));
    }
}
