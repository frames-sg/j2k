// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_native::{encode, encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, Image};

#[test]
fn classic_cuda_plan_retains_irreversible_midpoint_reconstruction() {
    let pixels = j2k_test_support::gradient_u8(16, 16, 1);
    let bytes = encode(
        &pixels,
        16,
        16,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode irreversible grayscale");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let direct = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct plan");

    let cuda = CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
        .expect("CUDA plan");

    assert!(
        cuda.classic_subbands
            .iter()
            .all(|subband| subband.irreversible_midpoint),
        "every classic CUDA sub-band must retain the 9/7 reconstruction rule"
    );
}

#[test]
fn classic_cuda_plan_accepts_roi_maxshift() {
    let pixels = j2k_test_support::gradient_u8(16, 16, 1);
    let bytes = encode(
        &pixels,
        16,
        16,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            roi_component_shifts: vec![7],
            ..EncodeOptions::default()
        },
    )
    .expect("encode ROI maxshift grayscale");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let direct = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct ROI plan");

    let cuda = CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
        .expect("CUDA classic ROI plan");
    assert!(
        cuda.classic_code_blocks
            .iter()
            .all(|block| block.roi_shift == 7),
        "every classic CUDA code-block must retain the component ROI maxshift"
    );
}

#[test]
fn ht_cuda_plan_retains_irreversible_midpoint_reconstruction() {
    let pixels = j2k_test_support::gradient_u8(32, 32, 1);
    let bytes = encode_htj2k(
        &pixels,
        32,
        32,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            num_decomposition_levels: 3,
            ..EncodeOptions::default()
        },
    )
    .expect("encode irreversible HTJ2K");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let direct = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct HTJ2K plan");

    let cuda = CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
        .expect("CUDA HTJ2K plan");
    assert!(
        !cuda.code_blocks().is_empty()
            && cuda
                .code_blocks()
                .iter()
                .all(|block| block.irreversible_midpoint),
        "every HT CUDA code-block must retain the fixed-point reconstruction rule"
    );
}
