// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use burn_core::tensor::DType;
use j2k::{
    encode_j2k_lossless, prepare_batch, wrap_j2k_codestream, BatchDecodeOptions, BatchItemError,
    BatchLayout, CpuBatchDecoder, CpuBatchSamples, DecodeRequest, DecodeSettings, Downscale,
    EncodedImage, J2kBlockCodingMode, J2kChannelAssociation, J2kChannelDefinition, J2kChannelType,
    J2kEncodeValidation, J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions,
    J2kLosslessEncodeOptions, J2kLosslessSamples, Rect,
};
use j2k_core::Colorspace;
use j2k_ml::{BurnBatchTensor, BurnDecodeError, MetalUploadBurnDecoder};
use j2k_native::{encode, EncodeOptions};
use j2k_test_support::{
    generated_htj2k_rgba_fixture, htj2k_rgb8_97_fixture, metal_runtime_gate,
    openhtj2k_refinement_fixture, openhtj2k_refinement_odd_fixture, openhtj2k_refinement_pixels,
    Htj2kRgbaAlpha, Htj2kRgbaSampleProfile,
};

fn wrap_rgba_jph(codestream: &[u8], alpha: Htj2kRgbaAlpha) -> Vec<u8> {
    let alpha_type = match alpha {
        Htj2kRgbaAlpha::Straight => J2kChannelType::Opacity,
        Htj2kRgbaAlpha::Premultiplied => J2kChannelType::PremultipliedOpacity,
    };
    let channel_definitions = [
        J2kChannelDefinition {
            channel_index: 0,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 1 },
        },
        J2kChannelDefinition {
            channel_index: 1,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 2 },
        },
        J2kChannelDefinition {
            channel_index: 2,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 3 },
        },
        J2kChannelDefinition {
            channel_index: 3,
            channel_type: alpha_type,
            association: J2kChannelAssociation::WholeImage,
        },
    ];
    wrap_j2k_codestream(
        codestream,
        J2kFileWrapOptions::jph()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: None,
                component_mappings: &[],
                channel_definitions: &channel_definitions,
            }),
    )
    .expect("wrap explicit HTJ2K RGBA image")
}

fn unsupported_classic_roi_rgb() -> Arc<[u8]> {
    let pixels = (0..4_u8)
        .flat_map(|index| [index * 17, index * 29 + 3, index * 41 + 5])
        .collect::<Vec<_>>();
    Arc::from(
        encode(
            &pixels,
            2,
            2,
            3,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                roi_component_shifts: vec![3, 0, 0],
                ..EncodeOptions::default()
            },
        )
        .expect("encode classic RGB8 with unsupported RGN maxshift"),
    )
}

fn encode_gray12(samples: &[u16]) -> Vec<u8> {
    let bytes = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&bytes, 2, 2, 1, 12, false).expect("Gray12 samples");
    encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode Gray12")
    .codestream
}

fn encode_signed_gray12(samples: &[i16]) -> Vec<u8> {
    let bytes = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let samples =
        J2kLosslessSamples::new(&bytes, 2, 2, 1, 12, true).expect("signed Gray12 samples");
    encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode signed Gray12")
    .codestream
}

fn encode_gray8(samples: &[u8], width: u32, height: u32) -> Vec<u8> {
    let samples =
        J2kLosslessSamples::new(samples, width, height, 1, 8, false).expect("Gray8 samples");
    encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode Gray8")
    .codestream
}

fn encode_rgb_u8(width: u32, height: u32, offset: u8) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            for pattern in [x * 3 + y * 5, x * 7 + y * 11 + 13, x * 17 + y * 19 + 29] {
                let pattern = u8::try_from(pattern & 0x3f).expect("six-bit RGB pattern");
                bytes.push(offset.wrapping_add(pattern) & 0x3f);
            }
        }
    }
    let samples =
        J2kLosslessSamples::new(&bytes, width, height, 3, 6, false).expect("six-bit RGB samples");
    encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode HT RGB U8")
    .codestream
}

fn encode_rgb_u16(width: u32, height: u32, offset: u16) -> Vec<u8> {
    encode_rgb_u16_with_mode(width, height, offset, J2kBlockCodingMode::HighThroughput)
}

fn encode_classic_rgb_u16(width: u32, height: u32, offset: u16) -> Vec<u8> {
    encode_rgb_u16_with_mode(width, height, offset, J2kBlockCodingMode::Classic)
}

fn encode_rgb_u16_with_mode(
    width: u32,
    height: u32,
    offset: u16,
    block_coding_mode: J2kBlockCodingMode,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(width as usize * height as usize * 3 * 2);
    for y in 0..height {
        for x in 0..width {
            for pattern in [
                x * 193 + y * 257,
                x * 313 + y * 97 + 31,
                x * 71 + y * 401 + 63,
            ] {
                let pattern = u16::try_from(pattern & 0x07ff).expect("twelve-bit RGB pattern");
                let sample = offset.wrapping_add(pattern);
                bytes.extend_from_slice(&(sample & 0x0fff).to_le_bytes());
            }
        }
    }
    let samples = J2kLosslessSamples::new(&bytes, width, height, 3, 12, false)
        .expect("twelve-bit RGB samples");
    encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(block_coding_mode)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode RGB U16")
    .codestream
}

#[path = "metal/native_color.rs"]
mod native_color;
#[path = "metal/requests.rs"]
mod requests;
#[path = "metal/sessions.rs"]
mod sessions;
