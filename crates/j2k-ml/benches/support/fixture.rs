// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    encode_j2k_lossless, wrap_j2k_codestream, J2kBlockCodingMode, J2kChannelAssociation,
    J2kChannelDefinition, J2kChannelType, J2kEncodeValidation, J2kFileBoxMetadata,
    J2kFileColorSpec, J2kFileWrapOptions, J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use j2k_core::Colorspace;

pub(super) fn encode_ht_fixture(
    width: u32,
    height: u32,
    components: u16,
    precision: u8,
    signed: bool,
    variant: usize,
) -> Vec<u8> {
    let sample_count = usize::try_from(width).expect("benchmark width fits usize")
        * usize::try_from(height).expect("benchmark height fits usize")
        * usize::from(components);
    let mask = if precision == 16 {
        u16::MAX
    } else {
        (1_u16 << precision) - 1
    };
    let bytes_per_sample = if precision <= 8 { 1 } else { 2 };
    let mut bytes = Vec::with_capacity(sample_count * bytes_per_sample);
    let variant = u32::try_from(variant).expect("benchmark fixture variant fits u32");
    for index in 0..sample_count {
        let value = u16::try_from(
            u32::try_from(index)
                .expect("benchmark sample index fits u32")
                .wrapping_mul(2_653)
                .wrapping_add(variant.wrapping_mul(4_051))
                .wrapping_add(17)
                & u32::from(u16::MAX),
        )
        .expect("masked benchmark value fits u16")
            & mask;
        if signed {
            let sign_bit = 1_u16 << (precision - 1);
            let sign_extended = if value & sign_bit == 0 {
                value
            } else {
                value | !mask
            };
            if precision <= 8 {
                bytes.push(sign_extended.to_le_bytes()[0]);
            } else {
                bytes.extend_from_slice(&sign_extended.to_le_bytes());
            }
        } else if precision <= 8 {
            bytes.push(u8::try_from(value).expect("8-bit benchmark sample"));
        } else {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let samples = J2kLosslessSamples::new(&bytes, width, height, components, precision, signed)
        .expect("valid HT benchmark samples");
    let codestream = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode HT benchmark fixture")
    .codestream;
    if components == 4 {
        wrap_benchmark_rgba_jph(&codestream)
    } else {
        codestream
    }
}

fn wrap_benchmark_rgba_jph(codestream: &[u8]) -> Vec<u8> {
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
            channel_type: J2kChannelType::Opacity,
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
    .expect("wrap benchmark RGBA HTJ2K as JPH with straight alpha")
}
