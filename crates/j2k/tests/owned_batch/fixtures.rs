// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    wrap_j2k_codestream, J2kChannelAssociation, J2kChannelDefinition, J2kChannelType,
    J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions,
};
use j2k_core::Colorspace;
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use j2k_test_support::Htj2kRgbaAlpha;

pub(super) fn htj2k_gray8_fixture(width: u32, height: u32) -> Vec<u8> {
    htj2k_gray8_fixture_with_levels(width, height, 1)
}

pub(super) fn classic_gray8_fixture(width: u32, height: u32) -> Vec<u8> {
    classic_gray8_fixture_with_tile_size(width, height, None)
}

pub(super) fn classic_gray8_fixture_with_tile_size(
    width: u32,
    height: u32,
    tile_size: Option<(u32, u32)>,
) -> Vec<u8> {
    let pixels = (0..width * height)
        .map(|index| (index & 0xff) as u8)
        .collect::<Vec<_>>();
    encode(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            tile_size,
            ..EncodeOptions::default()
        },
    )
    .expect("encode classic J2K gray8")
}

pub(super) fn htj2k_gray8_fixture_with_levels(
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
) -> Vec<u8> {
    let pixels = (0..width * height)
        .map(|index| (index & 0xff) as u8)
        .collect::<Vec<_>>();
    encode_htj2k(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels,
            ..EncodeOptions::default()
        },
    )
    .expect("encode HTJ2K gray8")
}

pub(super) fn rewrite_component_descriptor(bytes: &mut [u8], component: usize, descriptor: u8) {
    let siz_marker = bytes
        .windows(2)
        .position(|marker| marker == [0xff, 0x51])
        .expect("SIZ marker");
    bytes[siz_marker + 40 + component * 3] = descriptor;
}

pub(super) fn rgb8_fixture() -> Vec<u8> {
    let pixels = (0_u8..48).collect::<Vec<_>>();
    encode(
        &pixels,
        4,
        4,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .expect("encode RGB8")
}

pub(super) fn four_component_fixture() -> Vec<u8> {
    let pixels = (0_u8..64).collect::<Vec<_>>();
    encode(
        &pixels,
        4,
        4,
        4,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode four-component image")
}

fn rgba_channel_definitions(alpha_type: J2kChannelType) -> [J2kChannelDefinition; 4] {
    [
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
    ]
}

pub(super) fn wrap_rgba_jph(codestream: &[u8], alpha: Htj2kRgbaAlpha) -> Vec<u8> {
    let alpha_type = match alpha {
        Htj2kRgbaAlpha::Straight => J2kChannelType::Opacity,
        Htj2kRgbaAlpha::Premultiplied => J2kChannelType::PremultipliedOpacity,
    };
    let channel_definitions = rgba_channel_definitions(alpha_type);
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

pub(super) fn signed_gray16_fixture(samples: &[i16], width: u32, height: u32) -> Vec<u8> {
    let bytes = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    encode(
        &bytes,
        width,
        height,
        1,
        16,
        true,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode signed gray16")
}

pub(super) fn htj2k_native_fixture(
    components: u16,
    precision: u8,
    signed: bool,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let sample_count = width as usize * height as usize * components as usize;
    let bytes: Vec<u8> = if signed {
        let magnitude = 1_i32 << u32::from(precision - 1);
        (0..sample_count)
            .flat_map(|index| {
                let value = (i32::try_from(index).expect("small fixture") * 193 + 17)
                    % (magnitude * 2)
                    - magnitude;
                i16::try_from(value)
                    .expect("signed fixture value")
                    .to_le_bytes()
            })
            .collect()
    } else if precision <= 8 {
        (0..sample_count)
            .map(|index| {
                u8::try_from((index * 47 + 13) & 0xff).expect("masked 8-bit fixture value")
            })
            .collect()
    } else {
        let modulus = 1_u32 << u32::from(precision);
        (0..sample_count)
            .flat_map(|index| {
                u16::try_from((u32::try_from(index).expect("small fixture") * 977 + 31) % modulus)
                    .expect("unsigned fixture value")
                    .to_le_bytes()
            })
            .collect()
    };
    encode_htj2k(
        &bytes,
        width,
        height,
        components,
        precision,
        signed,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode native HTJ2K request fixture")
}
