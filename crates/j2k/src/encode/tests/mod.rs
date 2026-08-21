// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{
    encode_j2k_lossless, j2k_lossless_decomposition_levels_for_options, native_lossless_options,
    DecodeSettings, EncodeBackendPreference, Image, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kProgressionOrder, ReversibleTransform,
};

fn cod_mct(codestream: &[u8]) -> u8 {
    let cod_offset = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    codestream[cod_offset + 8]
}

#[test]
fn lossless_encode_can_disable_component_transform() {
    let pixels: Vec<u8> = (0..4 * 4 * 3)
        .map(|value| u8::try_from((value * 17) & 0xFF).expect("masked fixture byte"))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 4, 4, 3, 8, false).unwrap();
    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            block_coding_mode: J2kBlockCodingMode::Classic,
            progression: J2kProgressionOrder::Lrcp,
            max_decomposition_levels: Some(0),
            reversible_transform: ReversibleTransform::None53,
            validation: J2kEncodeValidation::CpuRoundTrip,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .unwrap();

    assert_eq!(cod_mct(&encoded.codestream), 0);
}

#[test]
fn explicit_decomposition_levels_override_default_lrcp_policy() {
    let pixels = vec![0; 128 * 128];
    let samples = J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).unwrap();

    let levels = j2k_lossless_decomposition_levels_for_options(
        samples,
        J2kLosslessEncodeOptions {
            block_coding_mode: J2kBlockCodingMode::Classic,
            progression: J2kProgressionOrder::Lrcp,
            max_decomposition_levels: Some(5),
            ..J2kLosslessEncodeOptions::default()
        },
    );

    assert_eq!(levels, 5);
}

#[test]
fn facade_native_options_skip_internal_ht_validation_for_external_validation() {
    let pixels = vec![0; 64 * 64];
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    let external = native_lossless_options(
        samples,
        J2kLosslessEncodeOptions {
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
            ..J2kLosslessEncodeOptions::default()
        },
    );
    let roundtrip = native_lossless_options(
        samples,
        J2kLosslessEncodeOptions {
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::CpuRoundTrip,
            ..J2kLosslessEncodeOptions::default()
        },
    );

    assert!(!external.validate_high_throughput_codestream);
    assert!(!roundtrip.validate_high_throughput_codestream);
}

#[test]
fn lossless_facade_roundtrips_four_component_via_public_api() {
    let width: u32 = 32;
    let height: u32 = 24;
    let components: u16 = 4;

    let mut pixels = Vec::with_capacity((width * height * u32::from(components)) as usize);
    for y in 0..height {
        for x in 0..width {
            for c in 0..u32::from(components) {
                let value = (x.wrapping_mul(7) ^ y.wrapping_mul(13)).wrapping_add(c * 41);
                pixels.push((value & 0xFF) as u8);
            }
        }
    }

    let samples = J2kLosslessSamples::new(&pixels, width, height, components, 8, false)
        .expect("4-component samples must be accepted by the public constructor");
    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            validation: J2kEncodeValidation::CpuRoundTrip,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("4-component CPU lossless encode must succeed");

    assert_eq!(encoded.components, components);
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native decode of 4-component codestream must construct")
        .decode_native()
        .expect("native decode of 4-component codestream must succeed");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.num_components, components);
    assert_eq!(decoded.bit_depth, 8);
    assert_eq!(
        decoded.data, pixels,
        "4-component pixels must round-trip exactly"
    );

    let two_component = vec![0u8; (width * height * 2) as usize];
    let two_component = J2kLosslessSamples::new(&two_component, width, height, 2, 8, false)
        .expect("2-component samples must be accepted by the public constructor");
    assert_eq!(two_component.components, 2);
}
