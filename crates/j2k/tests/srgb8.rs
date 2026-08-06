// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    wrap_j2k_codestream, J2kDecoder, J2kError, J2kFileColorSpec, J2kFileWrapOptions, J2kSrgb8Layout,
};
use j2k_core::Colorspace;
use j2k_native::{encode, EncodeOptions};

fn encode_fixture(pixels: &[u8], width: u32, height: u32, components: u16) -> Vec<u8> {
    encode(
        pixels,
        width,
        height,
        components,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            ..EncodeOptions::default()
        },
    )
    .expect("encode fixture")
}

#[test]
fn srgb8_exposes_private_rgb_storage_through_accessors() {
    let pixels = [5, 17, 29, 101, 151, 211];
    let codestream = encode_fixture(&pixels, 2, 1, 3);

    let image = J2kDecoder::new(&codestream)
        .expect("decoder")
        .decode_srgb8()
        .expect("sRGB decode");

    assert_eq!(image.dimensions(), (2, 1));
    assert_eq!(image.layout(), J2kSrgb8Layout::Rgb);
    assert_eq!(image.data(), pixels);
    assert_eq!(image.into_data(), pixels);
}

#[test]
fn srgb8_preserves_srgb_gray_and_alpha_layouts() {
    let gray = [0, 63, 127, 255];
    let gray_codestream = encode_fixture(&gray, 2, 2, 1);
    let gray_image = J2kDecoder::new(&gray_codestream)
        .expect("gray decoder")
        .decode_srgb8()
        .expect("gray decode");
    assert_eq!(gray_image.layout(), J2kSrgb8Layout::Gray);
    assert_eq!(gray_image.data(), gray);

    let rgba = [11, 29, 47, 67, 89, 107, 131, 149];
    let rgba_codestream = encode_fixture(&rgba, 2, 1, 4);
    let jp2 = wrap_j2k_codestream(
        &rgba_codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb)),
    )
    .expect("wrap RGBA JP2");
    let rgba_image = J2kDecoder::new(&jp2)
        .expect("RGBA decoder")
        .decode_srgb8()
        .expect("RGBA decode");
    assert_eq!(rgba_image.layout(), J2kSrgb8Layout::Rgba);
    assert_eq!(rgba_image.data(), rgba);
}

#[test]
fn srgb8_rejects_a_malformed_primary_icc_profile() {
    let codestream = encode_fixture(&[17, 31, 47], 1, 1, 3);
    let jp2 = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(b"not-an-icc-profile")),
    )
    .expect("wrap malformed ICC fixture");

    let error = J2kDecoder::new(&jp2)
        .expect("container remains structurally decodable")
        .decode_srgb8()
        .expect_err("malformed ICC must not be treated as RGB");

    assert_eq!(error, J2kError::InvalidIccProfile);
}

#[test]
fn srgb8_converts_a_restricted_rgb_icc_profile() {
    let romm_rgb = [128, 64, 32, 200, 120, 80, 32, 160, 220];
    let codestream = encode_fixture(&romm_rgb, 3, 1, 3);
    let profile = moxcms::ColorProfile::new_pro_photo_rgb()
        .encode()
        .expect("encode restricted ProPhoto RGB profile");
    let jp2 = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(&profile)),
    )
    .expect("wrap restricted ICC fixture");

    let image = J2kDecoder::new(&jp2)
        .expect("ICC decoder")
        .decode_srgb8()
        .expect("ICC to sRGB conversion");

    // Independent Little CMS 2.17 reference values for this pinned profile.
    let reference = [191_u8, 53, 29, 255, 114, 89, 0, 192, 234];
    assert_eq!(image.layout(), J2kSrgb8Layout::Rgb);
    assert!(
        image
            .data()
            .iter()
            .zip(reference)
            .all(|(&actual, expected)| actual.abs_diff(expected) <= 2),
        "actual {:?}, reference {reference:?}",
        image.data()
    );
}

#[test]
fn srgb8_converts_a_restricted_monochrome_icc_profile() {
    let gamma_18 = [0_u8, 32, 64, 128, 192, 255];
    let codestream = encode_fixture(&gamma_18, 6, 1, 1);
    let profile = moxcms::ColorProfile::new_gray_with_gamma(1.8)
        .encode()
        .expect("encode restricted monochrome profile");
    let jp2 = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(&profile)),
    )
    .expect("wrap monochrome ICC fixture");

    let image = J2kDecoder::new(&jp2)
        .expect("ICC decoder")
        .decode_srgb8()
        .expect("monochrome ICC to sRGB-grey conversion");

    let reference = [0_u8, 43, 81, 146, 203, 255];
    assert_eq!(image.layout(), J2kSrgb8Layout::Gray);
    assert!(
        image
            .data()
            .iter()
            .zip(reference)
            .all(|(&actual, expected)| actual.abs_diff(expected) <= 1),
        "actual {:?}, reference {reference:?}",
        image.data()
    );
}
