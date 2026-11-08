// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    wrap_j2k_codestream, J2kDecoder, J2kError, J2kFileColorSpec, J2kFileWrapOptions, J2kSrgb8Layout,
};
use j2k_core::Colorspace;
use j2k_native::{encode, EncodeOptions};

#[path = "support/codestream_corpus.rs"]
mod codestream_corpus;

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

fn premultiplied_rgba(color: J2kFileColorSpec<'_>) -> Vec<u8> {
    use j2k::{J2kChannelAssociation, J2kChannelDefinition, J2kChannelType, J2kFileBoxMetadata};
    let definitions = [
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
            channel_type: J2kChannelType::PremultipliedOpacity,
            association: J2kChannelAssociation::WholeImage,
        },
    ];
    let codestream = encode_fixture(&[64, 32, 16, 128, 7, 9, 11, 0], 2, 1, 4);
    wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2()
            .with_color(color)
            .with_metadata(J2kFileBoxMetadata {
                channel_definitions: &definitions,
                ..J2kFileBoxMetadata::empty()
            }),
    )
    .expect("wrap premultiplied alpha")
}

#[test]
fn srgb8_makes_premultiplied_alpha_straight_before_colour_conversion() {
    let plain = premultiplied_rgba(J2kFileColorSpec::Enumerated(Colorspace::SRgb));
    let mut ordinary = [0_u8; 8];
    J2kDecoder::new(&plain)
        .expect("decoder")
        .decode_into(&mut ordinary, 8, j2k::PixelFormat::Rgba8)
        .expect("ordinary output");
    assert_eq!(ordinary, [64, 32, 16, 128, 7, 9, 11, 0]);
    let normalized = J2kDecoder::new(&plain)
        .expect("decoder")
        .decode_srgb8()
        .expect("normalized");
    assert_eq!(normalized.data(), [128, 64, 32, 128, 0, 0, 0, 0]);

    let srgb = moxcms::ColorProfile::new_srgb()
        .encode()
        .expect("sRGB profile");
    let tagged_srgb = premultiplied_rgba(J2kFileColorSpec::IccProfile(&srgb));
    J2kDecoder::new(&tagged_srgb)
        .expect("sRGB ICC decoder")
        .decode_into(&mut ordinary, 8, j2k::PixelFormat::Rgba8)
        .expect("ordinary ICC output");
    assert_eq!(ordinary, [64, 32, 16, 128, 7, 9, 11, 0]);

    let profile = moxcms::ColorProfile::new_pro_photo_rgb()
        .encode()
        .expect("profile");
    let tagged = premultiplied_rgba(J2kFileColorSpec::IccProfile(&profile));
    let normalized = J2kDecoder::new(&tagged)
        .expect("decoder")
        .decode_srgb8()
        .expect("normalized ICC");
    // Same independent Little CMS reference as the RGB test, after undoing
    // association in the source profile's code values, before the ICC transform.
    for (&actual, expected) in normalized.data()[..3].iter().zip([191_u8, 53, 29]) {
        assert!(actual.abs_diff(expected) <= 2, "{actual} versus {expected}");
    }
    assert_eq!(normalized.data()[3..], [128, 0, 0, 0, 0]);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "direct, paletted and sub-8-bit sources share fixtures with the eight-bit palette case"
)]
fn srgb8_unassociates_premultiplied_samples_at_source_precision() {
    use j2k::{J2kFileBoxMetadata, J2kPaletteColumn, J2kPaletteMetadata, Rect};
    let samples = [128_u16, 256, 384, 512];
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let options = EncodeOptions {
        reversible: true,
        use_mct: false,
        num_decomposition_levels: 0,
        ..EncodeOptions::default()
    };
    let definitions: Vec<_> = (0..4_u16)
        .map(|channel| j2k::J2kChannelDefinition {
            channel_index: channel,
            channel_type: if channel == 3 {
                j2k::J2kChannelType::PremultipliedOpacity
            } else {
                j2k::J2kChannelType::Color
            },
            association: if channel == 3 {
                j2k::J2kChannelAssociation::WholeImage
            } else {
                j2k::J2kChannelAssociation::Color { index: channel + 1 }
            },
        })
        .collect();
    let direct = encode(&pixels, 1, 1, 4, 16, false, &options).expect("RGBA16");
    let direct = wrap_j2k_codestream(
        &direct,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb)),
    )
    .expect("RGBA16 container");
    let indices = encode(&[0], 1, 1, 1, 8, false, &options).expect("palette index");
    let palette = J2kPaletteMetadata {
        columns: vec![
            J2kPaletteColumn {
                bit_depth: 16,
                signed: false
            };
            4
        ],
        entries: vec![samples.into_iter().map(u64::from).collect()],
    };
    let paletted = wrap_j2k_codestream(
        &indices,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: Some(&palette),
                channel_definitions: &definitions,
                ..J2kFileBoxMetadata::empty()
            }),
    )
    .expect("RGBA16 palette");
    let sub8 = encode(&[63, 0, 0, 64], 1, 1, 4, 7, false, &options).expect("RGBA7");
    let sub8 = wrap_j2k_codestream(
        &sub8,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb)),
    )
    .expect("RGBA7 container");
    // Dividing original 16-bit samples gives straight RGB [64, 128, 191],
    // alpha 2; quantizing first would give [0, 128, 128]. Seven-bit
    // [63, 0, 0, 64] gives red 251 (quantizing first gives 249) and alpha
    // 64 / 127 of full scale, 129.
    let expected = [[64_u8, 128, 191, 2], [64, 128, 191, 2], [251, 0, 0, 129]];
    for (index, mut bytes) in [direct, paletted, sub8].into_iter().enumerate() {
        let cdef = bytes
            .windows(4)
            .position(|tag| tag == b"cdef")
            .expect("RGBA channel definitions");
        let alpha_type = cdef + 4 + 2 + 3 * 6 + 2;
        bytes[alpha_type..alpha_type + 2].copy_from_slice(&2_u16.to_be_bytes());
        if index == 0 {
            let mut native = [0_u8; 8];
            J2kDecoder::new(&bytes)
                .expect("decoder")
                .decode_into(&mut native, 8, j2k::PixelFormat::Rgba16)
                .expect("native packed precision");
            assert_eq!(
                native
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|sample| u16::from_ne_bytes(*sample))
                    .collect::<Vec<_>>(),
                samples
            );
        }
        let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
        assert_eq!(
            decoder.decode_srgb8().expect("normalized").data(),
            expected[index],
            "source {index}"
        );
        decoder
            .srgb8_region_decode_plan(Rect::full((1, 1)), 0)
            .expect("region plan");
        assert_eq!(
            decoder
                .decode_region_srgb8(Rect::full((1, 1)))
                .expect("normalized region")
                .data(),
            expected[index],
            "source {index}"
        );
    }

    // The source index precision is not the mapped colour precision. A 16-bit
    // index into an 8-bit RGBA palette still has lossless 8-bit unassociation.
    let index16 = encode(&0_u16.to_le_bytes(), 1, 1, 1, 16, false, &options).expect("index16");
    let eight_bit_palette = J2kPaletteMetadata {
        columns: vec![
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false
            };
            4
        ],
        entries: vec![vec![32, 64, 96, 128]],
    };
    let mut bytes = wrap_j2k_codestream(
        &index16,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: Some(&eight_bit_palette),
                channel_definitions: &definitions,
                ..J2kFileBoxMetadata::empty()
            }),
    )
    .expect("mapped8-bit colour");
    let cdef = bytes
        .windows(4)
        .position(|tag| tag == b"cdef")
        .expect("cdef");
    bytes[cdef + 4 + 2 + 3 * 6 + 2..cdef + 4 + 2 + 3 * 6 + 4].copy_from_slice(&2_u16.to_be_bytes());
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    assert_eq!(
        decoder
            .decode_srgb8()
            .expect("mapped precision controls support")
            .data(),
        [64, 128, 191, 128]
    );
    decoder
        .srgb8_region_decode_plan(Rect::full((1, 1)), 0)
        .expect("mapped8-bit region plan");
    assert_eq!(
        decoder
            .decode_region_srgb8(Rect::full((1, 1)))
            .expect("mapped8-bit region")
            .data(),
        [64, 128, 191, 128]
    );
}

#[test]
fn srgb8_unassociates_premultiplied_sycc_before_colour_conversion() {
    let codestream = encode_fixture(&[1, 128, 127, 2], 1, 1, 4);
    let mut sycc = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb)),
    )
    .expect("RGBA wrapper");
    let colr = sycc
        .windows(4)
        .position(|tag| tag == b"colr")
        .expect("colr");
    sycc[colr + 7..colr + 11].copy_from_slice(&18_u32.to_be_bytes());
    let cdef = sycc
        .windows(4)
        .position(|tag| tag == b"cdef")
        .expect("cdef");
    sycc[cdef + 4 + 2 + 3 * 6 + 2..cdef + 4 + 2 + 3 * 6 + 4].copy_from_slice(&2_u16.to_be_bytes());
    let mut decoder = J2kDecoder::new(&sycc).expect("sYCC decoder");
    // In source YCC, unassociation about the chroma midpoint before
    // conversion gives RGB [0, 219, 128]. Converting and quantizing first
    // instead loses the green fraction.
    assert_eq!(
        decoder.decode_srgb8().expect("normalized sYCC").data(),
        [0, 219, 128, 2]
    );
    let region = j2k::Rect::full((1, 1));
    decoder
        .srgb8_region_decode_plan(region, 0)
        .expect("sYCC region plan");
    assert_eq!(
        decoder
            .decode_region_srgb8(region)
            .expect("normalized sYCC region")
            .data(),
        [0, 219, 128, 2]
    );

    let codestream = encode_fixture(&[64, 32, 16, 128], 1, 1, 4);
    let colors = [
        J2kFileColorSpec::Enumerated(Colorspace::SRgb),
        J2kFileColorSpec::Enumerated(Colorspace::YCbCr),
    ];
    let mut rgb = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color_specs(&colors),
    )
    .expect("secondary colour box");
    let cdef = rgb.windows(4).position(|tag| tag == b"cdef").expect("cdef");
    rgb[cdef + 4 + 2 + 3 * 6 + 2..cdef + 4 + 2 + 3 * 6 + 4].copy_from_slice(&2_u16.to_be_bytes());
    let mut decoder = J2kDecoder::new(&rgb).expect("primary RGB decoder");
    assert_eq!(
        decoder
            .decode_srgb8()
            .expect("secondary sYCC does not select conversion")
            .data(),
        [128, 64, 32, 128]
    );
    decoder
        .srgb8_region_decode_plan(region, 0)
        .expect("primary RGB plan");
    assert_eq!(
        decoder
            .decode_region_srgb8(region)
            .expect("primary RGB region")
            .data(),
        [128, 64, 32, 128]
    );
}

/// `decode_srgb8` and `decode_srgb8_with_control` with a default control
/// decode the same images to the same output, and the controlled form never
/// refuses an image the plain one decodes. The corpus adds JP2 wrappers with
/// enumerated sRGB, a restricted ICC profile and premultiplied alpha.
#[test]
fn plain_and_default_controlled_srgb8_decodes_agree() {
    let mut corpus = codestream_corpus::parity_corpus();
    let profile = moxcms::ColorProfile::new_pro_photo_rgb()
        .encode()
        .expect("encode restricted ProPhoto RGB profile");
    let rgb = encode_fixture(&[128, 64, 32, 200, 120, 80, 32, 160, 220], 3, 1, 3);
    for (name, color) in [
        ("srgb jp2", J2kFileColorSpec::Enumerated(Colorspace::SRgb)),
        ("icc jp2", J2kFileColorSpec::IccProfile(&profile)),
    ] {
        corpus.push((
            name.into(),
            wrap_j2k_codestream(&rgb, J2kFileWrapOptions::jp2().with_color(color))
                .expect("wrap RGB fixture"),
        ));
    }
    corpus.push((
        "premultiplied srgb jp2".into(),
        premultiplied_rgba(J2kFileColorSpec::Enumerated(Colorspace::SRgb)),
    ));
    let control = j2k::J2kDecodeControl::default();
    let (mut compared, mut both_refused) = (0, 0);
    for (name, bytes) in corpus {
        let Ok(mut plain) = J2kDecoder::new(&bytes) else {
            continue;
        };
        let plain = plain.decode_srgb8();
        let controlled = J2kDecoder::new(&bytes)
            .expect("decoder")
            .decode_srgb8_with_control(&control);
        match (plain, controlled) {
            (Ok(plain), Ok(controlled)) => {
                assert_eq!(plain, controlled, "{name}");
                compared += 1;
            }
            (Err(plain), Err(controlled)) => {
                assert_eq!(
                    core::mem::discriminant(&plain),
                    core::mem::discriminant(&controlled),
                    "{name}: {plain} | {controlled}"
                );
                both_refused += 1;
            }
            (Ok(_), Err(error)) => panic!("{name}: controlled refuses a plain decode: {error}"),
            (Err(error), Ok(_)) => panic!("{name}: plain refuses a controlled decode: {error}"),
        }
    }
    assert!(
        compared > 10,
        "compared {compared}, both refused {both_refused}"
    );
}
