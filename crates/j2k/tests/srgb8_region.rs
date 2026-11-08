// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    wrap_j2k_codestream, BatchAlpha, J2kChannelAssociation, J2kChannelDefinition, J2kChannelType,
    J2kDecoder, J2kError, J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions,
    J2kSampleInterpretation, J2kSrgb8Layout, Rect,
};
use j2k_core::Colorspace;
use j2k_native::{encode, EncodeOptions};
use moxcms::ColorProfile;

const SIZE: (u32, u32) = (63, 45);
const ROI: Rect = Rect {
    x: 7,
    y: 5,
    w: 19,
    h: 17,
};

fn fixture(channels: u16, color: J2kFileColorSpec<'_>, alpha: Option<J2kChannelType>) -> Vec<u8> {
    let pixels: Vec<u8> = (0..SIZE.0 * SIZE.1 * u32::from(channels))
        .map(|i| u8::try_from((i * 29 + i / 7) % 256).expect("byte"))
        .collect();
    let codestream = encode(
        &pixels,
        SIZE.0,
        SIZE.1,
        channels,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 3,
            reversible: true,
            ..EncodeOptions::default()
        },
    )
    .expect("codestream");
    let mut definitions = Vec::new();
    if let Some(alpha) = alpha {
        for channel in 0..channels - 1 {
            definitions.push(J2kChannelDefinition {
                channel_index: channel,
                channel_type: J2kChannelType::Color,
                association: J2kChannelAssociation::Color { index: channel + 1 },
            });
        }
        definitions.push(J2kChannelDefinition {
            channel_index: channels - 1,
            channel_type: alpha,
            association: J2kChannelAssociation::WholeImage,
        });
    }
    wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2()
            .with_color(color)
            .with_metadata(J2kFileBoxMetadata {
                channel_definitions: &definitions,
                ..J2kFileBoxMetadata::empty()
            }),
    )
    .expect("JP2")
}

fn crop(data: &[u8], size: (u32, u32), region: Rect, channels: usize) -> Vec<u8> {
    let mut result = Vec::new();
    for y in region.y..region.y + region.h {
        let start = (y as usize * size.0 as usize + region.x as usize) * channels;
        result.extend_from_slice(&data[start..start + region.w as usize * channels]);
    }
    result
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one matrix compares native alpha metadata with explicit normalization across layouts and precision"
)]
fn ordinary_region_plans_distinguish_straight_and_premultiplied_alpha() {
    use j2k::{J2kScratchPool, PixelFormat};
    let straight_bytes = fixture(
        4,
        J2kFileColorSpec::Enumerated(Colorspace::SRgb),
        Some(J2kChannelType::Opacity),
    );
    let premultiplied_bytes = fixture(
        4,
        J2kFileColorSpec::Enumerated(Colorspace::SRgb),
        Some(J2kChannelType::PremultipliedOpacity),
    );
    let mut straight_decoder = J2kDecoder::new(&straight_bytes).expect("straight decoder");
    let mut premultiplied_decoder =
        J2kDecoder::new(&premultiplied_bytes).expect("premultiplied decoder");
    let straight = straight_decoder
        .region_decode_plan(PixelFormat::Rgba8, ROI, 0)
        .expect("straight plan");
    let premultiplied = premultiplied_decoder
        .region_decode_plan(PixelFormat::Rgba8, ROI, 0)
        .expect("premultiplied plan");
    assert_ne!(
        straight, premultiplied,
        "packed plans must distinguish their alpha convention"
    );
    assert_eq!(straight.alpha, BatchAlpha::Straight);
    assert_eq!(premultiplied.alpha, BatchAlpha::Premultiplied);
    assert_eq!(
        premultiplied_decoder
            .region_decode_plan(PixelFormat::Rgba16, ROI, 0)
            .expect("native-width alpha")
            .alpha,
        BatchAlpha::Premultiplied
    );
    assert_eq!(
        premultiplied_decoder
            .srgb8_region_decode_plan(ROI, 0)
            .expect("normalized alpha")
            .alpha,
        BatchAlpha::Straight
    );
    assert_eq!(
        straight_decoder
            .srgb8_region_decode_plan(ROI, 0)
            .expect("normalized straight alpha")
            .alpha,
        BatchAlpha::Straight
    );

    let rgb_plan = premultiplied_decoder
        .region_decode_plan(PixelFormat::Rgb8, ROI, 0)
        .expect("drop alpha plan");
    assert_eq!(rgb_plan.alpha, BatchAlpha::None);
    let mut rgba = vec![0_u8; premultiplied.buffer_bytes];
    let mut rgb = vec![0_u8; rgb_plan.buffer_bytes];
    let mut pool = J2kScratchPool::new();
    premultiplied_decoder
        .decode_region_into(
            &mut pool,
            &mut rgba,
            premultiplied.row_bytes,
            PixelFormat::Rgba8,
            ROI,
        )
        .expect("native RGBA");
    premultiplied_decoder
        .decode_region_into(
            &mut pool,
            &mut rgb,
            rgb_plan.row_bytes,
            PixelFormat::Rgb8,
            ROI,
        )
        .expect("native RGB drops alpha");
    for (rgba, rgb) in rgba.as_chunks::<4>().0.iter().zip(rgb.as_chunks::<3>().0) {
        assert_eq!(
            &rgba[..3],
            rgb,
            "dropping alpha must not unassociate colour"
        );
    }

    let opaque_bytes = fixture(3, J2kFileColorSpec::Enumerated(Colorspace::SRgb), None);
    let mut opaque_decoder = J2kDecoder::new(&opaque_bytes).expect("opaque decoder");
    for format in [PixelFormat::Rgba8, PixelFormat::Rgba16] {
        assert_eq!(
            opaque_decoder
                .region_decode_plan(format, ROI, 0)
                .expect("opaque alpha fill")
                .alpha,
            BatchAlpha::Straight
        );
    }
    assert_eq!(
        opaque_decoder
            .srgb8_region_decode_plan(ROI, 0)
            .expect("no normalized alpha")
            .alpha,
        BatchAlpha::None
    );
    let gray = fixture(1, J2kFileColorSpec::Enumerated(Colorspace::SGray), None);
    assert_eq!(
        J2kDecoder::new(&gray)
            .expect("gray decoder")
            .region_decode_plan(PixelFormat::Gray8, ROI, 0)
            .expect("gray plan")
            .alpha,
        BatchAlpha::None
    );

    let samples: Vec<u8> = [128_u16, 256, 384, 512]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let codestream = encode(
        &samples,
        1,
        1,
        4,
        16,
        false,
        &EncodeOptions {
            reversible: true,
            use_mct: false,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        },
    )
    .expect("RGBA16 source");
    let mut wide = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb)),
    )
    .expect("RGBA16 container");
    let cdef = wide
        .windows(4)
        .position(|tag| tag == b"cdef")
        .expect("channel definitions");
    wide[cdef + 4 + 2 + 3 * 6 + 2..cdef + 4 + 2 + 3 * 6 + 4].copy_from_slice(&2_u16.to_be_bytes());
    let mut wide_decoder = J2kDecoder::new(&wide).expect("RGBA16 decoder");
    let pixel = Rect::full((1, 1));
    assert_eq!(
        wide_decoder
            .region_decode_plan(PixelFormat::Rgba16, pixel, 0)
            .expect("native RGBA16 remains supported")
            .alpha,
        BatchAlpha::Premultiplied
    );
    // Normalization unassociates the 16-bit source values before quantizing.
    assert_eq!(
        wide_decoder
            .srgb8_region_decode_plan(pixel, 0)
            .expect("premultiplied RGBA16 sRGB plan")
            .alpha,
        BatchAlpha::Straight
    );
    assert_eq!(
        wide_decoder
            .decode_region_srgb8(pixel)
            .expect("premultiplied RGBA16 sRGB region")
            .data(),
        [64, 128, 191, 2]
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one cross-colour matrix compares region and full normalization at every tested reduction"
)]
fn normalized_regions_match_correctly_reduced_full_reference() {
    let rgb_profile = ColorProfile::new_pro_photo_rgb()
        .encode()
        .expect("RGB profile");
    let gray_profile = ColorProfile::new_gray_with_gamma(1.8)
        .encode()
        .expect("gray profile");
    for (channels, color, alpha, layout, output_channels) in [
        (
            1,
            J2kFileColorSpec::Enumerated(Colorspace::SGray),
            None,
            J2kSrgb8Layout::Gray,
            1,
        ),
        (
            3,
            J2kFileColorSpec::Enumerated(Colorspace::SRgb),
            None,
            J2kSrgb8Layout::Rgb,
            3,
        ),
        (
            3,
            J2kFileColorSpec::IccProfile(&rgb_profile),
            None,
            J2kSrgb8Layout::Rgb,
            3,
        ),
        (
            1,
            J2kFileColorSpec::IccProfile(&gray_profile),
            None,
            J2kSrgb8Layout::Gray,
            1,
        ),
        (
            2,
            J2kFileColorSpec::IccProfile(&gray_profile),
            Some(J2kChannelType::Opacity),
            J2kSrgb8Layout::Rgba,
            4,
        ),
        (
            2,
            J2kFileColorSpec::IccProfile(&gray_profile),
            Some(J2kChannelType::PremultipliedOpacity),
            J2kSrgb8Layout::Rgba,
            4,
        ),
        (
            4,
            J2kFileColorSpec::IccProfile(&rgb_profile),
            Some(J2kChannelType::Opacity),
            J2kSrgb8Layout::Rgba,
            4,
        ),
        (
            4,
            J2kFileColorSpec::IccProfile(&rgb_profile),
            Some(J2kChannelType::PremultipliedOpacity),
            J2kSrgb8Layout::Rgba,
            4,
        ),
    ] {
        let bytes = fixture(channels, color, alpha);
        for levels in [0, 1, 2, 3] {
            let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
            let plan = decoder
                .srgb8_region_decode_plan(ROI, levels)
                .expect("normalized plan");
            assert_eq!(plan.sample_interpretation, J2kSampleInterpretation::Srgb8);
            let full = decoder
                .decode_region_srgb8_at_reduction(Rect::full(SIZE), levels)
                .expect("full reduced reference");
            let region = decoder
                .decode_region_srgb8_at_reduction(ROI, levels)
                .expect("normalized region");
            assert_eq!(
                region.dimensions(),
                (plan.output_region.w, plan.output_region.h)
            );
            assert_eq!(region.layout(), layout);
            assert_eq!(
                region.alpha(),
                if alpha.is_some() {
                    BatchAlpha::Straight
                } else {
                    BatchAlpha::None
                }
            );
            assert_eq!(region.data().len(), plan.buffer_bytes);
            assert_eq!(plan.alpha, region.alpha());
            assert_eq!(
                region.data(),
                crop(
                    full.data(),
                    full.dimensions(),
                    plan.output_region,
                    output_channels
                ),
                "{channels} channels at reduction {levels}"
            );
            if levels == 0 {
                let ordinary_full = decoder.decode_srgb8().expect("existing full normalization");
                assert_eq!(full, ordinary_full);
                assert_eq!(
                    region,
                    decoder
                        .decode_region_srgb8(ROI)
                        .expect("region convenience")
                );
                if channels == 2 && alpha == Some(J2kChannelType::PremultipliedOpacity) {
                    let pixel = decoder
                        .decode_region_srgb8(Rect {
                            x: 1,
                            y: 0,
                            w: 1,
                            h: 1,
                        })
                        .expect("gray-alpha pixel");
                    // Source 58 / alpha 87 unassociates to 170. Gamma 1.8 followed
                    // by the sRGB transfer function gives 184 independently.
                    assert!(pixel.data()[..3]
                        .iter()
                        .all(|&sample| sample.abs_diff(184) <= 1));
                    assert_eq!(pixel.data()[3], 87);
                }
            }
        }
    }
}

#[test]
fn normalized_region_preserves_profile_errors_and_rejects_unrepresentable_alpha() {
    let malformed = fixture(3, J2kFileColorSpec::IccProfile(b"invalid ICC"), None);
    let mut decoder = J2kDecoder::new(&malformed).expect("container");
    assert_eq!(
        decoder
            .srgb8_region_decode_plan(ROI, 0)
            .expect_err("invalid profile plan"),
        J2kError::InvalidIccProfile
    );
    assert_eq!(
        decoder
            .decode_region_srgb8(ROI)
            .expect_err("invalid profile"),
        J2kError::InvalidIccProfile
    );

    let rgb_profile = ColorProfile::new_srgb().encode().expect("RGB profile");
    // Too few color channels remain invalid; extra unspecified channels
    // without a cdef box are allowed and omitted from display output.
    let mismatched = fixture(1, J2kFileColorSpec::IccProfile(&rgb_profile), None);
    let mut decoder = J2kDecoder::new(&mismatched).expect("mismatched container");
    assert!(matches!(
        decoder.srgb8_region_decode_plan(ROI, 0),
        Err(J2kError::Unsupported(_))
    ));
    assert!(matches!(
        decoder.decode_region_srgb8(ROI),
        Err(J2kError::Unsupported(_))
    ));

    let bytes = fixture(
        4,
        J2kFileColorSpec::Enumerated(Colorspace::SRgb),
        Some(J2kChannelType::Opacity),
    );
    // A single opacity associated with only red cannot be represented by one
    // RGBA alpha applying to every colour. Keep the ordinary/native API separate.
    let mut per_channel = bytes.clone();
    let cdef = per_channel
        .windows(4)
        .position(|tag| tag == b"cdef")
        .expect("cdef");
    let association = cdef + 4 + 2 + 3 * 6 + 4;
    per_channel[association..association + 2].copy_from_slice(&1_u16.to_be_bytes());
    let mut decoder = J2kDecoder::new(&per_channel).expect("per-channel container");
    assert!(matches!(
        decoder.decode_region_srgb8(ROI),
        Err(J2kError::Unsupported(_))
    ));
    assert!(matches!(
        decoder.decode_srgb8(),
        Err(J2kError::Unsupported(_))
    ));

    let mut mixed = bytes;
    let blue_definition = cdef + 4 + 2 + 2 * 6;
    mixed[blue_definition + 2..blue_definition + 4].copy_from_slice(&2_u16.to_be_bytes());
    mixed[blue_definition + 4..blue_definition + 6].copy_from_slice(&0_u16.to_be_bytes());
    let mut decoder = J2kDecoder::new(&mixed).expect("mixed alpha container");
    assert!(matches!(
        decoder.decode_region_srgb8(ROI),
        Err(J2kError::Unsupported(_))
    ));
    assert!(matches!(
        decoder.decode_srgb8(),
        Err(J2kError::Unsupported(_))
    ));
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "paired palette and subsampled fixtures verify the same region support-window contract"
)]
fn normalized_regions_preserve_palette_and_subsampled_sycc_support_windows() {
    use j2k::{J2kPaletteColumn, J2kPaletteMetadata};
    use j2k_native::{encode_component_planes_53, EncodeComponentPlane};

    let indices: Vec<u8> = (0..SIZE.0 * SIZE.1)
        .map(|i| u8::try_from(128 + i % 4).expect("index"))
        .collect();
    let options = EncodeOptions {
        num_decomposition_levels: 3,
        reversible: true,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let indexed = encode(&indices, SIZE.0, SIZE.1, 1, 8, false, &options).expect("indices");
    let palette = J2kPaletteMetadata {
        columns: vec![
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false
            };
            3
        ],
        // Reduced wavelet samples need not land on the four original indices.
        // Every eight-bit index remains defined for the reduced reference.
        entries: (0..256_u64)
            .map(|i| vec![(i + 140) % 256, (i + 173) % 256, (i + 206) % 256])
            .collect(),
    };
    let paletted = wrap_j2k_codestream(
        &indexed,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: Some(&palette),
                ..J2kFileBoxMetadata::empty()
            }),
    )
    .expect("palette");

    let luma: Vec<u8> = (0..SIZE.0 * SIZE.1)
        .map(|i| u8::try_from(32 + i % 192).expect("luma"))
        .collect();
    let blue = vec![96_u8; (SIZE.0.div_ceil(2) * SIZE.1.div_ceil(2)) as usize];
    let red: Vec<u8> = (0..blue.len())
        .map(|i| u8::try_from(80 + i % 96).expect("chroma"))
        .collect();
    let sampled = encode_component_planes_53(
        &[
            EncodeComponentPlane {
                data: &luma,
                x_rsiz: 1,
                y_rsiz: 1,
            },
            EncodeComponentPlane {
                data: &blue,
                x_rsiz: 2,
                y_rsiz: 2,
            },
            EncodeComponentPlane {
                data: &red,
                x_rsiz: 2,
                y_rsiz: 2,
            },
        ],
        SIZE.0,
        SIZE.1,
        8,
        false,
        &options,
    )
    .expect("subsampled planes");
    let sycc = wrap_j2k_codestream(
        &sampled,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::Enumerated(Colorspace::YCbCr)),
    )
    .expect("sYCC");

    for (bytes, first_pixel) in [(paletted, [12, 45, 78]), (sycc, [0, 77, 0])] {
        for levels in [0, 1, 2] {
            let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
            let plan = decoder
                .srgb8_region_decode_plan(ROI, levels)
                .expect("colour plan");
            let full = decoder
                .decode_region_srgb8_at_reduction(Rect::full(SIZE), levels)
                .unwrap_or_else(|error| {
                    panic!("full colour reference at {levels} levels: {error:?}")
                });
            if levels == 0 {
                assert_eq!(full.data()[..3], first_pixel);
            }
            let region = decoder
                .decode_region_srgb8_at_reduction(ROI, levels)
                .expect("region colour conversion");
            assert_eq!(
                region.data(),
                crop(full.data(), full.dimensions(), plan.output_region, 3),
                "reduction {levels}"
            );
        }
    }

    // A valid full-resolution indexed image can reconstruct reduced values
    // outside a short palette. Preserve that reduced-data error rather than
    // falling back to a crop of the valid full-resolution normalization.
    let short_indices: Vec<u8> = indices.iter().map(|index| index - 128).collect();
    let indexed = encode(&short_indices, SIZE.0, SIZE.1, 1, 8, false, &options)
        .expect("short palette indices");
    let short_palette = J2kPaletteMetadata {
        columns: palette.columns,
        entries: vec![
            vec![12, 45, 78],
            vec![210, 180, 70],
            vec![0, 127, 255],
            vec![254, 30, 90],
        ],
    };
    let bytes = wrap_j2k_codestream(
        &indexed,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: Some(&short_palette),
                ..J2kFileBoxMetadata::empty()
            }),
    )
    .expect("short palette");
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    decoder
        .decode_srgb8()
        .expect("valid full-resolution palette");
    let full_error = decoder
        .decode_region_srgb8_at_reduction(Rect::full(SIZE), 1)
        .expect_err("reduced palette indices");
    assert!(matches!(full_error, J2kError::NativeDecode { .. }));
    let edge = Rect {
        x: 0,
        y: 0,
        w: 19,
        h: 17,
    };
    assert_eq!(
        decoder
            .decode_region_srgb8_at_reduction(edge, 1)
            .expect_err("reduced region palette indices"),
        full_error
    );
    decoder
        .decode_region_srgb8(edge)
        .expect("full-resolution region remains valid");
}

#[test]
fn normalized_regions_preserve_channel_reordering_and_lenient_warnings() {
    let bytes = fixture(
        4,
        J2kFileColorSpec::Enumerated(Colorspace::SRgb),
        Some(J2kChannelType::Opacity),
    );
    let cdef = bytes
        .windows(4)
        .position(|tag| tag == b"cdef")
        .expect("cdef");
    let mut reordered = bytes.clone();
    for (entry, source) in [2_u16, 0, 1, 3].into_iter().enumerate() {
        let offset = cdef + 4 + 2 + entry * 6;
        reordered[offset..offset + 2].copy_from_slice(&source.to_be_bytes());
    }
    let first = J2kDecoder::new(&reordered)
        .expect("reordered source")
        .decode_region_srgb8(Rect {
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        })
        .expect("reordered region");
    assert_eq!(first.data(), [58, 0, 29, 87]);
    assert_eq!(first.alpha(), BatchAlpha::Straight);

    let mut inferred = bytes;
    inferred[cdef..cdef + 4].copy_from_slice(b"free");
    let mut decoder = J2kDecoder::new_with_settings(&inferred, j2k::DecodeSettings::lenient())
        .expect("lenient source");
    let full = decoder.decode_srgb8().expect("full lenient");
    let region = decoder.decode_region_srgb8(ROI).expect("region lenient");
    assert_eq!(region.warnings(), full.warnings());
    assert_eq!(
        region.warnings(),
        [j2k::J2kDecodeWarning::LenientMetadataRecovery]
    );
    assert_eq!(region.data(), crop(full.data(), full.dimensions(), ROI, 4));
}

#[test]
fn normalized_region_controls_preserve_output_cancellation_and_bounded_cms_limits() {
    use core::num::NonZeroUsize;
    use j2k::{J2kCancellationToken, J2kDecodeControl};

    let profile = ColorProfile::new_pro_photo_rgb().encode().expect("profile");
    let bytes = fixture(3, J2kFileColorSpec::IccProfile(&profile), None);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let expected = decoder
        .decode_region_srgb8_at_reduction(ROI, 1)
        .expect("uncontrolled");
    let mut control = J2kDecodeControl::default();
    control.max_output_bytes = Some(expected.data().len() - 1);
    assert!(matches!(
        decoder.decode_region_srgb8_at_reduction_with_control(ROI, 1, &control),
        Err(J2kError::Buffer(
            j2k_core::BufferError::AllocationTooLarge { .. }
        ))
    ));
    control.max_output_bytes = Some(expected.data().len());
    assert_eq!(
        decoder
            .decode_region_srgb8_at_reduction_with_control(ROI, 1, &control)
            .expect("exact output cap"),
        expected
    );

    let token = J2kCancellationToken::new();
    token.cancel();
    control.cancellation = Some(token);
    assert_eq!(
        decoder
            .decode_region_srgb8_at_reduction_with_control(ROI, 1, &control)
            .expect_err("cancelled"),
        J2kError::Cancelled
    );
    assert_eq!(
        decoder
            .decode_region_srgb8_at_reduction(ROI, 1)
            .expect("next independent operation"),
        expected
    );

    // A matrix/TRC profile's colour-management state is charged against a
    // working cap: a generous one converts, a tiny one refuses before it.
    control = J2kDecodeControl::default();
    control.max_working_memory_bytes = NonZeroUsize::new(usize::MAX);
    assert_eq!(
        decoder
            .decode_region_srgb8_at_reduction_with_control(ROI, 1, &control)
            .expect("bounded ICC conversion"),
        expected
    );
    let mut tiny = J2kDecodeControl::default();
    tiny.max_working_memory_bytes = NonZeroUsize::new(64 << 10);
    assert!(matches!(
        decoder.decode_region_srgb8_at_reduction_with_control(ROI, 1, &tiny),
        Err(J2kError::ResourceLimit { .. })
    ));
    let malformed = fixture(3, J2kFileColorSpec::IccProfile(b"invalid ICC"), None);
    assert!(matches!(
        J2kDecoder::new(&malformed)
            .expect("container")
            .decode_region_srgb8_at_reduction_with_control(ROI, 0, &control),
        Err(J2kError::Unsupported(_))
    ));

    let plain = fixture(3, J2kFileColorSpec::Enumerated(Colorspace::SRgb), None);
    let mut decoder = J2kDecoder::new(&plain).expect("plain decoder");
    control.max_working_memory_bytes = NonZeroUsize::new(1);
    assert!(matches!(
        decoder.decode_region_srgb8_at_reduction_with_control(ROI, 0, &control),
        Err(J2kError::ResourceLimit { .. })
    ));
    let expected = decoder
        .decode_region_srgb8(ROI)
        .expect("refusal does not poison later operations");
    control.max_working_memory_bytes = NonZeroUsize::new(j2k_native::DEFAULT_MAX_DECODE_BYTES);
    assert_eq!(
        decoder
            .decode_region_srgb8_at_reduction_with_control(ROI, 0, &control)
            .expect("bounded plain normalization"),
        expected
    );
}

#[test]
fn normalized_icc_regions_with_odd_origins_match_openjpeg_reductions() {
    use j2k_test_support::{
        OPENJPEG_ORIGIN_23X17_AT_3_5, OPENJPH_ORIGIN_23X17_AT_3_5, ORIGIN_REDUCED_RGB,
    };
    let profile = ColorProfile::new_srgb().encode().expect("sRGB profile");
    let source = Rect {
        x: 1,
        y: 2,
        w: 13,
        h: 11,
    };
    for (codestream, options) in [
        (OPENJPEG_ORIGIN_23X17_AT_3_5, J2kFileWrapOptions::jp2()),
        (OPENJPH_ORIGIN_23X17_AT_3_5, J2kFileWrapOptions::jph()),
    ] {
        let bytes = wrap_j2k_codestream(
            codestream,
            options.with_color(J2kFileColorSpec::IccProfile(&profile)),
        )
        .expect("ICC wrapper");
        let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
        assert_eq!(decoder.reference_grid().image_origin, (3, 5));
        for (levels, (dimensions, reference)) in (1_u8..).zip(ORIGIN_REDUCED_RGB) {
            let plan = decoder
                .srgb8_region_decode_plan(source, levels)
                .expect("odd-origin plan");
            let result = decoder
                .decode_region_srgb8_at_reduction(source, levels)
                .expect("odd-origin normalized region");
            assert_eq!(plan.reduced_image_dimensions, dimensions);
            let expected = crop(reference, dimensions, plan.output_region, 3);
            assert_eq!(result.data().len(), expected.len());
            assert!(
                result
                    .data()
                    .iter()
                    .zip(expected)
                    .all(|(&actual, expected)| actual.abs_diff(expected) <= 1),
                "reduction {levels}"
            );
        }
    }
}

#[test]
fn normalized_bitmap_regions_follow_high_precision_native_capabilities() {
    let dimensions = (8, 8);
    let roi = Rect {
        x: 1,
        y: 2,
        w: 4,
        h: 3,
    };
    let maximum = (1_u32 << 25) - 1;
    let pixels: Vec<u8> = (0..64_u32)
        .flat_map(|index| (index * (maximum / 63)).to_le_bytes())
        .collect();
    for reversible in [false, true] {
        let bytes = encode(
            &pixels,
            dimensions.0,
            dimensions.1,
            1,
            25,
            false,
            &EncodeOptions {
                reversible,
                use_mct: false,
                num_decomposition_levels: 2,
                ..EncodeOptions::default()
            },
        )
        .expect("valid 25-bit image");
        let native = j2k_native::Image::new(&bytes, &j2k_native::DecodeSettings::strict())
            .expect("native image");
        let full = native
            .decode_with_context(&mut j2k_native::DecoderContext::default())
            .expect("full 25-bit bitmap");
        let mut decoder = J2kDecoder::new(&bytes).expect("facade");
        assert_eq!(
            decoder
                .decode_srgb8()
                .expect("full 25-bit normalization")
                .data(),
            full.data
        );
        let mut control = j2k::J2kDecodeControl::default();
        control.max_working_memory_bytes =
            core::num::NonZeroUsize::new(j2k_native::DEFAULT_MAX_DECODE_BYTES);
        assert_eq!(
            decoder
                .decode_srgb8_with_control(&control)
                .expect("controlled full 25-bit normalization")
                .data(),
            full.data
        );
        for levels in 0..=2 {
            let reduced = j2k_native::Image::new_with_reduction(
                &bytes,
                &j2k_native::DecodeSettings::strict(),
                levels,
            )
            .expect("exact reduced image");
            let full_reduced = reduced
                .decode_with_context(&mut j2k_native::DecoderContext::default())
                .expect("full reduced bitmap");
            let plan = decoder
                .srgb8_region_decode_plan(roi, levels)
                .expect("25-bit bounded region plan");
            let output = plan.output_region;
            let native_region = reduced
                .decode_region((output.x, output.y, output.w, output.h))
                .expect("25-bit bounded bitmap");
            assert_eq!(
                native_region.data,
                crop(
                    &full_reduced.data,
                    (reduced.width(), reduced.height()),
                    output,
                    1
                )
            );
            let region = decoder
                .decode_region_srgb8_at_reduction_with_control(roi, levels, &control)
                .expect("25-bit bounded normalization");
            assert_eq!(region.data(), native_region.data);
            assert_eq!(region.data().len(), plan.buffer_bytes);
        }
    }
}

#[test]
fn normalized_regions_of_subsampled_exact_integer_images_match_the_full_image() {
    // Twenty-five-bit samples take the exact integer route; the half-sampled
    // second and third components keep the region outside the window planner.
    let (width, height) = (40_u32, 30_u32);
    let full_plane = (0..width * height)
        .flat_map(|index| ((index * 104_729) & ((1 << 25) - 1)).to_le_bytes())
        .collect::<Vec<_>>();
    let half = |seed: u32| {
        (0..width.div_ceil(2) * height.div_ceil(2))
            .flat_map(move |index| ((index * seed) & ((1 << 25) - 1)).to_le_bytes())
            .collect::<Vec<_>>()
    };
    let (second, third) = (half(500_009), half(700_001));
    let plane = |data, sampling| j2k_native::EncodeTypedComponentPlane {
        data,
        x_rsiz: sampling,
        y_rsiz: sampling,
        bit_depth: 25,
        signed: false,
    };
    let codestream = j2k_native::encode_typed_component_planes_53(
        &[plane(&full_plane, 1), plane(&second, 2), plane(&third, 2)],
        width,
        height,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            tile_size: Some((16, 16)),
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("sampled 25-bit RGB");
    let bytes = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb)),
    )
    .expect("sampled RGB container");
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let full = decoder.decode_srgb8().expect("full normalized image");
    let region = Rect {
        x: 5,
        y: 3,
        w: 17,
        h: 13,
    };
    decoder
        .srgb8_region_decode_plan(region, 0)
        .expect("sampled exact-integer region plan");
    assert_eq!(
        decoder
            .decode_region_srgb8(region)
            .expect("sampled exact-integer region")
            .data(),
        crop(full.data(), (width, height), region, 3)
    );
}
