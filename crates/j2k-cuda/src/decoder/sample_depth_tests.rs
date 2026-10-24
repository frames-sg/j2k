// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    prepare_batch, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeRequest, DeviceDecodeRequest, EncodedImage,
};
use j2k_core::{Downscale, PixelFormat, Rect};

use super::color_batch::native_batch::{
    submit_native_color_resident_prepared_batch, NativeColorBatchInput,
};
use super::grayscale_batch::{
    decode_grayscale_cuda_resident_prepared_batch_surfaces_with_profile, GrayscaleBatchInput,
};
use super::{CudaSession, DecodeSettings, J2kDecoder};

const SIZE: u32 = 32;

fn source(
    components: u16,
    bits: u8,
    signed: bool,
    mct: bool,
    ht: bool,
    tile_size: Option<(u32, u32)>,
) -> Vec<u8> {
    let mask = (1_u32 << bits) - 1;
    let pixels = (0..SIZE * SIZE * u32::from(components))
        .flat_map(|i| {
            let raw = match i % 8 {
                0 => 1_u32 << (bits - 1),
                1 => mask,
                2 => 0,
                3 => (1_u32 << (bits - 1)) - 1,
                _ => (i * 113) & mask,
            };
            let bytes = raw.to_le_bytes();
            bytes[..if bits <= 8 { 1 } else { 2 }].to_vec()
        })
        .collect::<Vec<_>>();
    let encode = if ht {
        j2k_native::encode_htj2k
    } else {
        j2k_native::encode
    };
    encode(
        &pixels,
        SIZE,
        SIZE,
        components,
        bits,
        signed,
        &j2k_native::EncodeOptions {
            reversible: true,
            use_mct: mct,
            num_decomposition_levels: 2,
            tile_size,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("encode sample boundaries")
}

fn with_alpha(encoded: Vec<u8>, components: u16, ht: bool) -> Vec<u8> {
    if components != 4 {
        return encoded;
    }
    let definitions = core::array::from_fn::<_, 4, _>(|channel| j2k::J2kChannelDefinition {
        channel_index: u16::try_from(channel).expect("channel"),
        channel_type: if channel == 3 {
            j2k::J2kChannelType::Opacity
        } else {
            j2k::J2kChannelType::Color
        },
        association: if channel == 3 {
            j2k::J2kChannelAssociation::WholeImage
        } else {
            j2k::J2kChannelAssociation::Color {
                index: u16::try_from(channel + 1).expect("channel"),
            }
        },
    });
    j2k::wrap_j2k_codestream(
        &encoded,
        if ht {
            j2k::J2kFileWrapOptions::jph()
        } else {
            j2k::J2kFileWrapOptions::jp2()
        }
        .with_color(j2k::J2kFileColorSpec::Enumerated(
            j2k_core::Colorspace::SRgb,
        ))
        .with_metadata(j2k::J2kFileBoxMetadata {
            palette: None,
            component_mappings: &[],
            channel_definitions: &definitions,
        }),
    )
    .expect("wrap alpha")
}

fn requests() -> [(DecodeRequest, DeviceDecodeRequest); 4] {
    let roi = Rect {
        x: 2,
        y: 4,
        w: 20,
        h: 16,
    };
    let scale = Downscale::Half;
    [
        (DecodeRequest::Full, DeviceDecodeRequest::Full),
        (
            DecodeRequest::Region { roi },
            DeviceDecodeRequest::Region { roi },
        ),
        (
            DecodeRequest::Reduced { scale },
            DeviceDecodeRequest::Scaled { scale },
        ),
        (
            DecodeRequest::RegionReduced { roi, scale },
            DeviceDecodeRequest::RegionScaled { roi, scale },
        ),
    ]
}

#[test]
fn prepared_signed_gray_requested_formats_match_cpu() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let mut session = CudaSession::default();
    let mut comparisons = 0;
    for ht in [false, true] {
        for tiled in [false, true] {
            let bytes: Arc<[u8]> = source(1, 16, true, false, ht, tiled.then_some((16, 16))).into();
            for (request, _) in requests() {
                let prepared = prepare_batch(
                    vec![EncodedImage::new(bytes.clone(), request)],
                    BatchDecodeOptions::default(),
                )
                .expect("prepare gray");
                assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
                let image = &prepared.groups()[0].images()[0];
                for referenced in [false, true] {
                    if tiled && !referenced {
                        continue;
                    }
                    let input = GrayscaleBatchInput {
                        source_index: 0,
                        bytes: &bytes,
                        device_plan: Some(image.plan()),
                        referenced_plan: referenced
                            .then(|| image.htj2k_plan().map(j2k::PreparedHtj2kPlan::geometry))
                            .flatten(),
                        referenced_classic_plan: referenced
                            .then(|| image.classic_plan().map(j2k::PreparedClassicPlan::geometry))
                            .flatten(),
                    };
                    for fmt in [PixelFormat::Gray8, PixelFormat::Gray16] {
                        let dims = image.plan().output_dims();
                        let stride = dims.0 as usize * fmt.bytes_per_pixel();
                        let mut expected = vec![0; stride * dims.1 as usize];
                        j2k::J2kDecoder::new(&bytes)
                            .expect("CPU decoder")
                            .decode_region_scaled_into(
                                &mut j2k::J2kScratchPool::default(),
                                &mut expected,
                                stride,
                                fmt,
                                image.plan().source_rect(),
                                image.plan().scale(),
                            )
                            .expect("CPU requested format");
                        let (surfaces, _) =
                            decode_grayscale_cuda_resident_prepared_batch_surfaces_with_profile(
                                &[input, input],
                                DecodeSettings::strict(),
                                &mut session,
                                fmt,
                                false,
                            )
                            .expect("prepared CUDA requested format");
                        for surface in surfaces {
                            let mut actual = vec![0; expected.len()];
                            surface
                                .download_into(&mut actual, dims.0 as usize * fmt.bytes_per_pixel())
                                .expect("download");
                            assert!(
                                actual == expected,
                                "ht={ht} tiled={tiled} referenced={referenced} {request:?} {fmt:?}"
                            );
                            comparisons += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("prepared grayscale CPU byte comparisons: {comparisons}");
}

fn cpu_bytes(samples: &CpuBatchSamples) -> Vec<u8> {
    match samples {
        CpuBatchSamples::U8(values) => values.clone(),
        CpuBatchSamples::U16(values) => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        CpuBatchSamples::I16(values) => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        _ => panic!("unexpected native samples"),
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one parity matrix covers codec, layout, geometry and native sample conversion"
)]
fn prepared_native_color_matches_cpu_and_signed_format_conversions() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let mut session = CudaSession::default();
    let mut comparisons = 0;
    for ht in [false, true] {
        for mct in [false, true] {
            for components in [3, 4] {
                for (bits, signed) in [(8, false), (16, false), (8, true), (16, true)] {
                    for tiled in [false, true] {
                        let bytes: Arc<[u8]> = with_alpha(
                            source(components, bits, signed, mct, ht, tiled.then_some((16, 16))),
                            components,
                            ht,
                        )
                        .into();
                        for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
                            let options = BatchDecodeOptions {
                                layout,
                                ..BatchDecodeOptions::default()
                            };
                            for (request, _) in requests() {
                                let prepared = prepare_batch(
                                    vec![EncodedImage::new(bytes.clone(), request)],
                                    options,
                                )
                                .expect("prepare color");
                                assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
                                let group = &prepared.groups()[0];
                                let image = &group.images()[0];
                                let native_fmt =
                                    group.info().native_pixel_format().expect("native format");
                                let cpu = CpuBatchDecoder::new(options)
                                    .decode_prepared(&prepared)
                                    .expect("CPU native batch");
                                let samples = cpu.groups()[0].samples();
                                let expected_native = cpu_bytes(samples);
                                if signed && bits == 16 && request == DecodeRequest::Full {
                                    let CpuBatchSamples::I16(samples) = samples else {
                                        panic!("signed source must select I16")
                                    };
                                    for boundary in [i16::MIN, -1, 0, i16::MAX] {
                                        assert!(samples.contains(&boundary), "missing {boundary}");
                                    }
                                }
                                for referenced in [false, true] {
                                    if !referenced && (tiled || components == 4) {
                                        continue;
                                    }
                                    let input = NativeColorBatchInput {
                                        source_index: 0,
                                        bytes: &bytes,
                                        device_plan: image.plan(),
                                        settings: DecodeSettings::strict(),
                                        referenced_plan: referenced
                                            .then(|| {
                                                image
                                                    .htj2k_plan()
                                                    .map(j2k::PreparedHtj2kPlan::geometry)
                                            })
                                            .flatten(),
                                        referenced_classic_plan: referenced
                                            .then(|| {
                                                image
                                                    .classic_plan()
                                                    .map(j2k::PreparedClassicPlan::geometry)
                                            })
                                            .flatten(),
                                    };
                                    let mut formats = vec![native_fmt];
                                    // Public native groups select I16 for signed sources. Exercise the
                                    // internal unsigned stores too: native conversion clamps, without
                                    // a display-width rescale or an unsigned MCT level shift.
                                    if signed && request == DecodeRequest::Full {
                                        formats.push(match (components, bits) {
                                            (3, 8) => PixelFormat::Rgb8,
                                            (3, _) => PixelFormat::Rgb16,
                                            (_, 8) => PixelFormat::Rgba8,
                                            _ => PixelFormat::Rgba16,
                                        });
                                    }
                                    for fmt in formats {
                                        let expected = if fmt == native_fmt {
                                            expected_native.clone()
                                        } else {
                                            let CpuBatchSamples::I16(values) = samples else {
                                                unreachable!()
                                            };
                                            if matches!(fmt, PixelFormat::Rgb8 | PixelFormat::Rgba8)
                                            {
                                                values
                                                    .iter()
                                                    .map(|&v| {
                                                        u8::try_from(v.clamp(0, 255))
                                                            .expect("clamped")
                                                    })
                                                    .collect()
                                            } else {
                                                values
                                                    .iter()
                                                    .flat_map(|&v| v.max(0).to_ne_bytes())
                                                    .collect()
                                            }
                                        };
                                        let (output, _) =
                                            submit_native_color_resident_prepared_batch(
                                                &[input, input],
                                                &mut session,
                                                fmt,
                                                layout,
                                            )
                                            .expect("submit native color")
                                            .finish()
                                            .expect("finish native color");
                                        for range in output.ranges {
                                            let mut actual = vec![0; expected.len()];
                                            output
                                                .buffer
                                                .copy_range_to_host(range.offset, &mut actual)
                                                .expect("download native color");
                                            assert!(actual == expected, "ht={ht} mct={mct} signed={signed} tiled={tiled} referenced={referenced} {fmt:?} {layout:?} {request:?}");
                                            comparisons += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("prepared native color CPU byte comparisons: {comparisons}");
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one rejection matrix covers RGB and RGBA through every store route"
)]
fn mixed_signedness_is_rejected_on_single_and_batch_routes() {
    use PixelFormat::{Gray16, Rgb16, Rgb8, RgbI16, Rgba16, Rgba8, RgbaI16};
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let mut session = CudaSession::default();
    for ht in [false, true] {
        for mct in [false, true] {
            for components in [3, 4] {
                for tiled in [false, true] {
                    let mut bytes =
                        source(components, 16, true, mct, ht, tiled.then_some((16, 16)));
                    let siz = bytes
                        .windows(2)
                        .position(|v| v == [0xff, 0x51])
                        .expect("SIZ");
                    bytes[siz + 40 + 3] &= 0x7f; // Make the second component unsigned.
                    let bytes = with_alpha(bytes, components, ht);
                    // Native referenced-plan construction is also reachable below shared
                    // batch preflight. Its CUDA adapters must reject mixed sources themselves.
                    let native = j2k_native::Image::new(&bytes, &DecodeSettings::strict())
                        .expect("mixed native image");
                    let mut context = j2k_native::DecoderContext::default();
                    let ht_plan = ht.then(|| {
                        native
                            .build_referenced_htj2k_plan_region_with_context(
                                &mut context,
                                (0, 0, SIZE, SIZE),
                            )
                            .expect("native HT plan")
                    });
                    let classic_plan = (!ht).then(|| {
                        native
                            .build_referenced_classic_plan_region_with_context(
                                &mut context,
                                (0, 0, SIZE, SIZE),
                            )
                            .expect("native classic plan")
                    });
                    let input = NativeColorBatchInput {
                        source_index: 0,
                        bytes: &bytes,
                        device_plan: j2k::DeviceDecodePlan::for_image(
                            (SIZE, SIZE),
                            DeviceDecodeRequest::Full,
                        )
                        .expect("geometry"),
                        referenced_plan: ht_plan.as_ref(),
                        referenced_classic_plan: classic_plan.as_ref(),
                        settings: DecodeSettings::strict(),
                    };
                    let formats = if components == 3 {
                        [Rgb8, Rgb16, RgbI16]
                    } else {
                        [Rgba8, Rgba16, RgbaI16]
                    };
                    for fmt in formats {
                        assert!(
                            submit_native_color_resident_prepared_batch(
                                &[input],
                                &mut session,
                                fmt,
                                BatchLayout::Nhwc
                            )
                            .is_err(),
                            "accepted referenced mixed color {ht} {mct} {tiled} {fmt:?}"
                        );
                    }
                    for (request, device_request) in requests() {
                        let prepared = prepare_batch(
                            vec![EncodedImage::new(Arc::from(bytes.clone()), request)],
                            BatchDecodeOptions::default(),
                        )
                        .expect("indexed preflight rejection");
                        assert!(prepared.groups().is_empty());
                        assert!(prepared
                            .errors()
                            .iter()
                            .any(|e| e.source.to_string().contains("mixed component signedness")));
                        for fmt in [Gray16, Rgb8, Rgb16, Rgba8, Rgba16] {
                            assert!(
                                J2kDecoder::new(&bytes)
                                    .expect("parse mixed source")
                                    .decode_request_to_device_with_session(
                                        fmt,
                                        device_request,
                                        &mut session
                                    )
                                    .is_err(),
                                "accepted {ht} {mct} {tiled} {fmt:?} {request:?}"
                            );
                        }
                    }
                    for fmt in [Gray16, Rgb8, Rgb16, Rgba8, Rgba16] {
                        assert!(
                            J2kDecoder::decode_batch_to_device_with_session(
                                &[&bytes, &bytes],
                                fmt,
                                &mut session
                            )
                            .is_err(),
                            "accepted mixed batch {ht} {mct} {tiled} {fmt:?}"
                        );
                    }
                }
            }
        }
    }
}
