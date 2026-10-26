// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal device decodes return the CPU decoder's samples for every source
//! depth, signedness, output format and request (issue #135).

#![cfg(target_os = "macos")]

use std::sync::Arc;

use j2k::{J2kDecoder as CpuDecoder, J2kScratchPool};
use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};
use j2k_metal::{J2kDecoder, MetalDecodeRequest, MetalTileBatch, SurfaceResidency};
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use j2k_test_support::metal_runtime_gate;

const SIZE: u32 = 64;
const WHOLE: Rect = Rect {
    x: 0,
    y: 0,
    w: SIZE,
    h: SIZE,
};

/// Lossless ramp: sample `i` of component `c` is `17 i + 1000 c + offset` modulo
/// `2^bits`, read as two's complement for signed sources.
fn codestream(
    bits: u8,
    signed: bool,
    components: u16,
    mct: bool,
    ht: bool,
    tile_size: Option<(u32, u32)>,
    offset: u32,
) -> Vec<u8> {
    let mask = (1_u32 << bits) - 1;
    let bytes_per_sample = if bits <= 8 { 1 } else { 2 };
    let mut pixels = Vec::new();
    for i in 0..SIZE * SIZE {
        for c in 0..u32::from(components) {
            let raw = if bits == 16 && signed && i < 4 {
                [32768, 65535, 0, 32767][i as usize]
            } else {
                (17 * i + 1000 * c + offset) & mask
            };
            pixels.extend_from_slice(&raw.to_le_bytes()[..bytes_per_sample]);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct: mct,
        tile_size,
        ..EncodeOptions::default()
    };
    let encoder = if ht { encode_htj2k } else { encode };
    encoder(&pixels, SIZE, SIZE, components, bits, signed, &options).expect("encode source")
}

fn bytes_per_pixel(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Gray16 => 2,
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Rgb16 => 6,
        other => panic!("unexpected format {other:?}"),
    }
}

fn cpu_pixels(bytes: &[u8], format: PixelFormat, roi: Rect, scale: Downscale) -> Vec<u8> {
    let out = roi.scaled_covering(scale);
    let stride = out.w as usize * bytes_per_pixel(format);
    let mut pixels = vec![0u8; stride * out.h as usize];
    CpuDecoder::new(bytes)
        .expect("CPU decoder")
        .decode_region_scaled_into(
            &mut J2kScratchPool::default(),
            &mut pixels,
            stride,
            format,
            roi,
            scale,
        )
        .expect("CPU decode");
    pixels
}

fn batch_surfaces(
    first: &[u8],
    second: &[u8],
    request: MetalDecodeRequest,
    case: &str,
) -> Vec<(j2k_metal::Surface, usize)> {
    let input = Arc::<[u8]>::from(first);
    let mut surfaces = Vec::new();
    for (other, expected) in [(input.clone(), 0), (Arc::from(second), 1)] {
        let mut batch = MetalTileBatch::new();
        for bytes in [input.clone(), other] {
            batch
                .push_shared_tile_request(bytes, request)
                .expect("queue Metal batch decode");
        }
        let batch = batch
            .decode_all()
            .unwrap_or_else(|error| panic!("{case}: {error}"));
        assert_eq!(batch.len(), 2);
        surfaces.extend(batch.into_iter().zip([0, expected]));
    }
    surfaces
}

#[test]
fn metal_device_decode_matches_cpu_for_every_sample_depth() {
    if !metal_runtime_gate(module_path!()) {
        return;
    }

    let gray = [PixelFormat::Gray8, PixelFormat::Gray16];
    let color = [PixelFormat::Rgb8, PixelFormat::Rgb16, PixelFormat::Rgba8];
    // (components, mct, tile size, formats); tiled images take the fallback
    // route.
    let tiles = Some((SIZE / 2, SIZE / 2));
    let sources = [
        (1, false, None, &gray[..]),
        (3, true, None, &color[..]),
        (3, false, None, &color[..]),
        (1, false, tiles, &gray[..]),
        (3, true, tiles, &color[..]),
    ];
    let requests = [
        ("full", WHOLE, Downscale::None),
        (
            "region",
            Rect {
                x: 8,
                y: 8,
                w: 40,
                h: 40,
            },
            Downscale::None,
        ),
        ("half", WHOLE, Downscale::Half),
    ];
    let mut mismatches = Vec::new();
    for bits in [8, 10, 12, 16] {
        for signed in [false, true] {
            for ht in [false, true] {
                for (components, mct, tile_size, formats) in sources {
                    let first = codestream(bits, signed, components, mct, ht, tile_size, 0);
                    let second = codestream(bits, signed, components, mct, ht, tile_size, 1);
                    for &format in formats {
                        for (request, roi, scale) in requests {
                            let out = roi.scaled_covering(scale);
                            let stride = out.w as usize * bytes_per_pixel(format);
                            let cpu = [
                                cpu_pixels(&first, format, roi, scale),
                                cpu_pixels(&second, format, roi, scale),
                            ];
                            let metal_request = match request {
                                "full" => MetalDecodeRequest::full(format, BackendRequest::Metal),
                                "region" => {
                                    MetalDecodeRequest::region(format, roi, BackendRequest::Metal)
                                }
                                _ => MetalDecodeRequest::region_scaled(
                                    format,
                                    roi,
                                    scale,
                                    BackendRequest::Metal,
                                ),
                            };
                            let surface = J2kDecoder::new(&first)
                                .expect("decoder")
                                .decode_request_to_device(metal_request)
                                .expect("Metal decode");
                            let case = format!(
                                "{request} {bits}-bit signed={signed} ht={ht} \
                                 components={components} mct={mct} tiles={tile_size:?} {format:?}"
                            );
                            let mut surfaces = vec![(surface, 0)];
                            // Tiled images retain single-image fallback coverage above;
                            // the direct batch API requires single-tile geometry.
                            if tile_size.is_none() {
                                surfaces.extend(batch_surfaces(
                                    &first,
                                    &second,
                                    metal_request,
                                    &case,
                                ));
                            }
                            for (index, (surface, expected)) in surfaces.into_iter().enumerate() {
                                let cpu = &cpu[expected];
                                let mut metal = vec![0u8; cpu.len()];
                                surface
                                    .download_into(&mut metal, stride)
                                    .expect("download Metal surface");
                                let differing =
                                    cpu.iter().zip(&metal).filter(|(a, b)| a != b).count();
                                let residency = surface.residency();
                                if differing != 0
                                    || residency != SurfaceResidency::MetalResidentDecode
                                {
                                    mismatches.push(format!(
                                        "output={index} {case} {residency:?}: {differing} bytes differ"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

fn native_source(
    bits: u8,
    signed: bool,
    components: u16,
    mct: bool,
    ht: bool,
    tile_size: Option<(u32, u32)>,
) -> Arc<[u8]> {
    let bytes = codestream(bits, signed, components, mct, ht, tile_size, 0);
    wrap_native_source(bytes, components, ht).into()
}

fn wrap_native_source(bytes: Vec<u8>, components: u16, ht: bool) -> Vec<u8> {
    if components != 4 {
        return bytes;
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
        &bytes,
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
    .expect("wrap RGBA")
}

#[test]
fn prepared_native_samples_match_cpu_for_signed_and_unsigned_sources() {
    use j2k::{
        prepare_batch, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
        DecodeRequest, EncodedImage,
    };
    if !metal_runtime_gate(module_path!()) {
        return;
    }
    let roi = Rect {
        x: 8,
        y: 8,
        w: 40,
        h: 40,
    };
    let scale = Downscale::Half;
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region { roi },
        DecodeRequest::Reduced { scale },
        DecodeRequest::RegionReduced { roi, scale },
    ];
    let mut comparisons = 0;
    for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
        let options = {
            let mut options = BatchDecodeOptions::default();
            options.layout = layout;
            options
        };
        let mut decoder = j2k_metal::MetalBatchDecoder::system_default_with_options(options)
            .expect("Metal session");
        let mut cpu = CpuBatchDecoder::new(options);
        for ht in [false, true] {
            for mct in [false, true] {
                for components in [1, 3, 4] {
                    if components == 1 && mct {
                        continue;
                    }
                    for (bits, signed) in [(8, false), (16, false), (16, true)] {
                        for tiled in [false, true] {
                            let source = native_source(
                                bits,
                                signed,
                                components,
                                mct,
                                ht,
                                tiled.then_some((32, 32)),
                            );
                            for request in requests {
                                let prepared = prepare_batch(
                                    vec![EncodedImage::new(source.clone(), request); 2],
                                    options,
                                )
                                .expect("prepare native samples");
                                assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
                                let oracle =
                                    cpu.decode_prepared(&prepared).expect("CPU native samples");
                                let expected = match oracle.groups()[0].samples() {
                                    CpuBatchSamples::U8(v) => v.clone(),
                                    CpuBatchSamples::U16(v) => {
                                        v.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>()
                                    }
                                    CpuBatchSamples::I16(v) => {
                                        if request == DecodeRequest::Full {
                                            for value in [i16::MIN, -1, 0, i16::MAX] {
                                                assert!(v.contains(&value), "missing {value}");
                                            }
                                        }
                                        v.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>()
                                    }
                                    _ => panic!("unexpected native samples"),
                                };
                                let output = decoder
                                    .decode_prepared(&prepared)
                                    .expect("Metal prepared native samples");
                                assert!(
                                    output.group_errors().is_empty(),
                                    "{:?}",
                                    output.group_errors()
                                );
                                let resident = output.groups()[0]
                                    .resident_batch()
                                    .expect("dense native output");
                                assert_eq!(resident.byte_len(), expected.len());
                                // SAFETY: decode_prepared completed; the retained output is read without a writer.
                                let actual = unsafe {
                                    j2k_metal_support::checked_buffer_read_vec::<u8>(
                                        resident.metal_buffer(),
                                        resident.byte_offset(),
                                        resident.byte_len(),
                                    )
                                    .expect("read native samples")
                                };
                                assert!(actual == expected, "{layout:?} {request:?} ht={ht} mct={mct} components={components} bits={bits} signed={signed} tiled={tiled}");
                                comparisons += 2;
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("prepared Metal native CPU byte comparisons: {comparisons}");
}

#[test]
fn mixed_signedness_is_rejected_before_metal_single_and_batch_stores() {
    use j2k::{prepare_batch, BatchDecodeOptions, DecodeRequest, EncodedImage};
    if !metal_runtime_gate(module_path!()) {
        return;
    }
    let roi = Rect {
        x: 8,
        y: 8,
        w: 40,
        h: 40,
    };
    for ht in [false, true] {
        for mct in [false, true] {
            for components in [3, 4] {
                for tiled in [false, true] {
                    let mut bytes =
                        codestream(16, true, components, mct, ht, tiled.then_some((32, 32)), 0);
                    let siz = bytes
                        .windows(2)
                        .position(|v| v == [0xff, 0x51])
                        .expect("SIZ");
                    bytes[siz + 43] &= 0x7f;
                    let bytes = wrap_native_source(bytes, components, ht);
                    let prepared = prepare_batch(
                        vec![EncodedImage::new(
                            Arc::from(bytes.clone()),
                            DecodeRequest::Full,
                        )],
                        BatchDecodeOptions::default(),
                    )
                    .expect("indexed rejection");
                    assert!(prepared.groups().is_empty());
                    assert!(prepared
                        .errors()
                        .iter()
                        .any(|e| e.source.to_string().contains("mixed component signedness")));
                    for fmt in [
                        PixelFormat::Gray16,
                        PixelFormat::Rgb8,
                        PixelFormat::Rgb16,
                        PixelFormat::Rgba8,
                    ] {
                        let requests = [
                            MetalDecodeRequest::full(fmt, BackendRequest::Metal),
                            MetalDecodeRequest::region(fmt, roi, BackendRequest::Metal),
                            MetalDecodeRequest::scaled(fmt, Downscale::Half, BackendRequest::Metal),
                            MetalDecodeRequest::region_scaled(
                                fmt,
                                roi,
                                Downscale::Half,
                                BackendRequest::Metal,
                            ),
                        ];
                        for request in requests {
                            // Rejected decodes may leave command buffers autoreleased on this
                            // thread. Unoptimized builds keep them until a pool drains, and a
                            // queue blocks once 64 are outstanding, so drain per attempt.
                            objc2::rc::autoreleasepool(|_| {
                                assert!(
                                    J2kDecoder::new(&bytes)
                                        .expect("mixed decoder")
                                        .decode_request_to_device(request)
                                        .is_err(),
                                    "accepted mixed source {ht} {mct} {tiled} {fmt:?}"
                                );
                                let mut batch = MetalTileBatch::new();
                                batch
                                    .push_shared_tile_request(Arc::from(bytes.clone()), request)
                                    .expect("queue");
                                assert!(
                                    batch.decode_all().is_err(),
                                    "accepted mixed batch {ht} {mct} {tiled} {fmt:?}"
                                );
                            });
                        }
                    }
                }
            }
        }
    }
}
