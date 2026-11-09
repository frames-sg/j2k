// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA device decodes return the CPU decoder's samples for every source
//! depth, signedness and output format (issue #135).

#![cfg(feature = "cuda-runtime")]

use j2k::DeviceDecodeRequest;
use j2k_core::{
    BackendRequest, Downscale, ImageDecode, ImageDecodeDevice, PixelFormat, Rect,
    TileBatchDecodeManyDevice,
};
use j2k_cuda::{
    Codec, CudaSession, J2kContext, J2kDecoder, J2kScratchPool, Surface, SurfaceResidency,
};
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use j2k_test_support::cuda_runtime_and_strict_oxide_gate;

const SIZE: u32 = 64;

/// Sample ramp: sample `i` of component `c` is `17 i + 1000 c + offset` modulo
/// `2^bits`, read as two's complement for signed sources.
fn codestream(
    bits: u8,
    signed: bool,
    components: u16,
    ht: bool,
    offset: u32,
    options: &EncodeOptions,
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
    let encoder = if ht { encode_htj2k } else { encode };
    encoder(&pixels, SIZE, SIZE, components, bits, signed, options).expect("encode source")
}

fn bytes_per_pixel(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Gray16 => 2,
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Rgb16 => 6,
        PixelFormat::Rgba16 => 8,
        other => panic!("unexpected format {other:?}"),
    }
}

/// Bytes of `surface` that differ from the CPU decode of `source`, and whether
/// the surface was decoded on the device.
fn compare_with_cpu(surface: &Surface, source: &[u8], format: PixelFormat) -> (usize, bool) {
    let stride = SIZE as usize * bytes_per_pixel(format);
    let cpu = cpu_pixels(source, format);
    let mut cuda = vec![0u8; stride * SIZE as usize];
    surface
        .download_into(&mut cuda, stride)
        .expect("download CUDA surface");
    let differing = cpu.iter().zip(&cuda).filter(|(a, b)| a != b).count();
    (
        differing,
        surface.residency() == SurfaceResidency::CudaResidentDecode,
    )
}

fn cpu_pixels(bytes: &[u8], format: PixelFormat) -> Vec<u8> {
    let stride = SIZE as usize * bytes_per_pixel(format);
    let mut pixels = vec![0u8; stride * SIZE as usize];
    J2kDecoder::new(bytes)
        .expect("decoder")
        .decode_into(&mut pixels, stride, format)
        .expect("CPU decode");
    pixels
}

#[test]
fn cuda_device_decode_matches_cpu_for_every_sample_depth() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let gray = [PixelFormat::Gray8, PixelFormat::Gray16];
    let color = [
        PixelFormat::Rgb8,
        PixelFormat::Rgb16,
        PixelFormat::Rgba8,
        PixelFormat::Rgba16,
    ];
    let sources = [
        (1, false, &gray[..]),
        (3, true, &color[..]),
        (3, false, &color[..]),
    ];
    let mut session = CudaSession::default();
    let mut mismatches = Vec::new();
    for reversible in [true, false] {
        for bits in 1..=16 {
            for signed in [false, true] {
                for ht in [false, true] {
                    for (components, mct, formats) in sources {
                        let options = EncodeOptions {
                            reversible,
                            num_decomposition_levels: 2,
                            use_mct: mct,
                            ..EncodeOptions::default()
                        };
                        let first = codestream(bits, signed, components, ht, 0, &options);
                        let second = codestream(bits, signed, components, ht, 1, &options);
                        for &format in formats {
                            let surface = J2kDecoder::new(&first)
                                .expect("decoder")
                                .decode_to_device(format, BackendRequest::Cuda)
                                .expect("CUDA decode");
                            let mut surfaces = J2kDecoder::decode_batch_to_device_with_session(
                                &[first.as_slice(), second.as_slice()],
                                format,
                                &mut session,
                            )
                            .expect("CUDA batch decode");
                            assert_eq!(surfaces.len(), 2);
                            surfaces.insert(0, surface);
                            for (index, surface) in surfaces.into_iter().enumerate() {
                                let source = if index == 2 { &second } else { &first };
                                let (differing, on_device) =
                                    compare_with_cpu(&surface, source, format);
                                if differing != 0 || !on_device {
                                    mismatches.push(format!(
                                        "output={index} {bits}-bit signed={signed} ht={ht} reversible={reversible} mct={mct} {format:?}: \
                                         on device={on_device}, {differing} bytes differ"
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

/// A 64x64 reversible codestream of full-range noise whose components are
/// `bits` (17 to 24) wide, three little-endian bytes per sample.
fn wide_codestream(bits: u8, components: u16, ht: bool) -> Vec<u8> {
    let mask = (1_u32 << bits) - 1;
    let mut state = 0x2545_f491_u32;
    let pixels: Vec<u8> = (0..SIZE * SIZE * u32::from(components))
        .flat_map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let [low, middle, high, _] = (state & mask).to_le_bytes();
            [low, middle, high]
        })
        .collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 5,
        ..EncodeOptions::default()
    };
    let encoder = if ht { encode_htj2k } else { encode };
    encoder(&pixels, SIZE, SIZE, components, bits, false, &options).expect("encode wide source")
}

fn assert_rejects_wide_precision<T>(result: Result<T, j2k_cuda::Error>, label: &str) {
    match result {
        Err(error) => assert!(
            error
                .to_string()
                .contains("component precisions of at most 16 bits"),
            "{label}: unexpected error {error}"
        ),
        Ok(_) => panic!("{label}: explicit CUDA must refuse components wider than 16 bits"),
    }
}

/// Components wider than 16 bits leave the integers the f32 CUDA 5/3, RCT
/// and stores reproduce exactly, and the output scales assume 16 bits.
/// Explicit CUDA once decoded them anyway: a 17-bit RGB image to Rgb16
/// differed from the CPU in 15268 of 24576 bytes. Every CUDA route now
/// refuses them, and Auto decodes them on the CPU.
#[test]
fn cuda_device_decode_refuses_components_wider_than_sixteen_bits() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let mut session = CudaSession::default();
    let region = Rect {
        x: 8,
        y: 8,
        w: 32,
        h: 32,
    };
    for (components, bits, ht) in [
        (1_u16, 17_u8, false),
        (1, 17, true),
        (3, 17, false),
        (3, 17, true),
        (1, 24, false),
        (3, 24, false),
    ] {
        let source = wide_codestream(bits, components, ht);
        let formats: &[PixelFormat] = if components == 1 {
            &[PixelFormat::Gray8, PixelFormat::Gray16]
        } else {
            &[PixelFormat::Rgb8, PixelFormat::Rgb16]
        };
        for &format in formats {
            let label = format!("{bits}-bit {components}-component ht={ht} {format:?}");
            let decoder = || J2kDecoder::new(&source).expect("decoder");
            assert_rejects_wide_precision(
                decoder().decode_to_device(format, BackendRequest::Cuda),
                &label,
            );
            for request in [
                DeviceDecodeRequest::Region { roi: region },
                DeviceDecodeRequest::Scaled {
                    scale: Downscale::Half,
                },
            ] {
                assert_rejects_wide_precision(
                    decoder().decode_request_to_device_with_session(format, request, &mut session),
                    &label,
                );
            }
            let inputs = [source.as_slice(), source.as_slice()];
            assert_rejects_wide_precision(
                J2kDecoder::decode_batch_to_device_with_session(&inputs, format, &mut session),
                &label,
            );
            let mut ctx = J2kContext::default();
            let mut pool = J2kScratchPool::new();
            assert_rejects_wide_precision(
                Codec::decode_tiles_to_device(
                    &mut ctx,
                    &mut pool,
                    &inputs,
                    format,
                    BackendRequest::Cuda,
                ),
                &label,
            );

            let auto = decoder()
                .decode_to_device(format, BackendRequest::Auto)
                .expect("Auto decode");
            let (differing, on_device) = compare_with_cpu(&auto, &source, format);
            assert_eq!((differing, on_device), (0, false), "{label} Auto");
            let tiles = Codec::decode_tiles_to_device(
                &mut ctx,
                &mut pool,
                &inputs,
                format,
                BackendRequest::Auto,
            )
            .expect("Auto tile batch");
            for tile in &tiles {
                let (differing, on_device) = compare_with_cpu(tile, &source, format);
                assert_eq!((differing, on_device), (0, false), "{label} Auto tile");
            }
        }
    }
}

#[test]
fn cuda_batch_of_signed_and_unsigned_sources_matches_cpu() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let cases = [
        (1, false, PixelFormat::Gray8),
        (1, false, PixelFormat::Gray16),
        (3, true, PixelFormat::Rgb8),
        (3, true, PixelFormat::Rgb16),
        (3, false, PixelFormat::Rgba16),
    ];
    let mut session = CudaSession::default();
    let mut mismatches = Vec::new();
    for bits in [8, 12, 16] {
        for ht in [false, true] {
            for (components, mct, format) in cases {
                let options = EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 2,
                    use_mct: mct,
                    ..EncodeOptions::default()
                };
                let sources = [
                    codestream(bits, false, components, ht, 0, &options),
                    codestream(bits, true, components, ht, 1, &options),
                ];
                let surfaces = J2kDecoder::decode_batch_to_device_with_session(
                    &[sources[0].as_slice(), sources[1].as_slice()],
                    format,
                    &mut session,
                )
                .expect("CUDA batch decode");
                assert_eq!(surfaces.len(), 2);
                for (surface, signed) in surfaces.iter().zip([false, true]) {
                    let source = &sources[usize::from(signed)];
                    let (differing, on_device) = compare_with_cpu(surface, source, format);
                    if differing != 0 || !on_device {
                        mismatches.push(format!(
                            "{bits}-bit signed={signed} ht={ht} mct={mct} {format:?} \
                             on device={on_device}: {differing} bytes differ"
                        ));
                    }
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}
