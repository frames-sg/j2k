// SPDX-License-Identifier: MIT OR Apache-2.0

//! Same-byte marker, source-sample and independent `OpenJPEG` checks.

use j2k::{encode_j2k_lossy, J2kLossyEncodeOptions, J2kLossySamples};
use j2k_compare::openjpeg;
use j2k_native::{
    encode, encode_typed_component_planes_53, DecodeSettings, EncodeOptions,
    EncodeTypedComponentPlane, Image, IrreversibleQuantizationSubbandScales,
};

// Read length-delimited main-header segments independently of the codec parser.
fn marker(bytes: &[u8], wanted: u8) -> &[u8] {
    assert_eq!(&bytes[..2], &[0xff, 0x4f]);
    let mut offset = 2;
    while bytes[offset + 1] != 0x90 {
        assert_eq!(bytes[offset], 0xff);
        let len = usize::from(u16::from_be_bytes([bytes[offset + 2], bytes[offset + 3]]));
        assert!(len >= 2);
        if bytes[offset + 1] == wanted {
            return &bytes[offset + 4..offset + 2 + len];
        }
        offset += 2 + len;
    }
    panic!("missing main-header marker {wanted:x}");
}

fn steps(bytes: &[u8], kind: u8) -> Vec<(u8, u16)> {
    let body = marker(bytes, kind);
    let body = if kind == 0x5d { &body[1..] } else { body };
    assert_eq!(body[0] & 0x1f, 2, "scalar expounded quantization");
    assert!((body.len() - 1).is_multiple_of(2));
    body[1..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let stored = u16::from_be_bytes([pair[0], pair[1]]);
            (u8::try_from(stored >> 11).unwrap(), stored & 2047)
        })
        .collect()
}

#[test]
fn edge_tile_profiles_signal_resolved_depth_and_decode_in_openjpeg() {
    for (width, height, tile, depth) in [
        (65_u32, 64_u32, 64_u32, 0_u8),
        (64, 65, 64, 0),
        (80, 64, 64, 4),
        (64, 80, 64, 4),
        (128, 128, 64, 5),
        (64, 64, 128, 5),
    ] {
        let pixels: Vec<_> = (0..width * height)
            .map(|i| u8::try_from((i * 17 + i / width * 3) % 251).unwrap())
            .collect();
        let samples = J2kLossySamples::new(&pixels, width, height, 1, 8, false).unwrap();
        let mut options = J2kLossyEncodeOptions::default()
            .with_cpu_only_backend()
            .with_max_decomposition_levels(Some(5))
            .with_tile_size(Some((tile, tile)));
        let plain = encode_j2k_lossy(samples, &options).unwrap().codestream;
        options.quantization_profile = Some(vec![vec![1.0; 1 + 3 * usize::from(depth)]]);
        assert_eq!(
            encode_j2k_lossy(samples, &options).unwrap().codestream,
            plain
        );
        options.quantization_profile.as_mut().unwrap()[0][0] = 1.5;
        let encoded = encode_j2k_lossy(samples, &options).unwrap().codestream;
        assert_eq!(
            marker(&encoded, 0x52)[5],
            depth,
            "COD depth {width}x{height}"
        );
        // The profile scales LL by 1.5 and leaves the default detail steps.
        let qcd = steps(&plain, 0x5c);
        let qcc = steps(&encoded, 0x5d);
        assert_eq!(qcc.len(), 1 + 3 * usize::from(depth));
        assert_eq!(qcc[1..], qcd[1..]);
        let step = |(exponent, mantissa): (u8, u16)| {
            2.0_f64.powi(-i32::from(exponent)) * (1.0 + f64::from(mantissa) / 2048.0)
        };
        let ll_ratio = step(qcc[0]) / step(qcd[0]);
        assert!(
            (ll_ratio - 1.5).abs() <= 1e-3,
            "{width}x{height}: LL ratio {ll_ratio}"
        );
        let reference =
            openjpeg::decode_components(&encoded).expect("OpenJPEG decodes the same encoded bytes");
        assert_eq!(reference.dimensions, (width, height));
        assert_eq!(reference.components[0].bit_depth, 8);
        assert!(!reference.components[0].signed);
        let ours = Image::new(&encoded, &DecodeSettings::strict())
            .unwrap()
            .decode_native()
            .unwrap();
        let mut max_source_error = 0_i32;
        let mut max_decoder_error = 0_i32;
        let mut squared_error = 0.0;
        for ((&source, &actual), &external) in pixels
            .iter()
            .zip(&ours.data)
            .zip(&reference.components[0].samples)
        {
            let error = external - i32::from(source);
            max_source_error = max_source_error.max(error.abs());
            max_decoder_error = max_decoder_error.max((external - i32::from(actual)).abs());
            squared_error += f64::from(error).powi(2);
        }
        // A fine (LL <=0.375, detail <=1) deadzone quantizer with up to five
        // 9/7 levels may differ by a few integer codes after reconstruction.
        assert!(
            max_source_error <= 4,
            "{width}x{height}: source error {max_source_error}"
        );
        assert!(
            max_decoder_error <= 1,
            "{width}x{height}: decoder error {max_decoder_error}"
        );
        let mse = squared_error / f64::from(width * height);
        let psnr = 10.0 * (255.0_f64.powi(2) / mse).log10();
        assert!(psnr >= 45.0, "{width}x{height}: PSNR {psnr}");
        eprintln!("OpenJPEG {}: {width}x{height} tile={tile} depth={depth} 8-bit 9/7 LL weight=1.5: source max={max_source_error}, decoder max={max_decoder_error}, PSNR={psnr:.3} dB", openjpeg::version());
        options.quantization_profile = Some(vec![vec![1.0; 4 + 3 * usize::from(depth)]]);
        assert!(encode_j2k_lossy(samples, &options).is_err());
    }
}

#[test]
fn sampled_signed_high_bit_tiles_match_openjpeg_source_samples() {
    for (width, height, tile, sampling, depth) in [
        (111_u32, 83_u32, (33, 25), (3_u8, 2_u8), 2_u8),
        (64, 64, (32, 32), (3, 2), 3),
        (64, 64, (128, 128), (3, 2), 4),
        (65, 64, (64, 64), (1, 1), 0),
        (64, 65, (64, 64), (1, 1), 0),
    ] {
        let component_width = width.div_ceil(u32::from(sampling.0));
        let component_height = height.div_ceil(u32::from(sampling.1));
        for signed in [false, true] {
            let source: Vec<i32> = (0..component_width * component_height)
                .map(|i| {
                    let raw = i.wrapping_mul(7_000_019) & ((1 << 29) - 1);
                    i32::try_from(raw).unwrap() - if signed { 1 << 28 } else { 0 }
                })
                .collect();
            let input: Vec<_> = source.iter().flat_map(|v| v.to_le_bytes()).collect();
            let options = EncodeOptions {
                tile_size: Some(tile),
                num_decomposition_levels: 5,
                use_mct: false,
                ..Default::default()
            };
            let bytes = encode_typed_component_planes_53(
                &[EncodeTypedComponentPlane {
                    data: &input,
                    bit_depth: 29,
                    signed,
                    x_rsiz: sampling.0,
                    y_rsiz: sampling.1,
                }],
                width,
                height,
                &options,
            )
            .unwrap();
            assert_eq!(marker(&bytes, 0x52)[5], depth);
            assert_eq!(
                &marker(&bytes, 0x51)[36..39],
                &[0x1c | if signed { 0x80 } else { 0 }, sampling.0, sampling.1]
            );
            let external =
                openjpeg::decode_components(&bytes).expect("OpenJPEG 29-bit component decode");
            let component = &external.components[0];
            assert_eq!(component.dimensions, (component_width, component_height));
            assert_eq!(
                component.sampling,
                (u32::from(sampling.0), u32::from(sampling.1))
            );
            assert_eq!(component.bit_depth, 29);
            assert_eq!(component.signed, signed);
            assert_eq!(
                component.samples, source,
                "OpenJPEG source samples: {width}x{height} {tile:?} {sampling:?} signed={signed}"
            );
            let ours = Image::new(&bytes, &DecodeSettings::strict())
                .unwrap()
                .decode_native_components()
                .unwrap();
            let plane = &ours.planes()[0];
            // This single sampled component is exposed on its own grid.
            // Both sides use signed i32 values; native bytes are little-endian.
            assert_eq!(plane.dimensions(), (component_width, component_height));
            let actual: Vec<_> = plane
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| i32::from_le_bytes(*v))
                .collect();
            assert_eq!(actual, source);
            eprintln!("OpenJPEG {}: {width}x{height}, tile={tile:?}, sampling={sampling:?}, 29-bit signed={signed}, depth={depth}: exact source equality", openjpeg::version());
        }
    }
}

#[test]
fn compensated_profile_product_writes_expected_steps_and_decodes_independently() {
    let pixels = [128_u8; 16];
    let options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 0,
        use_mct: false,
        irreversible_quantization_scale: 1e20,
        irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales {
            low_low: 1e20,
            ..Default::default()
        },
        irreversible_quantization_profile: Some(vec![vec![1e-40]]),
        ..Default::default()
    };
    let encoded = encode(&pixels, 4, 4, 1, 8, false, &options).unwrap();
    let mut plain_options = options.clone();
    plain_options.irreversible_quantization_profile = None;
    let plain = encode(&pixels, 4, 4, 1, 8, false, &plain_options).unwrap();
    assert_eq!(
        steps(&plain, 0x5c),
        [(0, 2047)],
        "global-only QCD saturates"
    );
    assert_eq!(
        steps(&encoded, 0x5c),
        [(10, 0)],
        "component zero also supplies QCD"
    );
    assert_eq!(steps(&encoded, 0x5d), [(10, 0)], "weighted QCC is 0.25");
    let decoded = openjpeg::decode_components(&encoded).unwrap();
    assert_eq!(decoded.components[0].samples, vec![128; 16]);
    assert_eq!(
        Image::new(&encoded, &DecodeSettings::strict())
            .unwrap()
            .decode_native()
            .unwrap()
            .data,
        pixels
    );
    eprintln!("OpenJPEG {}: 4x4, unsigned 8-bit, depth=0, 9/7, scale=1e20, LL scale=1e20, weight=1e-40: exact constant 128", openjpeg::version());
    let source = [0_u8, 1, 1, 0];
    let options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 0,
        irreversible_quantization_scale: f32::from_bits(1),
        irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales {
            low_low: f32::MAX,
            ..Default::default()
        },
        ..Default::default()
    };
    let bytes = encode(&source, 4, 1, 1, 1, false, &options).unwrap();
    assert_eq!(steps(&bytes, 0x5c), [(24, 0)]);
    let external = openjpeg::decode_components(&bytes).unwrap();
    assert_eq!(external.components[0].samples, [0, 1, 1, 0]);
    assert_eq!(
        Image::new(&bytes, &DecodeSettings::strict())
            .unwrap()
            .decode_native()
            .unwrap()
            .data,
        source
    );
    eprintln!("OpenJPEG {}: 4x1 unsigned 1-bit, depth=0, 9/7, smallest positive global scale * largest finite LL scale: exact source equality", openjpeg::version());
}

#[test]
fn clipped_high_bit_baseline_has_an_explicit_active_profile_contract() {
    let pixels: Vec<_> = [1_u32 << 29; 16]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let mut options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 0,
        use_mct: false,
        ..Default::default()
    };
    let plain = encode(&pixels, 4, 4, 1, 30, false, &options).unwrap();
    assert_eq!(marker(&plain, 0x51)[36], 29);
    assert_eq!(steps(&plain, 0x5c), [(31, 0)]); // R=30: delta=0.5
    options.irreversible_quantization_profile = Some(vec![vec![1.0]]);
    assert_eq!(
        encode(&pixels, 4, 4, 1, 30, false, &options).unwrap(),
        plain
    );
    for weight in [0.99, 1.01, 1.5] {
        options.irreversible_quantization_profile = Some(vec![vec![weight]]);
        assert!(matches!(
            encode(&pixels, 4, 4, 1, 30, false, &options),
            Err(j2k_native::EncodeError::InvalidInput { .. })
        ));
    }
    for (weight, mantissa) in [(2.0, 0), (2.03125, 32)] {
        options.irreversible_quantization_profile = Some(vec![vec![weight]]);
        let bytes = encode(&pixels, 4, 4, 1, 30, false, &options).unwrap();
        assert_eq!(steps(&bytes, 0x5c), [(31, mantissa)]);
        let decoded = openjpeg::decode_components(&bytes).unwrap();
        assert_eq!(decoded.components[0].bit_depth, 30);
        assert_eq!(decoded.components[0].samples, vec![1 << 29; 16]);
    }
    eprintln!("OpenJPEG {}: 4x4 unsigned 30-bit, depth=0, 9/7, active weights 2 and 2.03125: exact constant 2^29; nearby nominal weights rejected", openjpeg::version());
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "keep source, independent decoder, and requested-format assertions together for each matrix case"
)]
fn native_and_requested_formats_have_distinct_precision_contracts() {
    use j2k_core::PixelFormat;
    for bits in [7_u8, 8, 9, 16] {
        let unsigned_max = (1_i32 << bits) - 1;
        for signed in [false, true] {
            let values = if signed {
                [-(1 << (bits - 1)), -1, 0, (1 << (bits - 1)) - 1]
            } else {
                [0, 1, unsigned_max / 2, unsigned_max]
            };
            for (channels, mct) in [(1, false), (3, false), (3, true), (4, false), (4, true)] {
                let source: Vec<i32> = (0..4)
                    .flat_map(|x| (0..channels).map(move |c| values[(x + c) % 4]))
                    .collect();
                let sample_bytes = if bits <= 8 { 1 } else { 2 };
                let pixels: Vec<u8> = source
                    .iter()
                    .flat_map(|v| v.to_le_bytes()[..sample_bytes].to_vec())
                    .collect();
                let bytes = encode(
                    &pixels,
                    4,
                    1,
                    u16::try_from(channels).unwrap(),
                    bits,
                    signed,
                    &EncodeOptions {
                        num_decomposition_levels: 0,
                        use_mct: mct,
                        ..Default::default()
                    },
                )
                .unwrap();
                let siz = marker(&bytes, 0x51);
                assert_eq!(
                    usize::from(u16::from_be_bytes([siz[34], siz[35]])),
                    channels
                );
                for entry in siz[36..].as_chunks::<3>().0 {
                    assert_eq!(*entry, [(bits - 1) | if signed { 0x80 } else { 0 }, 1, 1]);
                }
                let bytes = if channels == 4 {
                    rgba_container(&bytes)
                } else {
                    bytes
                };
                let independent = openjpeg::decode_components(&bytes).unwrap();
                for (c, component) in independent.components.iter().enumerate() {
                    assert_eq!(
                        component.samples,
                        source
                            .iter()
                            .skip(c)
                            .step_by(channels)
                            .copied()
                            .collect::<Vec<_>>()
                    );
                }
                let native = Image::new(&bytes, &DecodeSettings::strict())
                    .unwrap()
                    .decode_native()
                    .unwrap();
                assert_eq!(
                    native.data, pixels,
                    "native {bits}-bit signed={signed} channels={channels} mct={mct}"
                );
                let formats = match channels {
                    1 => [PixelFormat::Gray8, PixelFormat::Gray16],
                    3 => [PixelFormat::Rgb8, PixelFormat::Rgb16],
                    _ => [PixelFormat::Rgba8, PixelFormat::Rgba16],
                };
                for format in formats {
                    let mut actual = vec![0; 4 * format.bytes_per_pixel()];
                    j2k::J2kDecoder::new(&bytes)
                        .unwrap()
                        .decode_into(&mut actual, 4 * format.bytes_per_pixel(), format)
                        .unwrap();
                    let expected: Vec<u8> = source
                        .iter()
                        .flat_map(|&v| {
                            if format.bytes_per_sample() == 1 {
                                let value = u8::try_from(
                                    (v.max(0) * 255 + unsigned_max / 2) / unsigned_max,
                                )
                                .unwrap();
                                vec![value]
                            } else if bits <= 8 {
                                let storage = i32::from(v.to_le_bytes()[0]);
                                let value = u16::try_from(
                                    ((storage * 65535 + unsigned_max / 2) / unsigned_max)
                                        .min(65535),
                                )
                                .unwrap();
                                value.to_le_bytes().to_vec()
                            } else {
                                v.to_le_bytes()[..2].to_vec()
                            }
                        })
                        .collect();
                    assert_eq!(
                        actual, expected,
                        "requested {format:?} {bits}-bit signed={signed} mct={mct}"
                    );
                }
            }
        }
    }
    eprintln!("OpenJPEG {}: 7/8/9/16-bit signed and unsigned, gray/RGB/RGBA, 5/3 with and without MCT: exact native source samples and explicit requested-format bytes", openjpeg::version());
}

fn rgba_container(bytes: &[u8]) -> Vec<u8> {
    let definitions: Vec<_> = (0..4_u16)
        .map(|c| j2k::J2kChannelDefinition {
            channel_index: c,
            channel_type: if c == 3 {
                j2k::J2kChannelType::Opacity
            } else {
                j2k::J2kChannelType::Color
            },
            association: if c == 3 {
                j2k::J2kChannelAssociation::WholeImage
            } else {
                j2k::J2kChannelAssociation::Color { index: c + 1 }
            },
        })
        .collect();
    j2k::wrap_j2k_codestream(
        bytes,
        j2k::J2kFileWrapOptions::jp2()
            .with_color(j2k::J2kFileColorSpec::Enumerated(
                j2k_core::Colorspace::SRgb,
            ))
            .with_metadata(j2k::J2kFileBoxMetadata {
                palette: None,
                component_mappings: &[],
                channel_definitions: &definitions,
            }),
    )
    .unwrap()
}
