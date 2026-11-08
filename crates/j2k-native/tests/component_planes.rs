use j2k_native::{
    encode, encode_typed_component_planes_53, inspect_j2k_codestream_header, DecodeError,
    DecodeSettings, DecoderContext, EncodeOptions, EncodeProgressionOrder,
    EncodeTypedComponentPlane, Image,
};

#[test]
fn sampled_high_bit_tiles_preserve_absolute_component_origins() {
    let (width, height) = (111_u32, 83_u32);
    let full: Vec<u8> = (0..width * height)
        .flat_map(|index| (index.wrapping_mul(1_000_003) & ((1 << 29) - 1)).to_le_bytes())
        .collect();
    let sampled: Vec<u8> = (0..width.div_ceil(3) * height.div_ceil(2))
        .flat_map(|index| (index.wrapping_mul(7_000_019) & ((1 << 29) - 1)).to_le_bytes())
        .collect();
    let planes = [
        EncodeTypedComponentPlane {
            data: &sampled,
            x_rsiz: 3,
            y_rsiz: 2,
            bit_depth: 29,
            signed: false,
        },
        EncodeTypedComponentPlane {
            data: &full,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 29,
            signed: false,
        },
    ];
    for use_ht_block_coding in [false, true] {
        let options = EncodeOptions {
            num_decomposition_levels: 2,
            tile_size: Some((33, 25)),
            code_block_width_exp: 1,
            code_block_height_exp: 1,
            precinct_exponents: vec![(3, 3), (4, 4), (4, 4)],
            progression_order: EncodeProgressionOrder::Pcrl,
            use_ht_block_coding,
            use_mct: false,
            ..EncodeOptions::default()
        };
        let encoded = encode_typed_component_planes_53(&planes, width, height, &options)
            .expect("encode sampled high-bit tiles");
        let decoded = Image::new(&encoded, &DecodeSettings::strict())
            .and_then(|image| image.decode_native_components())
            .expect("decode sampled high-bit tiles");
        // Native component output is expanded onto the reference grid.
        assert_eq!(decoded.planes()[0].dimensions(), (width, height));
        for y in 0..height {
            for x in 0..width {
                let output = ((y * width + x) * 4) as usize;
                let input = (((y / 2) * width.div_ceil(3) + x / 3) * 4) as usize;
                assert_eq!(
                    &decoded.planes()[0].data()[output..output + 4],
                    &sampled[input..input + 4],
                    "sampled ({x}, {y}), HT={use_ht_block_coding}"
                );
                assert_eq!(
                    &decoded.planes()[1].data()[output..output + 4],
                    &full[output..output + 4],
                    "full ({x}, {y}), HT={use_ht_block_coding}"
                );
            }
        }
    }
}

fn fixture() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode")
}

fn rewrite_component_descriptor(bytes: &mut [u8], component: usize, ssiz: u8) {
    let siz_offset = bytes
        .windows(2)
        .position(|marker| marker == [0xff, 0x51])
        .expect("SIZ marker");
    bytes[siz_offset + 40 + component * 3] = ssiz;
}

fn signed_12_bytes(sample: i16) -> [u8; 2] {
    let raw = u16::try_from(i32::from(sample) & 0x0fff).expect("masked 12-bit sample fits u16");
    raw.to_le_bytes()
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "decoded fixture samples are integral and constrained to their declared u8 range"
)]
fn rounded_u8(sample: f32) -> u8 {
    sample.round() as u8
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "decoded fixture samples are integral and constrained to their declared i8 range"
)]
fn rounded_i8(sample: f32) -> i8 {
    sample.round() as i8
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "decoded fixture samples are integral and constrained to their declared i16 range"
)]
fn rounded_i16(sample: f32) -> i16 {
    sample.round() as i16
}

#[test]
fn decoded_components_expose_component_planes() {
    let bytes = fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let bitmap = image
        .decode_with_context(&mut DecoderContext::default())
        .expect("bitmap");
    let planes = image
        .decode_components_with_context(&mut context)
        .expect("component decode");

    assert_eq!(planes.dimensions(), (2, 2));
    assert_eq!(planes.planes().len(), 3);
    assert_eq!(planes.planes()[0].bit_depth(), 8);
    assert!(planes
        .planes()
        .iter()
        .all(|plane| plane.samples().len() == 4));

    let mut interleaved = Vec::with_capacity(12);
    for idx in 0..4 {
        for plane in planes.planes() {
            interleaved.push(rounded_u8(plane.samples()[idx]));
        }
    }
    assert_eq!(interleaved, bitmap.data);
}

#[test]
fn decode_native_rejects_mixed_component_bit_depths() {
    let mut bytes = fixture();
    rewrite_component_descriptor(&mut bytes, 1, 11);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image parses");

    let Err(err) = image.decode_native() else {
        panic!("mixed bit depths cannot be packed as RawBitmap");
    };

    assert!(matches!(err, DecodeError::Decoding(_)));
    assert!(err
        .to_string()
        .contains("decode_native requires uniform component bit depths"));
}

#[test]
fn decode_native_components_handles_mixed_component_metadata() {
    let mut bytes = fixture();
    rewrite_component_descriptor(&mut bytes, 1, 11);
    rewrite_component_descriptor(&mut bytes, 2, 0x87);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image parses");

    let decoded = image
        .decode_native_components()
        .expect("native component decode");

    assert_eq!(decoded.dimensions(), (2, 2));
    assert_eq!(decoded.planes().len(), 3);
    assert_eq!(decoded.planes()[0].bit_depth(), 8);
    assert_eq!(decoded.planes()[0].bytes_per_sample(), 1);
    assert!(!decoded.planes()[0].signed());
    assert_eq!(decoded.planes()[0].data().len(), 4);
    assert_eq!(decoded.planes()[1].bit_depth(), 12);
    assert_eq!(decoded.planes()[1].bytes_per_sample(), 2);
    assert!(!decoded.planes()[1].signed());
    assert_eq!(decoded.planes()[1].data().len(), 8);
    assert_eq!(decoded.planes()[2].bit_depth(), 8);
    assert_eq!(decoded.planes()[2].bytes_per_sample(), 1);
    assert!(decoded.planes()[2].signed());
    assert_eq!(decoded.planes()[2].data().len(), 4);
}

#[test]
fn typed_component_plane_encode_preserves_mixed_metadata_for_classic_and_htj2k() {
    let unsigned = [0_u8, 7, 128, 255];
    let signed_samples = [-32_i16, -1, 0, 1023];
    let signed = signed_samples
        .iter()
        .flat_map(|sample| signed_12_bytes(*sample))
        .collect::<Vec<_>>();
    let planes = [
        EncodeTypedComponentPlane {
            data: &unsigned,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 8,
            signed: false,
        },
        EncodeTypedComponentPlane {
            data: &signed,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 12,
            signed: true,
        },
    ];

    for use_ht_block_coding in [false, true] {
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            use_mct: false,
            use_ht_block_coding,
            validate_high_throughput_codestream: false,
            ..EncodeOptions::default()
        };
        let bytes = encode_typed_component_planes_53(&planes, 2, 2, &options)
            .expect("typed component-plane encode");

        let header = inspect_j2k_codestream_header(&bytes).expect("inspect encoded header");
        assert_eq!(header.component_info.len(), 2);
        assert_eq!(header.component_info[0].bit_depth, 8);
        assert!(!header.component_info[0].signed);
        assert_eq!(header.component_info[1].bit_depth, 12);
        assert!(header.component_info[1].signed);
        assert!(
            bytes.windows(2).any(|marker| marker == [0xff, 0x5d]),
            "mixed component precision should write QCC"
        );

        let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse encoded image");
        let mut context = DecoderContext::default();
        let decoded = image
            .decode_components_with_context(&mut context)
            .expect("decode typed component codestream");

        assert_eq!(decoded.planes().len(), 2);
        assert_eq!(decoded.planes()[0].bit_depth(), 8);
        assert!(!decoded.planes()[0].signed());
        assert_eq!(decoded.planes()[1].bit_depth(), 12);
        assert!(decoded.planes()[1].signed());
        let decoded_unsigned = decoded.planes()[0]
            .samples()
            .iter()
            .copied()
            .map(rounded_u8)
            .collect::<Vec<_>>();
        let decoded_signed = decoded.planes()[1]
            .samples()
            .iter()
            .copied()
            .map(rounded_i16)
            .collect::<Vec<_>>();

        assert_eq!(decoded_unsigned, unsigned);
        assert_eq!(decoded_signed, signed_samples);
    }
}

#[test]
fn decode_native_region_components_preserves_plane_metadata() {
    let mut bytes = fixture();
    rewrite_component_descriptor(&mut bytes, 1, 15);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image parses");

    let decoded = image
        .decode_native_region_components((1, 0, 1, 2))
        .expect("native component region decode");

    assert_eq!(decoded.dimensions(), (1, 2));
    assert_eq!(decoded.planes().len(), 3);
    assert!(decoded
        .planes()
        .iter()
        .all(|plane| plane.dimensions() == (1, 2)));
    assert_eq!(decoded.planes()[0].data().len(), 2);
    assert_eq!(decoded.planes()[1].bit_depth(), 16);
    assert_eq!(decoded.planes()[1].bytes_per_sample(), 2);
    assert_eq!(decoded.planes()[1].data().len(), 4);
    assert_eq!(decoded.planes()[2].data().len(), 2);
}

#[test]
fn decode_native_accepts_gt24_bit_integer_packed_output() {
    let samples = [0_u32, 1, (1_u32 << 28) + 7, (1_u32 << 29) - 1];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_29_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 2, 2, 1, 29, false, &options).expect("encode gray29");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image parses");

    let decoded = image.decode_native().expect("native high-bit decode");

    assert_eq!(decoded.bit_depth, 29);
    assert_eq!(decoded.bytes_per_sample, 4);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn decode_components_rejects_gt24_bit_float_planes() {
    let mut bytes = fixture();
    for component in 0..3 {
        rewrite_component_descriptor(&mut bytes, component, 24);
    }
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image parses");
    let mut context = DecoderContext::default();

    let Err(err) = image.decode_components_with_context(&mut context) else {
        panic!(">24-bit samples cannot be represented exactly as f32 planes");
    };

    assert!(matches!(err, DecodeError::Decoding(_)));
    assert!(err
        .to_string()
        .contains("decode_components currently supports component planes up to 24 bits"));
}

#[test]
fn decoded_region_components_expose_cropped_component_planes() {
    let bytes = fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let bitmap = image
        .decode_region_with_context((1, 0, 1, 2), &mut DecoderContext::default())
        .expect("bitmap");
    let planes = image
        .decode_region_components_with_context((1, 0, 1, 2), &mut context)
        .expect("component region decode");

    assert_eq!(planes.dimensions(), (1, 2));
    assert_eq!(planes.planes().len(), 3);
    assert!(planes
        .planes()
        .iter()
        .all(|plane| plane.samples().len() == 2));

    let mut interleaved = Vec::with_capacity(6);
    for idx in 0..2 {
        for plane in planes.planes() {
            interleaved.push(rounded_u8(plane.samples()[idx]));
        }
    }
    assert_eq!(interleaved, bitmap.data);
}

fn five_component_pixels() -> Vec<u8> {
    (0..3 * 2 * 5)
        .map(|idx| {
            u8::try_from((idx * 17 + idx / 3) & 0xff).expect("test pattern is masked to one byte")
        })
        .collect()
}

#[test]
fn classic_encode_roundtrips_more_than_four_components() {
    let pixels = five_component_pixels();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 3, 2, 5, 8, false, &options).expect("encode five components");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let native = image.decode_native().expect("native decode");
    let mut context = DecoderContext::default();
    let components = image
        .decode_components_with_context(&mut context)
        .expect("component decode");

    assert_eq!(native.width, 3);
    assert_eq!(native.height, 2);
    assert_eq!(native.num_components, 5);
    assert_eq!(native.data, pixels);
    assert_eq!(components.planes().len(), 5);
    assert!(components
        .planes()
        .iter()
        .all(|plane| plane.samples().len() == 6));
}

#[test]
fn htj2k_encode_roundtrips_more_than_four_components() {
    let pixels = five_component_pixels();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let bytes =
        j2k_native::encode_htj2k(&pixels, 3, 2, 5, 8, false, &options).expect("encode five HT");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let native = image.decode_native().expect("native decode");

    assert_eq!(native.width, 3);
    assert_eq!(native.height, 2);
    assert_eq!(native.num_components, 5);
    assert_eq!(native.data, pixels);
}

#[test]
fn signed_gray8_roundtrips_through_component_planes_and_native_bytes() {
    let pixels = [(-10_i8).cast_unsigned(), (-1_i8).cast_unsigned(), 0, 12];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 2, 2, 1, 8, true, &options).expect("encode signed gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let native = image.decode_native().expect("native decode");
    let mut context = DecoderContext::default();
    let components = image
        .decode_components_with_context(&mut context)
        .expect("component decode");

    assert!(native.signed);
    assert_eq!(native.component_signed, [true]);
    assert_eq!(native.data, pixels);
    assert_eq!(components.planes().len(), 1);
    assert!(components.planes()[0].signed());
    assert_eq!(components.planes()[0].bit_depth(), 8);
    let decoded = components.planes()[0]
        .samples()
        .iter()
        .copied()
        .map(rounded_i8)
        .collect::<Vec<_>>();
    assert_eq!(decoded, [-10, -1, 0, 12]);
}

#[test]
fn signed_gray16_roundtrips_through_native_bytes() {
    let samples = [-300_i16, -1, 0, 300];
    let pixels = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 2, 2, 1, 16, true, &options).expect("encode signed gray16");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let native = image.decode_native().expect("native decode");
    let mut context = DecoderContext::default();
    let components = image
        .decode_components_with_context(&mut context)
        .expect("component decode");

    assert!(native.signed);
    assert_eq!(native.component_signed, [true]);
    assert_eq!(native.data, pixels);
    assert!(components.planes()[0].signed());
    assert_eq!(components.planes()[0].bit_depth(), 16);
    let decoded = components.planes()[0]
        .samples()
        .iter()
        .copied()
        .map(rounded_i16)
        .collect::<Vec<_>>();
    assert_eq!(decoded, samples);
}

fn unsigned_24_bytes(sample: u32) -> [u8; 3] {
    [
        (sample & 0xff) as u8,
        ((sample >> 8) & 0xff) as u8,
        ((sample >> 16) & 0xff) as u8,
    ]
}

fn signed_24_bytes(sample: i32) -> [u8; 3] {
    let raw = sample.cast_unsigned() & 0x00ff_ffff;
    unsigned_24_bytes(raw)
}

fn unsigned_38_bytes(sample: u64) -> [u8; 5] {
    [
        (sample & 0xff) as u8,
        ((sample >> 8) & 0xff) as u8,
        ((sample >> 16) & 0xff) as u8,
        ((sample >> 24) & 0xff) as u8,
        ((sample >> 32) & 0x3f) as u8,
    ]
}

fn unsigned_31_bytes(sample: u32) -> [u8; 4] {
    [
        (sample & 0xff) as u8,
        ((sample >> 8) & 0xff) as u8,
        ((sample >> 16) & 0xff) as u8,
        ((sample >> 24) & 0x7f) as u8,
    ]
}

fn unsigned_29_bytes(sample: u32) -> [u8; 4] {
    [
        (sample & 0xff) as u8,
        ((sample >> 8) & 0xff) as u8,
        ((sample >> 16) & 0xff) as u8,
        ((sample >> 24) & 0x1f) as u8,
    ]
}

fn signed_29_bytes(sample: i32) -> [u8; 4] {
    sample.to_le_bytes()
}

#[test]
fn unsigned_gray24_roundtrips_through_native_bytes() {
    let samples = [0_u32, 1, 0x12_34_56, 0xff_ff_ff];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_24_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 2, 2, 1, 24, false, &options).expect("encode gray24");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let native = image.decode_native().expect("native decode");

    assert!(!native.signed);
    assert_eq!(native.component_signed, [false]);
    assert_eq!(native.bit_depth, 24);
    assert_eq!(native.bytes_per_sample, 3);
    assert_eq!(native.data, pixels);
}

#[test]
fn signed_gray24_roundtrips_through_native_bytes() {
    let samples = [-8_388_608_i32, -1, 0, 8_388_607];
    let pixels = samples
        .iter()
        .flat_map(|sample| signed_24_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 2, 2, 1, 24, true, &options).expect("encode signed gray24");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let native = image.decode_native().expect("native decode");

    assert!(native.signed);
    assert_eq!(native.component_signed, [true]);
    assert_eq!(native.bit_depth, 24);
    assert_eq!(native.bytes_per_sample, 3);
    assert_eq!(native.data, pixels);
}

#[test]
fn classic_reversible_i64_encode_writes_29_bit_codestream_metadata() {
    let samples = [
        0_u32,
        1,
        (1_u32 << 28) + 17,
        (1_u32 << 29) - 1,
        0x1234_5678,
        0x03ff_ffff,
        42,
        (1_u32 << 27) - 3,
        (1_u32 << 26) + 99,
    ];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_29_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct: false,
        ..EncodeOptions::default()
    };

    let bytes = encode(&pixels, 3, 3, 1, 29, false, &options).expect("encode gray29");
    let header = inspect_j2k_codestream_header(&bytes).expect("inspect encoded gray29");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse encoded gray29");

    assert_eq!(header.dimensions, (3, 3));
    assert_eq!(header.components, 1);
    assert_eq!(header.bit_depth, 29);
    assert_eq!(header.component_info[0].bit_depth, 29);
    assert!(!header.component_info[0].signed);
    assert_eq!(image.width(), 3);
    assert_eq!(image.height(), 3);
}

#[test]
fn classic_reversible_i64_encode_rejects_38_bit_beyond_no_quant_bitplane_limit() {
    let samples = [
        0_u64,
        1,
        (1_u64 << 37) + 17,
        (1_u64 << 38) - 1,
        0x12_3456_789a,
        0x03_ffff_ffff,
        42,
        (1_u64 << 36) - 3,
        (1_u64 << 35) + 99,
    ];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_38_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct: false,
        ..EncodeOptions::default()
    };

    let err = encode(&pixels, 3, 3, 1, 38, false, &options)
        .expect_err("gray38 encode must not emit a truncated-bitplane codestream");
    assert!(matches!(
        err,
        j2k_native::EncodeError::Unsupported {
            what: "25-38 bit reversible encode exceeds the current no-quantization guard/exponent signaling limit"
        }
    ));
}

#[test]
fn classic_reversible_i64_decode_round_trips_29_bit_native_bytes() {
    let samples = [
        0_u32,
        1,
        (1_u32 << 28) + 17,
        (1_u32 << 29) - 1,
        0x1234_5678,
        0x03ff_ffff,
        42,
        (1_u32 << 27) - 3,
        (1_u32 << 26) + 99,
    ];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_29_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct: false,
        ..EncodeOptions::default()
    };

    let bytes = encode(&pixels, 3, 3, 1, 29, false, &options).expect("encode gray29");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse encoded gray29");
    let native = image.decode_native().expect("decode gray29");

    assert_eq!(native.width, 3);
    assert_eq!(native.height, 3);
    assert_eq!(native.num_components, 1);
    assert_eq!(native.bit_depth, 29);
    assert_eq!(native.bytes_per_sample, 4);
    assert!(!native.signed);
    assert_eq!(native.data, pixels);
}

#[test]
fn classic_reversible_i64_decode_round_trips_31_bit_native_bytes_without_dwt() {
    let samples = [0_u32, 1, (1_u32 << 30) + 17, (1_u32 << 31) - 1];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_31_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 0,
        use_mct: false,
        ..EncodeOptions::default()
    };

    let bytes = encode(&pixels, 2, 2, 1, 31, false, &options).expect("encode gray31");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse encoded gray31");
    let native = image.decode_native().expect("decode gray31");

    assert_eq!(native.bit_depth, 31);
    assert_eq!(native.bytes_per_sample, 4);
    assert_eq!(native.data, pixels);
}

#[test]
fn htj2k_reversible_i64_decode_round_trips_29_bit_native_bytes_without_dwt() {
    let samples = [
        0_u32,
        1,
        (1_u32 << 28) + 17,
        (1_u32 << 29) - 1,
        0x1234_5678,
        0x03ff_ffff,
        42,
        (1_u32 << 27) - 3,
        (1_u32 << 26) + 99,
    ];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_29_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 0,
        use_mct: false,
        ..EncodeOptions::default()
    };

    let bytes = encode(&pixels, 3, 3, 1, 29, false, &options).expect("encode HT gray29");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse encoded HT gray29");
    let native = image.decode_native().expect("decode HT gray29");

    assert_eq!(native.bit_depth, 29);
    assert_eq!(native.bytes_per_sample, 4);
    assert_eq!(native.data, pixels);
}

#[test]
fn htj2k_reversible_i64_decode_round_trips_31_bit_native_bytes_without_dwt() {
    let samples = [
        0_u32,
        1,
        255,
        65_535,
        16_777_216,
        (1_u32 << 30) + 17,
        (1_u32 << 30) - 1,
        1_u32 << 30,
        (1_u32 << 31) - 1,
    ];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_31_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 0,
        use_mct: false,
        ..EncodeOptions::default()
    };

    let bytes = encode(&pixels, 3, 3, 1, 31, false, &options).expect("encode HT gray31");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse encoded HT gray31");
    let native = image.decode_native().expect("decode HT gray31");

    assert_eq!(native.bit_depth, 31);
    assert_eq!(native.bytes_per_sample, 4);
    assert_eq!(native.data, pixels);
}

#[test]
fn classic_reversible_i64_decode_native_components_preserves_29_bit_plane() {
    let samples = [
        0_u32,
        1,
        (1_u32 << 28) + 17,
        (1_u32 << 29) - 1,
        0x1234_5678,
        0x03ff_ffff,
        42,
        (1_u32 << 27) - 3,
        (1_u32 << 26) + 99,
    ];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_29_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct: false,
        ..EncodeOptions::default()
    };

    let bytes = encode(&pixels, 3, 3, 1, 29, false, &options).expect("encode gray29");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse encoded gray29");
    let components = image
        .decode_native_components()
        .expect("decode native gray29 components");

    assert_eq!(components.dimensions(), (3, 3));
    assert_eq!(components.planes().len(), 1);
    assert_eq!(components.planes()[0].dimensions(), (3, 3));
    assert_eq!(components.planes()[0].bit_depth(), 29);
    assert_eq!(components.planes()[0].bytes_per_sample(), 4);
    assert!(!components.planes()[0].signed());
    assert_eq!(components.planes()[0].data(), pixels);
}

#[test]
fn classic_reversible_i64_decode_round_trips_signed_29_bit_native_bytes() {
    let samples = [
        -(1_i32 << 28),
        -65_537,
        -1,
        0,
        1,
        65_537,
        (1_i32 << 28) - 1,
        123_456,
        -123_456,
    ];
    let pixels = samples
        .iter()
        .flat_map(|sample| signed_29_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct: false,
        ..EncodeOptions::default()
    };

    let bytes = encode(&pixels, 3, 3, 1, 29, true, &options).expect("encode signed gray29");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse signed gray29");
    let native = image.decode_native().expect("decode signed gray29");
    let components = image
        .decode_native_components()
        .expect("decode native signed gray29 components");

    assert!(native.signed);
    assert_eq!(native.component_signed, [true]);
    assert_eq!(native.bit_depth, 29);
    assert_eq!(native.bytes_per_sample, 4);
    assert_eq!(native.data, pixels);
    assert_eq!(components.planes()[0].bit_depth(), 29);
    assert!(components.planes()[0].signed());
    assert_eq!(components.planes()[0].data(), pixels);
}

#[test]
fn classic_reversible_i64_decode_round_trips_rgb29_rct_native_bytes() {
    let mut pixels = Vec::new();
    for y in 0..3_u32 {
        for x in 0..3_u32 {
            for c in 0..3_u32 {
                let sample = ((x * 17_000_003 + y * 9_000_001 + c * 33_333_331)
                    & ((1_u32 << 29) - 1))
                    .min((1_u32 << 29) - 1);
                pixels.extend_from_slice(&unsigned_29_bytes(sample));
            }
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct: true,
        ..EncodeOptions::default()
    };

    let bytes = encode(&pixels, 3, 3, 3, 29, false, &options).expect("encode rgb29");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse rgb29");
    let native = image.decode_native().expect("decode rgb29");

    assert_eq!(native.width, 3);
    assert_eq!(native.height, 3);
    assert_eq!(native.num_components, 3);
    assert_eq!(native.bit_depth, 29);
    assert_eq!(native.bytes_per_sample, 4);
    assert_eq!(native.data, pixels);
}

#[test]
fn classic_reversible_i64_decode_native_region_crops_29_bit_bytes() {
    let samples = [
        0_u32,
        1,
        2,
        3,
        (1_u32 << 28) + 4,
        (1_u32 << 28) + 5,
        (1_u32 << 29) - 3,
        (1_u32 << 29) - 2,
        (1_u32 << 29) - 1,
    ];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_29_bytes(*sample))
        .collect::<Vec<_>>();
    let expected = [samples[4], samples[5]]
        .iter()
        .flat_map(|sample| unsigned_29_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct: false,
        ..EncodeOptions::default()
    };

    let bytes = encode(&pixels, 3, 3, 1, 29, false, &options).expect("encode gray29");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse gray29");
    let native = image
        .decode_native_region((1, 1, 2, 1))
        .expect("decode high-bit native region");
    let components = image
        .decode_native_region_components((1, 1, 2, 1))
        .expect("decode high-bit native component region");

    assert_eq!(native.width, 2);
    assert_eq!(native.height, 1);
    assert_eq!(native.bit_depth, 29);
    assert_eq!(native.data, expected);
    assert_eq!(components.dimensions(), (2, 1));
    assert_eq!(components.planes()[0].dimensions(), (2, 1));
    assert_eq!(components.planes()[0].bit_depth(), 29);
    assert_eq!(components.planes()[0].data(), expected);
}

fn exact_region_pixels(size: (u32, u32), components: u16, bit_depth: u8, signed: bool) -> Vec<u8> {
    let mask = (1_u64 << bit_depth) - 1;
    let bytes_per_sample = usize::from(bit_depth.div_ceil(8));
    (0..u64::from(size.0) * u64::from(size.1) * u64::from(components))
        .flat_map(|index| {
            let value = match index % 31 {
                0 => 0,
                1 => mask,
                _ => ((index * 17_000_003) ^ ((index / u64::from(components)) * 9_000_001)) & mask,
            };
            let sample = i64::try_from(value).expect("fixture value fits i64")
                - if signed { 1_i64 << (bit_depth - 1) } else { 0 };
            sample.to_le_bytes().into_iter().take(bytes_per_sample)
        })
        .collect()
}

fn assert_exact_native_regions(image: &Image<'_>, pixels: Option<&[u8]>, tile_edge: (u32, u32)) {
    let full = image.decode_native().expect("decode full exact image");
    if let Some(pixels) = pixels {
        assert_eq!(full.data, pixels, "full image must equal the reference");
    }
    let (width, height) = (image.width(), image.height());
    let bytes_per_sample = usize::from(full.bytes_per_sample);
    let pixel_stride = usize::from(full.num_components) * bytes_per_sample;
    let row_stride = width as usize * pixel_stride;
    for roi @ (x, y, roi_width, roi_height) in [
        (0, 0, 1, 1),
        (width - 1, height - 1, 1, 1),
        (0, height / 2, width, 1),
        (width / 2, 0, 1, height),
        (
            width / 3,
            height / 3,
            (width - width / 3).min(5),
            (height - height / 3).min(4),
        ),
        (tile_edge.0 - 1, tile_edge.1 - 1, 3, 3),
        (0, 0, width, height),
    ] {
        let expected: Vec<u8> = full
            .data
            .chunks_exact(row_stride)
            .skip(y as usize)
            .take(roi_height as usize)
            .flat_map(|row| {
                row[x as usize * pixel_stride..(x + roi_width) as usize * pixel_stride]
                    .iter()
                    .copied()
            })
            .collect();
        let native = image.decode_native_region(roi).expect("decode exact ROI");
        assert_eq!((native.width, native.height), (roi_width, roi_height));
        assert_eq!(native.bit_depth, full.bit_depth);
        assert_eq!(native.signed, full.signed);
        assert_eq!(native.data, expected, "native ROI {roi:?}");
        let components = image
            .decode_native_region_components(roi)
            .expect("decode exact component ROI");
        assert_eq!(components.dimensions(), (roi_width, roi_height));
        assert_eq!(components.planes().len(), usize::from(full.num_components));
        for (index, plane) in components.planes().iter().enumerate() {
            let expected_plane: Vec<u8> = expected
                .chunks_exact(pixel_stride)
                .flat_map(|pixel| {
                    pixel[index * bytes_per_sample..(index + 1) * bytes_per_sample]
                        .iter()
                        .copied()
                })
                .collect();
            assert_eq!(plane.dimensions(), (roi_width, roi_height));
            assert_eq!(plane.bit_depth(), full.bit_depth);
            assert_eq!(plane.signed(), full.signed);
            assert_eq!(
                plane.data(),
                expected_plane,
                "component {index}, ROI {roi:?}"
            );
        }
    }
}

#[test]
fn exact_native_regions_match_full_decode_across_tiles_and_reductions() {
    let size = (73, 57);
    for use_ht_block_coding in [false, true] {
        for signed in [false, true] {
            for components in [1, 3] {
                let pixels = exact_region_pixels(size, components, 29, signed);
                let levels = if use_ht_block_coding { 0 } else { 2 };
                let options = EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: levels,
                    tile_size: Some((33, 25)),
                    use_mct: components == 3,
                    use_ht_block_coding,
                    ..EncodeOptions::default()
                };
                let bytes = encode(&pixels, size.0, size.1, components, 29, signed, &options)
                    .expect("encode tiled exact image");
                for reduction in 0..=levels {
                    let image =
                        Image::new_with_reduction(&bytes, &DecodeSettings::strict(), reduction)
                            .expect("parse tiled exact image");
                    assert_exact_native_regions(
                        &image,
                        (reduction == 0).then_some(pixels.as_slice()),
                        (
                            33_u32.div_ceil(1 << reduction),
                            25_u32.div_ceil(1 << reduction),
                        ),
                    );
                }
            }
        }
    }
}

#[test]
fn unsigned_35_bit_rct_regions_preserve_exact_native_bytes() {
    let size = (73, 57);
    let pixels = exact_region_pixels(size, 3, 35, false);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 0,
        tile_size: Some((33, 25)),
        use_mct: true,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, size.0, size.1, 3, 35, false, &options)
        .expect("encode unsigned 35-bit RCT image");
    let image =
        Image::new(&bytes, &DecodeSettings::strict()).expect("parse unsigned 35-bit RCT image");
    assert_exact_native_regions(&image, Some(&pixels), (33, 25));
}

#[test]
fn exact_native_regions_match_openjpeg_at_nonzero_image_origin() {
    let bytes = include_bytes!(
        "../../j2k-test-support/fixtures/openjpeg_origin/gray29_23x17_origin_3_3.j2k"
    );
    let source: Vec<u8> = (0..17_u32)
        .flat_map(|y| {
            (0..23_u32).flat_map(move |x| {
                unsigned_29_bytes((1 << 28) + x * 17_003 + y * 9_001 + x * y * 997)
            })
        })
        .collect();
    let references: [&[u8]; 3] = [
        &source,
        include_bytes!("../../j2k-test-support/fixtures/openjpeg_origin/gray29_reduced_r1.raw"),
        include_bytes!("../../j2k-test-support/fixtures/openjpeg_origin/gray29_reduced_r2.raw"),
    ];
    for (reduction, (reference, dimensions)) in references
        .iter()
        .zip([(23, 17), (11, 8), (6, 4)])
        .enumerate()
    {
        let image = Image::new_with_reduction(
            bytes,
            &DecodeSettings::strict(),
            u8::try_from(reduction).expect("fixture reduction fits u8"),
        )
        .expect("parse exact image with odd origin");
        assert_eq!((image.width(), image.height()), dimensions);
        assert_exact_native_regions(
            &image,
            Some(reference),
            (image.width().div_ceil(2), image.height().div_ceil(2)),
        );
    }
}

#[test]
fn subsampled_nonzero_origin_decodes_full_and_reduced_regions() {
    // Minimized wsi-rs open_jp2k_codestream_bytes fuzz regression: XOsiz=2,
    // and component 1 has vertical sampling 47 with two decomposition levels.
    let bytes = include_bytes!("../fixtures/subsampled-origin.j2c");
    for reduction in 0..=2 {
        let image = Image::new_with_reduction(bytes, &DecodeSettings::strict(), reduction)
            .expect("parse subsampled nonzero-origin codestream");
        let region = image
            .decode_region((0, 0, image.width(), image.height()))
            .expect("decode full image region without coordinate underflow");
        let full = image.decode().expect("decode full subsampled image");
        assert_eq!(region.data, full, "reduction {reduction}");

        let width = image.width() as usize;
        let height = image.height() as usize;
        let cropped = image
            .decode_region((1, 1, image.width() - 2, image.height() - 2))
            .expect("decode clipped subsampled region");
        let expected: Vec<u8> = full
            .chunks_exact(width * 3)
            .skip(1)
            .take(height - 2)
            .flat_map(|row| row[3..(width - 1) * 3].iter().copied())
            .collect();
        assert_eq!(cropped.data, expected, "cropped reduction {reduction}");
    }
}

struct SampledPrecinctCase {
    size: (u32, u32),
    bit_depth: u8,
    sampling: [(u8, u8); 2],
    levels: u8,
    precincts: &'static [(u8, u8)],
    code_block_exp: u8,
}

fn sampled_extent(case: &SampledPrecinctCase, component: usize) -> (u32, u32) {
    let (x_rsiz, y_rsiz) = case.sampling[component];
    (
        case.size.0.div_ceil(u32::from(x_rsiz)),
        case.size.1.div_ceil(u32::from(y_rsiz)),
    )
}

fn sampled_plane_bytes(case: &SampledPrecinctCase, component: usize) -> Vec<u8> {
    let (width, height) = sampled_extent(case, component);
    let mask = (1_u64 << case.bit_depth) - 1;
    let bytes_per_sample = usize::from(case.bit_depth.div_ceil(8));
    let mut state = 0x9e37_79b9_u32 ^ u32::try_from(component).unwrap().wrapping_mul(0x85eb_ca6b);
    (0..width * height)
        .flat_map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let sample = u64::from(state) & mask;
            sample.to_le_bytes()[..bytes_per_sample].to_vec()
        })
        .collect()
}

/// Native-component decode returns reference-grid planes, repeating each
/// sampled sample over its `XRsiz` x `YRsiz` footprint.
fn sampled_reference_grid_bytes(
    case: &SampledPrecinctCase,
    component: usize,
    native: &[u8],
) -> Vec<u8> {
    let (x_rsiz, y_rsiz) = case.sampling[component];
    let (component_width, _) = sampled_extent(case, component);
    let bytes_per_sample = usize::from(case.bit_depth.div_ceil(8));
    (0..case.size.1)
        .flat_map(|y| (0..case.size.0).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            let index =
                usize::try_from((y / u32::from(y_rsiz)) * component_width + x / u32::from(x_rsiz))
                    .unwrap();
            native[index * bytes_per_sample..(index + 1) * bytes_per_sample].to_vec()
        })
        .collect()
}

/// Packets a conforming decoder expects for one tile at the origin: each
/// tile-component resolution is partitioned on its own extent (T.800 B.6).
fn sampled_expected_packet_count(case: &SampledPrecinctCase) -> usize {
    (0..case.sampling.len())
        .map(|component| {
            let (width, height) = sampled_extent(case, component);
            (0..=case.levels)
                .map(|resolution| {
                    let shift = u32::from(case.levels - resolution);
                    let (precinct_x, precinct_y) = case.precincts[usize::from(resolution)];
                    let columns = width.div_ceil(1 << shift).div_ceil(1 << precinct_x);
                    let rows = height.div_ceil(1 << shift).div_ceil(1 << precinct_y);
                    usize::try_from(columns * rows).unwrap()
                })
                .sum::<usize>()
        })
        .sum()
}

const P677: &[(u8, u8)] = &[(6, 6), (7, 7), (7, 7)];

const SAMPLED_PRECINCT_CASES: [SampledPrecinctCase; 4] = [
    // Smallest failing shape: the (2,2) component is one 4x4 precinct, while
    // the reference-grid tile would split it into four.
    SampledPrecinctCase {
        size: (8, 8),
        bit_depth: 8,
        sampling: [(2, 2), (1, 1)],
        levels: 0,
        precincts: &[(2, 2)],
        code_block_exp: 0,
    },
    SampledPrecinctCase {
        size: (8, 8),
        bit_depth: 8,
        sampling: [(1, 1), (2, 2)],
        levels: 0,
        precincts: &[(2, 2)],
        code_block_exp: 0,
    },
    SampledPrecinctCase {
        size: (256, 256),
        bit_depth: 16,
        sampling: [(1, 2), (1, 1)],
        levels: 2,
        precincts: P677,
        code_block_exp: 4,
    },
    SampledPrecinctCase {
        size: (256, 256),
        bit_depth: 29,
        sampling: [(2, 2), (1, 1)],
        levels: 2,
        precincts: P677,
        code_block_exp: 4,
    },
];

fn assert_sampled_precinct_round_trip(
    case: &SampledPrecinctCase,
    data: &[Vec<u8>],
    use_ht_block_coding: bool,
    progression_order: EncodeProgressionOrder,
) {
    let label = format!(
        "{:?} {}-bit {:?} ht={use_ht_block_coding} {progression_order:?}",
        case.size, case.bit_depth, case.sampling
    );
    let planes = data
        .iter()
        .zip(case.sampling)
        .map(|(data, (x_rsiz, y_rsiz))| EncodeTypedComponentPlane {
            data,
            x_rsiz,
            y_rsiz,
            bit_depth: case.bit_depth,
            signed: false,
        })
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: case.levels,
        use_mct: false,
        use_ht_block_coding,
        progression_order,
        write_sop: true,
        code_block_width_exp: case.code_block_exp,
        code_block_height_exp: case.code_block_exp,
        precinct_exponents: case.precincts.to_vec(),
        ..EncodeOptions::default()
    };
    let bytes = encode_typed_component_planes_53(&planes, case.size.0, case.size.1, &options)
        .unwrap_or_else(|error| panic!("{label}: encode: {error}"));
    assert_eq!(
        bytes
            .windows(2)
            .filter(|marker| *marker == [0xff, 0x91])
            .count(),
        sampled_expected_packet_count(case),
        "{label}: packet count"
    );
    let decoded = Image::new(&bytes, &DecodeSettings::default())
        .and_then(|image| image.decode_native_components())
        .unwrap_or_else(|error| panic!("{label}: decode: {error}"));
    for (component, native) in data.iter().enumerate() {
        let plane = &decoded.planes()[component];
        assert_eq!(
            plane.dimensions(),
            case.size,
            "{label}: component {component}"
        );
        assert!(
            plane.data() == sampled_reference_grid_bytes(case, component, native),
            "{label}: component {component} samples differ"
        );
    }
}

#[test]
fn sampled_components_with_custom_precincts_round_trip_exactly() {
    for case in &SAMPLED_PRECINCT_CASES {
        let data = (0..case.sampling.len())
            .map(|component| sampled_plane_bytes(case, component))
            .collect::<Vec<_>>();
        let block_codings: &[bool] = if case.bit_depth > 16 {
            &[false]
        } else {
            &[false, true]
        };
        for &use_ht_block_coding in block_codings {
            for progression_order in [
                EncodeProgressionOrder::Lrcp,
                EncodeProgressionOrder::Rlcp,
                EncodeProgressionOrder::Rpcl,
                EncodeProgressionOrder::Pcrl,
                EncodeProgressionOrder::Cprl,
            ] {
                assert_sampled_precinct_round_trip(
                    case,
                    &data,
                    use_ht_block_coding,
                    progression_order,
                );
            }
        }
    }
}

/// Whole-component ROI shifts reach the codestream at every typed bit depth,
/// including the >24-bit exact route, and the planes still round-trip exactly.
#[test]
fn typed_component_planes_keep_whole_component_roi_shifts_at_every_bit_depth() {
    // HTJ2K codes at most 31 bitplanes, so 29-bit samples plus the shift are
    // classic-only.
    for (bit_depth, block_coders) in [
        (16_u8, &[false, true][..]),
        (25, &[false, true]),
        (29, &[false]),
    ] {
        let samples: Vec<u32> = (0_u32..16 * 16)
            .map(|index| index.wrapping_mul(2_654_435_761) & ((1 << bit_depth) - 1))
            .collect();
        let data: Vec<u8> = samples
            .iter()
            .flat_map(|&sample| {
                if bit_depth <= 16 {
                    u16::try_from(sample)
                        .expect("16-bit fixture sample")
                        .to_le_bytes()
                        .to_vec()
                } else {
                    sample.to_le_bytes().to_vec()
                }
            })
            .collect();
        let planes = [
            EncodeTypedComponentPlane {
                data: &data,
                x_rsiz: 1,
                y_rsiz: 1,
                bit_depth,
                signed: false,
            },
            EncodeTypedComponentPlane {
                data: &data,
                x_rsiz: 1,
                y_rsiz: 1,
                bit_depth,
                signed: false,
            },
        ];
        for (&use_ht_block_coding, tile_size) in block_coders
            .iter()
            .flat_map(|ht| [(ht, None), (ht, Some((8, 8)))])
            // Only the >24-bit typed route encodes several tiles.
            .filter(|(_, tile_size)| bit_depth > 24 || tile_size.is_none())
        {
            let options = EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                use_mct: false,
                use_ht_block_coding,
                tile_size,
                roi_component_shifts: vec![0, 3],
                ..EncodeOptions::default()
            };
            let label = format!("{bit_depth}-bit, HT={use_ht_block_coding}, tiles {tile_size:?}");
            let encoded = encode_typed_component_planes_53(&planes, 16, 16, &options)
                .unwrap_or_else(|err| panic!("{label}: {err:?}"));
            let rgn: Vec<&[u8]> = encoded
                .windows(7)
                .filter(|segment| segment[..2] == [0xff, 0x5e])
                .collect();
            assert_eq!(
                rgn,
                [&[0xff, 0x5e, 0x00, 0x05, 0x01, 0x00, 0x03][..]],
                "{label}: one RGN for component 1"
            );
            let decoded = Image::new(&encoded, &DecodeSettings::strict())
                .and_then(|image| image.decode_native_components())
                .unwrap_or_else(|err| panic!("{label}: {err:?}"));
            for plane in decoded.planes() {
                assert_eq!(plane.data(), data.as_slice(), "{label}");
            }
        }
    }
}

/// A >24-bit typed ROI shift list is validated like every other route.
#[test]
fn typed_high_bit_component_planes_validate_roi_shifts() {
    let data: Vec<u8> = (0_u32..8 * 8)
        .flat_map(|index| ((index * 9_973) & ((1 << 25) - 1)).to_le_bytes())
        .collect();
    let planes = [EncodeTypedComponentPlane {
        data: &data,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 25,
        signed: false,
    }];
    let encode = |roi_component_shifts: Vec<u8>, use_ht_block_coding: bool| {
        encode_typed_component_planes_53(
            &planes,
            8,
            8,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                use_mct: false,
                use_ht_block_coding,
                roi_component_shifts,
                ..EncodeOptions::default()
            },
        )
    };
    assert!(matches!(
        encode(vec![1, 1], false),
        Err(j2k_native::EncodeError::InvalidInput {
            what: "ROI component shift count does not match component count",
        })
    ));
    assert!(matches!(
        encode(vec![30], true),
        Err(j2k_native::EncodeError::Unsupported {
            what: "ROI maxshift exceeds supported coded bitplane count",
        })
    ));
}
