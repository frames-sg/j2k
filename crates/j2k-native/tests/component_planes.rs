use j2k_native::{
    encode, encode_typed_component_planes_53, inspect_j2k_codestream_header, DecodeError,
    DecodeSettings, DecoderContext, EncodeOptions, EncodeTypedComponentPlane, Image,
};

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
    assert!(err.contains("no-quantization guard/exponent signaling limit"));
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
