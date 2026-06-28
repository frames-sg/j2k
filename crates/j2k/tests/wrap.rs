// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    encode_j2k_lossless, wrap_j2k_codestream, J2kChannelAssociation, J2kChannelDefinition,
    J2kChannelType, J2kColorSpec, J2kComponentMapping, J2kComponentMappingType, J2kDecoder,
    J2kError, J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions, J2kLosslessEncodeOptions,
    J2kLosslessSamples, J2kPaletteColumn, J2kPaletteMetadata, PixelFormat,
};
use j2k_core::{Colorspace, CompressedPayloadKind};
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use j2k_test_support::minimal_j2k_codestream;

fn classic_codestream() -> Vec<u8> {
    let pixels = [5_u8, 16, 27, 38];
    encode(&pixels, 2, 2, 1, 8, false, &EncodeOptions::default()).expect("encode classic")
}

fn ht_codestream() -> Vec<u8> {
    let pixels = [5_u8, 16, 27, 38];
    encode_htj2k(&pixels, 2, 2, 1, 8, false, &EncodeOptions::default()).expect("encode ht")
}

fn rewrite_component_descriptor(bytes: &mut [u8], component: usize, ssiz: u8) {
    let siz_offset = bytes
        .windows(2)
        .position(|marker| marker == [0xff, 0x51])
        .expect("SIZ marker");
    bytes[siz_offset + 40 + component * 3] = ssiz;
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn push_box(out: &mut Vec<u8>, box_type: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(&(8_u32 + payload.len() as u32).to_be_bytes());
    out.extend_from_slice(box_type);
    out.extend_from_slice(payload);
}

fn jp2_with_palette_mapping(codestream: &[u8]) -> Vec<u8> {
    let mut jp2 = Vec::new();
    jp2.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0d, 0x0a, 0x87, 0x0a]);

    let mut ftyp = Vec::new();
    ftyp.extend_from_slice(b"jp2 ");
    ftyp.extend_from_slice(&0_u32.to_be_bytes());
    ftyp.extend_from_slice(b"jp2 ");
    push_box(&mut jp2, b"ftyp", &ftyp);

    let mut jp2h = Vec::new();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&3_u16.to_be_bytes());
    ihdr.extend_from_slice(&[7, 7, 0, 0]);
    push_box(&mut jp2h, b"ihdr", &ihdr);

    let mut colr = Vec::new();
    colr.extend_from_slice(&[1, 0, 0]);
    colr.extend_from_slice(&16_u32.to_be_bytes());
    push_box(&mut jp2h, b"colr", &colr);

    let mut pclr = Vec::new();
    pclr.extend_from_slice(&2_u16.to_be_bytes());
    pclr.push(3);
    pclr.extend_from_slice(&[7, 7, 7]);
    pclr.extend_from_slice(&[10, 20, 30]);
    pclr.extend_from_slice(&[200, 210, 220]);
    push_box(&mut jp2h, b"pclr", &pclr);

    let mut cmap = Vec::new();
    for column in 0..3_u8 {
        cmap.extend_from_slice(&0_u16.to_be_bytes());
        cmap.push(1);
        cmap.push(column);
    }
    push_box(&mut jp2h, b"cmap", &cmap);

    let mut cdef = Vec::new();
    cdef.extend_from_slice(&3_u16.to_be_bytes());
    for (channel, association) in [(0_u16, 1_u16), (1, 2), (2, 3)] {
        cdef.extend_from_slice(&channel.to_be_bytes());
        cdef.extend_from_slice(&0_u16.to_be_bytes());
        cdef.extend_from_slice(&association.to_be_bytes());
    }
    push_box(&mut jp2h, b"cdef", &cdef);

    push_box(&mut jp2, b"jp2h", &jp2h);
    push_box(&mut jp2, b"jp2c", codestream);
    jp2
}

fn jp2_with_signed_palette_mapping(codestream: &[u8]) -> Vec<u8> {
    let mut jp2 = Vec::new();
    jp2.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0d, 0x0a, 0x87, 0x0a]);

    let mut ftyp = Vec::new();
    ftyp.extend_from_slice(b"jp2 ");
    ftyp.extend_from_slice(&0_u32.to_be_bytes());
    ftyp.extend_from_slice(b"jp2 ");
    push_box(&mut jp2, b"ftyp", &ftyp);

    let mut jp2h = Vec::new();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&1_u16.to_be_bytes());
    ihdr.extend_from_slice(&[0x87, 7, 0, 0]);
    push_box(&mut jp2h, b"ihdr", &ihdr);

    let mut colr = Vec::new();
    colr.extend_from_slice(&[1, 0, 0]);
    colr.extend_from_slice(&17_u32.to_be_bytes());
    push_box(&mut jp2h, b"colr", &colr);

    let mut pclr = Vec::new();
    pclr.extend_from_slice(&2_u16.to_be_bytes());
    pclr.push(1);
    pclr.push(0x87);
    pclr.extend_from_slice(&[0xfe, 0x02]);
    push_box(&mut jp2h, b"pclr", &pclr);

    let mut cmap = Vec::new();
    cmap.extend_from_slice(&0_u16.to_be_bytes());
    cmap.push(1);
    cmap.push(0);
    push_box(&mut jp2h, b"cmap", &cmap);

    push_box(&mut jp2, b"jp2h", &jp2h);
    push_box(&mut jp2, b"jp2c", codestream);
    jp2
}

fn jp2_with_header_payload(codestream: &[u8], jp2h: &[u8]) -> Vec<u8> {
    let mut jp2 = Vec::new();
    jp2.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0d, 0x0a, 0x87, 0x0a]);

    let mut ftyp = Vec::new();
    ftyp.extend_from_slice(b"jp2 ");
    ftyp.extend_from_slice(&0_u32.to_be_bytes());
    ftyp.extend_from_slice(b"jp2 ");
    push_box(&mut jp2, b"ftyp", &ftyp);
    push_box(&mut jp2, b"jp2h", jp2h);
    push_box(&mut jp2, b"jp2c", codestream);
    jp2
}

#[test]
fn wrap_classic_codestream_as_jp2_inspects_and_decodes() {
    let codestream = classic_codestream();
    let jp2 = wrap_j2k_codestream(&codestream, J2kFileWrapOptions::jp2()).expect("wrap jp2");

    let support = J2kDecoder::inspect_support(&jp2).expect("inspect support");
    assert_eq!(support.payload_kind, CompressedPayloadKind::Jp2File);
    assert_eq!(support.info.dimensions, (2, 2));
    assert_eq!(support.info.colorspace, Colorspace::SGray);
    let file_metadata = support.file_metadata.expect("JP2 metadata");
    assert!(matches!(
        file_metadata.color_specs.as_slice(),
        [J2kColorSpec::Enumerated { value: 17 }]
    ));
    assert!(file_metadata.bits_per_component.is_empty());
    assert!(!file_metadata.has_channel_definition);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut output = [0_u8; 4];
    decoder
        .decode_into(&mut output, 2, PixelFormat::Gray8)
        .expect("decode");
    assert_eq!(output, [5, 16, 27, 38]);
}

#[test]
fn inspect_and_decode_jp2_palette_component_mapping_metadata() {
    let codestream = encode(
        &[0_u8, 1, 1, 0],
        2,
        2,
        1,
        8,
        false,
        &EncodeOptions::default(),
    )
    .expect("encode palette indices");
    let jp2 = jp2_with_palette_mapping(&codestream);

    let support = J2kDecoder::inspect_support(&jp2).expect("inspect paletted JP2");
    assert_eq!(support.info.colorspace, Colorspace::SRgb);
    let metadata = support.file_metadata.expect("JP2 metadata");
    assert!(metadata.has_palette);
    assert!(metadata.has_component_mapping);
    assert!(metadata.has_channel_definition);

    let palette = metadata.palette.expect("palette metadata");
    assert_eq!(palette.columns.len(), 3);
    assert_eq!(palette.columns[0].bit_depth, 8);
    assert_eq!(palette.entries, vec![vec![10, 20, 30], vec![200, 210, 220]]);
    assert_eq!(metadata.component_mappings.len(), 3);
    assert_eq!(
        metadata.component_mappings[1].mapping_type,
        J2kComponentMappingType::Palette { column: 1 }
    );
    assert_eq!(metadata.channel_definitions.len(), 3);
    assert_eq!(
        metadata.channel_definitions[2].channel_type,
        J2kChannelType::Color
    );
    assert_eq!(
        metadata.channel_definitions[2].association,
        J2kChannelAssociation::Color { index: 3 }
    );

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut output = [0_u8; 12];
    decoder
        .decode_into(&mut output, 2 * 3, PixelFormat::Rgb8)
        .expect("decode paletted JP2");
    assert_eq!(
        output,
        [10, 20, 30, 200, 210, 220, 200, 210, 220, 10, 20, 30]
    );
}

#[test]
fn inspect_and_decode_jp2_signed_palette_metadata() {
    let codestream = encode(
        &[0_u8, 1, 1, 0],
        2,
        2,
        1,
        8,
        false,
        &EncodeOptions::default(),
    )
    .expect("encode signed palette indices");
    let jp2 = jp2_with_signed_palette_mapping(&codestream);

    let support = J2kDecoder::inspect_support(&jp2).expect("inspect signed paletted JP2");
    assert_eq!(support.info.colorspace, Colorspace::SGray);
    let metadata = support.file_metadata.expect("JP2 metadata");
    assert!(metadata.has_palette);
    assert!(metadata.has_component_mapping);

    let palette = metadata.palette.expect("palette metadata");
    assert_eq!(
        palette.columns,
        vec![J2kPaletteColumn {
            bit_depth: 8,
            signed: true,
        }]
    );
    assert_eq!(palette.entries, vec![vec![0xfe], vec![0x02]]);
    assert_eq!(metadata.component_mappings.len(), 1);
    assert_eq!(
        metadata.component_mappings[0].mapping_type,
        J2kComponentMappingType::Palette { column: 0 }
    );

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let components = decoder.decode_components().expect("decode components");
    let planes = components.planes();
    assert_eq!(planes.len(), 1);
    assert_eq!(planes[0].bit_depth(), 8);
    assert!(planes[0].signed());
    assert_eq!(planes[0].samples(), &[-2.0, 2.0, 2.0, -2.0]);
}

#[test]
fn wrap_writes_palette_component_mapping_and_channel_definitions() {
    let codestream = encode(
        &[0_u8, 1, 1, 0],
        2,
        2,
        1,
        8,
        false,
        &EncodeOptions::default(),
    )
    .expect("encode palette indices");
    let palette = J2kPaletteMetadata {
        columns: vec![
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false,
            },
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false,
            },
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false,
            },
        ],
        entries: vec![vec![10, 20, 30], vec![200, 210, 220]],
    };
    let mappings = [
        J2kComponentMapping {
            component_index: 0,
            mapping_type: J2kComponentMappingType::Palette { column: 0 },
        },
        J2kComponentMapping {
            component_index: 0,
            mapping_type: J2kComponentMappingType::Palette { column: 1 },
        },
        J2kComponentMapping {
            component_index: 0,
            mapping_type: J2kComponentMappingType::Palette { column: 2 },
        },
    ];
    let channels = [
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
    ];

    let jp2 = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: Some(&palette),
                component_mappings: &mappings,
                channel_definitions: &channels,
            }),
    )
    .expect("wrap paletted JP2");

    let support = J2kDecoder::inspect_support(&jp2).expect("inspect wrapped paletted JP2");
    let metadata = support.file_metadata.expect("JP2 metadata");
    assert_eq!(metadata.palette.expect("palette").entries, palette.entries);
    assert_eq!(metadata.component_mappings, mappings);
    assert_eq!(metadata.channel_definitions, channels);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut output = [0_u8; 12];
    decoder
        .decode_into(&mut output, 2 * 3, PixelFormat::Rgb8)
        .expect("decode wrapped paletted JP2");
    assert_eq!(
        output,
        [10, 20, 30, 200, 210, 220, 200, 210, 220, 10, 20, 30]
    );
}

#[test]
fn wrap_ht_codestream_as_jph_inspects_and_decodes() {
    let codestream = ht_codestream();
    let jph = wrap_j2k_codestream(&codestream, J2kFileWrapOptions::jph()).expect("wrap jph");

    let support = J2kDecoder::inspect_support(&jph).expect("inspect support");
    assert_eq!(support.payload_kind, CompressedPayloadKind::JphFile);
    assert_eq!(support.info.dimensions, (2, 2));
    assert!(support.file_metadata.is_some());

    let mut decoder = J2kDecoder::new(&jph).expect("decoder");
    let components = decoder.decode_components().expect("component decode");
    assert_eq!(components.planes()[0].samples().len(), 4);
}

#[test]
fn wrap_jph_rejects_classic_codestream() {
    let err =
        wrap_j2k_codestream(&classic_codestream(), J2kFileWrapOptions::jph()).expect_err("reject");

    assert!(matches!(err, J2kError::Unsupported(_)));
}

#[test]
fn wrap_jp2_rejects_ht_codestream() {
    let err = wrap_j2k_codestream(&ht_codestream(), J2kFileWrapOptions::jp2()).expect_err("reject");

    assert!(matches!(err, J2kError::Unsupported(_)));
}

#[test]
fn inspect_rejects_jp2_file_type_with_ht_codestream() {
    let codestream = ht_codestream();
    let mut mislabeled =
        wrap_j2k_codestream(&codestream, J2kFileWrapOptions::jph()).expect("wrap JPH");
    let ftyp_payload = 12 + 8;
    mislabeled[ftyp_payload..ftyp_payload + 4].copy_from_slice(b"jp2 ");
    mislabeled[ftyp_payload + 8..ftyp_payload + 12].copy_from_slice(b"jp2 ");

    let err =
        J2kDecoder::inspect_support(&mislabeled).expect_err("JP2-branded HTJ2K file must reject");

    assert!(matches!(err, J2kError::InvalidBox { .. }));
}

#[test]
fn inspect_rejects_invalid_ihdr_compression_type() {
    let codestream = classic_codestream();
    let mut jp2h = Vec::new();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&1_u16.to_be_bytes());
    ihdr.extend_from_slice(&[7, 0, 0, 0]);
    push_box(&mut jp2h, b"ihdr", &ihdr);
    let mut colr = Vec::new();
    colr.extend_from_slice(&[1, 0, 0]);
    colr.extend_from_slice(&17_u32.to_be_bytes());
    push_box(&mut jp2h, b"colr", &colr);
    let jp2 = jp2_with_header_payload(&codestream, &jp2h);

    let err =
        J2kDecoder::inspect_support(&jp2).expect_err("invalid ihdr compression type must reject");

    assert!(matches!(err, J2kError::InvalidBox { .. }));
}

#[test]
fn inspect_rejects_missing_colr() {
    let codestream = classic_codestream();
    let mut jp2h = Vec::new();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&1_u16.to_be_bytes());
    ihdr.extend_from_slice(&[7, 7, 0, 0]);
    push_box(&mut jp2h, b"ihdr", &ihdr);
    let jp2 = jp2_with_header_payload(&codestream, &jp2h);

    let err = J2kDecoder::inspect_support(&jp2).expect_err("missing COLR must reject");

    assert!(matches!(
        err,
        J2kError::MissingRequiredBox { box_type: "colr" }
    ));
}

#[test]
fn inspect_rejects_ihdr_dimension_mismatch() {
    let codestream = classic_codestream();
    let mut jp2h = Vec::new();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&3_u32.to_be_bytes());
    ihdr.extend_from_slice(&1_u16.to_be_bytes());
    ihdr.extend_from_slice(&[7, 7, 0, 0]);
    push_box(&mut jp2h, b"ihdr", &ihdr);
    let mut colr = Vec::new();
    colr.extend_from_slice(&[1, 0, 0]);
    colr.extend_from_slice(&17_u32.to_be_bytes());
    push_box(&mut jp2h, b"colr", &colr);
    let jp2 = jp2_with_header_payload(&codestream, &jp2h);

    let err = J2kDecoder::inspect_support(&jp2).expect_err("IHDR dimensions must reject");

    assert!(matches!(err, J2kError::InvalidBox { .. }));
}

#[test]
fn inspect_rejects_ihdr_bpc_mismatch() {
    let codestream = classic_codestream();
    let mut jp2h = Vec::new();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&1_u16.to_be_bytes());
    ihdr.extend_from_slice(&[15, 7, 0, 0]);
    push_box(&mut jp2h, b"ihdr", &ihdr);
    let mut colr = Vec::new();
    colr.extend_from_slice(&[1, 0, 0]);
    colr.extend_from_slice(&17_u32.to_be_bytes());
    push_box(&mut jp2h, b"colr", &colr);
    let jp2 = jp2_with_header_payload(&codestream, &jp2h);

    let err = J2kDecoder::inspect_support(&jp2).expect_err("IHDR BPC mismatch must reject");

    assert!(matches!(err, J2kError::InvalidBox { .. }));
}

#[test]
fn inspect_rejects_bpcc_precision_mismatch() {
    let codestream = classic_codestream();
    let mut jp2h = Vec::new();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&2_u32.to_be_bytes());
    ihdr.extend_from_slice(&1_u16.to_be_bytes());
    ihdr.extend_from_slice(&[0xff, 7, 0, 0]);
    push_box(&mut jp2h, b"ihdr", &ihdr);
    push_box(&mut jp2h, b"bpcc", &[15]);
    let mut colr = Vec::new();
    colr.extend_from_slice(&[1, 0, 0]);
    colr.extend_from_slice(&17_u32.to_be_bytes());
    push_box(&mut jp2h, b"colr", &colr);
    let jp2 = jp2_with_header_payload(&codestream, &jp2h);

    let err = J2kDecoder::inspect_support(&jp2).expect_err("BPCC mismatch must reject");

    assert!(matches!(err, J2kError::InvalidBox { .. }));
}

#[test]
fn wrap_writes_bpcc_for_mixed_precision_and_signedness() {
    let mut codestream = minimal_j2k_codestream();
    rewrite_component_descriptor(&mut codestream, 0, 0x07);
    rewrite_component_descriptor(&mut codestream, 1, 0x8b);
    rewrite_component_descriptor(&mut codestream, 2, 0x0f);

    let jp2 = wrap_j2k_codestream(&codestream, J2kFileWrapOptions::jp2()).expect("wrap jp2");

    assert!(jp2.windows(4).any(|box_type| box_type == b"bpcc"));
    let support = J2kDecoder::inspect_support(&jp2).expect("inspect support");
    assert_eq!(support.components[0].bit_depth, 8);
    assert_eq!(support.components[1].bit_depth, 12);
    assert!(support.components[1].signed);
    assert_eq!(support.components[2].bit_depth, 16);
    let file_metadata = support.file_metadata.expect("JP2 metadata");
    assert_eq!(file_metadata.bits_per_component.len(), 3);
    assert_eq!(file_metadata.bits_per_component[0].bit_depth, 8);
    assert_eq!(file_metadata.bits_per_component[1].bit_depth, 12);
    assert!(file_metadata.bits_per_component[1].signed);
    assert_eq!(file_metadata.bits_per_component[2].bit_depth, 16);
}

#[test]
fn wrap_allows_icc_color_spec_for_non_enumerated_component_counts() {
    let mut codestream = minimal_j2k_codestream();
    let siz = codestream
        .windows(2)
        .position(|marker| marker == [0xff, 0x51])
        .expect("SIZ marker");
    codestream[siz + 38..siz + 40].copy_from_slice(&5_u16.to_be_bytes());
    codestream.splice(siz + 49..siz + 49, [0x07, 0x01, 0x01, 0x07, 0x01, 0x01]);
    let new_lsiz = u16::from_be_bytes([codestream[siz + 2], codestream[siz + 3]]) + 6;
    codestream[siz + 2..siz + 4].copy_from_slice(&new_lsiz.to_be_bytes());

    let jp2 = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(b"icc")),
    )
    .expect("wrap jp2");

    let support = J2kDecoder::inspect_support(&jp2).expect("inspect support");
    assert_eq!(support.payload_kind, CompressedPayloadKind::Jp2File);
    assert_eq!(support.info.colorspace, Colorspace::IccTagged);
    assert_eq!(support.info.components, 5);
    let file_metadata = support.file_metadata.expect("JP2 metadata");
    assert!(matches!(
        file_metadata.color_specs.as_slice(),
        [J2kColorSpec::IccProfile { profile }] if profile == b"icc"
    ));
    assert!(file_metadata.has_icc_profile());
}

#[test]
fn wrap_preserves_inspected_icc_metadata_when_rewrapping() {
    let codestream = classic_codestream();
    let original = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(b"test-icc")),
    )
    .expect("wrap ICC JP2");
    let original_support = J2kDecoder::inspect_support(&original).expect("inspect ICC JP2");
    let original_metadata = original_support.file_metadata.expect("file metadata");
    let color =
        J2kFileColorSpec::from_file_metadata(&original_metadata).expect("borrow ICC metadata");

    let rewrapped = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2()
            .with_color(color)
            .with_metadata(J2kFileBoxMetadata::from_file_metadata(&original_metadata)),
    )
    .expect("rewrap ICC JP2");

    let support = J2kDecoder::inspect_support(&rewrapped).expect("inspect rewrapped ICC JP2");
    let metadata = support.file_metadata.expect("file metadata");
    assert!(metadata.has_icc_profile());
    assert!(matches!(
        metadata.color_specs.as_slice(),
        [J2kColorSpec::IccProfile { profile }] if profile == b"test-icc"
    ));
}

#[test]
fn wrap_preserves_inspected_icc_metadata_when_rewrapping_jph() {
    let codestream = ht_codestream();
    let original = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jph().with_color(J2kFileColorSpec::IccProfile(b"test-icc")),
    )
    .expect("wrap ICC JPH");
    let original_support = J2kDecoder::inspect_support(&original).expect("inspect ICC JPH");
    let original_metadata = original_support.file_metadata.expect("file metadata");
    let color =
        J2kFileColorSpec::from_file_metadata(&original_metadata).expect("borrow ICC metadata");

    let rewrapped = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jph()
            .with_color(color)
            .with_metadata(J2kFileBoxMetadata::from_file_metadata(&original_metadata)),
    )
    .expect("rewrap ICC JPH");

    let support = J2kDecoder::inspect_support(&rewrapped).expect("inspect rewrapped ICC JPH");
    assert_eq!(support.payload_kind, CompressedPayloadKind::JphFile);
    let metadata = support.file_metadata.expect("file metadata");
    assert!(metadata.has_icc_profile());
    assert!(matches!(
        metadata.color_specs.as_slice(),
        [J2kColorSpec::IccProfile { profile }] if profile == b"test-icc"
    ));
}

#[test]
fn wrap_preserves_multiple_colr_boxes_when_rewrapping() {
    let codestream = classic_codestream();
    let original_colors = [
        J2kFileColorSpec::Enumerated(Colorspace::SRgb),
        J2kFileColorSpec::IccProfile(b"test-icc"),
    ];
    let original = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color_specs(&original_colors),
    )
    .expect("wrap JP2 with multiple COLR boxes");
    let original_support = J2kDecoder::inspect_support(&original).expect("inspect JP2");
    let original_metadata = original_support.file_metadata.expect("file metadata");
    let colors = original_metadata
        .color_specs
        .iter()
        .filter_map(J2kFileColorSpec::from_inspected)
        .collect::<Vec<_>>();

    let rewrapped = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2()
            .with_color_specs(&colors)
            .with_metadata(J2kFileBoxMetadata::from_file_metadata(&original_metadata)),
    )
    .expect("rewrap JP2 with multiple COLR boxes");

    let support = J2kDecoder::inspect_support(&rewrapped).expect("inspect rewrapped JP2");
    let metadata = support.file_metadata.expect("file metadata");
    assert!(metadata.has_icc_profile());
    assert!(matches!(
        metadata.color_specs.as_slice(),
        [
            J2kColorSpec::Enumerated { value: 16 },
            J2kColorSpec::IccProfile { profile },
        ] if profile == b"test-icc"
    ));
}

#[test]
fn wrap_writes_cdef_for_explicit_srgb_alpha() {
    let pixels = [0_u8; 4];
    let codestream =
        encode(&pixels, 1, 1, 4, 8, false, &EncodeOptions::default()).expect("encode rgba");
    let jp2 = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb)),
    )
    .expect("wrap jp2");

    let support = J2kDecoder::inspect_support(&jp2).expect("inspect support");
    let file_metadata = support.file_metadata.expect("JP2 metadata");
    assert!(file_metadata.has_channel_definition);
}

#[test]
fn wrap_preserves_premultiplied_alpha_cdef_and_decodes_rgba() {
    let pixels = [10_u8, 20, 30, 128];
    let codestream =
        encode(&pixels, 1, 1, 4, 8, false, &EncodeOptions::default()).expect("encode rgba");
    let channels = [
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

    let jp2 = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: None,
                component_mappings: &[],
                channel_definitions: &channels,
            }),
    )
    .expect("wrap premultiplied alpha JP2");

    let support = J2kDecoder::inspect_support(&jp2).expect("inspect JP2");
    let metadata = support.file_metadata.expect("metadata");
    assert_eq!(metadata.channel_definitions, channels);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut output = [0_u8; 4];
    decoder
        .decode_into(&mut output, 4, PixelFormat::Rgba8)
        .expect("decode RGBA");
    assert_eq!(output, pixels);
}

#[test]
fn lossless_encode_facade_roundtrips_more_than_four_components() {
    let pixels = (0..2 * 2 * 5)
        .map(|idx| ((idx * 29 + 3) & 0xff) as u8)
        .collect::<Vec<_>>();
    let samples =
        J2kLosslessSamples::new(&pixels, 2, 2, 5, 8, false).expect("five component samples");

    let encoded = encode_j2k_lossless(samples, &J2kLosslessEncodeOptions::default())
        .expect("encode five components");
    let mut decoder = J2kDecoder::new(&encoded.codestream).expect("decoder");
    let components = decoder.decode_components().expect("component decode");

    assert_eq!(encoded.components, 5);
    assert_eq!(components.planes().len(), 5);
    for plane in components.planes() {
        assert_eq!(plane.samples().len(), 4);
    }
}
