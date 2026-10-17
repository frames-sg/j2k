// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::auxiliary::{crg_marker, rgn_marker};
use super::coding::cod_marker;
use super::model::{CodeBlockStyle, Header, SizeData};
use super::progression::poc_marker;
use super::quantization::qcd_marker;
use super::size::size_marker;
use super::{
    coc_marker, decode_packet_lengths, plt_marker, qcc_marker, read_header, skip_marker_segment,
    CodingStyleComponent, CodingStyleDefault, PacketLengthMarker, ProgressionChange,
    ProgressionOrder, QuantizationInfo, RgnMarkerData, WaveletTransform,
};
use crate::error::{DecodeError, MarkerError, Result, ValidationError};
use crate::reader::BitReader;
use crate::{DecodeSettings, J2kWaveletTransform};

#[test]
fn wavelet_transform_converts_to_external_selector() {
    assert_eq!(
        J2kWaveletTransform::from(WaveletTransform::Reversible53),
        J2kWaveletTransform::Reversible53
    );
    assert_eq!(
        J2kWaveletTransform::from(WaveletTransform::Irreversible97),
        J2kWaveletTransform::Irreversible97
    );
}

#[test]
fn code_block_style_preserves_ht_and_mixed_bits() {
    let style = CodeBlockStyle::from_u8(0xc0);

    assert!(style.uses_high_throughput_block_coding());
    assert!(style.allows_mixed_block_coding());
}

#[test]
fn poc_marker_preserves_wide_component_bounds() {
    let mut marker = Vec::new();
    marker.extend_from_slice(&11_u16.to_be_bytes());
    marker.push(0); // RSpoc
    marker.extend_from_slice(&300_u16.to_be_bytes()); // CSpoc
    marker.extend_from_slice(&1_u16.to_be_bytes()); // LYEpoc
    marker.push(1); // REpoc
    marker.extend_from_slice(&512_u16.to_be_bytes()); // CEpoc
    marker.push(0); // LRCP

    let changes = poc_marker(&mut BitReader::new(&marker), 600, 1, usize::MAX).expect("POC parses");

    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].component_start, 300);
    assert_eq!(changes[0].component_end, 512);
    assert!(matches!(
        changes[0].progression_order,
        ProgressionOrder::LayerResolutionComponentPosition
    ));
}

#[test]
fn poc_marker_accepts_wide_all_components_sentinel() {
    let mut marker = Vec::new();
    marker.extend_from_slice(&11_u16.to_be_bytes());
    marker.push(0); // RSpoc
    marker.extend_from_slice(&0_u16.to_be_bytes()); // CSpoc
    marker.extend_from_slice(&1_u16.to_be_bytes()); // LYEpoc
    marker.push(1); // REpoc
    marker.extend_from_slice(&u16::MAX.to_be_bytes()); // CEpoc sentinel
    marker.push(4); // CPRL

    let changes = poc_marker(&mut BitReader::new(&marker), 600, 1, usize::MAX).expect("POC parses");

    assert_eq!(changes[0].component_end, u16::MAX);
    assert!(matches!(
        changes[0].progression_order,
        ProgressionOrder::ComponentPositionResolutionLayer
    ));
}

#[test]
fn checked_image_dimensions_reject_shrink_factor_overflow() {
    let size_data = SizeData {
        decoder_capabilities: 0,
        reference_grid_width: 1024,
        reference_grid_height: 1024,
        image_area_x_offset: 0,
        image_area_y_offset: 0,
        tile_width: 1024,
        tile_height: 1024,
        tile_x_offset: 0,
        tile_y_offset: 0,
        component_sizes: Vec::new(),
        x_shrink_factor: u32::MAX,
        y_shrink_factor: 1,
        x_resolution_shrink_factor: 2,
        y_resolution_shrink_factor: 1,
    };

    assert!(matches!(
        size_data.checked_image_width(),
        Err(DecodeError::Validation(ValidationError::InvalidDimensions))
    ));
    assert_eq!(size_data.checked_image_height().expect("height"), 1024);
}

#[test]
fn truncated_siz_keeps_parse_error_and_cursor_boundary() {
    let data = [0, 4, 0, 0, 0xA5];
    let mut reader = BitReader::new(&data);

    assert!(matches!(
        size_marker(&mut reader, usize::MAX),
        Err(DecodeError::Marker(MarkerError::ParseFailure("SIZ")))
    ));
    assert_eq!(reader.offset(), 4);
    assert_eq!(reader.tail(), Some(&data[4..]));
}

#[test]
fn invalid_cod_layer_count_stops_before_mct_byte() {
    let data = [0, 12, 0, 0, 0, 0, 0xA5];
    let mut reader = BitReader::new(&data);

    assert!(cod_marker(&mut reader).is_err());
    assert_eq!(reader.offset(), 6);
    assert_eq!(reader.tail(), Some(&data[6..]));
}

/// A COD segment body: LRCP, one layer, no MCT, one decomposition level, the
/// given code-block exponent offsets and, when given, precinct sizes.
fn cod_body(xcb: u8, ycb: u8, precincts: &[u8]) -> Vec<u8> {
    let mut body = u16::try_from(12 + precincts.len())
        .expect("COD length")
        .to_be_bytes()
        .to_vec();
    body.extend_from_slice(&[
        u8::from(!precincts.is_empty()),
        0,
        0,
        1,
        0,
        1,
        xcb,
        ycb,
        0,
        1,
    ]);
    body.extend_from_slice(precincts);
    body
}

fn parses_cod(body: &[u8]) -> bool {
    match cod_marker(&mut BitReader::new(body)) {
        Ok(_) => true,
        Err(DecodeError::Marker(MarkerError::ParseFailure("COD"))) => false,
        Err(other) => panic!("unexpected COD error: {other}"),
    }
}

#[test]
fn cod_code_block_sizes_follow_table_a18() {
    // Exponent offsets 0-8 (4 to 1024 samples), at most 4096 samples a block.
    for (xcb, ycb) in [(0, 0), (4, 4), (8, 0), (0, 8), (5, 3)] {
        assert!(parses_cod(&cod_body(xcb, ycb, &[])), "xcb {xcb} ycb {ycb}");
    }
    for (xcb, ycb) in [(9, 0), (0, 9), (5, 4), (8, 1), (57, 4)] {
        assert!(!parses_cod(&cod_body(xcb, ycb, &[])), "xcb {xcb} ycb {ycb}");
    }
}

#[test]
fn cod_precinct_exponents_are_zero_only_at_the_lowest_resolution() {
    // Table A.21. Above resolution 0, B-17 and B-18 use PPx - 1 and PPy - 1.
    assert!(parses_cod(&cod_body(4, 4, &[0x00, 0x11])));
    for above in [0x10, 0x01, 0x00] {
        assert!(!parses_cod(&cod_body(4, 4, &[0x00, above])), "{above:#04x}");
    }
}

#[test]
fn fuzzed_precinct_and_code_block_sizes_fail_to_parse() {
    // From region_scaled_fuzz: PPx = 0 at resolution 1 made PPx - 1
    // underflow, and with an out-of-range code-block exponent the code-block
    // width then wrapped to 0 and decode planning divided by it.
    let options = crate::EncodeOptions {
        num_decomposition_levels: 1,
        precinct_exponents: vec![(15, 15), (15, 15)],
        ..crate::EncodeOptions::default()
    };
    let pixels = (0..64).collect::<Vec<u8>>();
    let mut codestream = crate::encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode");
    let cod = codestream
        .windows(2)
        .position(|marker| marker == [0xff, 0x52])
        .expect("COD marker");
    // After FF52: Lcod, Scod, SGcod (4 bytes), decomposition levels, xcb.
    codestream[cod + 10] = 30;
    // Then ycb, style, transform and one precinct byte per resolution.
    codestream[cod + 15] &= 0xf0;

    assert!(matches!(
        crate::Image::new(&codestream, &DecodeSettings::default()),
        Err(DecodeError::Marker(MarkerError::ParseFailure("COD")))
    ));
}

#[test]
fn undersized_qcd_stops_after_style_byte() {
    let data = [0, 2, 0, 0xA5];
    let mut reader = BitReader::new(&data);

    assert!(qcd_marker(&mut reader).is_err());
    assert_eq!(reader.offset(), 3);
    assert_eq!(reader.tail(), Some(&data[3..]));
}

#[test]
fn invalid_poc_geometry_consumes_the_complete_change() {
    let data = [0, 9, 1, 0, 0, 1, 1, 2, 0, 0xA5];
    let mut reader = BitReader::new(&data);

    assert!(poc_marker(&mut reader, 3, 1, usize::MAX).is_err());
    assert_eq!(reader.offset(), 9);
    assert_eq!(reader.tail(), Some(&data[9..]));
}

#[test]
fn invalid_rgn_length_only_consumes_the_length_field() {
    let data = [0, 4, 1, 0, 3, 0xA5];
    let mut reader = BitReader::new(&data);

    assert!(rgn_marker(&mut reader, 3).is_none());
    assert_eq!(reader.offset(), 2);
    assert_eq!(reader.tail(), Some(&data[2..]));
}

#[test]
fn crg_marker_requires_one_registration_pair_per_component() {
    let valid = [0, 10, 0, 1, 0, 2, 0, 3, 0, 4];
    let mut reader = BitReader::new(&valid);
    assert_eq!(crg_marker(&mut reader, 2), Some(()));
    assert_eq!(reader.offset(), valid.len());

    let invalid_length = [0, 9, 0, 1, 0, 2, 0, 3, 0, 4];
    let mut reader = BitReader::new(&invalid_length);
    assert_eq!(crg_marker(&mut reader, 2), None);
    assert_eq!(reader.offset(), 2);

    let truncated = [0, 10, 0, 1, 0, 2, 0, 3];
    let mut reader = BitReader::new(&truncated);
    assert_eq!(crg_marker(&mut reader, 2), None);
}

#[test]
fn incomplete_packet_length_varint_is_rejected() {
    assert_eq!(
        decode_packet_lengths(&[0x81]),
        Err(crate::DecodeError::Marker(MarkerError::ParseFailure(
            "packet lengths"
        )))
    );
    assert_eq!(decode_packet_lengths(&[0x81, 0x01]), Ok(vec![129]));
}

#[test]
fn public_marker_function_signatures_stay_stable() {
    let _: for<'a> fn(
        &mut BitReader<'a>,
        &DecodeSettings,
        usize,
        Option<u8>,
    ) -> Result<Header<'a>> = read_header;
    let _: fn(&mut BitReader<'_>) -> Result<CodingStyleDefault> = cod_marker;
    let _: fn(&mut BitReader<'_>, u16) -> Result<(u16, CodingStyleComponent)> = coc_marker;
    let _: fn(&mut BitReader<'_>) -> Result<QuantizationInfo> = qcd_marker;
    let _: fn(&mut BitReader<'_>, u16) -> Result<(u16, QuantizationInfo)> = qcc_marker;
    let _: fn(&mut BitReader<'_>, u16, u8, usize) -> Result<Vec<ProgressionChange>> = poc_marker;
    let _: fn(&mut BitReader<'_>, usize) -> Result<PacketLengthMarker> = plt_marker;
    let _: fn(&mut BitReader<'_>, u16) -> Option<RgnMarkerData> = rgn_marker;
    let _: fn(&mut BitReader<'_>) -> Option<()> = skip_marker_segment;
    let _: fn(&[u8]) -> Result<Vec<u32>> = decode_packet_lengths;
}

#[test]
fn marker_owned_vectors_require_fallible_reservation() {
    const OWNERS: &[(&str, &str)] = &[
        ("coding", include_str!("coding.rs")),
        ("progression", include_str!("progression.rs")),
        ("quantization", include_str!("quantization.rs")),
        ("size", include_str!("size.rs")),
    ];

    for &(name, source) in OWNERS {
        assert!(
            source.contains("try_reserve_decode_elements"),
            "{name} must reserve untrusted marker-owned vectors fallibly"
        );
        assert!(
            !source.contains("Vec::with_capacity"),
            "{name} must not allocate marker-owned vectors infallibly"
        );
    }
}
