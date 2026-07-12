// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::auxiliary::rgn_marker;
use super::coding::cod_marker;
use super::model::{Header, SizeData};
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
    let _: for<'a> fn(&mut BitReader<'a>, &DecodeSettings, usize) -> Result<Header<'a>> =
        read_header;
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

#[test]
fn codestream_module_boundaries_stay_focused() {
    const MODULES: &[(&str, &str, usize)] = &[
        ("coordinator", include_str!("../codestream.rs"), 50),
        ("allocation", include_str!("allocation.rs"), 100),
        ("auxiliary", include_str!("auxiliary.rs"), 190),
        (
            "packet-length allocation",
            include_str!("auxiliary/packet_lengths.rs"),
            190,
        ),
        ("coding", include_str!("coding.rs"), 150),
        ("header", include_str!("header.rs"), 280),
        (
            "header allocation",
            include_str!("header/allocation.rs"),
            280,
        ),
        (
            "header allocation tests",
            include_str!("header/allocation/tests.rs"),
            120,
        ),
        (
            "header components",
            include_str!("header/components.rs"),
            120,
        ),
        ("markers", include_str!("markers.rs"), 120),
        ("model", include_str!("model.rs"), 460),
        ("progression", include_str!("progression.rs"), 90),
        ("quantization", include_str!("quantization.rs"), 130),
        ("size", include_str!("size.rs"), 220),
        ("validation", include_str!("validation.rs"), 70),
    ];

    for &(name, source, line_limit) in MODULES {
        let line_count = source.lines().count();
        assert!(
            line_count <= line_limit,
            "codestream {name} module grew to {line_count} lines (limit {line_limit})"
        );
        assert!(
            !source.contains("include!("),
            "codestream {name} must use real Rust modules"
        );
        assert!(
            !source.contains("allow(unused"),
            "codestream {name} must not restore a broad unused allowance"
        );
    }
}
