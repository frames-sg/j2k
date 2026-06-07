// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `Decoder::inspect`.

use signinum_jpeg::{
    find_scan_ranges, is_sof_marker, iter_segments, parse_dri, parse_sof_info,
    prepare_tiff_jpeg_tile, rewrite_sof_dimensions, ColorSpace, ColorTransform, DecodeOptions,
    Decoder, DuplicateTablePolicy, JpegError, JpegTilePrepareOptions, JpegView, McuGeometry,
    PreparedJpeg, RestartSegment, SofKind, UnsupportedReason,
};
use signinum_jpeg::{
    CompressedPayloadKind, CompressedTransferSyntax, PassthroughDecision, PassthroughRequirements,
};

mod fixtures;
use fixtures::progressive_8x8_jpeg;

fn minimal_baseline_jpeg() -> Vec<u8> {
    // Same construction as parse::header::tests — duplicated here because
    // integration tests cannot access pub(crate) helpers.
    let mut v = Vec::new();
    v.extend_from_slice(&[0xFF, 0xD8]);
    v.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0x00]);
    v.extend(core::iter::repeat_n(1u8, 64));
    v.extend_from_slice(&[
        0xFF,
        0xC0,
        0x00,
        17,
        8,
        0,
        16,
        0,
        16,
        3,
        1,
        (2 << 4) | 2,
        0,
        2,
        (1 << 4) | 1,
        0,
        3,
        (1 << 4) | 1,
        0,
    ]);
    // DHT length = 2 (length field) + 1 (Tc/Th) + 16 (bits[]) + 1 (value) = 20
    v.extend_from_slice(&[
        0xFF, 0xC4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xAA,
    ]);
    v.extend_from_slice(&[
        0xFF, 0xC4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xBB,
    ]);
    v.extend_from_slice(&[0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);
    v.extend_from_slice(&[0x00, 0xFF, 0xD9]);
    v
}

fn minimal_baseline_jpeg_with_restart_interval(interval: u16) -> Vec<u8> {
    let mut bytes = minimal_baseline_jpeg();
    let sos_pos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("SOS marker");
    bytes.splice(
        sos_pos..sos_pos,
        [
            0xff,
            0xdd,
            0x00,
            0x04,
            (interval >> 8) as u8,
            interval as u8,
        ],
    );
    bytes
}

fn minimal_jpeg_with_sof_marker(marker: u8) -> Vec<u8> {
    let mut bytes = minimal_baseline_jpeg();
    let pos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xc0])
        .expect("minimal fixture has SOF0 marker");
    bytes[pos + 1] = marker;
    bytes
}

fn restart_coded_grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff,
        0xc0,
        0x00,
        11,
        8,
        (height >> 8) as u8,
        height as u8,
        (width >> 8) as u8,
        width as u8,
        1,
        1,
        0x11,
        0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);

    let mcu_cols = u32::from(width).div_ceil(8);
    let mcu_rows = u32::from(height).div_ceil(8);
    let mcu_count = (mcu_cols * mcu_rows) as usize;
    for mcu in 0..mcu_count {
        bytes.push(0x00);
        if mcu + 1 != mcu_count {
            bytes.extend_from_slice(&[0xff, 0xd0 | ((mcu as u8) & 0x07)]);
        }
    }

    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn scan_data_offset(bytes: &[u8]) -> usize {
    let sos_pos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("SOS marker");
    let len = u16::from_be_bytes([bytes[sos_pos + 2], bytes[sos_pos + 3]]) as usize;
    sos_pos + 2 + len
}

fn restart_marker_offsets(bytes: &[u8]) -> Vec<usize> {
    bytes
        .windows(2)
        .enumerate()
        .filter_map(|(offset, window)| {
            (window[0] == 0xff && (0xd0..=0xd7).contains(&window[1])).then_some(offset)
        })
        .collect()
}

fn prepare_options() -> JpegTilePrepareOptions {
    JpegTilePrepareOptions {
        expected_dimensions: None,
        duplicate_table_policy: DuplicateTablePolicy::RejectConflicting,
        repair_zero_sof_dimensions: false,
        validate_restart_markers: false,
    }
}

fn zero_sof_jpeg() -> Vec<u8> {
    let mut bytes = minimal_baseline_jpeg();
    let sof = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xc0])
        .expect("SOF");
    bytes[sof + 5] = 0;
    bytes[sof + 6] = 0;
    bytes[sof + 7] = 0;
    bytes[sof + 8] = 0;
    bytes
}

fn split_tables_and_scan(bytes: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let ranges = find_scan_ranges(bytes).expect("scan ranges");
    let mut tables = Vec::new();
    tables.extend_from_slice(&[0xff, 0xd8]);
    tables.extend_from_slice(&bytes[2..ranges.sos_marker_offset]);
    tables.extend_from_slice(&[0xff, 0xd9]);

    let mut tile = Vec::new();
    tile.extend_from_slice(&bytes[ranges.sos_marker_offset..]);
    (tables, tile)
}

fn mutate_first_dqt_value(tables: &[u8]) -> Vec<u8> {
    let mut out = tables.to_vec();
    let dqt = out
        .windows(2)
        .position(|window| window == [0xff, 0xdb])
        .expect("DQT");
    out[dqt + 5] ^= 0x7f;
    out
}

fn restart_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[0xff, 0xc0, 0x00, 11, 8, 0, 8, 0, 16, 1, 1, 0x11, 0]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);
    bytes.extend_from_slice(&[0x00, 0xff, 0xd0, 0x00, 0xff, 0xd9]);
    bytes
}

fn segment_payload(bytes: &[u8], marker: u8) -> &[u8] {
    iter_segments(bytes)
        .find_map(|segment| {
            let segment = segment.expect("segment parses");
            (segment.marker == marker).then_some(segment.payload)
        })
        .expect("marker present")
}

#[test]
fn public_segment_iterator_reports_header_markers_without_stuffed_entropy_markers() {
    let mut bytes = minimal_baseline_jpeg();
    let eoi = bytes.len() - 2;
    bytes.splice(eoi..eoi, [0xff, 0x00, 0x7f]);
    let markers = iter_segments(&bytes)
        .map(|segment| segment.expect("segment parses").marker)
        .collect::<Vec<_>>();

    assert_eq!(markers, vec![0xd8, 0xdb, 0xc0, 0xc4, 0xc4, 0xda, 0xd9]);
}

#[test]
fn public_sof_and_dri_helpers_report_marker_facts() {
    for marker in [
        0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
    ] {
        assert!(is_sof_marker(marker), "FF{marker:02X}");
    }
    for marker in [0xc4, 0xd8, 0xd9, 0xda, 0xdb, 0xdd, 0xee] {
        assert!(!is_sof_marker(marker), "FF{marker:02X}");
    }

    let bytes = minimal_baseline_jpeg();
    let sof = parse_sof_info(0xc0, segment_payload(&bytes, 0xc0)).expect("SOF parses");
    assert_eq!(sof.sof_kind, SofKind::Baseline8);
    assert_eq!(sof.dimensions, (16, 16));
    assert_eq!(sof.sampling.components(), &[(2, 2), (1, 1), (1, 1)]);
    assert_eq!(sof.component_ids, vec![1, 2, 3]);
    assert_eq!(sof.quant_table_ids, vec![0, 0, 0]);

    assert_eq!(parse_dri(&[0x00, 0x00]).expect("zero DRI"), None);
    assert_eq!(parse_dri(&[0x00, 0x08]).expect("nonzero DRI"), Some(8));
}

#[test]
fn public_scan_ranges_and_sof_rewrite_helpers_use_absolute_offsets() {
    let bytes = minimal_baseline_jpeg();
    let ranges = find_scan_ranges(&bytes).expect("scan ranges");
    let sos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("SOS");
    assert_eq!(ranges.sos_marker_offset, sos);
    assert_eq!(ranges.sos_payload_range, sos + 4..sos + 14);
    assert_eq!(ranges.entropy_range, sos + 14..bytes.len() - 2);
    assert_eq!(ranges.eoi_marker_offset, Some(bytes.len() - 2));

    let rewritten = rewrite_sof_dimensions(&bytes, (32, 24)).expect("rewrite");
    let sof = parse_sof_info(0xc0, segment_payload(&rewritten, 0xc0)).expect("SOF parses");
    assert_eq!(sof.dimensions, (32, 24));
    assert_eq!(rewritten.len(), bytes.len());
}

#[test]
fn complete_tiff_jpeg_tile_preparation_returns_borrowed_bytes() {
    let bytes = minimal_baseline_jpeg();
    let prepared = prepare_tiff_jpeg_tile(&bytes, None, prepare_options()).expect("prepared");

    match prepared {
        PreparedJpeg::Borrowed(slice) => assert!(std::ptr::eq(slice.as_ptr(), bytes.as_ptr())),
        PreparedJpeg::Owned(_) => panic!("complete JPEG tile should stay borrowed"),
    }
}

#[test]
fn prepared_tiff_jpeg_bytes_decode_complete_tile() {
    let bytes = minimal_baseline_jpeg();
    let prepared = prepare_tiff_jpeg_tile(&bytes, None, prepare_options()).expect("prepared");
    let info = Decoder::inspect(prepared.as_bytes()).expect("inspect prepared");

    assert_eq!(info.dimensions, (16, 16));
}

#[test]
fn zero_sof_tiff_jpeg_dimensions_without_expected_dimensions_are_rejected() {
    let err = prepare_tiff_jpeg_tile(&zero_sof_jpeg(), None, prepare_options()).unwrap_err();

    assert!(matches!(
        err,
        JpegError::ZeroDimension {
            width: 0,
            height: 0
        } | JpegError::ExpectedDimensionsRequired { .. }
    ));
}

#[test]
fn tiff_jpeg_preparation_rejects_scan_without_sof() {
    let tile = [
        0xff, 0xd8, 0xff, 0xda, 0x00, 0x08, 1, 1, 0, 0, 63, 0, 0, 0xff, 0xd9,
    ];
    let err = prepare_tiff_jpeg_tile(&tile, None, prepare_options()).unwrap_err();

    assert!(matches!(
        err,
        JpegError::MissingMarker { .. } | JpegError::InvalidJpegAssembly { .. }
    ));
}

#[test]
fn abbreviated_tiff_jpeg_tile_with_jpeg_tables_assembles_decode_ready_stream() {
    let full = minimal_baseline_jpeg();
    let (tables, tile) = split_tables_and_scan(&full);
    let prepared =
        prepare_tiff_jpeg_tile(&tile, Some(&tables), prepare_options()).expect("prepared");

    assert!(matches!(prepared, PreparedJpeg::Owned(_)));
    let info = Decoder::inspect(prepared.as_bytes()).expect("assembled inspect");
    assert_eq!(info.dimensions, (16, 16));
    assert!(prepared.as_bytes().starts_with(&[0xff, 0xd8]));
    assert!(prepared.as_bytes().ends_with(&[0xff, 0xd9]));
}

#[test]
fn jpeg_tables_soi_and_eoi_are_normalized_to_one_interchange_stream() {
    let full = minimal_baseline_jpeg();
    let (tables, tile) = split_tables_and_scan(&full);
    let prepared =
        prepare_tiff_jpeg_tile(&tile, Some(&tables), prepare_options()).expect("prepared");
    let soi_count = prepared
        .as_bytes()
        .windows(2)
        .filter(|window| *window == [0xff, 0xd8])
        .count();
    let eoi_count = prepared
        .as_bytes()
        .windows(2)
        .filter(|window| *window == [0xff, 0xd9])
        .count();

    assert_eq!(soi_count, 1);
    assert_eq!(eoi_count, 1);
}

#[test]
fn identical_duplicate_jpeg_tables_are_deduplicated_under_allow_identical() {
    let full = minimal_baseline_jpeg();
    let (mut tables, tile) = split_tables_and_scan(&full);
    let dqt = tables
        .windows(2)
        .position(|window| window == [0xff, 0xdb])
        .expect("DQT");
    let dqt_len = u16::from_be_bytes([tables[dqt + 2], tables[dqt + 3]]) as usize + 2;
    let duplicate = tables[dqt..dqt + dqt_len].to_vec();
    tables.splice(dqt..dqt, duplicate);
    let mut opts = prepare_options();
    opts.duplicate_table_policy = DuplicateTablePolicy::AllowIdentical;

    let prepared = prepare_tiff_jpeg_tile(&tile, Some(&tables), opts).expect("prepared");
    let dqt_count = prepared
        .as_bytes()
        .windows(2)
        .filter(|window| *window == [0xff, 0xdb])
        .count();

    assert_eq!(dqt_count, 1);
}

#[test]
fn conflicting_duplicate_jpeg_tables_are_rejected() {
    let full = minimal_baseline_jpeg();
    let (tables, tile) = split_tables_and_scan(&full);
    let mut conflicting = mutate_first_dqt_value(&tables);
    let dqt = tables
        .windows(2)
        .position(|window| window == [0xff, 0xdb])
        .expect("DQT");
    let dqt_len = u16::from_be_bytes([tables[dqt + 2], tables[dqt + 3]]) as usize + 2;
    conflicting.splice(2..2, tables[dqt..dqt + dqt_len].iter().copied());

    let err = prepare_tiff_jpeg_tile(&tile, Some(&conflicting), prepare_options()).unwrap_err();
    assert!(matches!(err, JpegError::ConflictingDuplicateTable { .. }));
}

#[test]
fn zero_sof_dimensions_are_repaired_with_expected_dimensions() {
    let bytes = zero_sof_jpeg();
    let mut opts = prepare_options();
    opts.expected_dimensions = Some((16, 16));
    opts.repair_zero_sof_dimensions = true;
    let prepared = prepare_tiff_jpeg_tile(&bytes, None, opts).expect("prepared");
    let info = Decoder::inspect(prepared.as_bytes()).expect("inspect repaired");

    assert_eq!(info.dimensions, (16, 16));
    assert!(matches!(prepared, PreparedJpeg::Owned(_)));
}

#[test]
fn nonzero_sof_dimensions_conflicting_with_expected_dimensions_are_rejected() {
    let bytes = minimal_baseline_jpeg();
    let mut opts = prepare_options();
    opts.expected_dimensions = Some((32, 16));

    let err = prepare_tiff_jpeg_tile(&bytes, None, opts).unwrap_err();
    assert!(matches!(
        err,
        JpegError::ConflictingExpectedDimensions { .. }
    ));
}

#[test]
fn dri_survives_preparation_and_restart_validation_accepts_ordered_rst_markers() {
    let bytes = restart_jpeg();
    let mut opts = prepare_options();
    opts.validate_restart_markers = true;
    let prepared = prepare_tiff_jpeg_tile(&bytes, None, opts).expect("prepared");
    let info = Decoder::inspect(prepared.as_bytes()).expect("inspect");

    assert_eq!(info.restart_interval, Some(1));
}

#[test]
fn restart_validation_rejects_out_of_order_rst_marker() {
    let mut bytes = restart_jpeg();
    let rst = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xd0])
        .expect("RST0");
    bytes[rst + 1] = 0xd3;
    let mut opts = prepare_options();
    opts.validate_restart_markers = true;

    let err = prepare_tiff_jpeg_tile(&bytes, None, opts).unwrap_err();
    assert!(matches!(err, JpegError::RestartMismatch { .. }));
}

#[test]
fn inspect_returns_info_for_valid_baseline_jpeg() {
    let info = Decoder::inspect(&minimal_baseline_jpeg()).unwrap();
    assert_eq!(info.dimensions, (16, 16));
    assert_eq!(info.sof_kind, SofKind::Baseline8);
    assert_eq!(info.color_space, ColorSpace::YCbCr);
    assert_eq!(info.bit_depth, 8);
    assert!(info.restart_interval.is_none());
    assert_eq!(
        info.mcu_geometry,
        McuGeometry {
            width: 16,
            height: 16,
            columns: 1,
            rows: 1,
            count: 1,
        }
    );
    assert_eq!(info.scan_count, 1, "single SOS → scan_count must be 1");
}

#[test]
fn decode_options_color_transform_setter_round_trips() {
    let mut options = DecodeOptions::default();
    options.set_color_transform(ColorTransform::ForceRgb);
    assert!(matches!(
        options.color_transform(),
        ColorTransform::ForceRgb
    ));
}

#[test]
fn inspect_with_options_forces_three_component_color_space() {
    let bytes = minimal_baseline_jpeg();
    let auto = Decoder::inspect(&bytes).unwrap();
    assert_eq!(auto.color_space, ColorSpace::YCbCr);

    let force_rgb = Decoder::inspect_with_options(
        &bytes,
        DecodeOptions::default().with_color_transform(ColorTransform::ForceRgb),
    )
    .unwrap();
    assert_eq!(force_rgb.color_space, ColorSpace::Rgb);

    let force_ycbcr = Decoder::inspect_with_options(
        &bytes,
        DecodeOptions::default().with_color_transform(ColorTransform::ForceYCbCr),
    )
    .unwrap();
    assert_eq!(force_ycbcr.color_space, ColorSpace::YCbCr);
}

#[test]
fn inspect_returns_typed_error_for_empty_input() {
    let err = Decoder::inspect(&[]).unwrap_err();
    assert!(matches!(err, JpegError::Truncated { .. }));
}

#[test]
fn inspect_returns_typed_error_for_missing_sof() {
    // SOI + EOI, nothing between
    let bytes = &[0xFF, 0xD8, 0xFF, 0xD9];
    let err = Decoder::inspect(bytes).unwrap_err();
    assert!(matches!(err, JpegError::MissingMarker { .. }));
}

#[test]
fn inspect_returns_typed_error_for_future_sof_classes() {
    for (marker, expected_reason) in [
        (0xc9, UnsupportedReason::ArithmeticCoding),
        (0xc5, UnsupportedReason::DifferentialBaseline),
        (0xc6, UnsupportedReason::Hierarchical),
        (0xcd, UnsupportedReason::ArithmeticAndHierarchical),
    ] {
        let bytes = minimal_jpeg_with_sof_marker(marker);
        let err = Decoder::inspect(&bytes).unwrap_err();
        assert!(matches!(
            err,
            JpegError::UnsupportedSof { marker: got_marker, reason }
                if got_marker == marker && reason == expected_reason
        ));
        assert!(err.is_unsupported());
    }
}

#[test]
fn inspect_is_api_misuse_predicate_negative_for_all_parse_errors() {
    // Parse errors are never API misuse.
    let err = Decoder::inspect(&[]).unwrap_err();
    assert!(!err.is_api_misuse());
}

#[test]
fn inspect_reports_all_progressive_scans() {
    let info = Decoder::inspect(&progressive_8x8_jpeg()).unwrap();
    assert_eq!(info.sof_kind, SofKind::Progressive8);
    assert_eq!(info.scan_count, 10);
}

#[test]
fn inspect_treats_dri_zero_as_no_restart_interval() {
    let info = Decoder::inspect(&minimal_baseline_jpeg_with_restart_interval(0)).unwrap();
    assert!(info.restart_interval.is_none());
}

#[test]
fn inspect_reports_restart_interval_and_mcu_geometry_for_wsi_planning() {
    let info = Decoder::inspect(&fixtures::baseline_420_restart_32x16_jpeg()).unwrap();

    assert_eq!(info.dimensions, (32, 16));
    assert_eq!(info.restart_interval, Some(2));
    assert_eq!(
        info.mcu_geometry,
        McuGeometry {
            width: 16,
            height: 16,
            columns: 2,
            rows: 1,
            count: 2,
        }
    );
}

#[test]
fn jpeg_view_restart_index_reports_original_byte_offsets() {
    let bytes = restart_coded_grayscale_jpeg(24, 8);
    let view = JpegView::parse(&bytes).expect("view");
    let index = view
        .restart_index()
        .expect("restart index")
        .expect("DRI should produce an index");
    let scan_data_offset = scan_data_offset(&bytes);
    let rst_offsets = restart_marker_offsets(&bytes);

    assert_eq!(index.scan_data_offset, scan_data_offset);
    assert_eq!(index.interval_mcus, 1);
    assert_eq!(
        index.segments,
        vec![
            RestartSegment {
                start_mcu: 0,
                entropy_offset: scan_data_offset,
                marker_offset: None,
                marker: None,
            },
            RestartSegment {
                start_mcu: 1,
                entropy_offset: rst_offsets[0] + 2,
                marker_offset: Some(rst_offsets[0]),
                marker: Some(0xd0),
            },
            RestartSegment {
                start_mcu: 2,
                entropy_offset: rst_offsets[1] + 2,
                marker_offset: Some(rst_offsets[1]),
                marker: Some(0xd1),
            },
        ]
    );

    let decoder_index = Decoder::new(&bytes)
        .expect("decoder")
        .restart_index()
        .expect("decoder restart index");
    assert_eq!(decoder_index, Some(index));
}

#[test]
fn restart_index_is_none_without_dri() {
    let bytes = minimal_baseline_jpeg();
    let view = JpegView::parse(&bytes).expect("view");
    assert_eq!(view.restart_index().expect("restart index"), None);
}

#[test]
fn jpeg_view_exposes_baseline_passthrough_candidate_with_original_bytes() {
    let bytes = minimal_baseline_jpeg();
    let view = JpegView::parse(&bytes).expect("view");
    let candidate = view
        .passthrough_candidate()
        .expect("baseline JPEG passthrough candidate");
    let requirements = PassthroughRequirements::new(
        CompressedTransferSyntax::JpegBaseline8,
        CompressedPayloadKind::JpegInterchange,
    )
    .with_dimensions((16, 16))
    .with_components(3)
    .with_bit_depth(8);

    assert_eq!(view.bytes(), bytes.as_slice());
    assert_eq!(
        candidate.transfer_syntax(),
        CompressedTransferSyntax::JpegBaseline8
    );
    assert_eq!(
        candidate.payload_kind(),
        CompressedPayloadKind::JpegInterchange
    );
    assert_eq!(
        candidate.evaluate(&requirements),
        PassthroughDecision::Copy {
            bytes: bytes.as_slice()
        }
    );
}

#[test]
fn jpeg_progressive_is_not_offered_as_active_passthrough_candidate() {
    let bytes = progressive_8x8_jpeg();
    let view = JpegView::parse(&bytes).expect("progressive view");

    assert!(view.passthrough_candidate().is_none());
}
