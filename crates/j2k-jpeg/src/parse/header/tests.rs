// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use super::{parse_header, ParsedProgressiveScan};
use crate::error::{JpegError, MarkerKind, Warning};
use crate::info::{ColorSpace, SofKind};

fn minimal_baseline_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(core::iter::repeat_n(1u8, 64));
    bytes.extend_from_slice(&[
        0xff,
        0xc0,
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
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xaa,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xbb,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);
    bytes.extend_from_slice(&[0x00, 0xff, 0xd9]);
    bytes
}

pub(super) fn progressive_two_scan_jpeg() -> Vec<u8> {
    let mut bytes = minimal_baseline_jpeg();
    let sof_offset = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xc0])
        .unwrap();
    bytes[sof_offset + 1] = 0xc2;
    let first_sos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .unwrap();
    // Initial DC at Al=1; the inserted scan legally refines Ah=1 to Al=0.
    bytes[first_sos + 12] = 0;
    bytes[first_sos + 13] = 1;
    let eoi = bytes
        .windows(2)
        .rposition(|window| window == [0xff, 0xd9])
        .unwrap();
    let second_scan = [
        0xff, 0xda, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 0, 0x10, 0x00,
    ];
    bytes.splice(eoi..eoi, second_scan);
    bytes
}

#[test]
fn parses_minimal_baseline_jpeg() {
    let header = parse_header(&minimal_baseline_jpeg()).unwrap();
    assert_eq!(header.dimensions, (16, 16));
    assert_eq!(header.sof_kind, SofKind::Baseline8);
    assert_eq!(header.color_space(), ColorSpace::YCbCr);
    assert_eq!(header.bit_depth, 8);
    assert_eq!(header.sampling.components(), &[(2, 2), (1, 1), (1, 1)]);
    assert!(header.quant_tables.entries[0].is_some());
    assert!(header.huffman_tables.dc[0].is_some());
    assert!(header.huffman_tables.ac[0].is_some());
    assert!(header.sos_offset.is_some());
    assert_eq!(header.scan_count, 1);
}

#[test]
fn rejects_missing_sof() {
    let error = parse_header(&[0xff, 0xd8, 0xff, 0xd9]).unwrap_err();
    assert!(matches!(
        error,
        JpegError::MissingMarker {
            marker: MarkerKind::Sof
        }
    ));
}

#[test]
fn rejects_duplicate_sof() {
    let mut bytes = minimal_baseline_jpeg();
    let sos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .unwrap();
    let duplicate = [
        0xff,
        0xc0,
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
    ];
    bytes.splice(sos..sos, duplicate);
    assert!(matches!(
        parse_header(&bytes),
        Err(JpegError::DuplicateMarker {
            marker: MarkerKind::Sof,
            ..
        })
    ));
}

#[test]
fn info_method_produces_expected_fields() {
    let info = parse_header(&minimal_baseline_jpeg()).unwrap().info();
    assert_eq!(info.dimensions, (16, 16));
    assert_eq!(info.sof_kind, SofKind::Baseline8);
    assert_eq!(info.scan_count, 1);
}

#[test]
fn app14_ycbcr_overrides_default() {
    let mut bytes = minimal_baseline_jpeg();
    let mut app14 = vec![0xff, 0xee, 0x00, 14];
    app14.extend_from_slice(b"Adobe");
    app14.extend_from_slice(&[0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x01]);
    bytes.splice(2..2, app14);
    assert_eq!(
        parse_header(&bytes).unwrap().color_space(),
        ColorSpace::YCbCr
    );
}

#[test]
fn app14_unknown_marks_rgb_for_3_components() {
    let mut bytes = minimal_baseline_jpeg();
    let mut app14 = vec![0xff, 0xee, 0x00, 14];
    app14.extend_from_slice(b"Adobe");
    app14.extend_from_slice(&[0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00]);
    bytes.splice(2..2, app14);
    assert_eq!(parse_header(&bytes).unwrap().color_space(), ColorSpace::Rgb);
}

#[test]
fn scan_count_tracks_progressive_sos_markers() {
    let header = parse_header(&progressive_two_scan_jpeg()).unwrap();
    assert_eq!(header.sof_kind, SofKind::Progressive8);
    assert_eq!(header.scan_count, 2);
    assert_eq!(header.progressive_scans.len(), 2);
}

#[test]
fn repeated_sos_reuses_versioned_table_state_without_snapshot_growth() {
    let header = parse_header(&progressive_two_scan_jpeg()).unwrap();
    assert_eq!(header.huffman_tables.versions.len(), 2);
    assert_eq!(header.quant_tables.versions.len(), 1);
    assert_eq!(
        header.progressive_scans[0].table_state,
        header.progressive_scans[1].table_state
    );
}

#[test]
fn progressive_scan_metadata_boundary_shares_the_context_cap() {
    let available =
        DEFAULT_MAX_HOST_ALLOCATION_BYTES - crate::context::MAX_DECODER_CONTEXT_ALLOCATION_BYTES;
    let max_scans = available / size_of::<ParsedProgressiveScan>();
    let exact = max_scans * size_of::<ParsedProgressiveScan>();
    crate::parse::allocation::ensure_retained_metadata_bytes(exact).unwrap();

    let one_over = (max_scans + 1) * size_of::<ParsedProgressiveScan>();
    assert!(matches!(
        crate::parse::allocation::ensure_retained_metadata_bytes(one_over),
        Err(JpegError::MemoryCapExceeded { requested, cap })
            if requested > cap && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
    ));
}

#[test]
fn warning_metadata_boundary_shares_the_context_cap() {
    let available =
        DEFAULT_MAX_HOST_ALLOCATION_BYTES - crate::context::MAX_DECODER_CONTEXT_ALLOCATION_BYTES;
    let max_warnings = available / size_of::<Warning>();
    crate::parse::allocation::ensure_retained_metadata_bytes(max_warnings * size_of::<Warning>())
        .unwrap();
    assert!(matches!(
        crate::parse::allocation::ensure_retained_metadata_bytes(
            (max_warnings + 1) * size_of::<Warning>(),
        ),
        Err(JpegError::MemoryCapExceeded { .. })
    ));
}

#[test]
fn sos_offset_points_at_first_entropy_byte() {
    let bytes = minimal_baseline_jpeg();
    let header = parse_header(&bytes).unwrap();
    let marker = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .unwrap();
    assert_eq!(header.sos_offset, Some(marker + 14));
}

#[test]
fn extracts_scan_component_table_selectors() {
    let header = parse_header(&minimal_baseline_jpeg()).unwrap();
    let scan = header.scan.as_ref().expect("SOS must be parsed");
    assert_eq!(scan.components.len(), 3);
    for component in &scan.components {
        assert_eq!(component.dc_table, 0);
        assert_eq!(component.ac_table, 0);
    }
    assert_eq!((scan.ss, scan.se, scan.ah, scan.al), (0, 63, 0, 0));
}

#[test]
fn rejects_malformed_sos_length() {
    let mut bytes = minimal_baseline_jpeg();
    let sos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .unwrap();
    bytes.drain(sos + 4..sos + 14);
    assert!(matches!(
        parse_header(&bytes),
        Err(JpegError::Truncated { .. } | JpegError::InvalidSegmentLength { .. })
    ));
}
