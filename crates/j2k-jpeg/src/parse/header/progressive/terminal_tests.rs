// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::parse_header;
use super::super::tests::progressive_two_scan_jpeg;
use crate::error::{JpegError, MarkerKind};

#[test]
fn trailing_ff_marker_prefix_is_truncated_not_missing_eoi() {
    let mut bytes = progressive_two_scan_jpeg();
    bytes.pop();

    assert!(matches!(
        parse_header(&bytes),
        Err(JpegError::Truncated {
            offset,
            expected: 1
        }) if offset == bytes.len()
    ));
}

#[test]
fn embedded_second_soi_after_entropy_is_a_duplicate_marker_error() {
    let mut bytes = progressive_two_scan_jpeg();
    *bytes.last_mut().expect("EOI code") = 0xd8;

    assert!(matches!(
        parse_header(&bytes),
        Err(JpegError::DuplicateMarker {
            marker: MarkerKind::Soi,
            ..
        })
    ));
}

#[test]
fn valid_multiscan_parser_records_each_exact_terminal() {
    let bytes = progressive_two_scan_jpeg();
    let second_sos = bytes
        .windows(2)
        .rposition(|window| window == [0xff, 0xda])
        .expect("second SOS");
    let eoi = bytes.len() - 2;
    let header = parse_header(&bytes).expect("valid multi-scan stream");

    assert_eq!(header.progressive_scans.len(), 2);
    assert_eq!(header.progressive_scans[0].terminal_offset, second_sos);
    assert_eq!(header.progressive_scans[0].terminal_code, 0xda);
    assert_eq!(header.progressive_scans[1].terminal_offset, eoi);
    assert_eq!(header.progressive_scans[1].terminal_code, 0xd9);
}
