// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::parse_header;
use super::super::tests::progressive_two_scan_jpeg;
use crate::error::{JpegError, ProgressiveScanStateError, Warning};

#[test]
fn eof_without_eoi_is_retained_as_a_missing_eoi_warning() {
    let mut bytes = progressive_two_scan_jpeg();
    bytes.truncate(bytes.len() - 2);

    let header = parse_header(&bytes).expect("otherwise-complete entropy tolerates missing EOI");
    assert!(header.warnings.contains(&Warning::MissingEoi));
    let terminal = header.progressive_scans.last().expect("last scan");
    assert_eq!(
        (terminal.terminal_offset, terminal.terminal_code),
        (bytes.len(), 0)
    );
}

#[test]
fn eof_still_requires_initial_dc_for_every_frame_component() {
    let mut bytes = progressive_two_scan_jpeg();
    let first_sos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("first SOS");
    let two_component_dc = [0xff, 0xda, 0, 10, 2, 1, 0, 2, 0, 0, 0, 0];
    bytes.splice(first_sos..first_sos + 14, two_component_dc);
    let next_sos = bytes
        .windows(2)
        .rposition(|window| window == [0xff, 0xda])
        .expect("next SOS");
    bytes.truncate(next_sos);

    assert!(matches!(
        parse_header(&bytes),
        Err(JpegError::InvalidProgressiveScanState {
            offset,
            marker: 0xd9,
            component: 3,
            coefficient: 0,
            state: ProgressiveScanStateError::MissingInitialDc,
        }) if offset == bytes.len()
    ));
}
