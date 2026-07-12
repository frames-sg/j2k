// SPDX-License-Identifier: MIT OR Apache-2.0

use super::finish_progressive_scan;
use crate::entropy::progressive::model::PreparedProgressiveScan;
use crate::error::{JpegError, ProgressiveScanTerminationError};
use crate::internal::bit_reader::BitReader;

#[test]
fn unexpected_intermediate_restart_does_not_match_the_parsed_eoi_boundary() {
    let bytes = &[0b1011_1111, 0xff, 0xd0, 0xff, 0xd9];
    let mut br = BitReader::new(bytes);
    br.read_bits(4).unwrap();
    let scan = PreparedProgressiveScan {
        component_start: 0,
        component_len: 1,
        ss: 1,
        se: 1,
        ah: 0,
        al: 0,
        entropy_offset: 700,
        terminal_offset: 703,
        terminal_code: 0xd9,
        restart_interval: None,
    };

    assert!(matches!(
        finish_progressive_scan(&mut br, bytes, &scan, 0),
        Err(JpegError::InvalidProgressiveScanTermination {
            state: ProgressiveScanTerminationError::TerminalMismatch {
                expected_offset: 703,
                expected_marker: Some(0xd9),
                found_offset: Some(701),
                found_marker: Some(0xd0),
            },
            ..
        })
    ));
}
