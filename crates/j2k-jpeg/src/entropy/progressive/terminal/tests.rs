// SPDX-License-Identifier: MIT OR Apache-2.0

use super::finish_progressive_scan;
use crate::entropy::progressive::model::PreparedProgressiveScan;
use crate::error::{JpegError, ProgressiveScanTerminationError};
use crate::internal::bit_reader::BitReader;

fn scan(
    entropy_offset: usize,
    terminal_offset: usize,
    terminal_code: u8,
) -> PreparedProgressiveScan {
    PreparedProgressiveScan {
        component_start: 0,
        component_len: 1,
        ss: 1,
        se: 1,
        ah: 0,
        al: 0,
        entropy_offset,
        terminal_offset,
        terminal_code,
        restart_interval: None,
    }
}

#[test]
fn valid_multiscan_boundaries_accept_exact_sos_and_eoi_markers() {
    for (entropy_offset, bytes, terminal_code) in [
        (100, &[0b1011_1111, 0xff, 0xda][..], 0xda),
        (200, &[0b1011_1111, 0xff, 0xd9][..], 0xd9),
    ] {
        let mut br = BitReader::new(bytes);
        assert_eq!(br.read_bits(4).unwrap(), 0b1011);
        finish_progressive_scan(
            &mut br,
            bytes,
            &scan(entropy_offset, entropy_offset + 1, terminal_code),
            0,
        )
        .unwrap();
    }
}

#[test]
fn physical_eof_allows_synthetic_lookahead_but_checks_real_padding() {
    let bytes = &[0b1011_1111];
    let mut br = BitReader::new_with_eof_padding(bytes, true);
    br.ensure_bits_padded(12).unwrap();
    br.consume_bits(4);

    finish_progressive_scan(&mut br, bytes, &scan(300, 301, 0), 0).unwrap();
}

#[test]
fn residual_eob_run_is_rejected_at_scan_end() {
    let bytes = &[0b1011_1111, 0xff, 0xd9];
    let mut br = BitReader::new(bytes);
    br.read_bits(4).unwrap();

    assert!(matches!(
        finish_progressive_scan(&mut br, bytes, &scan(400, 401, 0xd9), 2),
        Err(JpegError::InvalidProgressiveScanTermination {
            state: ProgressiveScanTerminationError::ResidualEobRun { remaining: 2 },
            ..
        })
    ));
}

#[test]
fn complete_excess_entropy_byte_is_rejected() {
    let bytes = &[0b1011_1111, 0xaa, 0xff, 0xd9];
    let mut br = BitReader::new(bytes);
    br.read_bits(4).unwrap();

    assert!(matches!(
        finish_progressive_scan(&mut br, bytes, &scan(500, 502, 0xd9), 0),
        Err(JpegError::InvalidProgressiveScanTermination {
            state: ProgressiveScanTerminationError::ExcessEntropy { .. },
            ..
        })
    ));
}

#[test]
fn non_one_terminal_padding_is_rejected() {
    let bytes = &[0b1011_0000, 0xff, 0xd9];
    let mut br = BitReader::new(bytes);
    br.read_bits(4).unwrap();

    assert!(matches!(
        finish_progressive_scan(&mut br, bytes, &scan(550, 551, 0xd9), 0),
        Err(JpegError::InvalidProgressiveScanTermination {
            state: ProgressiveScanTerminationError::InvalidPadding { unread_bits: 4 },
            ..
        })
    ));
}

#[test]
fn repeated_ff_fill_before_terminal_is_not_entropy() {
    let bytes = &[0b1011_1111, 0xff, 0xff, 0xff, 0xd9];
    let mut br = BitReader::new(bytes);
    br.read_bits(4).unwrap();

    finish_progressive_scan(&mut br, bytes, &scan(600, 603, 0xd9), 0).unwrap();
}
