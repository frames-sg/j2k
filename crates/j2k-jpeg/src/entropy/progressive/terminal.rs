// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact progressive entropy-boundary reconciliation.

use crate::error::{JpegError, ProgressiveScanTerminationError};
use crate::internal::bit_reader::BitReader;

use super::model::PreparedProgressiveScan;

pub(super) fn finish_progressive_scan(
    br: &mut BitReader<'_>,
    scan_bytes: &[u8],
    scan: &PreparedProgressiveScan,
    eob_run: u32,
) -> Result<(), JpegError> {
    if eob_run != 0 {
        return Err(termination_error(
            scan,
            ProgressiveScanTerminationError::ResidualEobRun { remaining: eob_run },
        ));
    }

    let expected_relative = scan
        .terminal_offset
        .checked_sub(scan.entropy_offset)
        .ok_or(JpegError::InternalInvariant {
            reason: "progressive terminal precedes its entropy offset",
        })?;
    if expected_relative > scan_bytes.len() {
        return Err(JpegError::InternalInvariant {
            reason: "progressive terminal is outside the JPEG input",
        });
    }

    reconcile_marker(br, scan_bytes, scan, expected_relative)?;
    let unread_bits = br.unread_real_bits();
    if unread_bits > 7 {
        return Err(termination_error(
            scan,
            ProgressiveScanTerminationError::ExcessEntropy {
                unread_bytes: usize::from(unread_bits / 8),
                unread_bits,
            },
        ));
    }
    if !br.unread_real_bits_are_ones() {
        return Err(termination_error(
            scan,
            ProgressiveScanTerminationError::InvalidPadding { unread_bits },
        ));
    }
    Ok(())
}

fn reconcile_marker(
    br: &mut BitReader<'_>,
    scan_bytes: &[u8],
    scan: &PreparedProgressiveScan,
    expected_relative: usize,
) -> Result<(), JpegError> {
    let expected_marker = (scan.terminal_code != 0).then_some(scan.terminal_code);
    if br.observed_marker().is_none()
        && expected_marker.is_none()
        && br.position() < expected_relative
    {
        return Err(termination_error(
            scan,
            ProgressiveScanTerminationError::ExcessEntropy {
                unread_bytes: expected_relative - br.position(),
                unread_bits: br.unread_real_bits(),
            },
        ));
    }
    if br.observed_marker().is_none() && expected_marker.is_some() {
        let position = br.position();
        let fill = scan_bytes
            .get(position..expected_relative)
            .ok_or_else(|| terminal_mismatch(scan, br))?;
        if fill.iter().any(|&byte| byte != 0xff) {
            return Err(termination_error(
                scan,
                ProgressiveScanTerminationError::ExcessEntropy {
                    unread_bytes: fill.len(),
                    unread_bits: br.unread_real_bits(),
                },
            ));
        }
        br.observe_marker();
    }

    let found_marker = br.observed_marker();
    let found_relative = (found_marker.is_some()).then_some(br.position());
    if found_marker != expected_marker
        || found_relative.is_some_and(|offset| offset != expected_relative)
        || (found_marker.is_none() && br.position() != expected_relative)
    {
        return Err(terminal_mismatch(scan, br));
    }
    Ok(())
}

fn terminal_mismatch(scan: &PreparedProgressiveScan, br: &BitReader<'_>) -> JpegError {
    let found_marker = br.observed_marker();
    let found_offset = found_marker.and_then(|_| scan.entropy_offset.checked_add(br.position()));
    termination_error(
        scan,
        ProgressiveScanTerminationError::TerminalMismatch {
            expected_offset: scan.terminal_offset,
            expected_marker: (scan.terminal_code != 0).then_some(scan.terminal_code),
            found_offset,
            found_marker,
        },
    )
}

fn termination_error(
    scan: &PreparedProgressiveScan,
    state: ProgressiveScanTerminationError,
) -> JpegError {
    JpegError::InvalidProgressiveScanTermination {
        offset: scan.terminal_offset,
        scan_offset: scan.entropy_offset,
        state,
    }
}

#[cfg(test)]
mod mismatch_tests;
#[cfg(test)]
mod tests;
