// SPDX-License-Identifier: MIT OR Apache-2.0

//! SOF/SOS validation shared by the header and progressive-script walkers.

use crate::error::JpegError;
use crate::info::SofKind;
use crate::parse::scan::ParsedScan;
use crate::parse::sof::ParsedSof;

pub(super) fn validate_scan_parameters(
    sof_kind: SofKind,
    scan: &ParsedScan,
    offset: usize,
) -> Result<(), JpegError> {
    if matches!(sof_kind, SofKind::Baseline8 | SofKind::Extended8)
        && (scan.ss != 0 || scan.se != 63 || scan.ah != 0 || scan.al != 0)
    {
        return Err(JpegError::InvalidScanParameters {
            offset,
            ss: scan.ss,
            se: scan.se,
            ah: scan.ah,
            al: scan.al,
        });
    }
    Ok(())
}

pub(super) const fn normalize_restart_interval(interval: u16) -> Option<u16> {
    if interval == 0 {
        None
    } else {
        Some(interval)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "validated sequential scan component counts fit the JPEG SOS byte field"
)]
pub(super) fn validate_sequential_scan_components(
    sof: &ParsedSof,
    scan: &ParsedScan,
    offset: usize,
) -> Result<(), JpegError> {
    if !matches!(sof.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return Ok(());
    }

    validate_known_unique_components(sof, scan, offset)?;
    if scan.components.len() != sof.component_ids.len() {
        return Err(JpegError::InvalidSequentialComponentSet {
            offset,
            expected: sof.component_ids.len() as u8,
            found: scan.components.len() as u8,
        });
    }
    Ok(())
}

/// Whether `scan` opens a split (multi-scan) frame: a sequential or lossless
/// SOF whose first scan codes fewer components than the frame, the rest
/// following in later scans.
pub(super) fn is_split_frame_scan(sof: &ParsedSof, scan: &ParsedScan) -> bool {
    matches!(
        sof.sof_kind,
        SofKind::Baseline8 | SofKind::Extended8 | SofKind::Extended12 | SofKind::Lossless
    ) && scan.components.len() < sof.component_ids.len()
}

/// Validates one scan of a split frame over known, unique components:
/// sequential scans code the full spectrum with no successive
/// approximation; lossless scans select predictor 1..=7 with a point
/// transform below the precision.
pub(super) fn validate_split_scan(
    sof: &ParsedSof,
    scan: &ParsedScan,
    offset: usize,
) -> Result<(), JpegError> {
    let invalid = if sof.sof_kind == SofKind::Lossless {
        !(1..=7).contains(&scan.ss) || scan.se != 0 || scan.ah != 0 || scan.al >= sof.bit_depth
    } else {
        scan.ss != 0 || scan.se != 63 || scan.ah != 0 || scan.al != 0
    };
    if scan.components.is_empty() || invalid {
        return Err(JpegError::InvalidScanParameters {
            offset,
            ss: scan.ss,
            se: scan.se,
            ah: scan.ah,
            al: scan.al,
        });
    }
    validate_known_unique_components(sof, scan, offset)
}

pub(super) fn validate_progressive_scan_components(
    sof: &ParsedSof,
    scan: &ParsedScan,
    offset: usize,
) -> Result<(), JpegError> {
    if !matches!(sof.sof_kind, SofKind::Progressive8 | SofKind::Progressive12) {
        return Ok(());
    }
    if scan.components.is_empty()
        || scan.ss > scan.se
        || scan.se > 63
        || scan.ah > 13
        || scan.al > 13
        || (scan.ah != 0 && scan.ah != scan.al + 1)
        || (scan.ss == 0 && scan.se != 0)
        || (scan.ss > 0 && scan.components.len() != 1)
    {
        return Err(JpegError::InvalidScanParameters {
            offset,
            ss: scan.ss,
            se: scan.se,
            ah: scan.ah,
            al: scan.al,
        });
    }
    validate_known_unique_components(sof, scan, offset)
}

fn validate_known_unique_components(
    sof: &ParsedSof,
    scan: &ParsedScan,
    offset: usize,
) -> Result<(), JpegError> {
    for (index, component) in scan.components.iter().enumerate() {
        let component_offset = offset + 1 + index * 2;
        if !sof.component_ids.contains(&component.id) {
            return Err(JpegError::UnknownScanComponent {
                offset: component_offset,
                component: component.id,
            });
        }
        if scan.components.as_slice()[..index]
            .iter()
            .any(|seen| seen.id == component.id)
        {
            return Err(JpegError::DuplicateScanComponent {
                offset: component_offset,
                component: component.id,
            });
        }
    }
    Ok(())
}
