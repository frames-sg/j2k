// SPDX-License-Identifier: MIT OR Apache-2.0

use super::ProgressiveScriptState;
use crate::error::{JpegError, ProgressiveScanStateError};
use crate::parse::scan::{parse_scan_header, ParsedScan};
use crate::parse::sof::{parse_sof, ParsedSof};
use alloc::vec::Vec;

fn three_component_sof() -> ParsedSof {
    parse_sof(
        0xc2,
        &[8, 0, 8, 0, 8, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0],
        4,
    )
    .expect("valid progressive SOF fixture")
}

fn scan(components: &[u8], ss: u8, se: u8, ah: u8, al: u8) -> ParsedScan {
    let mut payload = Vec::with_capacity(1 + components.len() * 2 + 3);
    payload.push(u8::try_from(components.len()).expect("scan component count fits u8"));
    for &component in components {
        payload.extend_from_slice(&[component, 0]);
    }
    payload.extend_from_slice(&[ss, se, (ah << 4) | al]);
    parse_scan_header(&payload, 10).expect("valid progressive SOS fixture")
}

#[test]
fn valid_multiscan_script_advances_dc_and_partial_ac_state() {
    let sof = three_component_sof();
    let mut state = ProgressiveScriptState::new(&sof);

    for (offset, scan) in [
        (100, scan(&[1, 2, 3], 0, 0, 0, 2)),
        (120, scan(&[1, 2, 3], 0, 0, 2, 1)),
        (140, scan(&[1, 2, 3], 0, 0, 1, 0)),
        (160, scan(&[1], 1, 5, 0, 2)),
        (180, scan(&[1], 1, 5, 2, 1)),
        (200, scan(&[1], 1, 5, 1, 0)),
    ] {
        state.record_scan(offset, &scan).unwrap();
    }

    state.finish_terminal(220).unwrap();
}

#[test]
fn duplicate_initial_dc_reports_first_coefficient_and_prior_state() {
    let sof = three_component_sof();
    let mut state = ProgressiveScriptState::new(&sof);
    state.record_scan(100, &scan(&[1], 0, 0, 0, 3)).unwrap();

    assert_eq!(
        state.record_scan(120, &scan(&[1], 0, 0, 0, 1)),
        Err(JpegError::InvalidProgressiveScanState {
            offset: 120,
            marker: 0xda,
            component: 1,
            coefficient: 0,
            state: ProgressiveScanStateError::DuplicateInitial {
                previous_al: 3,
                al: 1,
            },
        })
    );
}

#[test]
fn overlapping_initial_ac_ranges_report_shared_boundary() {
    let sof = three_component_sof();
    let mut state = ProgressiveScriptState::new(&sof);
    state.record_scan(100, &scan(&[1], 1, 5, 0, 0)).unwrap();

    assert!(matches!(
        state.record_scan(120, &scan(&[1], 5, 63, 0, 0)),
        Err(JpegError::InvalidProgressiveScanState {
            component: 1,
            coefficient: 5,
            state: ProgressiveScanStateError::DuplicateInitial { .. },
            ..
        })
    ));
}

#[test]
fn refinement_before_initial_reports_coefficient_63_boundary() {
    let sof = three_component_sof();
    let mut state = ProgressiveScriptState::new(&sof);

    assert_eq!(
        state.record_scan(100, &scan(&[1], 63, 63, 1, 0)),
        Err(JpegError::InvalidProgressiveScanState {
            offset: 100,
            marker: 0xda,
            component: 1,
            coefficient: 63,
            state: ProgressiveScanStateError::RefinementBeforeInitial { ah: 1, al: 0 },
        })
    );
}

#[test]
fn skipped_refinement_reports_previous_and_requested_levels() {
    let sof = three_component_sof();
    let mut state = ProgressiveScriptState::new(&sof);
    state.record_scan(100, &scan(&[1], 63, 63, 0, 3)).unwrap();

    assert_eq!(
        state.record_scan(120, &scan(&[1], 63, 63, 2, 1)),
        Err(JpegError::InvalidProgressiveScanState {
            offset: 120,
            marker: 0xda,
            component: 1,
            coefficient: 63,
            state: ProgressiveScanStateError::RefinementMismatch {
                previous_al: 3,
                ah: 2,
                al: 1,
            },
        })
    );
}

#[test]
fn eoi_requires_initial_dc_for_each_frame_component_only() {
    let sof = three_component_sof();
    let mut state = ProgressiveScriptState::new(&sof);
    state.record_scan(100, &scan(&[1, 2], 0, 0, 0, 0)).unwrap();
    state.record_scan(120, &scan(&[1], 1, 1, 0, 0)).unwrap();

    assert_eq!(
        state.finish_terminal(140),
        Err(JpegError::InvalidProgressiveScanState {
            offset: 140,
            marker: 0xd9,
            component: 3,
            coefficient: 0,
            state: ProgressiveScanStateError::MissingInitialDc,
        })
    );
}
