// SPDX-License-Identifier: MIT OR Apache-2.0

//! Heap-free cross-scan progressive coefficient-state validation.

use crate::error::{JpegError, ProgressiveScanStateError};
use crate::parse::scan::ParsedScan;
use crate::parse::sof::ParsedSof;

const MAX_COMPONENTS: usize = 4;
const COEFFICIENTS: usize = 64;
const UNINITIALIZED: u8 = u8::MAX;

pub(super) struct ProgressiveScriptState<'a> {
    component_ids: &'a [u8],
    approximation: [[u8; COEFFICIENTS]; MAX_COMPONENTS],
}

impl<'a> ProgressiveScriptState<'a> {
    pub(super) fn new(sof: &'a ParsedSof) -> Self {
        Self {
            component_ids: sof.component_ids.as_slice(),
            approximation: [[UNINITIALIZED; COEFFICIENTS]; MAX_COMPONENTS],
        }
    }

    pub(super) fn record_scan(
        &mut self,
        marker_offset: usize,
        scan: &ParsedScan,
    ) -> Result<(), JpegError> {
        let start = usize::from(scan.ss);
        let end = usize::from(scan.se);
        let expected = if scan.ah == 0 { UNINITIALIZED } else { scan.ah };
        let mut component_indices = [0usize; MAX_COMPONENTS];
        for (slot, scan_component) in scan.components.iter().enumerate() {
            let component_index = self.component_index(scan_component.id, marker_offset)?;
            component_indices[slot] = component_index;
            let selected = &self.approximation[component_index][start..=end];
            let difference = selected.iter().fold(0u8, |difference, &previous| {
                difference | (previous ^ expected)
            });
            if difference != 0 {
                let offset = selected
                    .iter()
                    .position(|&previous| previous != expected)
                    .ok_or(JpegError::InternalInvariant {
                        reason: "progressive scan state reduction disagrees with its range",
                    })?;
                let coefficient = scan
                    .ss
                    .checked_add(u8::try_from(offset).map_err(|_| {
                        JpegError::InternalInvariant {
                            reason: "progressive coefficient offset exceeds the JPEG range",
                        }
                    })?)
                    .ok_or(JpegError::InternalInvariant {
                        reason: "progressive coefficient exceeds the JPEG range",
                    })?;
                return validate_transition(
                    marker_offset,
                    scan_component.id,
                    coefficient,
                    selected[offset],
                    scan,
                );
            }
        }
        for &component_index in &component_indices[..scan.components.len()] {
            self.approximation[component_index][start..=end].fill(scan.al);
        }
        Ok(())
    }

    pub(super) fn finish_terminal(&self, marker_offset: usize) -> Result<(), JpegError> {
        for (component_index, &component) in self.component_ids.iter().enumerate() {
            if self.approximation[component_index][0] == UNINITIALIZED {
                return Err(script_error(
                    marker_offset,
                    0xd9,
                    component,
                    0,
                    ProgressiveScanStateError::MissingInitialDc,
                ));
            }
        }
        Ok(())
    }

    fn component_index(&self, component: u8, marker_offset: usize) -> Result<usize, JpegError> {
        self.component_ids
            .iter()
            .position(|&id| id == component)
            .ok_or(JpegError::UnknownScanComponent {
                offset: marker_offset,
                component,
            })
    }
}

fn validate_transition(
    marker_offset: usize,
    component: u8,
    coefficient: u8,
    previous: u8,
    scan: &ParsedScan,
) -> Result<(), JpegError> {
    let state = if scan.ah == 0 {
        (previous != UNINITIALIZED).then_some(ProgressiveScanStateError::DuplicateInitial {
            previous_al: previous,
            al: scan.al,
        })
    } else if previous == UNINITIALIZED {
        Some(ProgressiveScanStateError::RefinementBeforeInitial {
            ah: scan.ah,
            al: scan.al,
        })
    } else {
        (previous != scan.ah).then_some(ProgressiveScanStateError::RefinementMismatch {
            previous_al: previous,
            ah: scan.ah,
            al: scan.al,
        })
    };
    match state {
        Some(state) => Err(script_error(
            marker_offset,
            0xda,
            component,
            coefficient,
            state,
        )),
        None => Ok(()),
    }
}

fn script_error(
    offset: usize,
    marker: u8,
    component: u8,
    coefficient: u8,
    state: ProgressiveScanStateError,
) -> JpegError {
    JpegError::InvalidProgressiveScanState {
        offset,
        marker,
        component,
        coefficient,
        state,
    }
}

#[cfg(test)]
mod tests;
