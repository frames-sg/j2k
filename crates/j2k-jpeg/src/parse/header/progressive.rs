// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive post-SOS marker walk and compact scan-script capture.

mod application;
mod pending;
mod script;

#[cfg(test)]
mod eof_tests;
#[cfg(test)]
mod terminal_tests;

use alloc::vec::Vec;

use self::application::application_warning;
use self::pending::PendingProgressiveScan;
use self::script::ProgressiveScriptState;

use super::markers::{marker_payload, segment_length};
use super::types::ParsedProgressiveScan;
use super::validation::{
    normalize_restart_interval, validate_progressive_scan_components, validate_scan_parameters,
};
use crate::error::{JpegError, MarkerKind, Warning};
use crate::parse::allocation::ParsedMetadataBudget;
use crate::parse::markers::next_marker_after_entropy;
use crate::parse::scan::{parse_scan_header, ParsedScan};
use crate::parse::sof::ParsedSof;
use crate::parse::tables::{
    parse_dht, parse_dqt, HuffmanTables, ProgressiveTableState, QuantTables,
};

pub(super) struct ProgressiveScanCollection<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) first_scan: ParsedScan,
    pub(super) first_marker_offset: usize,
    pub(super) first_entropy_offset: usize,
    pub(super) huffman_tables: &'a mut HuffmanTables,
    pub(super) quant_tables: &'a mut QuantTables,
    pub(super) restart_interval: &'a mut Option<u16>,
    pub(super) sof: &'a ParsedSof,
    pub(super) warnings: &'a mut Vec<Warning>,
    pub(super) allocation_budget: &'a mut ParsedMetadataBudget,
}

pub(super) fn collect_progressive_scans(
    request: ProgressiveScanCollection<'_>,
) -> Result<Vec<ParsedProgressiveScan>, JpegError> {
    ProgressiveCollector::new(request)?.collect()
}

struct ProgressiveCollector<'a> {
    bytes: &'a [u8],
    huffman_tables: &'a mut HuffmanTables,
    quant_tables: &'a mut QuantTables,
    restart_interval: &'a mut Option<u16>,
    sof: &'a ParsedSof,
    warnings: &'a mut Vec<Warning>,
    allocation_budget: &'a mut ParsedMetadataBudget,
    scans: Vec<ParsedProgressiveScan>,
    pending: Option<PendingProgressiveScan>,
    position: usize,
    script: ProgressiveScriptState<'a>,
}

impl<'a> ProgressiveCollector<'a> {
    fn new(request: ProgressiveScanCollection<'a>) -> Result<Self, JpegError> {
        let mut script = ProgressiveScriptState::new(request.sof);
        script.record_scan(request.first_marker_offset, &request.first_scan)?;
        let pending = PendingProgressiveScan::new(
            request.first_scan,
            request.first_entropy_offset,
            ProgressiveTableState::capture(request.huffman_tables, request.quant_tables),
            *request.restart_interval,
        );
        Ok(Self {
            bytes: request.bytes,
            huffman_tables: request.huffman_tables,
            quant_tables: request.quant_tables,
            restart_interval: request.restart_interval,
            sof: request.sof,
            warnings: request.warnings,
            allocation_budget: request.allocation_budget,
            scans: Vec::new(),
            pending: Some(pending),
            position: request.first_entropy_offset,
            script,
        })
    }

    fn collect(mut self) -> Result<Vec<ParsedProgressiveScan>, JpegError> {
        while let Some((marker_offset, code)) =
            next_marker_after_entropy(self.bytes, self.position)?
        {
            self.finish_pending(marker_offset, code)?;
            if self.handle_marker(marker_offset, code)? == MarkerFlow::Finish {
                return Ok(self.scans);
            }
        }
        let eof = self.bytes.len();
        self.finish_pending(eof, 0)?;
        self.script.finish_terminal(eof)?;
        self.allocation_budget
            .try_push(self.warnings, Warning::MissingEoi)?;
        Ok(self.scans)
    }

    fn finish_pending(
        &mut self,
        terminal_offset: usize,
        terminal_code: u8,
    ) -> Result<(), JpegError> {
        if let Some(scan) = self.pending.take() {
            self.allocation_budget
                .try_push(&mut self.scans, scan.finish(terminal_offset, terminal_code))?;
        }
        Ok(())
    }

    fn handle_marker(&mut self, marker_offset: usize, code: u8) -> Result<MarkerFlow, JpegError> {
        match code {
            0xd9 => {
                self.script.finish_terminal(marker_offset)?;
                Ok(MarkerFlow::Finish)
            }
            0xdb => {
                let (payload, next) = marker_payload(self.bytes, marker_offset, code)?;
                parse_dqt(
                    payload,
                    marker_offset + 4,
                    self.quant_tables,
                    self.allocation_budget,
                )?;
                self.position = next;
                Ok(MarkerFlow::Continue)
            }
            0xc4 => {
                let (payload, next) = marker_payload(self.bytes, marker_offset, code)?;
                parse_dht(
                    payload,
                    marker_offset + 4,
                    self.huffman_tables,
                    self.allocation_budget,
                )?;
                self.position = next;
                Ok(MarkerFlow::Continue)
            }
            0xdd => self.handle_restart_interval(marker_offset, code),
            0xda => self.handle_scan(marker_offset, code),
            0xe0..=0xef | 0xfe => self.handle_app_or_comment(marker_offset, code),
            0x01 => {
                self.position = marker_offset + 2;
                Ok(MarkerFlow::Continue)
            }
            0xd8 => Err(JpegError::DuplicateMarker {
                offset: marker_offset,
                marker: MarkerKind::Soi,
            }),
            _ => Err(JpegError::InvalidMarker {
                offset: marker_offset,
                marker: code,
            }),
        }
    }

    fn handle_restart_interval(
        &mut self,
        marker_offset: usize,
        code: u8,
    ) -> Result<MarkerFlow, JpegError> {
        let (payload, next) = marker_payload(self.bytes, marker_offset, code)?;
        if payload.len() != 2 {
            return Err(JpegError::InvalidSegmentLength {
                offset: marker_offset,
                marker: 0xdd,
                length: segment_length(payload),
            });
        }
        *self.restart_interval =
            normalize_restart_interval(u16::from_be_bytes([payload[0], payload[1]]));
        self.position = next;
        Ok(MarkerFlow::Continue)
    }

    fn handle_scan(&mut self, marker_offset: usize, code: u8) -> Result<MarkerFlow, JpegError> {
        let (payload, entropy_offset) = marker_payload(self.bytes, marker_offset, code)?;
        let scan = parse_scan_header(payload, marker_offset + 4)?;
        validate_scan_parameters(self.sof.sof_kind, &scan, marker_offset + 4)?;
        validate_progressive_scan_components(self.sof, &scan, marker_offset + 4)?;
        self.script.record_scan(marker_offset, &scan)?;
        self.pending = Some(PendingProgressiveScan::new(
            scan,
            entropy_offset,
            ProgressiveTableState::capture(self.huffman_tables, self.quant_tables),
            *self.restart_interval,
        ));
        self.position = entropy_offset;
        Ok(MarkerFlow::Continue)
    }

    fn handle_app_or_comment(
        &mut self,
        marker_offset: usize,
        code: u8,
    ) -> Result<MarkerFlow, JpegError> {
        let (warning, next) = application_warning(self.bytes, marker_offset, code)?;
        if let Some(warning) = warning {
            self.allocation_budget.try_push(self.warnings, warning)?;
        }
        self.position = next;
        Ok(MarkerFlow::Continue)
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum MarkerFlow {
    Continue,
    Finish,
}
