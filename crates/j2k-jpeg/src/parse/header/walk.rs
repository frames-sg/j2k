// SPDX-License-Identifier: MIT OR Apache-2.0

//! Header marker walkers for lightweight inspection and decode preparation.

use alloc::vec::Vec;

use crate::error::{JpegError, MarkerKind, Warning};
use crate::info::SofKind;
use crate::parse::adobe_app14::{parse_adobe_app14, AdobeTransform};
use crate::parse::allocation::ParsedMetadataBudget;
use crate::parse::markers::{Marker, MarkerWalker};
use crate::parse::scan::{parse_scan_header, ParsedScan};
use crate::parse::sof::{parse_sof, ParsedSof};
use crate::parse::tables::{parse_dht, parse_dqt, HuffmanTables, QuantTables};

use super::markers::count_scan_markers;
use super::progressive::{collect_progressive_scans, ProgressiveScanCollection};
use super::types::{ParsedHeader, ParsedProgressiveScan};
use super::validation::{
    normalize_restart_interval, validate_progressive_scan_components, validate_scan_parameters,
    validate_sequential_scan_components,
};

pub(crate) fn parse_header(bytes: &[u8]) -> Result<ParsedHeader, JpegError> {
    parse_header_with_external_live(bytes, 0)
}

pub(crate) fn parse_header_with_external_live(
    bytes: &[u8],
    external_live_bytes: usize,
) -> Result<ParsedHeader, JpegError> {
    HeaderParser::new(bytes, external_live_bytes)?.parse()
}

struct HeaderParser<'a> {
    bytes: &'a [u8],
    walker: MarkerWalker<'a>,
    sof: Option<ParsedSof>,
    quant_tables: QuantTables,
    huffman_tables: HuffmanTables,
    restart_interval: Option<u16>,
    adobe: Option<AdobeTransform>,
    warnings: Vec<Warning>,
    scan_count: u16,
    sos_offset: Option<usize>,
    scan: Option<ParsedScan>,
    progressive_scans: Vec<ParsedProgressiveScan>,
    allocation_budget: ParsedMetadataBudget,
}

impl<'a> HeaderParser<'a> {
    fn new(bytes: &'a [u8], external_live_bytes: usize) -> Result<Self, JpegError> {
        Ok(Self {
            bytes,
            walker: MarkerWalker::new(bytes),
            sof: None,
            quant_tables: QuantTables::default(),
            huffman_tables: HuffmanTables::default(),
            restart_interval: None,
            adobe: None,
            warnings: Vec::new(),
            scan_count: 0,
            sos_offset: None,
            scan: None,
            progressive_scans: Vec::new(),
            allocation_budget: ParsedMetadataBudget::with_external_live(external_live_bytes)?,
        })
    }

    fn parse(mut self) -> Result<ParsedHeader, JpegError> {
        self.walker.read_soi()?;
        while let Some(marker) = self.walker.next_marker()? {
            if self.handle_marker(marker)? {
                break;
            }
        }
        self.finish()
    }

    fn handle_marker(&mut self, marker: Marker<'a>) -> Result<bool, JpegError> {
        match marker.code {
            0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => {
                self.handle_sof(marker)?;
                Ok(false)
            }
            0xdb => {
                parse_dqt(
                    marker.payload,
                    marker.offset + 4,
                    &mut self.quant_tables,
                    &mut self.allocation_budget,
                )?;
                Ok(false)
            }
            0xc4 => {
                parse_dht(
                    marker.payload,
                    marker.offset + 4,
                    &mut self.huffman_tables,
                    &mut self.allocation_budget,
                )?;
                Ok(false)
            }
            0xdd => {
                self.restart_interval = parse_restart_interval(marker)?;
                Ok(false)
            }
            0xda => {
                self.handle_scan(marker)?;
                Ok(true)
            }
            0xee => {
                self.handle_adobe(marker)?;
                Ok(false)
            }
            0xe0 | 0xfe => Ok(false),
            0xe2 => {
                self.push_warning(Warning::IccProfileIgnored {
                    size: marker.payload.len(),
                })?;
                Ok(false)
            }
            0xe1..=0xef => {
                self.push_warning(Warning::UnknownAppMarker {
                    marker: marker.code,
                    size: marker.payload.len(),
                })?;
                Ok(false)
            }
            _ => Err(JpegError::InvalidMarker {
                offset: marker.offset,
                marker: marker.code,
            }),
        }
    }

    fn handle_sof(&mut self, marker: Marker<'a>) -> Result<(), JpegError> {
        if self.sof.is_some() {
            return Err(JpegError::DuplicateMarker {
                offset: marker.offset,
                marker: MarkerKind::Sof,
            });
        }
        self.sof = Some(parse_sof(marker.code, marker.payload, marker.offset + 4)?);
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the retained progressive scan count is saturated to the public u16 field"
    )]
    fn handle_scan(&mut self, marker: Marker<'a>) -> Result<(), JpegError> {
        let scan = parse_scan_header(marker.payload, marker.offset + 4)?;
        if let Some(sof) = self.sof.as_ref() {
            validate_scan_parameters(sof.sof_kind, &scan, marker.offset + 4)?;
            validate_sequential_scan_components(sof, &scan, marker.offset + 4)?;
            validate_progressive_scan_components(sof, &scan, marker.offset + 4)?;
        }
        let entropy_offset = self.walker.position();
        self.sos_offset = Some(entropy_offset);
        self.scan = Some(scan.clone());
        if matches!(
            self.sof.as_ref().map(|sof| sof.sof_kind),
            Some(SofKind::Progressive8 | SofKind::Progressive12)
        ) {
            let sof = self.sof.as_ref().ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sof,
            })?;
            self.progressive_scans = collect_progressive_scans(ProgressiveScanCollection {
                bytes: self.bytes,
                first_scan: scan,
                first_marker_offset: marker.offset,
                first_entropy_offset: entropy_offset,
                huffman_tables: &mut self.huffman_tables,
                quant_tables: &mut self.quant_tables,
                restart_interval: &mut self.restart_interval,
                sof,
                warnings: &mut self.warnings,
                allocation_budget: &mut self.allocation_budget,
            })?;
            self.scan_count = self.progressive_scans.len().min(usize::from(u16::MAX)) as u16;
        } else {
            self.scan_count = count_scan_markers(self.bytes, entropy_offset);
        }
        Ok(())
    }

    fn handle_adobe(&mut self, marker: Marker<'a>) -> Result<(), JpegError> {
        if let Some(transform) = parse_adobe_app14(marker.payload) {
            self.adobe = Some(transform);
            if transform == AdobeTransform::Unknown
                && marker.payload.len() >= 12
                && marker.payload[11] > 2
            {
                self.push_warning(Warning::AdobeApp14Ambiguous {
                    raw_transform: marker.payload[11],
                })?;
            }
        } else {
            self.push_warning(Warning::UnknownAppMarker {
                marker: 0xee,
                size: marker.payload.len(),
            })?;
        }
        Ok(())
    }

    fn push_warning(&mut self, warning: Warning) -> Result<(), JpegError> {
        self.allocation_budget.try_push(&mut self.warnings, warning)
    }

    fn finish(mut self) -> Result<ParsedHeader, JpegError> {
        let sof = self.sof.take().ok_or(JpegError::MissingMarker {
            marker: MarkerKind::Sof,
        })?;
        let header = ParsedHeader {
            sof_kind: sof.sof_kind,
            bit_depth: sof.bit_depth,
            dimensions: (u32::from(sof.width), u32::from(sof.height)),
            sampling: sof.sampling,
            component_ids: sof.component_ids,
            quant_table_ids: sof.quant_table_ids,
            quant_tables: self.quant_tables,
            huffman_tables: self.huffman_tables,
            restart_interval: self.restart_interval,
            adobe: self.adobe,
            scan_count: self.scan_count,
            warnings: self.warnings,
            sos_offset: self.sos_offset,
            scan: self.scan,
            progressive_scans: self.progressive_scans,
        };
        self.allocation_budget
            .finish(header.retained_allocation_bytes()?)?;
        Ok(header)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "DRI payload lengths are bounded by the JPEG 16-bit marker grammar"
)]
fn parse_restart_interval(marker: Marker<'_>) -> Result<Option<u16>, JpegError> {
    if marker.payload.len() != 2 {
        return Err(JpegError::InvalidSegmentLength {
            offset: marker.offset,
            marker: 0xdd,
            length: (marker.payload.len() + 2) as u16,
        });
    }
    Ok(normalize_restart_interval(u16::from_be_bytes([
        marker.payload[0],
        marker.payload[1],
    ])))
}
