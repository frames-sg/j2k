// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-only parsed JPEG header owners.

use alloc::vec::Vec;

use crate::allocation::checked_add_allocation_bytes;
use crate::error::{JpegError, Warning};
use crate::info::{ColorSpace, Info, McuGeometry, SamplingFactors, SofKind};
use crate::parse::adobe_app14::AdobeTransform;
use crate::parse::allocation::capacity_bytes;
use crate::parse::scan::ParsedScan;
use crate::parse::sof::FrameComponentValues;
use crate::parse::tables::{HuffmanTables, ProgressiveTableState, QuantTables};

#[derive(Debug)]
pub(crate) struct ParsedProgressiveScan {
    pub(crate) scan: ParsedScan,
    pub(crate) entropy_offset: usize,
    /// Absolute byte offset and marker code ending this entropy segment.
    /// Code zero denotes physical EOF without EOI.
    pub(crate) terminal_offset: usize,
    pub(crate) terminal_code: u8,
    pub(crate) table_state: ProgressiveTableState,
    pub(crate) restart_interval: Option<u16>,
}

#[derive(Debug)]
pub(crate) struct ParsedHeader {
    pub(crate) sof_kind: SofKind,
    pub(crate) bit_depth: u8,
    pub(crate) dimensions: (u32, u32),
    pub(crate) sampling: SamplingFactors,
    pub(crate) component_ids: FrameComponentValues,
    pub(crate) quant_table_ids: FrameComponentValues,
    pub(crate) quant_tables: QuantTables,
    pub(crate) huffman_tables: HuffmanTables,
    pub(crate) restart_interval: Option<u16>,
    pub(crate) adobe: Option<AdobeTransform>,
    pub(crate) scan_count: u16,
    pub(crate) warnings: Vec<Warning>,
    pub(crate) sos_offset: Option<usize>,
    pub(crate) scan: Option<ParsedScan>,
    pub(crate) progressive_scans: Vec<ParsedProgressiveScan>,
}

impl ParsedHeader {
    pub(crate) fn color_space(&self) -> ColorSpace {
        color_space_for_components(self.sampling.len(), self.adobe)
    }

    pub(crate) fn info(&self) -> Info {
        Info {
            dimensions: self.dimensions,
            color_space: self.color_space(),
            sampling: self.sampling,
            sof_kind: self.sof_kind,
            bit_depth: self.bit_depth,
            restart_interval: self.restart_interval,
            mcu_geometry: McuGeometry::from_sampling(self.dimensions, self.sampling),
            scan_count: self.scan_count,
        }
    }

    pub(crate) fn retained_allocation_bytes(&self) -> Result<usize, JpegError> {
        let warning_bytes = capacity_bytes::<Warning>(self.warnings.capacity())?;
        let scan_bytes =
            capacity_bytes::<ParsedProgressiveScan>(self.progressive_scans.capacity())?;
        let mut total = checked_add_allocation_bytes(warning_bytes, scan_bytes)?;
        total =
            checked_add_allocation_bytes(total, self.huffman_tables.retained_allocation_bytes()?)?;
        checked_add_allocation_bytes(total, self.quant_tables.retained_allocation_bytes()?)
    }
}

pub(super) fn color_space_for_components(
    component_count: usize,
    adobe: Option<AdobeTransform>,
) -> ColorSpace {
    match (component_count, adobe) {
        (1, _) => ColorSpace::Grayscale,
        (3, Some(AdobeTransform::Unknown)) => ColorSpace::Rgb,
        (4, Some(AdobeTransform::Ycck)) => ColorSpace::Ycck,
        (4, _) => ColorSpace::Cmyk,
        _ => ColorSpace::YCbCr,
    }
}
