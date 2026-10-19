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
    /// Whether a JFIF APP0 header preceded the first scan.
    pub(crate) jfif: bool,
    pub(crate) scan_count: u16,
    pub(crate) warnings: Vec<Warning>,
    pub(crate) sos_offset: Option<usize>,
    pub(crate) scan: Option<ParsedScan>,
    pub(crate) progressive_scans: Vec<ParsedProgressiveScan>,
}

impl ParsedHeader {
    pub(crate) fn color_space(&self) -> ColorSpace {
        color_space_for_frame(
            self.component_ids.as_slice(),
            self.sof_kind == SofKind::Lossless,
            self.adobe,
            self.jfif,
        )
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

/// Input color space as libjpeg-turbo's `default_decompress_parms` chooses
/// it: a JFIF marker means YCbCr; otherwise the Adobe transform decides;
/// otherwise three components are RGB when their IDs spell "RGB" or the
/// frame is lossless (libjpeg-turbo's lossless guess), else YCbCr.
pub(super) fn color_space_for_frame(
    component_ids: &[u8],
    lossless: bool,
    adobe: Option<AdobeTransform>,
    jfif: bool,
) -> ColorSpace {
    match component_ids.len() {
        1 => ColorSpace::Grayscale,
        3 if jfif => ColorSpace::YCbCr,
        3 => match adobe {
            Some(AdobeTransform::Unknown) => ColorSpace::Rgb,
            Some(AdobeTransform::YCbCr | AdobeTransform::Ycck | AdobeTransform::Reserved) => {
                ColorSpace::YCbCr
            }
            None if component_ids == b"RGB" || lossless => ColorSpace::Rgb,
            None => ColorSpace::YCbCr,
        },
        4 => match adobe {
            None | Some(AdobeTransform::Unknown) => ColorSpace::Cmyk,
            Some(AdobeTransform::YCbCr | AdobeTransform::Ycck | AdobeTransform::Reserved) => {
                ColorSpace::Ycck
            }
        },
        _ => ColorSpace::YCbCr,
    }
}
