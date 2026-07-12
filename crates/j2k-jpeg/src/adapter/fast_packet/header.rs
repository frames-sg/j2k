// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size packet metadata extracted from the already parsed JPEG header.

use super::error::{FastPacketError, TableKind};
use super::types::JpegHuffmanTable;
use crate::error::JpegError;
use crate::info::{ColorSpace, SamplingFactors, SofKind};
use crate::parse::header::ParsedHeader;
use crate::parse::scan::ScanComponent;
use crate::parse::tables::RawHuffmanTable;

#[derive(Debug, Clone, Copy)]
pub(super) struct FastLayout {
    sampling: &'static [(u8, u8)],
    allow_rgb: bool,
    mcu_width: u32,
    mcu_height: u32,
}

pub(super) const FAST420_LAYOUT: FastLayout = FastLayout {
    sampling: &[(2, 2), (1, 1), (1, 1)],
    allow_rgb: false,
    mcu_width: 16,
    mcu_height: 16,
};

pub(super) const FAST422_LAYOUT: FastLayout = FastLayout {
    sampling: &[(2, 1), (1, 1), (1, 1)],
    allow_rgb: false,
    mcu_width: 16,
    mcu_height: 8,
};

pub(super) const FAST444_LAYOUT: FastLayout = FastLayout {
    sampling: &[(1, 1), (1, 1), (1, 1)],
    allow_rgb: true,
    mcu_width: 8,
    mcu_height: 8,
};

#[derive(Debug)]
pub(super) struct ColorFastHeader {
    pub(super) dimensions: (u32, u32),
    pub(super) mcus_per_row: u32,
    pub(super) mcu_rows: u32,
    pub(super) total_mcus: u32,
    pub(super) restart_interval: Option<u16>,
    pub(super) entropy_offset: usize,
    pub(super) y_quant: [u16; 64],
    pub(super) cb_quant: [u16; 64],
    pub(super) cr_quant: [u16; 64],
    pub(super) y_dc_table: JpegHuffmanTable,
    pub(super) y_ac_table: JpegHuffmanTable,
    pub(super) cb_dc_table: JpegHuffmanTable,
    pub(super) cb_ac_table: JpegHuffmanTable,
    pub(super) cr_dc_table: JpegHuffmanTable,
    pub(super) cr_ac_table: JpegHuffmanTable,
}

impl ColorFastHeader {
    pub(super) fn inspect(
        header: &ParsedHeader,
        layout: FastLayout,
    ) -> Result<Self, FastPacketError> {
        validate_sequential_eight_bit(header)?;
        let color_space = header.color_space();
        if color_space != ColorSpace::YCbCr && !(layout.allow_rgb && color_space == ColorSpace::Rgb)
        {
            return Err(FastPacketError::UnsupportedColorSpace(color_space));
        }
        if header.sampling != SamplingFactors::from_validated_components(layout.sampling) {
            return Err(FastPacketError::UnsupportedSampling);
        }
        let scan = header.scan.as_ref().ok_or(FastPacketError::MissingScan)?;
        let [y_scan, cb_scan, cr_scan] =
            ordered_scan_triplet(&header.component_ids, &scan.components)?;
        let (width, height) = header.dimensions;
        let mcus_per_row = width.div_ceil(layout.mcu_width);
        let mcu_rows = height.div_ceil(layout.mcu_height);
        let total_mcus = mcus_per_row
            .checked_mul(mcu_rows)
            .ok_or(FastPacketError::Decode(JpegError::DimensionOverflow {
                width,
                height,
            }))?;

        Ok(Self {
            dimensions: header.dimensions,
            mcus_per_row,
            mcu_rows,
            total_mcus,
            restart_interval: header.restart_interval,
            entropy_offset: header.sos_offset.ok_or(FastPacketError::MissingScan)?,
            y_quant: quant_for_component(header, 0)?,
            cb_quant: quant_for_component(header, 1)?,
            cr_quant: quant_for_component(header, 2)?,
            y_dc_table: huffman_table(&header.huffman_tables.dc, TableKind::Dc, y_scan.dc_table)?,
            y_ac_table: huffman_table(&header.huffman_tables.ac, TableKind::Ac, y_scan.ac_table)?,
            cb_dc_table: huffman_table(&header.huffman_tables.dc, TableKind::Dc, cb_scan.dc_table)?,
            cb_ac_table: huffman_table(&header.huffman_tables.ac, TableKind::Ac, cb_scan.ac_table)?,
            cr_dc_table: huffman_table(&header.huffman_tables.dc, TableKind::Dc, cr_scan.dc_table)?,
            cr_ac_table: huffman_table(&header.huffman_tables.ac, TableKind::Ac, cr_scan.ac_table)?,
        })
    }
}

#[derive(Debug)]
pub(super) struct GrayFastHeader {
    pub(super) dimensions: (u32, u32),
    pub(super) restart_interval: Option<u16>,
    pub(super) entropy_offset: usize,
    pub(super) y_quant: [u16; 64],
    pub(super) y_dc_table: JpegHuffmanTable,
    pub(super) y_ac_table: JpegHuffmanTable,
}

impl GrayFastHeader {
    pub(super) fn inspect(header: &ParsedHeader) -> Result<Self, FastPacketError> {
        validate_sequential_eight_bit(header)?;
        if header.color_space() != ColorSpace::Grayscale {
            return Err(FastPacketError::UnsupportedColorSpace(header.color_space()));
        }
        if header.sampling != SamplingFactors::from_validated_components(&[(1, 1)]) {
            return Err(FastPacketError::UnsupportedSampling);
        }
        let scan = header.scan.as_ref().ok_or(FastPacketError::MissingScan)?;
        if header.component_ids.len() != 1
            || scan.components.len() != 1
            || scan.components[0].id != header.component_ids[0]
        {
            return Err(FastPacketError::UnsupportedComponentOrder);
        }
        let scan_component = scan.components[0];

        Ok(Self {
            dimensions: header.dimensions,
            restart_interval: header.restart_interval,
            entropy_offset: header.sos_offset.ok_or(FastPacketError::MissingScan)?,
            y_quant: quant_for_component(header, 0)?,
            y_dc_table: huffman_table(
                &header.huffman_tables.dc,
                TableKind::Dc,
                scan_component.dc_table,
            )?,
            y_ac_table: huffman_table(
                &header.huffman_tables.ac,
                TableKind::Ac,
                scan_component.ac_table,
            )?,
        })
    }
}

fn validate_sequential_eight_bit(header: &ParsedHeader) -> Result<(), FastPacketError> {
    if !matches!(header.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return Err(FastPacketError::UnsupportedSof(header.sof_kind));
    }
    if header.bit_depth != 8 {
        return Err(FastPacketError::Decode(JpegError::UnsupportedBitDepth {
            depth: header.bit_depth,
        }));
    }
    Ok(())
}

fn quant_for_component(
    header: &ParsedHeader,
    component_idx: usize,
) -> Result<[u16; 64], FastPacketError> {
    let slot = *header
        .quant_table_ids
        .get(component_idx)
        .ok_or(FastPacketError::UnsupportedComponentOrder)?;
    header
        .quant_tables
        .entries
        .get(usize::from(slot))
        .copied()
        .flatten()
        .ok_or(FastPacketError::MissingQuantTable { slot })
}

fn ordered_scan_triplet(
    component_ids: &[u8],
    scan_components: &[ScanComponent],
) -> Result<[ScanComponent; 3], FastPacketError> {
    if component_ids.len() != 3 || scan_components.len() != 3 {
        return Err(FastPacketError::UnsupportedComponentOrder);
    }
    let mut ordered = [None; 3];
    for (index, &component_id) in component_ids.iter().enumerate() {
        ordered[index] = scan_components
            .iter()
            .copied()
            .find(|component| component.id == component_id);
    }
    match ordered {
        [Some(first), Some(second), Some(third)] => Ok([first, second, third]),
        _ => Err(FastPacketError::UnsupportedComponentOrder),
    }
}

fn huffman_table(
    tables: &[Option<RawHuffmanTable>; 4],
    kind: TableKind,
    slot: u8,
) -> Result<JpegHuffmanTable, FastPacketError> {
    let raw = tables
        .get(usize::from(slot))
        .and_then(Option::as_ref)
        .ok_or(FastPacketError::MissingHuffmanTable { kind, slot })?;
    Ok(JpegHuffmanTable::from_raw(raw))
}
