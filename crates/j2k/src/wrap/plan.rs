// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free validation and exact byte planning for JP2/JPH output.

use super::{
    color::ColorSelection,
    metadata::{
        component_bpc, component_mapping_payload_len, validate_palette, ChannelDefinitionPlan,
        ResolvedComponents,
    },
};
use crate::{parse::ParsedImageInfo, J2kError, J2kFileBoxMetadata};
use j2k_core::{BufferError, Unsupported};

const BOX_HEADER_LEN: usize = 8;
const IMAGE_HEADER_PAYLOAD_LEN: usize = 14;
const FILE_TYPE_PAYLOAD_LEN: usize = 12;
const SIGNATURE_PAYLOAD_LEN: usize = 4;

pub(super) struct WrapPlan<'a> {
    pub(super) codestream: &'a [u8],
    pub(super) brand: [u8; 4],
    pub(super) parsed: &'a ParsedImageInfo,
    pub(super) metadata: J2kFileBoxMetadata<'a>,
    pub(super) colors: ColorSelection<'a>,
    pub(super) components: ResolvedComponents<'a>,
    pub(super) component_count: u16,
    pub(super) image_header_bpc: u8,
    pub(super) bpcc_payload_len: Option<usize>,
    pub(super) palette_payload_len: Option<usize>,
    pub(super) component_mapping_payload_len: Option<usize>,
    pub(super) channel_definitions: ChannelDefinitionPlan<'a>,
    pub(super) jp2_header_payload_len: usize,
    pub(super) total_len: usize,
}

impl<'a> WrapPlan<'a> {
    pub(super) fn build(
        codestream: &'a [u8],
        brand: [u8; 4],
        parsed: &'a ParsedImageInfo,
        colors: ColorSelection<'a>,
        metadata: J2kFileBoxMetadata<'a>,
    ) -> Result<Self, J2kError> {
        let components = ResolvedComponents::new(parsed, metadata)?;
        components.validate_precisions()?;
        let component_count = u16::try_from(components.len()).map_err(|_| {
            J2kError::Unsupported(Unsupported {
                what: "JP2/JPH resolved image component count exceeds u16",
            })
        })?;
        let uses_bpcc = components.uses_bpcc()?;
        let image_header_bpc = if uses_bpcc {
            0xff
        } else {
            components.component(0).map_or(0xff, component_bpc)
        };
        let bpcc_payload_len = uses_bpcc.then_some(components.len());
        let palette_payload_len = metadata.palette.map(validate_palette);
        let palette_payload_len = palette_payload_len.transpose()?;
        let component_mapping_payload_len = component_mapping_payload_len(parsed, metadata)?;

        let mut color_boxes_len = 0_usize;
        colors.for_each_resolved(parsed, |color| {
            let payload_len = color.payload_len()?;
            checked_include_box(&mut color_boxes_len, payload_len, "COLR")
        })?;

        let channel_definitions = ChannelDefinitionPlan::new(
            metadata.channel_definitions,
            colors.writes_rgba_cdef(parsed)?,
        )?;

        let mut jp2_header_payload_len = 0_usize;
        checked_include_box(
            &mut jp2_header_payload_len,
            IMAGE_HEADER_PAYLOAD_LEN,
            "IHDR",
        )?;
        if let Some(payload_len) = bpcc_payload_len {
            checked_include_box(&mut jp2_header_payload_len, payload_len, "BPCC")?;
        }
        checked_add_len(
            &mut jp2_header_payload_len,
            color_boxes_len,
            "JP2 header color boxes",
        )?;
        if let Some(payload_len) = palette_payload_len {
            checked_include_box(&mut jp2_header_payload_len, payload_len, "PCLR")?;
        }
        if let Some(payload_len) = component_mapping_payload_len {
            checked_include_box(&mut jp2_header_payload_len, payload_len, "CMAP")?;
        }
        if let Some(payload_len) = channel_definitions.payload_len()? {
            checked_include_box(&mut jp2_header_payload_len, payload_len, "CDEF")?;
        }

        let mut total_len = 0_usize;
        checked_include_box(&mut total_len, SIGNATURE_PAYLOAD_LEN, "signature")?;
        checked_include_box(&mut total_len, FILE_TYPE_PAYLOAD_LEN, "FTYP")?;
        checked_include_box(&mut total_len, jp2_header_payload_len, "JP2H")?;
        checked_include_box(&mut total_len, codestream.len(), "JP2C")?;

        Ok(Self {
            codestream,
            brand,
            parsed,
            metadata,
            colors,
            components,
            component_count,
            image_header_bpc,
            bpcc_payload_len,
            palette_payload_len,
            component_mapping_payload_len,
            channel_definitions,
            jp2_header_payload_len,
            total_len,
        })
    }
}

fn checked_include_box(
    total: &mut usize,
    payload_len: usize,
    box_name: &'static str,
) -> Result<(), J2kError> {
    let box_len = payload_len
        .checked_add(BOX_HEADER_LEN)
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
            what: "JP2/JPH box length",
        }))?;
    u32::try_from(box_len).map_err(|_| {
        J2kError::Unsupported(Unsupported {
            what: "JP2/JPH box payload exceeds 32-bit box length",
        })
    })?;
    checked_add_len(total, box_len, box_name)
}

fn checked_add_len(
    total: &mut usize,
    additional: usize,
    what: &'static str,
) -> Result<(), J2kError> {
    *total = total
        .checked_add(additional)
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow { what }))?;
    Ok(())
}
