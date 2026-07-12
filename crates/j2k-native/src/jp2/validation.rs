// SPDX-License-Identifier: MIT OR Apache-2.0

//! JP2/JPH wrapper and codestream consistency validation.

use crate::error::{bail, FormatError, Result};

use super::cmap::ComponentMappingType;
use super::container::Jp2FileKind;
use super::{ComponentDescriptor, ImageBoxes};

pub(super) fn validate_codestream_file_kind(
    file_kind: Jp2FileKind,
    header: &crate::j2c::Header<'_>,
) -> Result<()> {
    let high_throughput = header.component_infos.iter().any(|component| {
        component
            .code_block_style()
            .uses_high_throughput_block_coding()
    });
    match (file_kind, high_throughput) {
        (Jp2FileKind::Jph, false) | (Jp2FileKind::Jp2, true) => {
            bail!(FormatError::InvalidFileType);
        }
        _ => Ok(()),
    }
}

pub(super) fn validate_image_header_matches_codestream(
    boxes: &ImageBoxes,
    header: &crate::j2c::Header<'_>,
) -> Result<()> {
    let Some(image_header) = boxes.image_header else {
        bail!(FormatError::InvalidBox);
    };
    if image_header.width != header.size_data.reference_image_width()
        || image_header.height != header.size_data.reference_image_height()
    {
        bail!(FormatError::InvalidBox);
    }
    Ok(())
}

pub(super) fn validate_component_precision_metadata(
    boxes: &ImageBoxes,
    header: &crate::j2c::Header<'_>,
) -> Result<()> {
    let Some(image_header) = boxes.image_header else {
        bail!(FormatError::InvalidBox);
    };
    let resolved_count = resolved_image_component_count(boxes, header);
    if resolved_count != usize::from(image_header.components) {
        bail!(FormatError::InvalidBox);
    }

    if let Some(descriptor) = image_header.bits_per_component {
        if !boxes.bits_per_component.is_empty() {
            bail!(FormatError::InvalidBox);
        }
        for index in 0..resolved_count {
            let component = resolved_image_component_descriptor(boxes, header, index)
                .ok_or(FormatError::InvalidBox)?;
            if component != descriptor {
                bail!(FormatError::InvalidBox);
            }
        }
    } else {
        if boxes.bits_per_component.len() != usize::from(image_header.components) {
            bail!(FormatError::InvalidBox);
        }
        for (index, descriptor) in boxes.bits_per_component.iter().enumerate() {
            let component = resolved_image_component_descriptor(boxes, header, index)
                .ok_or(FormatError::InvalidBox)?;
            if component != *descriptor {
                bail!(FormatError::InvalidBox);
            }
        }
    }

    Ok(())
}

fn resolved_image_component_count(boxes: &ImageBoxes, header: &crate::j2c::Header<'_>) -> usize {
    if let Some(component_mapping) = boxes.component_mapping.as_ref() {
        return component_mapping.entries.len();
    }

    if let Some(palette) = boxes.palette.as_ref() {
        return palette.columns.len();
    }

    header.component_infos.len()
}

fn resolved_image_component_descriptor(
    boxes: &ImageBoxes,
    header: &crate::j2c::Header<'_>,
    index: usize,
) -> Option<ComponentDescriptor> {
    if let Some(component_mapping) = boxes.component_mapping.as_ref() {
        let entry = component_mapping.entries.get(index)?;
        return match entry.mapping_type {
            ComponentMappingType::Direct => {
                let component = header
                    .component_infos
                    .get(usize::from(entry.component_index))?;
                Some(component_descriptor_from_size_info(
                    component.size_info.precision,
                    component.size_info.signed,
                ))
            }
            ComponentMappingType::Palette { column } => {
                let palette = boxes.palette.as_ref()?;
                let column = palette.columns.get(usize::from(column))?;
                Some(component_descriptor_from_size_info(
                    column.bit_depth,
                    column.signed,
                ))
            }
            ComponentMappingType::Unknown { .. } => None,
        };
    }

    if let Some(palette) = boxes.palette.as_ref() {
        let column = palette.columns.get(index)?;
        return Some(component_descriptor_from_size_info(
            column.bit_depth,
            column.signed,
        ));
    }

    let component = header.component_infos.get(index)?;
    Some(component_descriptor_from_size_info(
        component.size_info.precision,
        component.size_info.signed,
    ))
}

fn component_descriptor_from_size_info(bit_depth: u8, signed: bool) -> ComponentDescriptor {
    ComponentDescriptor { bit_depth, signed }
}
