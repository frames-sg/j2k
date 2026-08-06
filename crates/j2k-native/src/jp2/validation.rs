// SPDX-License-Identifier: MIT OR Apache-2.0

//! JP2/JPH wrapper and codestream consistency validation.

use crate::error::{bail, FormatError, Result};

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
    let codestream_count = header.component_infos.len();
    if codestream_count != usize::from(image_header.components) {
        bail!(FormatError::InvalidBox);
    }

    if let Some(descriptor) = image_header.bits_per_component {
        if !boxes.bits_per_component.is_empty() {
            bail!(FormatError::InvalidBox);
        }
        for component in &header.component_infos {
            let component = component_descriptor_from_size_info(
                component.size_info.precision,
                component.size_info.signed,
            );
            if component != descriptor {
                bail!(FormatError::InvalidBox);
            }
        }
    } else {
        if boxes.bits_per_component.len() != usize::from(image_header.components) {
            bail!(FormatError::InvalidBox);
        }
        for (component, descriptor) in header.component_infos.iter().zip(&boxes.bits_per_component)
        {
            let component = component_descriptor_from_size_info(
                component.size_info.precision,
                component.size_info.signed,
            );
            if component != *descriptor {
                bail!(FormatError::InvalidBox);
            }
        }
    }

    Ok(())
}

fn component_descriptor_from_size_info(bit_depth: u8, signed: bool) -> ComponentDescriptor {
    ComponentDescriptor { bit_depth, signed }
}
