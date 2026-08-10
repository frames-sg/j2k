// SPDX-License-Identifier: MIT OR Apache-2.0

//! JP2 header child traversal and recoverable optional-metadata parsing.

use crate::error::{bail, FormatError, Result};
use crate::reader::BitReader;

use super::{read_recoverable_metadata_box, Jp2FileKind};
use crate::jp2::allocation;
use crate::jp2::cdef;
use crate::jp2::cmap;
use crate::jp2::colr;
use crate::jp2::image_header::{parse_bits_per_component, parse_image_header};
use crate::jp2::metadata::ImageBoxes;
use crate::jp2::pclr;
use crate::jp2::r#box;

pub(super) struct ParsedJp2Header {
    pub(super) boxes: ImageBoxes,
    pub(super) used_lenient_metadata_recovery: bool,
}

#[cfg(test)]
pub(in crate::jp2) fn parse_jp2_header_box(
    data: &[u8],
    strict: bool,
    retained_baseline_bytes: usize,
) -> Result<ImageBoxes> {
    Ok(
        parse_jp2_header_box_tracked(data, strict, retained_baseline_bytes, Jp2FileKind::Jp2)?
            .boxes,
    )
}

pub(super) fn parse_jp2_header_box_tracked(
    data: &[u8],
    strict: bool,
    retained_baseline_bytes: usize,
    file_kind: Jp2FileKind,
) -> Result<ParsedJp2Header> {
    let (color_spec_count, mut used_lenient_metadata_recovery) =
        count_color_specification_boxes(data, strict)?;
    let mut budget = allocation::Jp2AllocationBudget::from_live_bytes(retained_baseline_bytes)?;
    let mut boxes = ImageBoxes {
        color_specifications: budget.try_vec(color_spec_count, "JP2 COLR metadata")?,
        ..ImageBoxes::default()
    };
    let mut saw_image_header = false;
    let mut reader = BitReader::new(data);

    while !reader.at_end() {
        let Some(child_box) = read_recoverable_metadata_box(
            &mut reader,
            strict,
            &mut used_lenient_metadata_recovery,
        )?
        else {
            break;
        };
        match child_box.box_type {
            r#box::IMAGE_HEADER => {
                if saw_image_header {
                    bail!(FormatError::InvalidBox);
                }
                boxes.image_header = Some(parse_image_header(child_box.data)?);
                saw_image_header = true;
            }
            r#box::BITS_PER_COMPONENT => {
                let parsed = parse_bits_per_component(child_box.data, &mut budget)?;
                let replaced = core::mem::replace(&mut boxes.bits_per_component, parsed);
                budget.release_vec(&replaced)?;
            }
            r#box::CHANNEL_DEFINITION => {
                let mut attempt = budget;
                match cdef::parse(&mut boxes, child_box.data, &mut attempt) {
                    Ok(()) => budget = attempt,
                    Err(crate::DecodeError::Format(_)) if !strict => {
                        used_lenient_metadata_recovery = true;
                    }
                    Err(error) => return Err(error),
                }
            }
            r#box::COLOUR_SPECIFICATION => {
                colr::parse(&mut boxes, child_box.data, &mut budget)?;
            }
            r#box::PALETTE => {
                let mut attempt = budget;
                match pclr::parse(&mut boxes, child_box.data, &mut attempt) {
                    Ok(()) => budget = attempt,
                    Err(crate::DecodeError::Format(_)) if !strict => {
                        used_lenient_metadata_recovery = true;
                    }
                    Err(error) => return Err(error),
                }
            }
            r#box::COMPONENT_MAPPING => {
                cmap::parse(&mut boxes, child_box.data, &mut budget)?;
            }
            _ => {
                ldebug!("ignoring header box 0x{:08X}", child_box.box_type);
            }
        }
    }

    if !saw_image_header {
        bail!(FormatError::MissingRequiredBox("ihdr"));
    }
    if file_kind == Jp2FileKind::Jp2 && boxes.primary_color_specification().is_none() {
        bail!(FormatError::MissingRequiredBox("colr"));
    }
    Ok(ParsedJp2Header {
        boxes,
        used_lenient_metadata_recovery,
    })
}

fn count_color_specification_boxes(data: &[u8], strict: bool) -> Result<(usize, bool)> {
    let mut count = 0_usize;
    let mut used_lenient_metadata_recovery = false;
    let mut reader = BitReader::new(data);
    while !reader.at_end() {
        let Some(child_box) = read_recoverable_metadata_box(
            &mut reader,
            strict,
            &mut used_lenient_metadata_recovery,
        )?
        else {
            break;
        };
        if child_box.box_type == r#box::COLOUR_SPECIFICATION {
            count = count
                .checked_add(1)
                .ok_or(crate::DecodeError::AllocationTooLarge {
                    what: "JP2 COLR metadata",
                    requested: usize::MAX,
                    cap: crate::DEFAULT_MAX_DECODE_BYTES,
                })?;
        }
    }
    Ok((count, used_lenient_metadata_recovery))
}
