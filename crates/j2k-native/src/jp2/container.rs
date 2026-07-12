// SPDX-License-Identifier: MIT OR Apache-2.0

//! JP2/JPH container traversal and native decode parse orchestration.

use crate::error::{bail, FormatError, Result};
use crate::reader::BitReader;
use crate::{resolve_alpha_and_color_space, DecodeSettings, Image};

use super::allocation;
use super::cdef;
use super::cmap::{self, ComponentMappingBox, ComponentMappingEntry, ComponentMappingType};
use super::colr;
use super::image_header::{parse_bits_per_component, parse_image_header};
use super::metadata::{
    public_image_header, public_metadata_from_boxes, ImageBoxes, Jp2FileMetadata,
    Jp2ImageHeaderMetadata,
};
use super::pclr;
use super::r#box::{self, FILE_TYPE, JP2_SIGNATURE};
use super::validation::{
    validate_codestream_file_kind, validate_component_precision_metadata,
    validate_image_header_matches_codestream,
};

/// Parsed still-image file kind from the JP2/JPH file type box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jp2FileKind {
    /// JP2 still-image file.
    Jp2,
    /// JPH still-image file wrapping HTJ2K codestream data.
    Jph,
}

/// Native-owned JP2/JPH container parse summary.
#[derive(Debug)]
pub struct Jp2Container<'a> {
    /// Parsed still-image file kind.
    pub file_kind: Jp2FileKind,
    /// Byte offset of the codestream payload within the file.
    pub codestream_offset: usize,
    /// Contiguous codestream payload.
    pub codestream: &'a [u8],
    /// Parsed JP2 image header box.
    pub image_header: Jp2ImageHeaderMetadata,
    /// Parsed JP2 file metadata boxes.
    pub metadata: Jp2FileMetadata,
}

struct ParsedJp2Container<'a> {
    file_kind: Jp2FileKind,
    codestream_offset: usize,
    codestream: &'a [u8],
    boxes: ImageBoxes,
}
const JP2_SIGNATURE_PAYLOAD: [u8; 4] = [0x0D, 0x0A, 0x87, 0x0A];

/// Parse JP2/JPH container boxes without decoding the codestream.
///
/// # Errors
///
/// Returns an error when the wrapper is malformed or its metadata is inconsistent.
pub fn inspect_jp2_container(data: &[u8]) -> Result<Jp2Container<'_>> {
    let parsed = parse_jp2_container_with_strict(data, true)?;
    let image_header = parsed
        .boxes
        .image_header
        .ok_or(FormatError::MissingRequiredBox("ihdr"))?;
    let metadata = public_metadata_from_boxes(parsed.boxes)?;
    Ok(Jp2Container {
        file_kind: parsed.file_kind,
        codestream_offset: parsed.codestream_offset,
        codestream: parsed.codestream,
        image_header: public_image_header(image_header),
        metadata,
    })
}

/// Extract the contiguous codestream payload from a JP2/JPH wrapper.
///
/// # Errors
///
/// Returns an error when required JP2/JPH boxes are missing or malformed.
pub fn extract_jp2_codestream_payload(data: &[u8]) -> Result<(Jp2FileKind, usize, &[u8])> {
    if data.len() < 12 {
        bail!(FormatError::TooShort {
            need: 12,
            have: data.len(),
        });
    }

    let mut reader = BitReader::new(data);
    let signature_box = r#box::read_checked(&mut reader)?;
    if signature_box.box_type != JP2_SIGNATURE || signature_box.data != JP2_SIGNATURE_PAYLOAD {
        bail!(FormatError::InvalidSignature);
    }

    let file_type_box = r#box::read_checked(&mut reader)?;
    if file_type_box.box_type != FILE_TYPE {
        bail!(FormatError::InvalidFileType);
    }
    let file_kind = classify_file_type(file_type_box.data)?;

    while !reader.at_end() {
        let current_box = r#box::read_checked(&mut reader)?;
        if current_box.box_type == r#box::CONTIGUOUS_CODESTREAM {
            let codestream_offset = current_box.data.as_ptr() as usize - data.as_ptr() as usize;
            return Ok((file_kind, codestream_offset, current_box.data));
        }
    }

    bail!(FormatError::MissingCodestream);
}

pub(crate) fn parse(data: &[u8], settings: DecodeSettings) -> Result<Image<'_>> {
    parse_with_retained_baseline(data, settings, 0)
}

pub(crate) fn parse_with_retained_baseline(
    data: &[u8],
    mut settings: DecodeSettings,
    retained_baseline_bytes: usize,
) -> Result<Image<'_>> {
    let container = parse_jp2_container_with_strict_and_retained_baseline(
        data,
        settings.strict,
        retained_baseline_bytes,
    )?;
    if container.boxes.palette.is_some() {
        settings.target_resolution = None;
    }
    let mut image_boxes = container.boxes;
    let mut retained_box_bytes = image_boxes.allocated_bytes()?;
    allocation::checked_add_bytes(
        &mut retained_box_bytes,
        retained_baseline_bytes,
        "retained JP2 parse owners",
    )?;
    let parsed_codestream = crate::j2c::parse_raw_with_retained_baseline(
        container.codestream,
        &settings,
        retained_box_bytes,
    )?;
    validate_codestream_file_kind(container.file_kind, &parsed_codestream.header)?;
    validate_image_header_matches_codestream(&image_boxes, &parsed_codestream.header)?;
    validate_component_precision_metadata(&image_boxes, &parsed_codestream.header)?;

    let implicit_mapping_count = image_boxes
        .palette
        .as_ref()
        .filter(|_| image_boxes.component_mapping.is_none())
        .map_or(0, |palette| palette.columns.len());
    if implicit_mapping_count != 0 {
        // In theory, CMAP is required when PCLR is present. Some files omit it,
        // so map every palette column from codestream component zero.
        let retained_container_bytes = crate::image::retained_container_metadata_bytes(
            &parsed_codestream.header,
            &image_boxes,
        )?;
        let mut budget =
            implicit_mapping_budget(retained_container_bytes, retained_baseline_bytes)?;
        let mut mappings =
            budget.try_vec(implicit_mapping_count, "implicit JP2 component mappings")?;
        for index in 0..implicit_mapping_count {
            let column = u8::try_from(index).map_err(|_| FormatError::InvalidBox)?;
            mappings.push(ComponentMappingEntry {
                component_index: 0,
                mapping_type: ComponentMappingType::Palette { column },
            });
        }
        image_boxes.component_mapping = Some(ComponentMappingBox { entries: mappings });
    }

    let (color_space, has_alpha) = resolve_alpha_and_color_space(
        &image_boxes,
        &parsed_codestream.header,
        &settings,
        retained_baseline_bytes,
    )?;
    if retained_baseline_bytes == 0 {
        Image::from_parsed_parts(
            parsed_codestream.data,
            parsed_codestream.header,
            image_boxes,
            settings,
            color_space,
            has_alpha,
        )
    } else {
        Image::from_parsed_parts_with_retained_baseline(
            parsed_codestream.data,
            parsed_codestream.header,
            image_boxes,
            settings,
            color_space,
            has_alpha,
            retained_baseline_bytes,
        )
    }
}

pub(super) fn implicit_mapping_budget(
    retained_container_bytes: usize,
    retained_baseline_bytes: usize,
) -> Result<allocation::Jp2AllocationBudget> {
    let mut live_bytes = retained_container_bytes;
    allocation::checked_add_bytes(
        &mut live_bytes,
        retained_baseline_bytes,
        "retained JP2 parse owners",
    )?;
    allocation::Jp2AllocationBudget::from_live_bytes(live_bytes)
}

fn parse_jp2_container_with_strict(data: &[u8], strict: bool) -> Result<ParsedJp2Container<'_>> {
    parse_jp2_container_with_strict_and_retained_baseline(data, strict, 0)
}

fn parse_jp2_container_with_strict_and_retained_baseline(
    data: &[u8],
    strict: bool,
    retained_baseline_bytes: usize,
) -> Result<ParsedJp2Container<'_>> {
    let mut reader = BitReader::new(data);
    let signature_box = r#box::read_checked(&mut reader)?;
    if signature_box.box_type != JP2_SIGNATURE || signature_box.data != JP2_SIGNATURE_PAYLOAD {
        bail!(FormatError::InvalidSignature);
    }

    let file_type_box = r#box::read_checked(&mut reader)?;
    if file_type_box.box_type != FILE_TYPE {
        bail!(FormatError::InvalidFileType);
    }
    let file_kind = classify_file_type(file_type_box.data)?;

    let mut image_boxes = None;
    let mut codestream = None;
    while !reader.at_end() {
        let current_box = match r#box::read_checked(&mut reader) {
            Ok(current_box) => current_box,
            Err(error) if strict => return Err(error),
            Err(_) => break,
        };

        match current_box.box_type {
            r#box::JP2_HEADER => {
                if image_boxes.is_some() || codestream.is_some() {
                    bail!(FormatError::InvalidBox);
                }
                image_boxes = Some(parse_jp2_header_box(
                    current_box.data,
                    strict,
                    retained_baseline_bytes,
                )?);
            }
            r#box::CONTIGUOUS_CODESTREAM => {
                if image_boxes.is_none() || codestream.is_some() {
                    bail!(FormatError::InvalidBox);
                }
                let codestream_offset = current_box.data.as_ptr() as usize - data.as_ptr() as usize;
                codestream = Some((codestream_offset, current_box.data));
            }
            _ => {}
        }
    }

    let boxes = image_boxes.ok_or(FormatError::MissingRequiredBox("jp2h"))?;
    let (codestream_offset, codestream) = codestream.ok_or(FormatError::MissingCodestream)?;
    Ok(ParsedJp2Container {
        file_kind,
        codestream_offset,
        codestream,
        boxes,
    })
}

pub(super) fn parse_jp2_header_box(
    data: &[u8],
    strict: bool,
    retained_baseline_bytes: usize,
) -> Result<ImageBoxes> {
    let color_spec_count = count_color_specification_boxes(data, strict)?;
    let mut budget = allocation::Jp2AllocationBudget::from_live_bytes(retained_baseline_bytes)?;
    let mut boxes = ImageBoxes {
        color_specifications: budget.try_vec(color_spec_count, "JP2 COLR metadata")?,
        ..ImageBoxes::default()
    };
    let mut saw_image_header = false;
    let mut reader = BitReader::new(data);

    while !reader.at_end() {
        let child_box = match r#box::read_checked(&mut reader) {
            Ok(child_box) => child_box,
            Err(error) if strict => return Err(error),
            Err(_) => break,
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
                    Err(crate::DecodeError::Format(_)) if !strict => {}
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
                    Err(crate::DecodeError::Format(_)) if !strict => {}
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
    if boxes.primary_color_specification().is_none() {
        bail!(FormatError::MissingRequiredBox("colr"));
    }
    Ok(boxes)
}

fn count_color_specification_boxes(data: &[u8], strict: bool) -> Result<usize> {
    let mut count = 0_usize;
    let mut reader = BitReader::new(data);
    while !reader.at_end() {
        let child_box = match r#box::read_checked(&mut reader) {
            Ok(child_box) => child_box,
            Err(error) if strict => return Err(error),
            Err(_) => break,
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
    Ok(count)
}

fn classify_file_type(payload: &[u8]) -> Result<Jp2FileKind> {
    if payload.len() < 8 {
        bail!(FormatError::InvalidFileType);
    }
    if payload[..4] == *b"jph " {
        return Ok(Jp2FileKind::Jph);
    }
    if payload[8..]
        .chunks_exact(4)
        .any(|compatible_brand| compatible_brand == b"jph ")
    {
        return Ok(Jp2FileKind::Jph);
    }
    Ok(Jp2FileKind::Jp2)
}
