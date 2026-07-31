// SPDX-License-Identifier: MIT OR Apache-2.0

//! JP2/JPH container traversal and native decode parse orchestration.

mod header;

use crate::error::{bail, FormatError, Result};
use crate::image::{ImageProperties, ImageSource};
use crate::reader::BitReader;
use crate::{resolve_alpha_and_color_space, DecodeSettings, Image};

#[cfg(test)]
pub(super) use self::header::parse_jp2_header_box;
use self::header::parse_jp2_header_box_tracked;
use super::allocation;
use super::cmap::{ComponentMappingBox, ComponentMappingEntry, ComponentMappingType};
use super::metadata::{
    public_image_header, public_metadata_from_boxes, ImageBoxes, Jp2FileMetadata,
    Jp2ImageHeaderMetadata,
};
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
    used_lenient_metadata_recovery: bool,
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
    let mut codestream_settings = settings;
    codestream_settings.strict = true;
    let parsed_codestream = crate::j2c::parse_raw_with_retained_baseline(
        container.codestream,
        &codestream_settings,
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

    let (color_space, has_alpha, recovered_alpha_metadata) = resolve_alpha_and_color_space(
        &image_boxes,
        &parsed_codestream.header,
        &settings,
        retained_baseline_bytes,
    )?;
    let used_lenient_metadata_recovery =
        container.used_lenient_metadata_recovery || recovered_alpha_metadata;
    let properties = ImageProperties::new(
        image_boxes,
        settings,
        color_space,
        has_alpha,
        used_lenient_metadata_recovery,
    );
    if retained_baseline_bytes == 0 {
        Image::from_parsed_parts(
            ImageSource::new(data, parsed_codestream.data),
            parsed_codestream.header,
            properties,
        )
    } else {
        Image::from_parsed_parts_with_retained_baseline(
            ImageSource::new(data, parsed_codestream.data),
            parsed_codestream.header,
            properties,
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
    let mut used_lenient_metadata_recovery = false;
    while !reader.at_end() {
        let Some(current_box) = read_recoverable_metadata_box(
            &mut reader,
            strict,
            &mut used_lenient_metadata_recovery,
        )?
        else {
            break;
        };

        match current_box.box_type {
            r#box::JP2_HEADER => {
                if image_boxes.is_some() || codestream.is_some() {
                    bail!(FormatError::InvalidBox);
                }
                let parsed_header = parse_jp2_header_box_tracked(
                    current_box.data,
                    strict,
                    retained_baseline_bytes,
                )?;
                used_lenient_metadata_recovery |= parsed_header.used_lenient_metadata_recovery;
                image_boxes = Some(parsed_header.boxes);
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
        used_lenient_metadata_recovery,
    })
}

fn read_recoverable_metadata_box<'a>(
    reader: &mut BitReader<'a>,
    strict: bool,
    used_lenient_metadata_recovery: &mut bool,
) -> Result<Option<r#box::Jp2Box<'a>>> {
    match r#box::read_checked(reader) {
        Ok(parsed) => Ok(Some(parsed)),
        Err(error) if strict => Err(error),
        Err(_) => {
            *used_lenient_metadata_recovery = true;
            Ok(None)
        }
    }
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
