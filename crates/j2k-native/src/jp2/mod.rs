//! Reading a JP2 file, defined in Annex I.

use alloc::vec::Vec;

use crate::error::{bail, FormatError, Result};
use crate::j2c::ComponentData;
use crate::jp2::cdef::ChannelDefinitionBox;
use crate::jp2::cmap::{ComponentMappingBox, ComponentMappingEntry, ComponentMappingType};
use crate::jp2::colr::ColorSpecificationBox;
use crate::jp2::pclr::PaletteBox;
use crate::jp2::r#box::{FILE_TYPE, JP2_SIGNATURE};
use crate::reader::BitReader;
use crate::{resolve_alpha_and_color_space, DecodeSettings, Image};

pub(crate) mod r#box;
pub(crate) mod cdef;
pub(crate) mod cmap;
pub(crate) mod colr;
pub(crate) mod icc;
pub(crate) mod pclr;

#[derive(Debug, Clone, Default)]
pub(crate) struct ImageBoxes {
    pub(crate) image_header: Option<ImageHeaderBox>,
    pub(crate) bits_per_component: Vec<ComponentDescriptor>,
    pub(crate) color_specification: Option<ColorSpecificationBox>,
    pub(crate) channel_definition: Option<ChannelDefinitionBox>,
    pub(crate) palette: Option<PaletteBox>,
    pub(crate) component_mapping: Option<ComponentMappingBox>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ImageHeaderBox {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) components: u16,
    pub(crate) bits_per_component: Option<ComponentDescriptor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ComponentDescriptor {
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
}

pub(crate) struct DecodedImage<'a> {
    /// The raw decoded JPEG2000 codestream components.
    pub(crate) decoded_components: &'a mut Vec<ComponentData>,
    /// The JP2 boxes of the image. In the case of a raw codestream, we
    /// will synthesize the necessary boxes.
    pub(crate) boxes: ImageBoxes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StillImageFileKind {
    Jp2,
    Jph,
}

pub(crate) fn parse<'a>(data: &'a [u8], mut settings: DecodeSettings) -> Result<Image<'a>> {
    let mut reader = BitReader::new(data);
    let signature_box = r#box::read(&mut reader).ok_or(FormatError::InvalidBox)?;

    if signature_box.box_type != JP2_SIGNATURE {
        bail!(FormatError::InvalidSignature);
    }

    let file_type_box = r#box::read(&mut reader).ok_or(FormatError::InvalidBox)?;

    if file_type_box.box_type != FILE_TYPE {
        bail!(FormatError::InvalidFileType);
    }
    let file_kind = classify_file_type(file_type_box.data)?;

    let mut image_boxes: Option<ImageBoxes> = None;
    let mut parsed_codestream = None;

    // Read boxes until we find the JP2 Header box
    while !reader.at_end() {
        let Some(current_box) = r#box::read(&mut reader) else {
            if settings.strict {
                bail!(FormatError::InvalidBox);
            }

            break;
        };

        match current_box.box_type {
            r#box::JP2_HEADER => {
                let mut boxes = ImageBoxes::default();
                let mut saw_image_header = false;

                let mut jp2h_reader = BitReader::new(current_box.data);

                // Read child boxes within JP2 Header box
                while !jp2h_reader.at_end() {
                    let child_box = r#box::read(&mut jp2h_reader).ok_or(FormatError::InvalidBox)?;

                    match child_box.box_type {
                        r#box::IMAGE_HEADER => {
                            boxes.image_header = Some(parse_image_header(child_box.data)?);
                            saw_image_header = true;
                        }
                        r#box::BITS_PER_COMPONENT => {
                            boxes.bits_per_component = parse_bits_per_component(child_box.data)?;
                        }
                        r#box::CHANNEL_DEFINITION => {
                            if cdef::parse(&mut boxes, child_box.data).is_err() && settings.strict {
                                bail!(FormatError::InvalidBox);
                            }
                            // If not strict decoding, just assume default
                            // configuration.
                        }
                        r#box::COLOUR_SPECIFICATION => {
                            colr::parse(&mut boxes, child_box.data)?;
                        }
                        r#box::PALETTE => {
                            if pclr::parse(&mut boxes, child_box.data).is_err() && settings.strict {
                                bail!(FormatError::InvalidBox);
                            }

                            // If we have a palettized image, decoding at a
                            // lower resolution will corrupt it, so we can't do
                            // it in this case.
                            settings.target_resolution = None;
                        }
                        r#box::COMPONENT_MAPPING => {
                            cmap::parse(&mut boxes, child_box.data)?;
                        }
                        _ => {
                            ldebug!(
                                "ignoring header box {}",
                                r#box::tag_to_string(child_box.box_type)
                            );
                        }
                    }
                }

                if !saw_image_header {
                    bail!(FormatError::InvalidBox);
                }
                if boxes.color_specification.is_none() {
                    bail!(FormatError::InvalidBox);
                }
                image_boxes = Some(boxes);
            }
            r#box::CONTIGUOUS_CODESTREAM => {
                parsed_codestream = Some(crate::j2c::parse_raw(current_box.data, &settings)?);
            }
            _ => {}
        }
    }

    let mut image_boxes = image_boxes.ok_or(FormatError::InvalidBox)?;
    let parsed_codestream = parsed_codestream.ok_or(FormatError::MissingCodestream)?;
    validate_codestream_file_kind(file_kind, &parsed_codestream.header)?;
    validate_image_header_matches_codestream(&image_boxes, &parsed_codestream.header)?;
    validate_component_precision_metadata(&image_boxes, &parsed_codestream.header)?;

    if let Some(palette) = image_boxes.palette.as_ref() {
        if image_boxes.component_mapping.is_none() {
            // In theory, a cmap is required if we have pclr, but since there are
            // some files that don't seem to do so, we assume
            // that all channels are mapped via the palette in case not.
            let mappings = (0..palette.columns.len())
                .map(|i| ComponentMappingEntry {
                    component_index: 0,
                    mapping_type: ComponentMappingType::Palette { column: i as u8 },
                })
                .collect::<Vec<_>>();

            image_boxes.component_mapping = Some(ComponentMappingBox { entries: mappings });
        }
    }

    let (color_space, has_alpha) =
        resolve_alpha_and_color_space(&image_boxes, &parsed_codestream.header, &settings)?;
    Ok(Image {
        codestream: parsed_codestream.data,
        header: parsed_codestream.header,
        boxes: image_boxes,
        settings,
        color_space,
        has_alpha,
    })
}

fn classify_file_type(payload: &[u8]) -> Result<StillImageFileKind> {
    if payload.len() < 8 {
        bail!(FormatError::InvalidFileType);
    }
    if payload[..4] == *b"jph " {
        return Ok(StillImageFileKind::Jph);
    }
    if payload[8..]
        .chunks_exact(4)
        .any(|compatible_brand| compatible_brand == b"jph ")
    {
        return Ok(StillImageFileKind::Jph);
    }
    Ok(StillImageFileKind::Jp2)
}

fn validate_codestream_file_kind(
    file_kind: StillImageFileKind,
    header: &crate::j2c::Header<'_>,
) -> Result<()> {
    let high_throughput = header.component_infos.iter().any(|component| {
        component
            .code_block_style()
            .uses_high_throughput_block_coding()
    });
    match (file_kind, high_throughput) {
        (StillImageFileKind::Jph, false) | (StillImageFileKind::Jp2, true) => {
            bail!(FormatError::InvalidFileType);
        }
        _ => Ok(()),
    }
}

fn parse_image_header(data: &[u8]) -> Result<ImageHeaderBox> {
    if data.len() < 14 {
        bail!(FormatError::InvalidBox);
    }
    let compression_type = data[11];
    if compression_type != 7 {
        bail!(FormatError::InvalidBox);
    }
    Ok(ImageHeaderBox {
        height: u32::from_be_bytes([data[0], data[1], data[2], data[3]]),
        width: u32::from_be_bytes([data[4], data[5], data[6], data[7]]),
        components: u16::from_be_bytes([data[8], data[9]]),
        bits_per_component: if data[10] == 0xFF {
            None
        } else {
            Some(parse_component_descriptor(data[10])?)
        },
    })
}

fn parse_bits_per_component(data: &[u8]) -> Result<Vec<ComponentDescriptor>> {
    if data.is_empty() {
        bail!(FormatError::InvalidBox);
    }
    data.iter()
        .map(|descriptor| parse_component_descriptor(*descriptor))
        .collect()
}

fn parse_component_descriptor(descriptor: u8) -> Result<ComponentDescriptor> {
    let bit_depth = (descriptor & 0x7F) + 1;
    if bit_depth > 38 {
        bail!(FormatError::InvalidBox);
    }
    Ok(ComponentDescriptor {
        bit_depth,
        signed: (descriptor & 0x80) != 0,
    })
}

fn validate_image_header_matches_codestream(
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

fn validate_component_precision_metadata(
    boxes: &ImageBoxes,
    header: &crate::j2c::Header<'_>,
) -> Result<()> {
    let Some(image_header) = boxes.image_header else {
        bail!(FormatError::InvalidBox);
    };
    let resolved =
        resolved_image_component_descriptors(boxes, header).ok_or(FormatError::InvalidBox)?;
    if resolved.len() != image_header.components as usize {
        bail!(FormatError::InvalidBox);
    }

    match image_header.bits_per_component {
        Some(descriptor) => {
            if !boxes.bits_per_component.is_empty() {
                bail!(FormatError::InvalidBox);
            }
            if !resolved.iter().all(|component| *component == descriptor) {
                bail!(FormatError::InvalidBox);
            }
        }
        None => {
            if boxes.bits_per_component.len() != image_header.components as usize {
                bail!(FormatError::InvalidBox);
            }
            for (component, descriptor) in resolved.iter().zip(boxes.bits_per_component.iter()) {
                if component != descriptor {
                    bail!(FormatError::InvalidBox);
                }
            }
        }
    }

    Ok(())
}

fn resolved_image_component_descriptors(
    boxes: &ImageBoxes,
    header: &crate::j2c::Header<'_>,
) -> Option<Vec<ComponentDescriptor>> {
    if let Some(component_mapping) = boxes.component_mapping.as_ref() {
        let mut resolved = Vec::with_capacity(component_mapping.entries.len());
        for entry in &component_mapping.entries {
            match entry.mapping_type {
                ComponentMappingType::Direct => {
                    let component = header.component_infos.get(entry.component_index as usize)?;
                    resolved.push(component_descriptor_from_size_info(
                        component.size_info.precision,
                        component.size_info.signed,
                    ));
                }
                ComponentMappingType::Palette { column } => {
                    let palette = boxes.palette.as_ref()?;
                    let column = palette.columns.get(column as usize)?;
                    resolved.push(component_descriptor_from_size_info(
                        column.bit_depth,
                        column.signed,
                    ));
                }
            }
        }
        return Some(resolved);
    }

    if let Some(palette) = boxes.palette.as_ref() {
        return Some(
            palette
                .columns
                .iter()
                .map(|column| component_descriptor_from_size_info(column.bit_depth, column.signed))
                .collect(),
        );
    }

    Some(
        header
            .component_infos
            .iter()
            .map(|component| {
                component_descriptor_from_size_info(
                    component.size_info.precision,
                    component.size_info.signed,
                )
            })
            .collect(),
    )
}

fn component_descriptor_from_size_info(bit_depth: u8, signed: bool) -> ComponentDescriptor {
    ComponentDescriptor { bit_depth, signed }
}
