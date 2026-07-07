//! Reading a JP2 file, defined in Annex I.

use alloc::vec::Vec;

use crate::error::{bail, FormatError, Result};
use crate::j2c::ComponentData;
use crate::jp2::cdef::{ChannelAssociation, ChannelDefinitionBox, ChannelType};
use crate::jp2::cmap::{ComponentMappingBox, ComponentMappingEntry, ComponentMappingType};
use crate::jp2::colr::{ColorSpace as NativeColorSpace, ColorSpecificationBox};
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
    pub(crate) color_specifications: Vec<ColorSpecificationBox>,
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

/// Parsed still-image file kind from the JP2/JPH file type box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jp2FileKind {
    /// JP2 still-image file.
    Jp2,
    /// JPH still-image file wrapping HTJ2K codestream data.
    Jph,
}

/// Native-owned JP2/JPH container parse summary.
#[derive(Debug, Clone)]
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
    boxes: ImageBoxes,
}

/// Parsed JP2/JPH image header metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Jp2ImageHeaderMetadata {
    /// Width from the Image Header box.
    pub width: u32,
    /// Height from the Image Header box.
    pub height: u32,
    /// Component count from the Image Header box.
    pub components: u16,
    /// Explicit bits-per-component descriptor, or `None` when BPCC is required.
    pub bits_per_component: Option<Jp2ComponentMetadata>,
}

/// Parsed component precision metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Jp2ComponentMetadata {
    /// Significant bits in this component.
    pub bit_depth: u8,
    /// Whether this component stores signed sample values.
    pub signed: bool,
}

/// JP2/JPH file-wrapper metadata parsed from the JP2 Header box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jp2FileMetadata {
    /// Bits-per-component box entries, when BPCC is present.
    pub bits_per_component: Vec<Jp2ComponentMetadata>,
    /// Colour Specification boxes in file order.
    pub color_specs: Vec<Jp2ColorSpec>,
    /// Palette box metadata, when PCLR is present.
    pub palette: Option<Jp2PaletteMetadata>,
    /// Component Mapping box entries in file order.
    pub component_mappings: Vec<Jp2ComponentMapping>,
    /// Channel Definition box entries in file order.
    pub channel_definitions: Vec<Jp2ChannelDefinition>,
    /// Whether a Palette box is present.
    pub has_palette: bool,
    /// Whether a Component Mapping box is present.
    pub has_component_mapping: bool,
    /// Whether a Channel Definition box is present.
    pub has_channel_definition: bool,
}

/// Parsed JP2/JPH Colour Specification box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Jp2ColorSpec {
    /// Enumerated color space value from a method-1 COLR box.
    Enumerated {
        /// Raw JP2 enumerated color-space value.
        value: u32,
    },
    /// ICC profile bytes from a method-2 COLR box.
    IccProfile {
        /// ICC profile byte payload.
        profile: Vec<u8>,
    },
    /// Unknown or currently unsupported COLR method.
    Unknown {
        /// Raw COLR method byte.
        method: u8,
    },
}

/// Parsed JP2/JPH Palette box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jp2PaletteMetadata {
    /// Palette column descriptors in box order.
    pub columns: Vec<Jp2PaletteColumn>,
    /// Palette entries in row-major order: entry, then column.
    pub entries: Vec<Vec<u64>>,
}

/// Parsed JP2/JPH Palette column descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Jp2PaletteColumn {
    /// Significant bits in this palette column.
    pub bit_depth: u8,
    /// Whether this palette column stores signed values.
    pub signed: bool,
}

/// Parsed JP2/JPH Component Mapping box entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Jp2ComponentMapping {
    /// Source codestream component index.
    pub component_index: u16,
    /// Mapping operation for this output channel.
    pub mapping_type: Jp2ComponentMappingType,
}

/// JP2/JPH Component Mapping operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jp2ComponentMappingType {
    /// Directly map the codestream component.
    Direct,
    /// Map the codestream component through a palette column.
    Palette {
        /// Palette column index.
        column: u8,
    },
    /// Unknown mapping type preserved for inspection.
    Unknown {
        /// Raw mapping type value.
        value: u8,
        /// Raw palette-column byte carried by the entry.
        column: u8,
    },
}

/// Parsed JP2/JPH Channel Definition box entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Jp2ChannelDefinition {
    /// Output channel index.
    pub channel_index: u16,
    /// Channel type.
    pub channel_type: Jp2ChannelType,
    /// Channel association.
    pub association: Jp2ChannelAssociation,
}

/// JP2/JPH channel type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jp2ChannelType {
    /// Color channel.
    Color,
    /// Opacity channel.
    Opacity,
    /// Premultiplied opacity channel.
    PremultipliedOpacity,
    /// Channel type is unspecified.
    Unspecified,
    /// Unknown raw channel type.
    Unknown {
        /// Raw channel type value.
        value: u16,
    },
}

/// JP2/JPH channel association.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jp2ChannelAssociation {
    /// Applies to the whole image.
    WholeImage,
    /// Associated with a one-based color channel index from CDEF.
    Color {
        /// One-based color channel index.
        index: u16,
    },
    /// Association is unspecified.
    Unspecified,
}

pub(crate) struct DecodedImage<'a> {
    /// The raw decoded JPEG2000 codestream components.
    pub(crate) decoded_components: &'a mut Vec<ComponentData>,
    /// The JP2 boxes of the image. In the case of a raw codestream, we
    /// will synthesize the necessary boxes.
    pub(crate) boxes: ImageBoxes,
}

const JP2_SIGNATURE_PAYLOAD: [u8; 4] = [0x0D, 0x0A, 0x87, 0x0A];

/// Parse JP2/JPH container boxes without decoding the codestream.
pub fn inspect_jp2_container(data: &[u8]) -> Result<Jp2Container<'_>> {
    parse_jp2_container_with_strict(data, true)
}

/// Extract the contiguous codestream payload from a JP2/JPH wrapper.
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

pub(crate) fn parse<'a>(data: &'a [u8], mut settings: DecodeSettings) -> Result<Image<'a>> {
    let container = parse_jp2_container_with_strict(data, settings.strict)?;
    if container.metadata.has_palette {
        settings.target_resolution = None;
    }
    let mut image_boxes = container.boxes;
    let parsed_codestream = crate::j2c::parse_raw(container.codestream, &settings)?;
    validate_codestream_file_kind(container.file_kind, &parsed_codestream.header)?;
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

fn parse_jp2_container_with_strict(data: &[u8], strict: bool) -> Result<Jp2Container<'_>> {
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
                image_boxes = Some(parse_jp2_header_box(current_box.data, strict)?);
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
    let image_header = boxes
        .image_header
        .ok_or(FormatError::MissingRequiredBox("ihdr"))?;
    let (codestream_offset, codestream) = codestream.ok_or(FormatError::MissingCodestream)?;
    let metadata = public_metadata_from_boxes(&boxes);
    Ok(Jp2Container {
        file_kind,
        codestream_offset,
        codestream,
        image_header: public_image_header(image_header),
        metadata,
        boxes,
    })
}

fn parse_jp2_header_box(data: &[u8], strict: bool) -> Result<ImageBoxes> {
    let mut boxes = ImageBoxes::default();
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
                boxes.bits_per_component = parse_bits_per_component(child_box.data)?;
            }
            r#box::CHANNEL_DEFINITION => {
                if cdef::parse(&mut boxes, child_box.data).is_err() && strict {
                    bail!(FormatError::InvalidBox);
                }
            }
            r#box::COLOUR_SPECIFICATION => {
                colr::parse(&mut boxes, child_box.data)?;
            }
            r#box::PALETTE => {
                if pclr::parse(&mut boxes, child_box.data).is_err() && strict {
                    bail!(FormatError::InvalidBox);
                }
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
        bail!(FormatError::MissingRequiredBox("ihdr"));
    }
    if boxes.color_specification.is_none() {
        bail!(FormatError::MissingRequiredBox("colr"));
    }
    Ok(boxes)
}

fn public_image_header(header: ImageHeaderBox) -> Jp2ImageHeaderMetadata {
    Jp2ImageHeaderMetadata {
        width: header.width,
        height: header.height,
        components: header.components,
        bits_per_component: header.bits_per_component.map(public_component_metadata),
    }
}

fn public_metadata_from_boxes(boxes: &ImageBoxes) -> Jp2FileMetadata {
    Jp2FileMetadata {
        bits_per_component: boxes
            .bits_per_component
            .iter()
            .copied()
            .map(public_component_metadata)
            .collect(),
        color_specs: boxes
            .color_specifications
            .iter()
            .map(public_color_spec)
            .collect(),
        palette: boxes.palette.as_ref().map(public_palette_metadata),
        component_mappings: boxes
            .component_mapping
            .as_ref()
            .map(|mapping| {
                mapping
                    .entries
                    .iter()
                    .map(public_component_mapping)
                    .collect()
            })
            .unwrap_or_default(),
        channel_definitions: boxes
            .channel_definition
            .as_ref()
            .map(|definition| {
                definition
                    .channel_definitions
                    .iter()
                    .map(public_channel_definition)
                    .collect()
            })
            .unwrap_or_default(),
        has_palette: boxes.palette.is_some(),
        has_component_mapping: boxes.component_mapping.is_some(),
        has_channel_definition: boxes.channel_definition.is_some(),
    }
}

fn public_component_metadata(component: ComponentDescriptor) -> Jp2ComponentMetadata {
    Jp2ComponentMetadata {
        bit_depth: component.bit_depth,
        signed: component.signed,
    }
}

fn public_color_spec(color_spec: &ColorSpecificationBox) -> Jp2ColorSpec {
    match &color_spec.color_space {
        NativeColorSpace::Enumerated(_) => Jp2ColorSpec::Enumerated {
            value: color_spec.enumerated_value.unwrap_or(0),
        },
        NativeColorSpace::Icc(profile) => Jp2ColorSpec::IccProfile {
            profile: profile.clone(),
        },
        NativeColorSpace::Unknown => Jp2ColorSpec::Unknown {
            method: color_spec.method,
        },
    }
}

fn public_palette_metadata(palette: &PaletteBox) -> Jp2PaletteMetadata {
    Jp2PaletteMetadata {
        columns: palette
            .columns
            .iter()
            .map(|column| Jp2PaletteColumn {
                bit_depth: column.bit_depth,
                signed: column.signed,
            })
            .collect(),
        entries: palette.entries.clone(),
    }
}

fn public_component_mapping(mapping: &ComponentMappingEntry) -> Jp2ComponentMapping {
    Jp2ComponentMapping {
        component_index: mapping.component_index,
        mapping_type: match mapping.mapping_type {
            ComponentMappingType::Direct => Jp2ComponentMappingType::Direct,
            ComponentMappingType::Palette { column } => Jp2ComponentMappingType::Palette { column },
            ComponentMappingType::Unknown { value, column } => {
                Jp2ComponentMappingType::Unknown { value, column }
            }
        },
    }
}

fn public_channel_definition(
    definition: &crate::jp2::cdef::ChannelDefinition,
) -> Jp2ChannelDefinition {
    Jp2ChannelDefinition {
        channel_index: definition.channel_index,
        channel_type: match definition.channel_type {
            ChannelType::Colour => Jp2ChannelType::Color,
            ChannelType::Opacity => Jp2ChannelType::Opacity,
            ChannelType::PremultipliedOpacity => Jp2ChannelType::PremultipliedOpacity,
            ChannelType::Unspecified => Jp2ChannelType::Unspecified,
            ChannelType::Unknown(value) => Jp2ChannelType::Unknown { value },
        },
        association: match definition._association {
            ChannelAssociation::WholeImage => Jp2ChannelAssociation::WholeImage,
            ChannelAssociation::Colour(index) => Jp2ChannelAssociation::Color { index },
            ChannelAssociation::Unspecified => Jp2ChannelAssociation::Unspecified,
        },
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

fn validate_codestream_file_kind(
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
                ComponentMappingType::Unknown { .. } => return None,
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
