// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{codestream::parse_codestream, ParsedImageInfo};
use crate::{
    J2kChannelAssociation, J2kChannelDefinition, J2kChannelType, J2kColorSpec, J2kComponentInfo,
    J2kComponentMapping, J2kComponentMappingType, J2kError, J2kFileMetadata, J2kPaletteColumn,
    J2kPaletteMetadata,
};
use j2k_core::{Colorspace, CompressedPayloadKind, InputError};
use j2k_native::{
    extract_jp2_codestream_payload as native_extract_jp2_codestream_payload, inspect_jp2_container,
    DecodeError as NativeDecodeError, FormatError as NativeFormatError,
    Jp2ChannelAssociation as NativeChannelAssociation,
    Jp2ChannelDefinition as NativeChannelDefinition, Jp2ChannelType as NativeChannelType,
    Jp2ColorSpec as NativeColorSpec, Jp2ComponentMapping as NativeComponentMapping,
    Jp2ComponentMappingType as NativeComponentMappingType,
    Jp2ComponentMetadata as NativeComponentMetadata, Jp2FileKind, Jp2FileMetadata,
    Jp2ImageHeaderMetadata, Jp2PaletteColumn as NativePaletteColumn,
    Jp2PaletteMetadata as NativePaletteMetadata,
};

const JP2_SIGNATURE_PREFIX: [u8; 8] = [0, 0, 0, 12, b'j', b'P', b' ', b' '];

pub(crate) fn looks_like_jp2(input: &[u8]) -> bool {
    input.starts_with(&JP2_SIGNATURE_PREFIX)
}

pub(crate) fn extract_jp2_codestream_payload(
    input: &[u8],
) -> Result<(CompressedPayloadKind, usize, &[u8]), J2kError> {
    let (file_kind, codestream_offset, codestream) =
        native_extract_jp2_codestream_payload(input).map_err(map_native_jp2_error)?;
    Ok((
        payload_kind_from_native(file_kind),
        codestream_offset,
        codestream,
    ))
}

pub(crate) fn parse_jp2(input: &[u8]) -> Result<ParsedImageInfo, J2kError> {
    let container = inspect_jp2_container(input).map_err(map_native_jp2_error)?;
    let payload_kind = payload_kind_from_native(container.file_kind);
    let image_header = image_header_from_native(container.image_header);
    let file_metadata = file_metadata_from_native(&container.metadata);
    let parsed = parse_codestream(container.codestream)?;

    validate_ihdr_matches_codestream(image_header, &parsed.siz)?;
    validate_component_metadata(image_header, &file_metadata, &parsed.siz)?;
    if payload_kind == CompressedPayloadKind::JphFile && !parsed.cod.high_throughput {
        return Err(J2kError::InvalidBox {
            offset: 0,
            what: "JPH file type requires an HTJ2K codestream",
        });
    }
    if payload_kind == CompressedPayloadKind::Jp2File && parsed.cod.high_throughput {
        return Err(J2kError::InvalidBox {
            offset: 0,
            what: "JP2 file type must not wrap an HTJ2K codestream; use JPH",
        });
    }

    let colorspace = primary_colorspace_from_file_metadata(&file_metadata);
    let info = parsed.clone().into_info(colorspace);
    let components = parsed.siz.component_info.clone();
    Ok(ParsedImageInfo {
        info,
        transfer_syntax: parsed.transfer_syntax(),
        payload_kind,
        components,
        file_metadata: Some(file_metadata),
    })
}

#[derive(Debug, Clone, Copy)]
struct Jp2ImageHeader {
    offset: usize,
    width: u32,
    height: u32,
    components: u16,
    bits_per_component: Option<J2kComponentInfo>,
}

fn payload_kind_from_native(file_kind: Jp2FileKind) -> CompressedPayloadKind {
    match file_kind {
        Jp2FileKind::Jp2 => CompressedPayloadKind::Jp2File,
        Jp2FileKind::Jph => CompressedPayloadKind::JphFile,
    }
}

fn image_header_from_native(header: Jp2ImageHeaderMetadata) -> Jp2ImageHeader {
    Jp2ImageHeader {
        offset: 0,
        width: header.width,
        height: header.height,
        components: header.components,
        bits_per_component: header.bits_per_component.map(component_from_native),
    }
}

fn file_metadata_from_native(metadata: &Jp2FileMetadata) -> J2kFileMetadata {
    J2kFileMetadata {
        bits_per_component: metadata
            .bits_per_component
            .iter()
            .copied()
            .map(component_from_native)
            .collect(),
        color_specs: metadata
            .color_specs
            .iter()
            .map(color_spec_from_native)
            .collect(),
        palette: metadata.palette.as_ref().map(palette_from_native),
        component_mappings: metadata
            .component_mappings
            .iter()
            .copied()
            .map(component_mapping_from_native)
            .collect(),
        channel_definitions: metadata
            .channel_definitions
            .iter()
            .copied()
            .map(channel_definition_from_native)
            .collect(),
        has_palette: metadata.has_palette,
        has_component_mapping: metadata.has_component_mapping,
        has_channel_definition: metadata.has_channel_definition,
    }
}

fn component_from_native(component: NativeComponentMetadata) -> J2kComponentInfo {
    J2kComponentInfo {
        bit_depth: component.bit_depth,
        signed: component.signed,
        x_rsiz: 1,
        y_rsiz: 1,
    }
}

fn color_spec_from_native(color_spec: &NativeColorSpec) -> J2kColorSpec {
    match color_spec {
        NativeColorSpec::Enumerated { value } => J2kColorSpec::Enumerated { value: *value },
        NativeColorSpec::IccProfile { profile } => J2kColorSpec::IccProfile {
            profile: profile.clone(),
        },
        NativeColorSpec::Unknown { method } => J2kColorSpec::Unknown { method: *method },
    }
}

fn palette_from_native(palette: &NativePaletteMetadata) -> J2kPaletteMetadata {
    J2kPaletteMetadata {
        columns: palette
            .columns
            .iter()
            .copied()
            .map(palette_column_from_native)
            .collect(),
        entries: palette.entries.clone(),
    }
}

fn palette_column_from_native(column: NativePaletteColumn) -> J2kPaletteColumn {
    J2kPaletteColumn {
        bit_depth: column.bit_depth,
        signed: column.signed,
    }
}

fn component_mapping_from_native(mapping: NativeComponentMapping) -> J2kComponentMapping {
    J2kComponentMapping {
        component_index: mapping.component_index,
        mapping_type: match mapping.mapping_type {
            NativeComponentMappingType::Direct => J2kComponentMappingType::Direct,
            NativeComponentMappingType::Palette { column } => {
                J2kComponentMappingType::Palette { column }
            }
            NativeComponentMappingType::Unknown { value, column } => {
                J2kComponentMappingType::Unknown { value, column }
            }
        },
    }
}

fn channel_definition_from_native(definition: NativeChannelDefinition) -> J2kChannelDefinition {
    J2kChannelDefinition {
        channel_index: definition.channel_index,
        channel_type: match definition.channel_type {
            NativeChannelType::Color => J2kChannelType::Color,
            NativeChannelType::Opacity => J2kChannelType::Opacity,
            NativeChannelType::PremultipliedOpacity => J2kChannelType::PremultipliedOpacity,
            NativeChannelType::Unspecified => J2kChannelType::Unspecified,
            NativeChannelType::Unknown { value } => J2kChannelType::Unknown { value },
        },
        association: match definition.association {
            NativeChannelAssociation::WholeImage => J2kChannelAssociation::WholeImage,
            NativeChannelAssociation::Color { index } => J2kChannelAssociation::Color { index },
            NativeChannelAssociation::Unspecified => J2kChannelAssociation::Unspecified,
        },
    }
}

fn primary_colorspace_from_file_metadata(metadata: &J2kFileMetadata) -> Option<Colorspace> {
    metadata
        .color_specs
        .first()
        .map(|color_spec| match color_spec {
            J2kColorSpec::Enumerated { value: 16 } => Colorspace::SRgb,
            J2kColorSpec::Enumerated { value: 17 } => Colorspace::SGray,
            J2kColorSpec::Enumerated { value: 18 } => Colorspace::YCbCr,
            J2kColorSpec::Enumerated { .. } | J2kColorSpec::IccProfile { .. } => {
                Colorspace::IccTagged
            }
            J2kColorSpec::Unknown { .. } => Colorspace::IccTagged,
        })
}

fn validate_ihdr_matches_codestream(
    ihdr: Jp2ImageHeader,
    siz: &super::ParsedSiz,
) -> Result<(), J2kError> {
    if (ihdr.width, ihdr.height) != siz.dimensions {
        return Err(J2kError::InvalidBox {
            offset: ihdr.offset,
            what: "ihdr dimensions must match codestream image dimensions",
        });
    }
    Ok(())
}

fn validate_component_metadata(
    ihdr: Jp2ImageHeader,
    metadata: &J2kFileMetadata,
    siz: &super::ParsedSiz,
) -> Result<(), J2kError> {
    let resolved = resolved_image_component_metadata(metadata, siz);
    let resolved = resolved.as_deref().unwrap_or(&siz.component_info);
    if resolved.len() != ihdr.components as usize {
        return Err(J2kError::InvalidBox {
            offset: ihdr.offset,
            what: "ihdr component count must match resolved JP2 image components",
        });
    }

    if let Some(descriptor) = ihdr.bits_per_component {
        if !metadata.bits_per_component.is_empty() {
            return Err(J2kError::InvalidBox {
                offset: ihdr.offset,
                what: "bpcc must not be present when ihdr bpc is explicit",
            });
        }
        if !resolved
            .iter()
            .all(|component| same_component_precision(*component, descriptor))
        {
            return Err(J2kError::InvalidBox {
                offset: ihdr.offset,
                what: "ihdr bpc must match resolved JP2 image component precision",
            });
        }
    } else {
        if metadata.bits_per_component.len() != ihdr.components as usize {
            return Err(J2kError::InvalidBox {
                offset: ihdr.offset,
                what: "bpcc component count must match ihdr component count",
            });
        }
        for (component, descriptor) in resolved.iter().zip(metadata.bits_per_component.iter()) {
            if !same_component_precision(*component, *descriptor) {
                return Err(J2kError::InvalidBox {
                    offset: ihdr.offset,
                    what: "bpcc entries must match resolved JP2 image component precision",
                });
            }
        }
    }

    Ok(())
}

fn resolved_image_component_metadata(
    metadata: &J2kFileMetadata,
    siz: &super::ParsedSiz,
) -> Option<Vec<J2kComponentInfo>> {
    if metadata.component_mappings.is_empty() {
        if let Some(palette) = metadata.palette.as_ref() {
            return Some(
                palette
                    .columns
                    .iter()
                    .copied()
                    .map(component_from_palette_column)
                    .collect(),
            );
        }
        return Some(siz.component_info.clone());
    }

    let mut resolved = Vec::with_capacity(metadata.component_mappings.len());
    for mapping in &metadata.component_mappings {
        match mapping.mapping_type {
            J2kComponentMappingType::Direct => {
                let component = siz.component_info.get(mapping.component_index as usize)?;
                resolved.push(*component);
            }
            J2kComponentMappingType::Palette { column } => {
                let palette = metadata.palette.as_ref()?;
                let column = palette.columns.get(column as usize)?;
                resolved.push(component_from_palette_column(*column));
            }
            J2kComponentMappingType::Unknown { .. } => return None,
        }
    }

    Some(resolved)
}

fn component_from_palette_column(column: J2kPaletteColumn) -> J2kComponentInfo {
    J2kComponentInfo {
        bit_depth: column.bit_depth,
        signed: column.signed,
        x_rsiz: 1,
        y_rsiz: 1,
    }
}

fn same_component_precision(left: J2kComponentInfo, right: J2kComponentInfo) -> bool {
    left.bit_depth == right.bit_depth && left.signed == right.signed
}

fn map_native_jp2_error(error: NativeDecodeError) -> J2kError {
    match error {
        NativeDecodeError::Format(NativeFormatError::InvalidSignature) => J2kError::InvalidBox {
            offset: 0,
            what: "invalid JP2 signature box",
        },
        NativeDecodeError::Format(NativeFormatError::InvalidFileType) => J2kError::InvalidBox {
            offset: 0,
            what: "invalid JP2/JPH file type box",
        },
        NativeDecodeError::Format(NativeFormatError::TooShort { need, have }) => {
            InputError::TooShort { need, have }.into()
        }
        NativeDecodeError::Format(NativeFormatError::TruncatedAt { offset, segment }) => {
            InputError::TruncatedAt { offset, segment }.into()
        }
        NativeDecodeError::Format(NativeFormatError::MissingRequiredBox(box_type)) => {
            J2kError::MissingRequiredBox { box_type }
        }
        NativeDecodeError::Format(NativeFormatError::MissingCodestream) => {
            J2kError::MissingRequiredBox { box_type: "jp2c" }
        }
        NativeDecodeError::Format(NativeFormatError::InvalidBox) => J2kError::InvalidBox {
            offset: 0,
            what: "invalid JP2/JPH box",
        },
        NativeDecodeError::Format(NativeFormatError::Unsupported) => J2kError::InvalidBox {
            offset: 0,
            what: "unsupported JP2/JPH box metadata",
        },
        _ => J2kError::InvalidBox {
            offset: 0,
            what: "invalid JP2/JPH container",
        },
    }
}
