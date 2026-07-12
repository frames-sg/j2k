// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    allocation::{capacity_bytes, checked_add_bytes, ParseAllocationBudget},
    codestream::parse_codestream,
    ParsedImageInfo,
};
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
    let (file_metadata, metadata_bytes) = file_metadata_from_native(container.metadata)?;
    let parsed = parse_codestream(container.codestream, metadata_bytes)?;

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
    let (info, transfer_syntax, components) = parsed.into_parts(colorspace);
    Ok(ParsedImageInfo {
        info,
        transfer_syntax,
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

fn file_metadata_from_native(
    metadata: Jp2FileMetadata,
) -> Result<(J2kFileMetadata, usize), J2kError> {
    let mut budget =
        ParseAllocationBudget::from_live_bytes(native_metadata_allocated_bytes(&metadata)?)?;
    let Jp2FileMetadata {
        bits_per_component,
        color_specs,
        palette,
        component_mappings,
        channel_definitions,
        has_palette,
        has_component_mapping,
        has_channel_definition,
    } = metadata;

    let metadata = J2kFileMetadata {
        bits_per_component: components_from_native(bits_per_component, &mut budget)?,
        color_specs: color_specs_from_native(color_specs, &mut budget)?,
        palette: palette_from_native(palette, &mut budget)?,
        component_mappings: component_mappings_from_native(component_mappings, &mut budget)?,
        channel_definitions: channel_definitions_from_native(channel_definitions, &mut budget)?,
        has_palette,
        has_component_mapping,
        has_channel_definition,
    };
    let allocated_bytes = metadata_allocated_bytes(&metadata)?;
    if allocated_bytes != budget.live_bytes() {
        return Err(J2kError::InternalInvariant {
            what: "JP2 facade metadata allocation accounting mismatch",
        });
    }
    Ok((metadata, allocated_bytes))
}

fn component_from_native(component: NativeComponentMetadata) -> J2kComponentInfo {
    J2kComponentInfo {
        bit_depth: component.bit_depth,
        signed: component.signed,
        x_rsiz: 1,
        y_rsiz: 1,
    }
}

fn components_from_native(
    components: Vec<NativeComponentMetadata>,
    budget: &mut ParseAllocationBudget,
) -> Result<Vec<J2kComponentInfo>, J2kError> {
    let source_capacity = components.capacity();
    let mut facade = budget.try_vec(components.len(), "JP2 facade BPCC metadata")?;
    for component in components {
        facade.push(component_from_native(component));
    }
    budget.release_capacity::<NativeComponentMetadata>(source_capacity)?;
    Ok(facade)
}

fn color_specs_from_native(
    color_specs: Vec<NativeColorSpec>,
    budget: &mut ParseAllocationBudget,
) -> Result<Vec<J2kColorSpec>, J2kError> {
    let source_capacity = color_specs.capacity();
    let mut facade = budget.try_vec(color_specs.len(), "JP2 facade COLR metadata")?;
    for color_spec in color_specs {
        facade.push(color_spec_from_native(color_spec));
    }
    budget.release_capacity::<NativeColorSpec>(source_capacity)?;
    Ok(facade)
}

fn color_spec_from_native(color_spec: NativeColorSpec) -> J2kColorSpec {
    match color_spec {
        NativeColorSpec::Enumerated { value } => J2kColorSpec::Enumerated { value },
        NativeColorSpec::IccProfile { profile } => J2kColorSpec::IccProfile { profile },
        NativeColorSpec::Unknown { method } => J2kColorSpec::Unknown { method },
    }
}

fn palette_from_native(
    palette: Option<NativePaletteMetadata>,
    budget: &mut ParseAllocationBudget,
) -> Result<Option<J2kPaletteMetadata>, J2kError> {
    let Some(NativePaletteMetadata { columns, entries }) = palette else {
        return Ok(None);
    };
    let source_capacity = columns.capacity();
    let mut facade_columns = budget.try_vec(columns.len(), "JP2 facade palette columns")?;
    for column in columns {
        facade_columns.push(palette_column_from_native(column));
    }
    budget.release_capacity::<NativePaletteColumn>(source_capacity)?;
    Ok(Some(J2kPaletteMetadata {
        columns: facade_columns,
        entries,
    }))
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

fn component_mappings_from_native(
    mappings: Vec<NativeComponentMapping>,
    budget: &mut ParseAllocationBudget,
) -> Result<Vec<J2kComponentMapping>, J2kError> {
    let source_capacity = mappings.capacity();
    let mut facade = budget.try_vec(mappings.len(), "JP2 facade component mappings")?;
    for mapping in mappings {
        facade.push(component_mapping_from_native(mapping));
    }
    budget.release_capacity::<NativeComponentMapping>(source_capacity)?;
    Ok(facade)
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

fn channel_definitions_from_native(
    definitions: Vec<NativeChannelDefinition>,
    budget: &mut ParseAllocationBudget,
) -> Result<Vec<J2kChannelDefinition>, J2kError> {
    let source_capacity = definitions.capacity();
    let mut facade = budget.try_vec(definitions.len(), "JP2 facade channel definitions")?;
    for definition in definitions {
        facade.push(channel_definition_from_native(definition));
    }
    budget.release_capacity::<NativeChannelDefinition>(source_capacity)?;
    Ok(facade)
}

fn native_metadata_allocated_bytes(metadata: &Jp2FileMetadata) -> Result<usize, J2kError> {
    let mut bytes = capacity_bytes::<NativeComponentMetadata>(
        metadata.bits_per_component.capacity(),
        "native JP2 BPCC metadata",
    )?;
    checked_add_bytes(
        &mut bytes,
        capacity_bytes::<NativeColorSpec>(
            metadata.color_specs.capacity(),
            "native JP2 COLR metadata",
        )?,
        "native JP2 metadata",
    )?;
    for color_spec in &metadata.color_specs {
        if let NativeColorSpec::IccProfile { profile } = color_spec {
            checked_add_bytes(&mut bytes, profile.capacity(), "native JP2 ICC profile")?;
        }
    }
    if let Some(palette) = &metadata.palette {
        checked_add_bytes(
            &mut bytes,
            capacity_bytes::<NativePaletteColumn>(
                palette.columns.capacity(),
                "native JP2 palette columns",
            )?,
            "native JP2 metadata",
        )?;
        checked_add_bytes(
            &mut bytes,
            capacity_bytes::<Vec<u64>>(palette.entries.capacity(), "native JP2 palette rows")?,
            "native JP2 metadata",
        )?;
        for row in &palette.entries {
            checked_add_bytes(
                &mut bytes,
                capacity_bytes::<u64>(row.capacity(), "native JP2 palette entries")?,
                "native JP2 metadata",
            )?;
        }
    }
    checked_add_bytes(
        &mut bytes,
        capacity_bytes::<NativeComponentMapping>(
            metadata.component_mappings.capacity(),
            "native JP2 component mappings",
        )?,
        "native JP2 metadata",
    )?;
    checked_add_bytes(
        &mut bytes,
        capacity_bytes::<NativeChannelDefinition>(
            metadata.channel_definitions.capacity(),
            "native JP2 channel definitions",
        )?,
        "native JP2 metadata",
    )?;
    Ok(bytes)
}

pub(super) fn metadata_allocated_bytes(metadata: &J2kFileMetadata) -> Result<usize, J2kError> {
    let mut bytes = capacity_bytes::<J2kComponentInfo>(
        metadata.bits_per_component.capacity(),
        "JP2 BPCC metadata",
    )?;
    checked_add_bytes(
        &mut bytes,
        capacity_bytes::<J2kColorSpec>(metadata.color_specs.capacity(), "JP2 COLR metadata")?,
        "JP2 facade metadata",
    )?;
    for color_spec in &metadata.color_specs {
        if let J2kColorSpec::IccProfile { profile } = color_spec {
            checked_add_bytes(&mut bytes, profile.capacity(), "JP2 ICC profile")?;
        }
    }
    if let Some(palette) = &metadata.palette {
        checked_add_bytes(
            &mut bytes,
            capacity_bytes::<J2kPaletteColumn>(palette.columns.capacity(), "JP2 palette columns")?,
            "JP2 facade metadata",
        )?;
        checked_add_bytes(
            &mut bytes,
            capacity_bytes::<Vec<u64>>(palette.entries.capacity(), "JP2 palette rows")?,
            "JP2 facade metadata",
        )?;
        for row in &palette.entries {
            checked_add_bytes(
                &mut bytes,
                capacity_bytes::<u64>(row.capacity(), "JP2 palette entries")?,
                "JP2 facade metadata",
            )?;
        }
    }
    checked_add_bytes(
        &mut bytes,
        capacity_bytes::<J2kComponentMapping>(
            metadata.component_mappings.capacity(),
            "JP2 component mappings",
        )?,
        "JP2 facade metadata",
    )?;
    checked_add_bytes(
        &mut bytes,
        capacity_bytes::<J2kChannelDefinition>(
            metadata.channel_definitions.capacity(),
            "JP2 channel definitions",
        )?,
        "JP2 facade metadata",
    )?;
    Ok(bytes)
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
    let source = resolved_component_source(metadata, siz);
    let resolved_count = resolved_component_count(source, metadata, siz);
    if resolved_count != usize::from(ihdr.components) {
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
        for index in 0..resolved_count {
            let component = resolved_component_at(source, metadata, siz, index).ok_or(
                J2kError::InvalidBox {
                    offset: ihdr.offset,
                    what: "JP2 component metadata could not be resolved",
                },
            )?;
            if !same_component_precision(component, descriptor) {
                return Err(J2kError::InvalidBox {
                    offset: ihdr.offset,
                    what: "ihdr bpc must match resolved JP2 image component precision",
                });
            }
        }
    } else {
        if metadata.bits_per_component.len() != usize::from(ihdr.components) {
            return Err(J2kError::InvalidBox {
                offset: ihdr.offset,
                what: "bpcc component count must match ihdr component count",
            });
        }
        for (index, descriptor) in metadata.bits_per_component.iter().enumerate() {
            let component = resolved_component_at(source, metadata, siz, index).ok_or(
                J2kError::InvalidBox {
                    offset: ihdr.offset,
                    what: "JP2 component metadata could not be resolved",
                },
            )?;
            if !same_component_precision(component, *descriptor) {
                return Err(J2kError::InvalidBox {
                    offset: ihdr.offset,
                    what: "bpcc entries must match resolved JP2 image component precision",
                });
            }
        }
    }

    Ok(())
}

#[derive(Clone, Copy)]
enum ResolvedComponentSource {
    Codestream,
    Palette,
    Mappings,
}

fn resolved_component_source(
    metadata: &J2kFileMetadata,
    siz: &super::ParsedSiz,
) -> ResolvedComponentSource {
    if metadata.component_mappings.is_empty() {
        if metadata.palette.is_some() {
            return ResolvedComponentSource::Palette;
        }
        return ResolvedComponentSource::Codestream;
    }

    let resolvable = metadata
        .component_mappings
        .iter()
        .all(|mapping| match mapping.mapping_type {
            J2kComponentMappingType::Direct => siz
                .component_info
                .get(usize::from(mapping.component_index))
                .is_some(),
            J2kComponentMappingType::Palette { column } => metadata
                .palette
                .as_ref()
                .and_then(|palette| palette.columns.get(usize::from(column)))
                .is_some(),
            J2kComponentMappingType::Unknown { .. } => false,
        });
    if resolvable {
        ResolvedComponentSource::Mappings
    } else {
        ResolvedComponentSource::Codestream
    }
}

fn resolved_component_count(
    source: ResolvedComponentSource,
    metadata: &J2kFileMetadata,
    siz: &super::ParsedSiz,
) -> usize {
    match source {
        ResolvedComponentSource::Codestream => siz.component_info.len(),
        ResolvedComponentSource::Palette => metadata
            .palette
            .as_ref()
            .map_or(0, |palette| palette.columns.len()),
        ResolvedComponentSource::Mappings => metadata.component_mappings.len(),
    }
}

fn resolved_component_at(
    source: ResolvedComponentSource,
    metadata: &J2kFileMetadata,
    siz: &super::ParsedSiz,
    index: usize,
) -> Option<J2kComponentInfo> {
    match source {
        ResolvedComponentSource::Codestream => siz.component_info.get(index).copied(),
        ResolvedComponentSource::Palette => metadata
            .palette
            .as_ref()?
            .columns
            .get(index)
            .copied()
            .map(component_from_palette_column),
        ResolvedComponentSource::Mappings => {
            let mapping = metadata.component_mappings.get(index)?;
            match mapping.mapping_type {
                J2kComponentMappingType::Direct => siz
                    .component_info
                    .get(usize::from(mapping.component_index))
                    .copied(),
                J2kComponentMappingType::Palette { column } => metadata
                    .palette
                    .as_ref()?
                    .columns
                    .get(usize::from(column))
                    .copied()
                    .map(component_from_palette_column),
                J2kComponentMappingType::Unknown { .. } => None,
            }
        }
    }
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
        source @ (NativeDecodeError::AllocationTooLarge { .. }
        | NativeDecodeError::HostAllocationFailed { .. }) => J2kError::NativeDecode {
            context: "JP2/JPH metadata inspection failed",
            source: crate::NativeBackendError::decode(source),
        },
        _ => J2kError::InvalidBox {
            offset: 0,
            what: "invalid JP2/JPH container",
        },
    }
}
