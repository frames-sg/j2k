// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use j2k_core::{Colorspace, CompressedPayloadKind, CompressedTransferSyntax, Unsupported};

use crate::{
    parse::parse_image_info, J2kChannelAssociation, J2kChannelDefinition, J2kChannelType,
    J2kColorSpec, J2kComponentMapping, J2kComponentMappingType, J2kError, J2kFileMetadata,
    J2kPaletteColumn, J2kPaletteMetadata,
};

const JP2_SIGNATURE_PAYLOAD: &[u8; 4] = &[0x0d, 0x0a, 0x87, 0x0a];
const JP2_BRAND: [u8; 4] = *b"jp2 ";
const JPH_BRAND: [u8; 4] = *b"jph ";
const JP2_COMPRESSION_TYPE: u8 = 7;

/// Color metadata to write into a JP2/JPH Colour Specification box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum J2kFileColorSpec<'a> {
    /// Infer an enumerated JP2 colorspace from the codestream component count
    /// and parsed JPEG 2000 color transform metadata.
    Infer,
    /// Write an enumerated JP2 colorspace.
    Enumerated(Colorspace),
    /// Write an ICC-profile Colour Specification box.
    IccProfile(&'a [u8]),
}

impl<'a> J2kFileColorSpec<'a> {
    /// Return a directly representable JP2/JPH colour specification borrowed
    /// from inspected file metadata.
    #[must_use]
    pub fn from_inspected(color_spec: &'a J2kColorSpec) -> Option<Self> {
        file_color_spec_from_inspected_colr(color_spec)
    }

    /// Return the first directly representable JP2/JPH colour specification
    /// from inspected file metadata.
    #[must_use]
    pub fn from_file_metadata(metadata: &'a J2kFileMetadata) -> Option<Self> {
        metadata.color_specs.iter().find_map(Self::from_inspected)
    }
}

/// Optional JP2/JPH metadata boxes to write in the JP2 Header box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kFileBoxMetadata<'a> {
    /// Palette box metadata to write as `pclr`.
    pub palette: Option<&'a J2kPaletteMetadata>,
    /// Component Mapping box entries to write as `cmap`.
    ///
    /// When [`Self::palette`] is present and this slice is empty, the writer
    /// emits standard palette mappings from codestream component 0 to each
    /// palette column.
    pub component_mappings: &'a [J2kComponentMapping],
    /// Channel Definition box entries to write as `cdef`.
    pub channel_definitions: &'a [J2kChannelDefinition],
}

impl J2kFileBoxMetadata<'_> {
    /// Empty JP2/JPH metadata-box selection.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            palette: None,
            component_mappings: &[],
            channel_definitions: &[],
        }
    }
}

impl<'a> J2kFileBoxMetadata<'a> {
    /// Borrow directly representable JP2/JPH metadata boxes from inspected
    /// file metadata for rewrapping.
    #[must_use]
    pub fn from_file_metadata(metadata: &'a J2kFileMetadata) -> Self {
        Self {
            palette: metadata.palette.as_ref(),
            component_mappings: &metadata.component_mappings,
            channel_definitions: &metadata.channel_definitions,
        }
    }
}

/// Options for wrapping a raw JPEG 2000 / HTJ2K codestream as a JP2/JPH file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kFileWrapOptions<'a> {
    /// Desired file wrapper.
    pub payload_kind: CompressedPayloadKind,
    /// Color metadata to place in the JP2 Header box.
    pub color: J2kFileColorSpec<'a>,
    /// Explicit Colour Specification boxes to place in the JP2 Header box.
    ///
    /// When non-empty, these are written in order and [`Self::color`] is used
    /// only for legacy single-COLR callers.
    pub color_specs: &'a [J2kFileColorSpec<'a>],
    /// Optional JP2/JPH file metadata boxes to place in the JP2 Header box.
    pub metadata: J2kFileBoxMetadata<'a>,
}

impl J2kFileWrapOptions<'_> {
    /// Create standard JP2 wrapper options.
    #[must_use]
    pub const fn jp2() -> Self {
        Self {
            payload_kind: CompressedPayloadKind::Jp2File,
            color: J2kFileColorSpec::Infer,
            color_specs: &[],
            metadata: J2kFileBoxMetadata::empty(),
        }
    }

    /// Create standard JPH wrapper options for an HTJ2K codestream.
    #[must_use]
    pub const fn jph() -> Self {
        Self {
            payload_kind: CompressedPayloadKind::JphFile,
            color: J2kFileColorSpec::Infer,
            color_specs: &[],
            metadata: J2kFileBoxMetadata::empty(),
        }
    }
}

impl<'a> J2kFileWrapOptions<'a> {
    /// Return options with explicit color metadata.
    #[must_use]
    pub const fn with_color(mut self, color: J2kFileColorSpec<'a>) -> Self {
        self.color = color;
        self.color_specs = &[];
        self
    }

    /// Return options with explicit ordered JP2/JPH Colour Specification boxes.
    #[must_use]
    pub const fn with_color_specs(mut self, color_specs: &'a [J2kFileColorSpec<'a>]) -> Self {
        self.color_specs = color_specs;
        self
    }

    /// Return options with explicit JP2/JPH metadata boxes.
    #[must_use]
    pub const fn with_metadata(mut self, metadata: J2kFileBoxMetadata<'a>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Wrap a raw JPEG 2000 / HTJ2K codestream in a JP2 or JPH file container.
///
/// The codestream is inspected with the existing parser before any boxes are
/// written. Component precision, signedness, and sampling metadata come from
/// the codestream SIZ marker. Mixed precision or mixed signedness is emitted via
/// the JP2 `bpcc` box.
pub fn wrap_j2k_codestream(
    codestream: &[u8],
    options: J2kFileWrapOptions<'_>,
) -> Result<Vec<u8>, J2kError> {
    let parsed = parse_image_info(codestream)?;
    if parsed.payload_kind != CompressedPayloadKind::Jpeg2000Codestream {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH wrapping expects a raw JPEG 2000 codestream",
        }));
    }
    validate_wrapper_kind(options.payload_kind, parsed.transfer_syntax)?;

    let brand = match options.payload_kind {
        CompressedPayloadKind::Jp2File => JP2_BRAND,
        CompressedPayloadKind::JphFile => JPH_BRAND,
        _ => {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JP2/JPH wrapping requires Jp2File or JphFile output",
            }))
        }
    };

    let mut output = Vec::new();
    push_box(&mut output, b"jP  ", JP2_SIGNATURE_PAYLOAD)?;
    push_box(&mut output, b"ftyp", &file_type_payload(brand))?;
    let jp2h = jp2_header_payload(
        &parsed,
        options.color,
        options.color_specs,
        options.metadata,
    )?;
    push_box(&mut output, b"jp2h", &jp2h)?;
    push_box(&mut output, b"jp2c", codestream)?;
    Ok(output)
}

fn validate_wrapper_kind(
    payload_kind: CompressedPayloadKind,
    transfer_syntax: CompressedTransferSyntax,
) -> Result<(), J2kError> {
    let htj2k = matches!(
        transfer_syntax,
        CompressedTransferSyntax::HtJpeg2000Lossless | CompressedTransferSyntax::HtJpeg2000Lossy
    );
    match (payload_kind, htj2k) {
        (CompressedPayloadKind::Jp2File, false) | (CompressedPayloadKind::JphFile, true) => Ok(()),
        (CompressedPayloadKind::Jp2File, true) => Err(J2kError::Unsupported(Unsupported {
            what: "HTJ2K codestreams should be wrapped as JPH files",
        })),
        (CompressedPayloadKind::JphFile, false) => Err(J2kError::Unsupported(Unsupported {
            what: "JPH wrapping requires an HTJ2K codestream",
        })),
        _ => Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH wrapping requires Jp2File or JphFile output",
        })),
    }
}

fn file_type_payload(brand: [u8; 4]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(12);
    payload.extend_from_slice(&brand);
    payload.extend_from_slice(&0_u32.to_be_bytes());
    payload.extend_from_slice(&brand);
    payload
}

fn jp2_header_payload(
    parsed: &crate::parse::ParsedImageInfo,
    color: J2kFileColorSpec<'_>,
    color_specs: &[J2kFileColorSpec<'_>],
    metadata: J2kFileBoxMetadata<'_>,
) -> Result<Vec<u8>, J2kError> {
    let mut payload = Vec::new();
    push_box(
        &mut payload,
        b"ihdr",
        &image_header_payload(parsed, metadata)?,
    )?;
    if uses_bpcc(parsed, metadata)? {
        push_box(
            &mut payload,
            b"bpcc",
            &bits_per_component_payload(parsed, metadata)?,
        )?;
    }
    if color_specs.is_empty() {
        push_box(&mut payload, b"colr", &color_spec_payload(parsed, color)?)?;
    } else {
        for color_spec in color_specs {
            push_box(
                &mut payload,
                b"colr",
                &color_spec_payload(parsed, *color_spec)?,
            )?;
        }
    }

    if let Some(palette) = metadata.palette {
        push_box(&mut payload, b"pclr", &palette_payload(palette)?)?;
    }
    if metadata.palette.is_some() || !metadata.component_mappings.is_empty() {
        push_box(
            &mut payload,
            b"cmap",
            &component_mapping_payload(parsed, metadata)?,
        )?;
    }
    if !metadata.channel_definitions.is_empty() {
        push_box(
            &mut payload,
            b"cdef",
            &channel_definition_payload(metadata.channel_definitions)?,
        )?;
    } else if should_write_srgb_alpha_cdef(parsed, color_specs.first().copied().unwrap_or(color)) {
        push_box(&mut payload, b"cdef", &rgba_channel_definition_payload())?;
    }
    Ok(payload)
}

fn image_header_payload(
    parsed: &crate::parse::ParsedImageInfo,
    metadata: J2kFileBoxMetadata<'_>,
) -> Result<Vec<u8>, J2kError> {
    let components = resolved_file_components(parsed, metadata)?;
    let component_count = u16::try_from(components.len()).map_err(|_| {
        J2kError::Unsupported(Unsupported {
            what: "JP2/JPH resolved image component count exceeds u16",
        })
    })?;
    let mut payload = Vec::with_capacity(14);
    let (width, height) = parsed.info.dimensions;
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.extend_from_slice(&component_count.to_be_bytes());
    payload.push(image_header_bpc(&components));
    payload.push(JP2_COMPRESSION_TYPE);
    payload.push(0);
    payload.push(0);
    Ok(payload)
}

fn image_header_bpc(components: &[crate::J2kComponentInfo]) -> u8 {
    if components_use_bpcc(components) {
        return 0xff;
    }
    components.first().copied().map_or(0xff, component_bpc)
}

fn uses_bpcc(
    parsed: &crate::parse::ParsedImageInfo,
    metadata: J2kFileBoxMetadata<'_>,
) -> Result<bool, J2kError> {
    Ok(components_use_bpcc(&resolved_file_components(
        parsed, metadata,
    )?))
}

fn components_use_bpcc(components: &[crate::J2kComponentInfo]) -> bool {
    let Some(first) = components.first() else {
        return false;
    };
    components
        .iter()
        .any(|component| component.bit_depth != first.bit_depth || component.signed != first.signed)
}

fn bits_per_component_payload(
    parsed: &crate::parse::ParsedImageInfo,
    metadata: J2kFileBoxMetadata<'_>,
) -> Result<Vec<u8>, J2kError> {
    Ok(resolved_file_components(parsed, metadata)?
        .iter()
        .copied()
        .map(component_bpc)
        .collect())
}

fn component_bpc(component: crate::J2kComponentInfo) -> u8 {
    let precision = component.bit_depth.saturating_sub(1) & 0x7f;
    precision | if component.signed { 0x80 } else { 0 }
}

fn resolved_file_components(
    parsed: &crate::parse::ParsedImageInfo,
    metadata: J2kFileBoxMetadata<'_>,
) -> Result<Vec<crate::J2kComponentInfo>, J2kError> {
    if metadata.component_mappings.is_empty() {
        if let Some(palette) = metadata.palette {
            return Ok(palette
                .columns
                .iter()
                .copied()
                .map(component_from_palette_column)
                .collect());
        }
        return Ok(parsed.components.clone());
    }

    let mut components = Vec::with_capacity(metadata.component_mappings.len());
    for mapping in metadata.component_mappings {
        validate_component_mapping(parsed, metadata.palette, *mapping)?;
        match mapping.mapping_type {
            J2kComponentMappingType::Direct => {
                components.push(parsed.components[mapping.component_index as usize]);
            }
            J2kComponentMappingType::Palette { column } => {
                let palette = metadata.palette.ok_or(J2kError::Unsupported(Unsupported {
                    what: "JP2/JPH palette component mapping requires a palette box",
                }))?;
                components.push(component_from_palette_column(
                    palette.columns[column as usize],
                ));
            }
            J2kComponentMappingType::Unknown { .. } => {
                return Err(J2kError::Unsupported(Unsupported {
                    what:
                        "JP2/JPH unknown component mappings cannot define image component precision",
                }))
            }
        }
    }
    Ok(components)
}

fn component_from_palette_column(column: J2kPaletteColumn) -> crate::J2kComponentInfo {
    crate::J2kComponentInfo {
        bit_depth: column.bit_depth,
        signed: column.signed,
        x_rsiz: 1,
        y_rsiz: 1,
    }
}

fn color_spec_payload(
    parsed: &crate::parse::ParsedImageInfo,
    color: J2kFileColorSpec<'_>,
) -> Result<Vec<u8>, J2kError> {
    let mut payload = Vec::new();
    match color {
        J2kFileColorSpec::Infer => push_enumerated_colr(
            &mut payload,
            inferred_enumerated_colorspace(parsed.info.components, parsed.info.colorspace)?,
        ),
        J2kFileColorSpec::Enumerated(colorspace) => {
            push_enumerated_colr(&mut payload, enumerated_colorspace_code(colorspace)?);
        }
        J2kFileColorSpec::IccProfile(profile) => {
            payload.extend_from_slice(&[2, 0, 0]);
            payload.extend_from_slice(profile);
        }
    }
    Ok(payload)
}

fn push_enumerated_colr(payload: &mut Vec<u8>, colorspace_code: u32) {
    payload.extend_from_slice(&[1, 0, 0]);
    payload.extend_from_slice(&colorspace_code.to_be_bytes());
}

fn inferred_enumerated_colorspace(
    components: u16,
    colorspace: Colorspace,
) -> Result<u32, J2kError> {
    match colorspace {
        Colorspace::Grayscale | Colorspace::SGray => Ok(17),
        Colorspace::YCbCr => Ok(18),
        Colorspace::Rgb | Colorspace::SRgb | Colorspace::Rct | Colorspace::Ict => Ok(16),
        Colorspace::IccTagged if components == 1 => Ok(17),
        Colorspace::IccTagged if components == 3 || components == 4 => Ok(16),
        _ => Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH wrapping for this colorspace requires an ICC profile",
        })),
    }
}

fn enumerated_colorspace_code(colorspace: Colorspace) -> Result<u32, J2kError> {
    match colorspace {
        Colorspace::Grayscale | Colorspace::SGray => Ok(17),
        Colorspace::YCbCr => Ok(18),
        Colorspace::Rgb | Colorspace::SRgb | Colorspace::Rct | Colorspace::Ict => Ok(16),
        _ => Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH enumerated colorspace must be sRGB, sGray, or YCbCr",
        })),
    }
}

fn file_color_spec_from_inspected_colr(color_spec: &J2kColorSpec) -> Option<J2kFileColorSpec<'_>> {
    match color_spec {
        J2kColorSpec::Enumerated { value } => {
            inspected_enumerated_colorspace(*value).map(J2kFileColorSpec::Enumerated)
        }
        J2kColorSpec::IccProfile { profile } => Some(J2kFileColorSpec::IccProfile(profile)),
        J2kColorSpec::Unknown { .. } => None,
    }
}

fn inspected_enumerated_colorspace(value: u32) -> Option<Colorspace> {
    match value {
        16 => Some(Colorspace::SRgb),
        17 => Some(Colorspace::SGray),
        18 => Some(Colorspace::YCbCr),
        _ => None,
    }
}

fn should_write_srgb_alpha_cdef(
    parsed: &crate::parse::ParsedImageInfo,
    color: J2kFileColorSpec<'_>,
) -> bool {
    if parsed.info.components != 4 {
        return false;
    }
    match color {
        J2kFileColorSpec::Infer => matches!(
            parsed.info.colorspace,
            Colorspace::Rgb | Colorspace::SRgb | Colorspace::Rct | Colorspace::Ict
        ),
        J2kFileColorSpec::Enumerated(colorspace) => matches!(
            colorspace,
            Colorspace::Rgb | Colorspace::SRgb | Colorspace::Rct | Colorspace::Ict
        ),
        J2kFileColorSpec::IccProfile(_) => false,
    }
}

fn rgba_channel_definition_payload() -> Vec<u8> {
    let mut payload = Vec::with_capacity(2 + 4 * 6);
    payload.extend_from_slice(&4_u16.to_be_bytes());
    for (channel, channel_type, association) in [
        (0_u16, 0_u16, 1_u16),
        (1_u16, 0_u16, 2_u16),
        (2_u16, 0_u16, 3_u16),
        (3_u16, 1_u16, 0_u16),
    ] {
        payload.extend_from_slice(&channel.to_be_bytes());
        payload.extend_from_slice(&channel_type.to_be_bytes());
        payload.extend_from_slice(&association.to_be_bytes());
    }
    payload
}

fn palette_payload(palette: &J2kPaletteMetadata) -> Result<Vec<u8>, J2kError> {
    if palette.entries.is_empty() || palette.columns.is_empty() {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH palette metadata requires non-empty entries and columns",
        }));
    }
    let entry_count = u16::try_from(palette.entries.len()).map_err(|_| {
        J2kError::Unsupported(Unsupported {
            what: "JP2/JPH palette entry count exceeds u16",
        })
    })?;
    let column_count = u8::try_from(palette.columns.len()).map_err(|_| {
        J2kError::Unsupported(Unsupported {
            what: "JP2/JPH palette column count exceeds u8",
        })
    })?;

    let mut payload = Vec::new();
    payload.extend_from_slice(&entry_count.to_be_bytes());
    payload.push(column_count);
    for column in &palette.columns {
        payload.push(palette_column_bpc(*column)?);
    }
    for row in &palette.entries {
        if row.len() != palette.columns.len() {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JP2/JPH palette rows must match palette column count",
            }));
        }
        for (&value, column) in row.iter().zip(palette.columns.iter()) {
            push_palette_value(&mut payload, value, *column)?;
        }
    }
    Ok(payload)
}

fn palette_column_bpc(column: J2kPaletteColumn) -> Result<u8, J2kError> {
    if !(1..=38).contains(&column.bit_depth) {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH palette column precision must be 1-38 bits",
        }));
    }
    Ok((column.bit_depth - 1) | if column.signed { 0x80 } else { 0 })
}

fn push_palette_value(
    out: &mut Vec<u8>,
    value: u64,
    column: J2kPaletteColumn,
) -> Result<(), J2kError> {
    let max_value = (1_u64 << column.bit_depth) - 1;
    if value > max_value {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH palette entry exceeds column precision",
        }));
    }
    let bytes = usize::from(column.bit_depth).div_ceil(8).max(1);
    for shift in (0..bytes).rev().map(|byte| byte * 8) {
        out.push(((value >> shift) & 0xff) as u8);
    }
    Ok(())
}

fn component_mapping_payload(
    parsed: &crate::parse::ParsedImageInfo,
    metadata: J2kFileBoxMetadata<'_>,
) -> Result<Vec<u8>, J2kError> {
    if metadata.component_mappings.is_empty() {
        let Some(palette) = metadata.palette else {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JP2/JPH component mapping metadata is empty",
            }));
        };
        let mut payload = Vec::new();
        for column in 0..palette.columns.len() {
            payload.extend_from_slice(&0_u16.to_be_bytes());
            payload.push(1);
            payload.push(u8::try_from(column).map_err(|_| {
                J2kError::Unsupported(Unsupported {
                    what: "JP2/JPH palette column count exceeds cmap field width",
                })
            })?);
        }
        return Ok(payload);
    }

    let mut payload = Vec::with_capacity(metadata.component_mappings.len() * 4);
    for mapping in metadata.component_mappings {
        validate_component_mapping(parsed, metadata.palette, *mapping)?;
        payload.extend_from_slice(&mapping.component_index.to_be_bytes());
        match mapping.mapping_type {
            J2kComponentMappingType::Direct => {
                payload.push(0);
                payload.push(0);
            }
            J2kComponentMappingType::Palette { column } => {
                payload.push(1);
                payload.push(column);
            }
            J2kComponentMappingType::Unknown { value, column } => {
                payload.push(value);
                payload.push(column);
            }
        }
    }
    Ok(payload)
}

fn validate_component_mapping(
    parsed: &crate::parse::ParsedImageInfo,
    palette: Option<&J2kPaletteMetadata>,
    mapping: J2kComponentMapping,
) -> Result<(), J2kError> {
    if mapping.component_index >= parsed.info.components {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH component mapping references a missing codestream component",
        }));
    }
    if let J2kComponentMappingType::Palette { column } = mapping.mapping_type {
        let Some(palette) = palette else {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JP2/JPH palette component mapping requires a palette box",
            }));
        };
        if usize::from(column) >= palette.columns.len() {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JP2/JPH palette component mapping references a missing palette column",
            }));
        }
    }
    Ok(())
}

fn channel_definition_payload(definitions: &[J2kChannelDefinition]) -> Result<Vec<u8>, J2kError> {
    if definitions.is_empty() {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH channel definition metadata is empty",
        }));
    }
    let count = u16::try_from(definitions.len()).map_err(|_| {
        J2kError::Unsupported(Unsupported {
            what: "JP2/JPH channel definition count exceeds u16",
        })
    })?;
    let mut payload = Vec::with_capacity(2 + definitions.len() * 6);
    payload.extend_from_slice(&count.to_be_bytes());
    for definition in definitions {
        payload.extend_from_slice(&definition.channel_index.to_be_bytes());
        payload.extend_from_slice(&raw_channel_type(definition.channel_type).to_be_bytes());
        payload.extend_from_slice(&raw_channel_association(definition.association).to_be_bytes());
    }
    Ok(payload)
}

fn raw_channel_type(channel_type: J2kChannelType) -> u16 {
    match channel_type {
        J2kChannelType::Color => 0,
        J2kChannelType::Opacity => 1,
        J2kChannelType::PremultipliedOpacity => 2,
        J2kChannelType::Unspecified => u16::MAX,
        J2kChannelType::Unknown { value } => value,
    }
}

fn raw_channel_association(association: J2kChannelAssociation) -> u16 {
    match association {
        J2kChannelAssociation::WholeImage => 0,
        J2kChannelAssociation::Color { index } => index,
        J2kChannelAssociation::Unspecified => u16::MAX,
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn push_box(out: &mut Vec<u8>, box_type: &[u8; 4], payload: &[u8]) -> Result<(), J2kError> {
    let len = payload.len().checked_add(8).ok_or(J2kError::InvalidBox {
        offset: out.len(),
        what: "box length overflow",
    })?;
    let len = u32::try_from(len).map_err(|_| {
        J2kError::Unsupported(Unsupported {
            what: "JP2/JPH box payload exceeds 32-bit box length",
        })
    })?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(box_type);
    out.extend_from_slice(payload);
    Ok(())
}
