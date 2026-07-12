// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free JP2/JPH component and optional metadata-box validation.

use crate::{
    parse::ParsedImageInfo, J2kChannelDefinition, J2kComponentInfo, J2kComponentMapping,
    J2kComponentMappingType, J2kError, J2kFileBoxMetadata, J2kPaletteColumn, J2kPaletteMetadata,
};
use j2k_core::{BufferError, Unsupported};

const RGBA_CDEF_PAYLOAD_LEN: usize = 2 + 4 * 6;

#[derive(Clone, Copy)]
pub(super) enum ChannelDefinitionPlan<'a> {
    None,
    Explicit(&'a [J2kChannelDefinition]),
    Rgba,
}

#[derive(Clone, Copy)]
pub(super) enum ResolvedComponents<'a> {
    Codestream(&'a [J2kComponentInfo]),
    Palette(&'a [J2kPaletteColumn]),
    Mappings {
        codestream: &'a [J2kComponentInfo],
        palette: Option<&'a J2kPaletteMetadata>,
        mappings: &'a [J2kComponentMapping],
    },
}

impl<'a> ChannelDefinitionPlan<'a> {
    pub(super) fn new(
        definitions: &'a [J2kChannelDefinition],
        write_rgba: bool,
    ) -> Result<Self, J2kError> {
        if !definitions.is_empty() {
            u16::try_from(definitions.len()).map_err(|_| {
                J2kError::Unsupported(Unsupported {
                    what: "JP2/JPH channel definition count exceeds u16",
                })
            })?;
            return Ok(Self::Explicit(definitions));
        }
        Ok(if write_rgba { Self::Rgba } else { Self::None })
    }

    pub(super) fn payload_len(self) -> Result<Option<usize>, J2kError> {
        match self {
            Self::None => Ok(None),
            Self::Explicit(definitions) => definitions
                .len()
                .checked_mul(6)
                .and_then(|bytes| bytes.checked_add(2))
                .map(Some)
                .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
                    what: "JP2/JPH channel definition payload",
                })),
            Self::Rgba => Ok(Some(RGBA_CDEF_PAYLOAD_LEN)),
        }
    }
}

impl<'a> ResolvedComponents<'a> {
    pub(super) fn new(
        parsed: &'a ParsedImageInfo,
        metadata: J2kFileBoxMetadata<'a>,
    ) -> Result<Self, J2kError> {
        if metadata.component_mappings.is_empty() {
            return Ok(metadata
                .palette
                .map_or(Self::Codestream(&parsed.components), |palette| {
                    Self::Palette(&palette.columns)
                }));
        }
        for &mapping in metadata.component_mappings {
            validate_component_mapping(parsed, metadata.palette, mapping)?;
            if matches!(
                mapping.mapping_type,
                J2kComponentMappingType::Unknown { .. }
            ) {
                return Err(J2kError::Unsupported(Unsupported {
                    what:
                        "JP2/JPH unknown component mappings cannot define image component precision",
                }));
            }
        }
        Ok(Self::Mappings {
            codestream: &parsed.components,
            palette: metadata.palette,
            mappings: metadata.component_mappings,
        })
    }

    pub(super) fn len(self) -> usize {
        match self {
            Self::Codestream(components) => components.len(),
            Self::Palette(columns) => columns.len(),
            Self::Mappings { mappings, .. } => mappings.len(),
        }
    }

    pub(super) fn component(self, index: usize) -> Option<J2kComponentInfo> {
        match self {
            Self::Codestream(components) => components.get(index).copied(),
            Self::Palette(columns) => columns.get(index).copied().map(component_from_palette),
            Self::Mappings {
                codestream,
                palette,
                mappings,
            } => {
                let mapping = mappings.get(index)?;
                match mapping.mapping_type {
                    J2kComponentMappingType::Direct => codestream
                        .get(usize::from(mapping.component_index))
                        .copied(),
                    J2kComponentMappingType::Palette { column } => palette
                        .and_then(|palette| palette.columns.get(usize::from(column)))
                        .copied()
                        .map(component_from_palette),
                    J2kComponentMappingType::Unknown { .. } => None,
                }
            }
        }
    }

    pub(super) fn uses_bpcc(self) -> Result<bool, J2kError> {
        let Some(first) = self.component(0) else {
            return Ok(false);
        };
        for index in 1..self.len() {
            let component = self.component(index).ok_or(J2kError::InternalInvariant {
                what: "validated JP2/JPH component mapping became unresolved",
            })?;
            if component.bit_depth != first.bit_depth || component.signed != first.signed {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(super) fn validate_precisions(self) -> Result<(), J2kError> {
        for index in 0..self.len() {
            let component = self.component(index).ok_or(J2kError::InternalInvariant {
                what: "validated JP2/JPH component became unresolved",
            })?;
            if !(1..=38).contains(&component.bit_depth) {
                return Err(J2kError::Unsupported(Unsupported {
                    what: "JP2/JPH component precision must be 1-38 bits",
                }));
            }
        }
        Ok(())
    }
}

pub(super) fn validate_palette(palette: &J2kPaletteMetadata) -> Result<usize, J2kError> {
    if palette.entries.is_empty() || palette.columns.is_empty() {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH palette metadata requires non-empty entries and columns",
        }));
    }
    u16::try_from(palette.entries.len()).map_err(|_| {
        J2kError::Unsupported(Unsupported {
            what: "JP2/JPH palette entry count exceeds u16",
        })
    })?;
    u8::try_from(palette.columns.len()).map_err(|_| {
        J2kError::Unsupported(Unsupported {
            what: "JP2/JPH palette column count exceeds u8",
        })
    })?;

    let mut row_bytes = 0_usize;
    for &column in &palette.columns {
        validate_palette_column(column)?;
        row_bytes = row_bytes
            .checked_add(usize::from(column.bit_depth).div_ceil(8).max(1))
            .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
                what: "JP2/JPH palette row",
            }))?;
    }
    for row in &palette.entries {
        if row.len() != palette.columns.len() {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JP2/JPH palette rows must match palette column count",
            }));
        }
        for (&value, &column) in row.iter().zip(&palette.columns) {
            let max_value = (1_u64 << column.bit_depth) - 1;
            if value > max_value {
                return Err(J2kError::Unsupported(Unsupported {
                    what: "JP2/JPH palette entry exceeds column precision",
                }));
            }
        }
    }
    let entries_bytes = row_bytes
        .checked_mul(palette.entries.len())
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
            what: "JP2/JPH palette entries",
        }))?;
    3_usize
        .checked_add(palette.columns.len())
        .and_then(|bytes| bytes.checked_add(entries_bytes))
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
            what: "JP2/JPH palette payload",
        }))
}

pub(super) fn component_mapping_payload_len(
    parsed: &ParsedImageInfo,
    metadata: J2kFileBoxMetadata<'_>,
) -> Result<Option<usize>, J2kError> {
    if metadata.palette.is_none() && metadata.component_mappings.is_empty() {
        return Ok(None);
    }
    let count = if metadata.component_mappings.is_empty() {
        metadata.palette.map_or(0, |palette| palette.columns.len())
    } else {
        for &mapping in metadata.component_mappings {
            validate_component_mapping(parsed, metadata.palette, mapping)?;
        }
        metadata.component_mappings.len()
    };
    count
        .checked_mul(4)
        .map(Some)
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
            what: "JP2/JPH component mapping payload",
        }))
}

fn validate_component_mapping(
    parsed: &ParsedImageInfo,
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

fn validate_palette_column(column: J2kPaletteColumn) -> Result<(), J2kError> {
    if !(1..=38).contains(&column.bit_depth) {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH palette column precision must be 1-38 bits",
        }));
    }
    Ok(())
}

fn component_from_palette(column: J2kPaletteColumn) -> J2kComponentInfo {
    J2kComponentInfo {
        bit_depth: column.bit_depth,
        signed: column.signed,
        x_rsiz: 1,
        y_rsiz: 1,
    }
}

pub(super) fn component_bpc(component: J2kComponentInfo) -> u8 {
    let precision = component.bit_depth - 1;
    precision | if component.signed { 0x80 } else { 0 }
}
