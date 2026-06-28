// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use j2k_core::{CompressedPayloadKind, CompressedTransferSyntax, Info};

/// Per-component JPEG 2000 SIZ metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kComponentInfo {
    /// Significant bits in this component.
    pub bit_depth: u8,
    /// Whether this component stores signed sample values.
    pub signed: bool,
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
}

/// Full parsed JPEG 2000 / HTJ2K support metadata.
///
/// This preserves the existing compact [`Info`] summary while exposing fields
/// needed to reason about full Part 1 / Part 15 support surfaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kSupportInfo {
    /// Backward-compatible metadata summary used by shared codec traits.
    pub info: Info,
    /// Parsed compressed transfer syntax.
    pub transfer_syntax: CompressedTransferSyntax,
    /// Parsed raw codestream or file-wrapper shape.
    pub payload_kind: CompressedPayloadKind,
    /// Exact per-component SIZ metadata in codestream order.
    pub components: Vec<J2kComponentInfo>,
    /// JP2/JPH file-wrapper metadata when the input is a still-image file.
    pub file_metadata: Option<J2kFileMetadata>,
}

impl J2kSupportInfo {
    /// Return the exact codestream component count.
    #[must_use]
    pub fn component_count(&self) -> u16 {
        self.info.components
    }

    /// Return whether any component uses signed sample values.
    #[must_use]
    pub fn has_signed_components(&self) -> bool {
        self.components.iter().any(|component| component.signed)
    }

    /// Return whether the codestream uses mixed component precision.
    #[must_use]
    pub fn has_mixed_bit_depths(&self) -> bool {
        let Some(first) = self.components.first() else {
            return false;
        };
        self.components
            .iter()
            .any(|component| component.bit_depth != first.bit_depth)
    }

    /// Return whether component sampling factors differ from full resolution.
    #[must_use]
    pub fn has_component_subsampling(&self) -> bool {
        self.components
            .iter()
            .any(|component| component.x_rsiz != 1 || component.y_rsiz != 1)
    }
}

/// JP2/JPH file-wrapper metadata preserved by public inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kFileMetadata {
    /// Bits-per-component box entries, when BPCC is present.
    pub bits_per_component: Vec<J2kComponentInfo>,
    /// Colour Specification boxes in file order.
    pub color_specs: Vec<J2kColorSpec>,
    /// Palette box metadata, when PCLR is present.
    pub palette: Option<J2kPaletteMetadata>,
    /// Component Mapping box entries in file order.
    pub component_mappings: Vec<J2kComponentMapping>,
    /// Channel Definition box entries in file order.
    pub channel_definitions: Vec<J2kChannelDefinition>,
    /// Whether a Palette box is present.
    pub has_palette: bool,
    /// Whether a Component Mapping box is present.
    pub has_component_mapping: bool,
    /// Whether a Channel Definition box is present.
    pub has_channel_definition: bool,
}

impl J2kFileMetadata {
    /// Return whether the wrapper preserves an ICC profile in any COLR box.
    #[must_use]
    pub fn has_icc_profile(&self) -> bool {
        self.color_specs
            .iter()
            .any(|color_spec| matches!(color_spec, J2kColorSpec::IccProfile { .. }))
    }
}

/// Parsed JP2/JPH Palette box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kPaletteMetadata {
    /// Palette column descriptors in box order.
    pub columns: Vec<J2kPaletteColumn>,
    /// Palette entries in row-major order: entry, then column.
    pub entries: Vec<Vec<u64>>,
}

/// Parsed JP2/JPH Palette column descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kPaletteColumn {
    /// Significant bits in this palette column.
    pub bit_depth: u8,
    /// Whether this palette column stores signed values.
    pub signed: bool,
}

/// Parsed JP2/JPH Component Mapping box entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kComponentMapping {
    /// Source codestream component index.
    pub component_index: u16,
    /// Mapping operation for this output channel.
    pub mapping_type: J2kComponentMappingType,
}

/// JP2/JPH Component Mapping operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kComponentMappingType {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kChannelDefinition {
    /// Output channel index.
    pub channel_index: u16,
    /// Channel type.
    pub channel_type: J2kChannelType,
    /// Channel association.
    pub association: J2kChannelAssociation,
}

/// JP2/JPH channel type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kChannelType {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kChannelAssociation {
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

/// Parsed JP2/JPH Colour Specification box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum J2kColorSpec {
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
