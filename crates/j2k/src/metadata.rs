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

/// Where an image and its tiles lie on the JPEG 2000 reference grid (the SIZ
/// marker's origins and sizes), in full-resolution reference-grid units.
///
/// Image coordinates elsewhere in this crate count from the image area's own
/// corner, `image_origin`. A reduction by `s` spans
/// `[ceil(origin / s), ceil((origin + size) / s))` on each axis (ITU-T T.800
/// B.5), so an origin that is not a multiple of `s` moves the reduced grid
/// against the image's own; see [`crate::DeviceDecodePlan::for_reference_grid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct J2kReferenceGrid {
    /// Offset of the image area from the grid's origin (`XOsiz`, `YOsiz`).
    pub image_origin: (u32, u32),
    /// Width and height of the image area (`Xsiz - XOsiz`, `Ysiz - YOsiz`).
    pub image_size: (u32, u32),
    /// Offset of the first tile from the grid's origin (`XTOsiz`, `YTOsiz`).
    pub tile_origin: (u32, u32),
    /// Nominal width and height of a tile (`XTsiz`, `YTsiz`).
    pub tile_size: (u32, u32),
}

impl J2kReferenceGrid {
    /// The grid of `image`, a native image at any resolution of a
    /// full-resolution image of `image_size`.
    pub(crate) fn of_native(image: &j2k_native::Image<'_>, image_size: (u32, u32)) -> Self {
        let (tile_origin, tile_size) = image.tile_grid();
        Self {
            image_origin: image.image_origin(),
            image_size,
            tile_origin,
            tile_size,
        }
    }

    /// The grid of an image of `image_size` at the grid's origin, in one tile.
    #[must_use]
    pub const fn at_origin(image_size: (u32, u32)) -> Self {
        Self {
            image_origin: (0, 0),
            image_size,
            tile_origin: (0, 0),
            tile_size: image_size,
        }
    }

    /// The far corner of the image area (`Xsiz`, `Ysiz`).
    #[must_use]
    pub const fn extent(&self) -> (u64, u64) {
        (
            self.image_origin.0 as u64 + self.image_size.0 as u64,
            self.image_origin.1 as u64 + self.image_size.1 as u64,
        )
    }

    /// The samples of the image reduced by `levels` resolution levels whose
    /// reference-grid footprints overlap `region` of the full-resolution
    /// image, in reduced image coordinates: what a reduced decode of
    /// `region` covers. `None` when `region` is empty or leaves the image, or
    /// `levels` is 32 or more.
    #[must_use]
    pub fn reduced_region(&self, region: j2k_core::Rect, levels: u8) -> Option<j2k_core::Rect> {
        let denominator = 1_u32.checked_shl(u32::from(levels))?;
        (region.w > 0 && region.h > 0 && region.is_within(self.image_size)).then(|| {
            crate::adapter::device_plan::reference_grid_covering(
                self.image_origin,
                self.image_size,
                region,
                denominator,
            )
        })
    }
}

/// Full parsed JPEG 2000 / HTJ2K support metadata.
///
/// This preserves the existing compact [`Info`] summary while exposing fields
/// needed to reason about the implemented Part 1 / Part 15 support surfaces.
#[derive(Debug, PartialEq, Eq)]
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
#[derive(Debug, PartialEq, Eq)]
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
#[derive(Debug, PartialEq, Eq)]
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

impl J2kChannelDefinition {
    /// Raw CDEF channel-type value, including application-defined values.
    #[must_use]
    pub const fn channel_type_raw(&self) -> u16 {
        match self.channel_type {
            J2kChannelType::Color => 0,
            J2kChannelType::Opacity => 1,
            J2kChannelType::PremultipliedOpacity => 2,
            J2kChannelType::Unspecified => u16::MAX,
            J2kChannelType::Unknown { value } => value,
        }
    }

    /// Raw CDEF channel-association value, including application-defined values.
    #[must_use]
    pub const fn association_raw(&self) -> u16 {
        match self.association {
            J2kChannelAssociation::WholeImage => 0,
            J2kChannelAssociation::Color { index } => index,
            J2kChannelAssociation::Unspecified => u16::MAX,
        }
    }
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
#[derive(Debug, PartialEq, Eq)]
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
