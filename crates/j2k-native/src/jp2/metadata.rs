// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native and public JP2/JPH metadata models and move-only conversions.

use alloc::vec::Vec;

use crate::error::Result;
use crate::j2c::ComponentData;

use super::allocation;
use super::cdef::{ChannelAssociation, ChannelDefinitionBox, ChannelType};
use super::cmap::{ComponentMappingBox, ComponentMappingEntry, ComponentMappingType};
use super::colr::{ColorSpace as NativeColorSpace, ColorSpecificationBox};
use super::pclr::PaletteBox;

#[derive(Debug, Default)]
pub(crate) struct ImageBoxes {
    pub(crate) image_header: Option<ImageHeaderBox>,
    pub(crate) bits_per_component: Vec<ComponentDescriptor>,
    pub(crate) color_specifications: Vec<ColorSpecificationBox>,
    pub(crate) channel_definition: Option<ChannelDefinitionBox>,
    pub(crate) palette: Option<PaletteBox>,
    pub(crate) component_mapping: Option<ComponentMappingBox>,
}

impl ImageBoxes {
    pub(crate) fn try_with_synthetic_color_specification(
        header: &crate::j2c::Header<'_>,
        color_specification: ColorSpecificationBox,
        retained_baseline_bytes: usize,
    ) -> Result<Self> {
        let mut retained_header_bytes =
            crate::j2c::codestream::allocation::retained_header_bytes(header)?;
        allocation::checked_add_bytes(
            &mut retained_header_bytes,
            retained_baseline_bytes,
            "retained raw-codestream parse owners",
        )?;
        let mut budget = allocation::Jp2AllocationBudget::from_live_bytes(retained_header_bytes)?;
        let mut color_specifications =
            budget.try_vec(1, "synthetic raw-codestream color specification")?;
        color_specifications.push(color_specification);
        Ok(Self {
            color_specifications,
            ..Self::default()
        })
    }

    pub(crate) fn primary_color_specification(&self) -> Option<&ColorSpecificationBox> {
        self.color_specifications.first()
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize> {
        use allocation::{capacity_bytes, checked_add_bytes};

        let mut bytes = capacity_bytes::<ComponentDescriptor>(
            self.bits_per_component.capacity(),
            "JP2 BPCC metadata",
        )?;
        checked_add_bytes(
            &mut bytes,
            capacity_bytes::<ColorSpecificationBox>(
                self.color_specifications.capacity(),
                "JP2 COLR metadata",
            )?,
            "JP2 metadata",
        )?;
        for color_spec in &self.color_specifications {
            if let NativeColorSpace::Icc(profile) = &color_spec.color_space {
                checked_add_bytes(&mut bytes, profile.capacity(), "JP2 ICC metadata")?;
            }
        }
        if let Some(palette) = &self.palette {
            checked_add_bytes(
                &mut bytes,
                capacity_bytes::<crate::jp2::pclr::PaletteColumn>(
                    palette.columns.capacity(),
                    "JP2 palette columns",
                )?,
                "JP2 metadata",
            )?;
            checked_add_bytes(
                &mut bytes,
                capacity_bytes::<Vec<u64>>(palette.entries.capacity(), "JP2 palette rows")?,
                "JP2 metadata",
            )?;
            for row in &palette.entries {
                checked_add_bytes(
                    &mut bytes,
                    capacity_bytes::<u64>(row.capacity(), "JP2 palette entries")?,
                    "JP2 metadata",
                )?;
            }
        }
        if let Some(mapping) = &self.component_mapping {
            checked_add_bytes(
                &mut bytes,
                capacity_bytes::<ComponentMappingEntry>(
                    mapping.entries.capacity(),
                    "JP2 component mappings",
                )?,
                "JP2 metadata",
            )?;
        }
        if let Some(definition) = &self.channel_definition {
            checked_add_bytes(
                &mut bytes,
                capacity_bytes::<crate::jp2::cdef::ChannelDefinition>(
                    definition.channel_definitions.capacity(),
                    "JP2 channel definitions",
                )?,
                "JP2 metadata",
            )?;
        }
        Ok(bytes)
    }
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
#[derive(Debug, PartialEq, Eq)]
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
#[derive(Debug, PartialEq, Eq)]
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
#[derive(Debug, PartialEq, Eq)]
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

pub(crate) struct DecodedImage<'components, 'boxes> {
    /// The raw decoded JPEG2000 codestream components.
    pub(crate) decoded_components: &'components mut Vec<ComponentData>,
    /// The JP2 boxes of the image. In the case of a raw codestream, we
    /// will synthesize the necessary boxes.
    pub(crate) boxes: &'boxes ImageBoxes,
}
pub(super) fn public_image_header(header: ImageHeaderBox) -> Jp2ImageHeaderMetadata {
    Jp2ImageHeaderMetadata {
        width: header.width,
        height: header.height,
        components: header.components,
        bits_per_component: header.bits_per_component.map(public_component_metadata),
    }
}

pub(super) fn public_metadata_from_boxes(boxes: ImageBoxes) -> Result<Jp2FileMetadata> {
    let mut budget = allocation::Jp2AllocationBudget::from_live_bytes(boxes.allocated_bytes()?)?;
    let has_palette = boxes.palette.is_some();
    let has_component_mapping = boxes.component_mapping.is_some();
    let has_channel_definition = boxes.channel_definition.is_some();
    let ImageBoxes {
        image_header: _,
        bits_per_component,
        color_specifications,
        channel_definition,
        palette,
        component_mapping,
    } = boxes;

    Ok(Jp2FileMetadata {
        bits_per_component: public_component_metadata_list(bits_per_component, &mut budget)?,
        color_specs: public_color_specs(color_specifications, &mut budget)?,
        palette: public_palette_metadata(palette, &mut budget)?,
        component_mappings: public_component_mappings(component_mapping, &mut budget)?,
        channel_definitions: public_channel_definitions(channel_definition, &mut budget)?,
        has_palette,
        has_component_mapping,
        has_channel_definition,
    })
}

fn public_component_metadata(component: ComponentDescriptor) -> Jp2ComponentMetadata {
    Jp2ComponentMetadata {
        bit_depth: component.bit_depth,
        signed: component.signed,
    }
}

fn public_component_metadata_list(
    components: Vec<ComponentDescriptor>,
    budget: &mut allocation::Jp2AllocationBudget,
) -> Result<Vec<Jp2ComponentMetadata>> {
    let source_capacity = components.capacity();
    let mut public = budget.try_vec(components.len(), "public JP2 BPCC metadata")?;
    for component in components {
        public.push(public_component_metadata(component));
    }
    budget.release_capacity::<ComponentDescriptor>(source_capacity)?;
    Ok(public)
}

fn public_color_specs(
    color_specs: Vec<ColorSpecificationBox>,
    budget: &mut allocation::Jp2AllocationBudget,
) -> Result<Vec<Jp2ColorSpec>> {
    let source_capacity = color_specs.capacity();
    let mut public = budget.try_vec(color_specs.len(), "public JP2 COLR metadata")?;
    for color_spec in color_specs {
        public.push(public_color_spec(color_spec));
    }
    budget.release_capacity::<ColorSpecificationBox>(source_capacity)?;
    Ok(public)
}

fn public_color_spec(color_spec: ColorSpecificationBox) -> Jp2ColorSpec {
    match color_spec.color_space {
        NativeColorSpace::Enumerated(_) => Jp2ColorSpec::Enumerated {
            value: color_spec.enumerated_value.unwrap_or(0),
        },
        NativeColorSpace::Icc(profile) => Jp2ColorSpec::IccProfile { profile },
        NativeColorSpace::Unknown => Jp2ColorSpec::Unknown {
            method: color_spec.method,
        },
    }
}

fn public_palette_metadata(
    palette: Option<PaletteBox>,
    budget: &mut allocation::Jp2AllocationBudget,
) -> Result<Option<Jp2PaletteMetadata>> {
    let Some(PaletteBox { entries, columns }) = palette else {
        return Ok(None);
    };
    let source_capacity = columns.capacity();
    let mut public_columns = budget.try_vec(columns.len(), "public JP2 palette columns")?;
    for column in columns {
        public_columns.push(Jp2PaletteColumn {
            bit_depth: column.bit_depth,
            signed: column.signed,
        });
    }
    budget.release_capacity::<crate::jp2::pclr::PaletteColumn>(source_capacity)?;
    Ok(Some(Jp2PaletteMetadata {
        columns: public_columns,
        entries,
    }))
}

fn public_component_mappings(
    mapping: Option<ComponentMappingBox>,
    budget: &mut allocation::Jp2AllocationBudget,
) -> Result<Vec<Jp2ComponentMapping>> {
    let Some(mapping) = mapping else {
        return Ok(Vec::new());
    };
    let source_capacity = mapping.entries.capacity();
    let mut public = budget.try_vec(mapping.entries.len(), "public JP2 component mappings")?;
    for entry in mapping.entries {
        public.push(public_component_mapping(&entry));
    }
    budget.release_capacity::<ComponentMappingEntry>(source_capacity)?;
    Ok(public)
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

fn public_channel_definitions(
    definition: Option<ChannelDefinitionBox>,
    budget: &mut allocation::Jp2AllocationBudget,
) -> Result<Vec<Jp2ChannelDefinition>> {
    let Some(definition) = definition else {
        return Ok(Vec::new());
    };
    let source_capacity = definition.channel_definitions.capacity();
    let mut public = budget.try_vec(
        definition.channel_definitions.len(),
        "public JP2 channel definitions",
    )?;
    for definition in definition.channel_definitions {
        public.push(public_channel_definition(&definition));
    }
    budget.release_capacity::<crate::jp2::cdef::ChannelDefinition>(source_capacity)?;
    Ok(public)
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
        association: match definition.association {
            ChannelAssociation::WholeImage => Jp2ChannelAssociation::WholeImage,
            ChannelAssociation::Colour(index) => Jp2ChannelAssociation::Color { index },
            ChannelAssociation::Unspecified => Jp2ChannelAssociation::Unspecified,
        },
    }
}
