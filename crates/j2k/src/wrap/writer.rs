// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single-allocation, bounds-checked JP2/JPH serialization.

use alloc::vec::Vec;

use super::{
    allocation::allocate_output,
    color::PlannedColorSpec,
    metadata::{component_bpc, ChannelDefinitionPlan, ResolvedComponents},
    plan::WrapPlan,
    JP2_COMPRESSION_TYPE, JP2_SIGNATURE_PAYLOAD,
};
use crate::{
    J2kChannelAssociation, J2kChannelDefinition, J2kChannelType, J2kComponentMappingType, J2kError,
    J2kPaletteColumn,
};

pub(super) fn write(plan: &WrapPlan<'_>, retained_bytes: usize) -> Result<Vec<u8>, J2kError> {
    let output = allocate_output(plan.total_len, retained_bytes)?;
    let mut writer = CheckedWriter::new(output, plan.total_len);

    writer.box_header(*b"jP  ", JP2_SIGNATURE_PAYLOAD.len())?;
    writer.bytes(JP2_SIGNATURE_PAYLOAD)?;

    writer.box_header(*b"ftyp", 12)?;
    writer.bytes(&plan.brand)?;
    writer.bytes(&0_u32.to_be_bytes())?;
    writer.bytes(&plan.brand)?;

    writer.box_header(*b"jp2h", plan.jp2_header_payload_len)?;
    write_image_header(&mut writer, plan)?;
    if let Some(payload_len) = plan.bpcc_payload_len {
        write_bits_per_component(&mut writer, plan.components, payload_len)?;
    }
    plan.colors.for_each_resolved(plan.parsed, |color| {
        write_color_specification(&mut writer, color)
    })?;
    if let (Some(palette), Some(payload_len)) = (plan.metadata.palette, plan.palette_payload_len) {
        write_palette(&mut writer, palette, payload_len)?;
    }
    if let Some(payload_len) = plan.component_mapping_payload_len {
        write_component_mappings(&mut writer, plan, payload_len)?;
    }
    write_channel_definitions(&mut writer, plan.channel_definitions)?;

    writer.box_header(*b"jp2c", plan.codestream.len())?;
    writer.bytes(plan.codestream)?;
    writer.finish()
}

struct CheckedWriter {
    output: Vec<u8>,
    planned_len: usize,
}

impl CheckedWriter {
    fn new(output: Vec<u8>, planned_len: usize) -> Self {
        Self {
            output,
            planned_len,
        }
    }

    fn byte(&mut self, value: u8) -> Result<(), J2kError> {
        self.ensure_space(1)?;
        self.output.push(value);
        Ok(())
    }

    fn bytes(&mut self, values: &[u8]) -> Result<(), J2kError> {
        self.ensure_space(values.len())?;
        self.output.extend_from_slice(values);
        Ok(())
    }

    fn box_header(&mut self, box_type: [u8; 4], payload_len: usize) -> Result<(), J2kError> {
        let box_len = payload_len
            .checked_add(8)
            .and_then(|len| u32::try_from(len).ok())
            .ok_or(J2kError::InternalInvariant {
                what: "validated JP2/JPH box length changed during writing",
            })?;
        self.bytes(&box_len.to_be_bytes())?;
        self.bytes(&box_type)
    }

    fn ensure_space(&self, additional: usize) -> Result<(), J2kError> {
        let new_len =
            self.output
                .len()
                .checked_add(additional)
                .ok_or(J2kError::InternalInvariant {
                    what: "JP2/JPH writer length overflow after planning",
                })?;
        if new_len > self.planned_len || new_len > self.output.capacity() {
            return Err(J2kError::InternalInvariant {
                what: "JP2/JPH writer exceeded its exact allocation plan",
            });
        }
        Ok(())
    }

    fn finish(self) -> Result<Vec<u8>, J2kError> {
        if self.output.len() != self.planned_len {
            return Err(J2kError::InternalInvariant {
                what: "JP2/JPH writer produced a non-planned output length",
            });
        }
        Ok(self.output)
    }
}

fn write_image_header(writer: &mut CheckedWriter, plan: &WrapPlan<'_>) -> Result<(), J2kError> {
    writer.box_header(*b"ihdr", 14)?;
    let (width, height) = plan.parsed.info.dimensions;
    writer.bytes(&height.to_be_bytes())?;
    writer.bytes(&width.to_be_bytes())?;
    writer.bytes(&plan.component_count.to_be_bytes())?;
    writer.byte(plan.image_header_bpc)?;
    writer.byte(JP2_COMPRESSION_TYPE)?;
    writer.byte(0)?;
    writer.byte(0)
}

fn write_bits_per_component(
    writer: &mut CheckedWriter,
    components: ResolvedComponents<'_>,
    payload_len: usize,
) -> Result<(), J2kError> {
    writer.box_header(*b"bpcc", payload_len)?;
    for index in 0..components.len() {
        let component = components
            .component(index)
            .ok_or(J2kError::InternalInvariant {
                what: "validated BPCC component became unresolved",
            })?;
        writer.byte(component_bpc(component))?;
    }
    Ok(())
}

fn write_color_specification(
    writer: &mut CheckedWriter,
    color: PlannedColorSpec<'_>,
) -> Result<(), J2kError> {
    writer.box_header(*b"colr", color.payload_len()?)?;
    match color {
        PlannedColorSpec::Enumerated(value) => {
            writer.bytes(&[1, 0, 0])?;
            writer.bytes(&value.to_be_bytes())
        }
        PlannedColorSpec::IccProfile(profile) => {
            writer.bytes(&[2, 0, 0])?;
            writer.bytes(profile)
        }
    }
}

fn write_palette(
    writer: &mut CheckedWriter,
    palette: &crate::J2kPaletteMetadata,
    payload_len: usize,
) -> Result<(), J2kError> {
    writer.box_header(*b"pclr", payload_len)?;
    let entries =
        u16::try_from(palette.entries.len()).map_err(|_| J2kError::InternalInvariant {
            what: "validated palette entry count changed during writing",
        })?;
    let columns = u8::try_from(palette.columns.len()).map_err(|_| J2kError::InternalInvariant {
        what: "validated palette column count changed during writing",
    })?;
    writer.bytes(&entries.to_be_bytes())?;
    writer.byte(columns)?;
    for &column in &palette.columns {
        writer.byte(palette_column_bpc(column))?;
    }
    for row in &palette.entries {
        for (&value, &column) in row.iter().zip(&palette.columns) {
            write_palette_value(writer, value, column)?;
        }
    }
    Ok(())
}

fn palette_column_bpc(column: J2kPaletteColumn) -> u8 {
    (column.bit_depth - 1) | if column.signed { 0x80 } else { 0 }
}

fn write_palette_value(
    writer: &mut CheckedWriter,
    value: u64,
    column: J2kPaletteColumn,
) -> Result<(), J2kError> {
    let bytes = usize::from(column.bit_depth).div_ceil(8).max(1);
    for byte in (0..bytes).rev() {
        let output = u8::try_from((value >> (byte * 8)) & 0xff).map_err(|_| {
            J2kError::InternalInvariant {
                what: "masked palette byte did not fit u8",
            }
        })?;
        writer.byte(output)?;
    }
    Ok(())
}

fn write_component_mappings(
    writer: &mut CheckedWriter,
    plan: &WrapPlan<'_>,
    payload_len: usize,
) -> Result<(), J2kError> {
    writer.box_header(*b"cmap", payload_len)?;
    if plan.metadata.component_mappings.is_empty() {
        let palette = plan.metadata.palette.ok_or(J2kError::InternalInvariant {
            what: "planned implicit CMAP lost its palette",
        })?;
        for column in 0..palette.columns.len() {
            writer.bytes(&0_u16.to_be_bytes())?;
            writer.byte(1)?;
            writer.byte(
                u8::try_from(column).map_err(|_| J2kError::InternalInvariant {
                    what: "validated implicit CMAP column no longer fits u8",
                })?,
            )?;
        }
        return Ok(());
    }

    for mapping in plan.metadata.component_mappings {
        writer.bytes(&mapping.component_index.to_be_bytes())?;
        match mapping.mapping_type {
            J2kComponentMappingType::Direct => {
                writer.byte(0)?;
                writer.byte(0)?;
            }
            J2kComponentMappingType::Palette { column } => {
                writer.byte(1)?;
                writer.byte(column)?;
            }
            J2kComponentMappingType::Unknown { value, column } => {
                writer.byte(value)?;
                writer.byte(column)?;
            }
        }
    }
    Ok(())
}

fn write_channel_definitions(
    writer: &mut CheckedWriter,
    plan: ChannelDefinitionPlan<'_>,
) -> Result<(), J2kError> {
    match plan {
        ChannelDefinitionPlan::None => Ok(()),
        ChannelDefinitionPlan::Explicit(definitions) => {
            let payload_len = definitions
                .len()
                .checked_mul(6)
                .and_then(|bytes| bytes.checked_add(2))
                .ok_or(J2kError::InternalInvariant {
                    what: "validated CDEF size changed during writing",
                })?;
            writer.box_header(*b"cdef", payload_len)?;
            let count =
                u16::try_from(definitions.len()).map_err(|_| J2kError::InternalInvariant {
                    what: "validated CDEF count no longer fits u16",
                })?;
            writer.bytes(&count.to_be_bytes())?;
            for definition in definitions {
                write_channel_definition(writer, *definition)?;
            }
            Ok(())
        }
        ChannelDefinitionPlan::Rgba => {
            writer.box_header(*b"cdef", 2 + 4 * 6)?;
            writer.bytes(&4_u16.to_be_bytes())?;
            for (channel, channel_type, association) in [
                (0_u16, 0_u16, 1_u16),
                (1_u16, 0_u16, 2_u16),
                (2_u16, 0_u16, 3_u16),
                (3_u16, 1_u16, 0_u16),
            ] {
                writer.bytes(&channel.to_be_bytes())?;
                writer.bytes(&channel_type.to_be_bytes())?;
                writer.bytes(&association.to_be_bytes())?;
            }
            Ok(())
        }
    }
}

fn write_channel_definition(
    writer: &mut CheckedWriter,
    definition: J2kChannelDefinition,
) -> Result<(), J2kError> {
    writer.bytes(&definition.channel_index.to_be_bytes())?;
    writer.bytes(&raw_channel_type(definition.channel_type).to_be_bytes())?;
    writer.bytes(&raw_channel_association(definition.association).to_be_bytes())
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
