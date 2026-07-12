//! The component mapping box (cmap), defined in I.5.3.5.

use alloc::vec::Vec;

use crate::error::{bail, FormatError, Result};
use crate::jp2::{allocation::Jp2AllocationBudget, ImageBoxes};
use crate::reader::BitReader;

pub(super) fn parse(
    boxes: &mut ImageBoxes,
    data: &[u8],
    budget: &mut Jp2AllocationBudget,
) -> Result<()> {
    if data.is_empty() || !data.len().is_multiple_of(4) {
        bail!(FormatError::InvalidBox);
    }

    let mut reader = BitReader::new(data);
    let mut entries = budget.try_vec(data.len() / 4, "JP2 component mappings")?;

    while !reader.at_end() {
        let component_index = reader.read_u16().ok_or(FormatError::InvalidBox)?;
        let mapping_type = reader.read_byte().ok_or(FormatError::InvalidBox)?;
        let palette_column = reader.read_byte().ok_or(FormatError::InvalidBox)?;

        let mapping_type = match mapping_type {
            0 => ComponentMappingType::Direct,
            1 => ComponentMappingType::Palette {
                column: palette_column,
            },
            value => ComponentMappingType::Unknown {
                value,
                column: palette_column,
            },
        };

        entries.push(ComponentMappingEntry {
            component_index,
            mapping_type,
        });
    }

    let replaced = boxes
        .component_mapping
        .replace(ComponentMappingBox { entries });
    if let Some(replaced) = replaced {
        budget.release_vec(&replaced.entries)?;
    }

    Ok(())
}

#[derive(Debug)]
pub(crate) struct ComponentMappingBox {
    pub(crate) entries: Vec<ComponentMappingEntry>,
}

#[derive(Debug, Clone)]
pub(crate) struct ComponentMappingEntry {
    pub(crate) component_index: u16,
    pub(crate) mapping_type: ComponentMappingType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComponentMappingType {
    Direct,
    Palette { column: u8 },
    Unknown { value: u8, column: u8 },
}
