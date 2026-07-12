//! The palette box (pclr), defined in I.5.3.4.

use alloc::vec::Vec;

use crate::error::{bail, FormatError, Result};
use crate::jp2::{allocation::Jp2AllocationBudget, ImageBoxes};
use crate::reader::BitReader;

pub(super) fn parse(
    boxes: &mut ImageBoxes,
    data: &[u8],
    budget: &mut Jp2AllocationBudget,
) -> Result<()> {
    let mut reader = BitReader::new(data);
    let num_entries = usize::from(reader.read_u16().ok_or(FormatError::InvalidBox)?);
    let num_components = usize::from(reader.read_byte().ok_or(FormatError::InvalidBox)?);

    if num_entries == 0 || num_components == 0 {
        bail!(FormatError::InvalidBox);
    }

    let mut columns = budget.try_vec(num_components, "JP2 palette columns")?;
    for _ in 0..num_components {
        let descriptor = reader.read_byte().ok_or(FormatError::InvalidBox)?;
        let bit_depth = (descriptor & 0x7F)
            .checked_add(1)
            .ok_or(FormatError::InvalidBox)?;
        let signed = (descriptor & 0x80) != 0;

        columns.push(PaletteColumn { bit_depth, signed });
    }

    let mut entries = budget.try_vec(num_entries, "JP2 palette rows")?;

    for _ in 0..num_entries {
        let mut row = budget.try_vec(num_components, "JP2 palette entries")?;

        for column in &columns {
            let num_bytes = usize::from(column.bit_depth).div_ceil(8).max(1);
            let raw_bytes = reader
                .read_bytes(num_bytes)
                .ok_or(FormatError::InvalidBox)?;
            let mut raw_value = 0_u64;
            for &byte in raw_bytes {
                raw_value = (raw_value << 8) | u64::from(byte);
            }

            row.push(raw_value);
        }

        entries.push(row);
    }

    let replaced = boxes.palette.replace(PaletteBox { entries, columns });
    if let Some(replaced) = replaced {
        budget.release_vec(&replaced.columns)?;
        for row in &replaced.entries {
            budget.release_vec(row)?;
        }
        budget.release_vec(&replaced.entries)?;
    }

    Ok(())
}

#[derive(Debug)]
pub(crate) struct PaletteBox {
    pub(crate) entries: Vec<Vec<u64>>,
    pub(crate) columns: Vec<PaletteColumn>,
}

impl PaletteBox {
    #[inline]
    pub(crate) fn map(&self, entry: usize, column: usize) -> Option<u64> {
        self.entries
            .get(entry)
            .and_then(|row| row.get(column))
            .copied()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PaletteColumn {
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
}
