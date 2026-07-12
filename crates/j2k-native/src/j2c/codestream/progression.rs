// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;

use super::{ProgressionChange, ProgressionOrder};
use crate::error::{MarkerError, Result, ValidationError};
use crate::reader::BitReader;
use crate::try_reserve_decode_elements;

/// POC marker (A.6.6).
pub(crate) fn poc_marker(
    reader: &mut BitReader<'_>,
    csiz: u16,
    _num_layers: u8,
    max_owned_bytes: usize,
) -> Result<Vec<ProgressionChange>> {
    let length = reader.read_u16().ok_or(MarkerError::ParseFailure("POC"))?;
    let remaining_bytes = length
        .checked_sub(2)
        .ok_or(MarkerError::ParseFailure("POC"))?;
    let component_index_size = if csiz < 257 { 1u16 } else { 2u16 };
    let change_size = 1 + component_index_size + 2 + 1 + component_index_size + 1;
    if remaining_bytes == 0 || remaining_bytes % change_size != 0 {
        return Err(MarkerError::ParseFailure("POC").into());
    }

    let change_count = usize::from(remaining_bytes / change_size);
    let owned_bytes = change_count
        .checked_mul(size_of::<ProgressionChange>())
        .ok_or(ValidationError::ImageTooLarge)?;
    if owned_bytes > max_owned_bytes {
        return Err(ValidationError::ImageTooLarge.into());
    }
    let mut changes = Vec::new();
    try_reserve_decode_elements(&mut changes, change_count)?;
    for _ in 0..change_count {
        let resolution_start = reader.read_byte().ok_or(MarkerError::ParseFailure("POC"))?;
        let component_start =
            read_component_index(reader, csiz).ok_or(MarkerError::ParseFailure("POC"))?;
        let layer_end = reader.read_u16().ok_or(MarkerError::ParseFailure("POC"))?;
        let resolution_end = reader.read_byte().ok_or(MarkerError::ParseFailure("POC"))?;
        let component_end =
            read_component_index(reader, csiz).ok_or(MarkerError::ParseFailure("POC"))?;
        let progression_order =
            ProgressionOrder::from_u8(reader.read_byte().ok_or(MarkerError::ParseFailure("POC"))?)
                .map_err(|_| MarkerError::ParseFailure("POC"))?;

        if resolution_start >= resolution_end
            || component_start >= component_end
            || component_start >= csiz
            || layer_end == 0
            || layer_end > u16::from(u8::MAX)
        {
            return Err(MarkerError::ParseFailure("POC").into());
        }

        changes.push(ProgressionChange {
            resolution_start,
            component_start,
            layer_end: u8::try_from(layer_end).map_err(|_| MarkerError::ParseFailure("POC"))?,
            resolution_end,
            component_end,
            progression_order,
        });
    }

    Ok(changes)
}

pub(super) fn read_component_index(reader: &mut BitReader<'_>, csiz: u16) -> Option<u16> {
    if csiz < 257 {
        Some(u16::from(reader.read_byte()?))
    } else {
        reader.read_u16()
    }
}
