// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{ProgressionChange, ProgressionOrder};
use crate::reader::BitReader;

/// POC marker (A.6.6).
pub(crate) fn poc_marker(
    reader: &mut BitReader<'_>,
    csiz: u16,
    _num_layers: u8,
) -> Option<Vec<ProgressionChange>> {
    let length = reader.read_u16()?;
    let remaining_bytes = length.checked_sub(2)?;
    let component_index_size = if csiz < 257 { 1u16 } else { 2u16 };
    let change_size = 1 + component_index_size + 2 + 1 + component_index_size + 1;
    if remaining_bytes == 0 || remaining_bytes % change_size != 0 {
        return None;
    }

    let change_count = remaining_bytes / change_size;
    let mut changes = Vec::with_capacity(change_count as usize);
    for _ in 0..change_count {
        let resolution_start = reader.read_byte()?;
        let component_start = read_component_index(reader, csiz)?;
        let layer_end = reader.read_u16()?;
        let resolution_end = reader.read_byte()?;
        let component_end = read_component_index(reader, csiz)?;
        let progression_order = ProgressionOrder::from_u8(reader.read_byte()?).ok()?;

        if resolution_start >= resolution_end
            || component_start >= component_end
            || component_start >= csiz
            || layer_end == 0
            || layer_end > u16::from(u8::MAX)
        {
            return None;
        }

        changes.push(ProgressionChange {
            resolution_start,
            component_start,
            layer_end: layer_end as u8,
            resolution_end,
            component_end,
            progression_order,
        });
    }

    Some(changes)
}

pub(super) fn read_component_index(reader: &mut BitReader<'_>, csiz: u16) -> Option<u16> {
    if csiz < 257 {
        Some(u16::from(reader.read_byte()?))
    } else {
        reader.read_u16()
    }
}
