// SPDX-License-Identifier: MIT OR Apache-2.0

//! JP2 Image Header and Bits Per Component parsing.

use alloc::vec::Vec;

use crate::error::{bail, FormatError, Result};

use super::allocation;
use super::{ComponentDescriptor, ImageHeaderBox};

pub(super) fn parse_image_header(data: &[u8]) -> Result<ImageHeaderBox> {
    if data.len() < 14 {
        bail!(FormatError::InvalidBox);
    }
    let compression_type = data[11];
    if compression_type != 7 {
        bail!(FormatError::InvalidBox);
    }
    Ok(ImageHeaderBox {
        height: u32::from_be_bytes([data[0], data[1], data[2], data[3]]),
        width: u32::from_be_bytes([data[4], data[5], data[6], data[7]]),
        components: u16::from_be_bytes([data[8], data[9]]),
        bits_per_component: if data[10] == 0xFF {
            None
        } else {
            Some(parse_component_descriptor(data[10])?)
        },
    })
}

pub(super) fn parse_bits_per_component(
    data: &[u8],
    budget: &mut allocation::Jp2AllocationBudget,
) -> Result<Vec<ComponentDescriptor>> {
    if data.is_empty() {
        bail!(FormatError::InvalidBox);
    }
    let mut components = budget.try_vec(data.len(), "JP2 BPCC metadata")?;
    for &descriptor in data {
        components.push(parse_component_descriptor(descriptor)?);
    }
    Ok(components)
}

fn parse_component_descriptor(descriptor: u8) -> Result<ComponentDescriptor> {
    let bit_depth = (descriptor & 0x7F) + 1;
    if bit_depth > 38 {
        bail!(FormatError::InvalidBox);
    }
    Ok(ComponentDescriptor {
        bit_depth,
        signed: (descriptor & 0x80) != 0,
    })
}
