// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained main-header allocation accounting.

use super::super::{
    ComponentInfo, ComponentSizeInfo, Header, PpmPacket, ProgressionChange, StepSize,
};
use crate::error::{Result, ValidationError};
use crate::DEFAULT_MAX_DECODE_BYTES;
use core::mem::size_of;

pub(crate) fn retained_header_bytes(header: &Header<'_>) -> Result<usize> {
    let mut bytes = size_of::<Header<'static>>();
    include_elements::<ComponentSizeInfo>(&mut bytes, header.size_data.component_sizes.capacity())?;
    include_elements::<(u8, u8)>(
        &mut bytes,
        header
            .global_coding_style
            .component_parameters
            .parameters
            .precinct_exponents
            .capacity(),
    )?;
    include_elements::<ComponentInfo>(&mut bytes, header.component_infos.capacity())?;
    for component in &header.component_infos {
        include_elements::<(u8, u8)>(
            &mut bytes,
            component
                .coding_style
                .parameters
                .precinct_exponents
                .capacity(),
        )?;
        include_elements::<StepSize>(
            &mut bytes,
            component.quantization_info.step_sizes.capacity(),
        )?;
    }
    include_elements::<ProgressionChange>(&mut bytes, header.progression_changes.capacity())?;
    include_elements::<u32>(&mut bytes, header.plm_packet_lengths.capacity())?;
    include_elements::<PpmPacket<'_>>(&mut bytes, header.ppm_packets.capacity())?;
    if bytes > DEFAULT_MAX_DECODE_BYTES {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(bytes)
}

fn include_elements<T>(bytes: &mut usize, count: usize) -> Result<()> {
    let additional = count
        .checked_mul(size_of::<T>())
        .ok_or(ValidationError::ImageTooLarge)?;
    *bytes = bytes
        .checked_add(additional)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok(())
}
