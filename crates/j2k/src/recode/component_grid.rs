// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible compaction of expanded native component planes to SIZ component grids.

use alloc::vec::Vec;

use super::allocation::RecodeAllocationBudget;
use crate::J2kError;
use j2k_core::Unsupported;

mod resolved;
pub(super) use resolved::resolved_plane_data;

pub(super) fn plane_data(
    data: &[u8],
    plane_dimensions: (u32, u32),
    reference_dimensions: (u32, u32),
    sampling: (u8, u8),
    bit_depth: u8,
    plane_index: usize,
    budget: &mut RecodeAllocationBudget,
) -> Result<Option<Vec<u8>>, J2kError> {
    let (x_rsiz, y_rsiz) = sampling;
    if x_rsiz == 0 || y_rsiz == 0 {
        return Err(J2kError::InvalidSamples {
            what: format!("component plane {plane_index} sampling factors must be non-zero"),
        });
    }
    let bytes_per_sample = bytes_per_sample(bit_depth)?;
    let component_width = reference_dimensions.0.div_ceil(u32::from(x_rsiz));
    let component_height = reference_dimensions.1.div_ceil(u32::from(y_rsiz));
    let expected_len = checked_plane_bytes(
        component_width,
        component_height,
        bytes_per_sample,
        plane_index,
    )?;
    if data.len() == expected_len {
        return Ok(None);
    }

    let expanded_len = checked_plane_bytes(
        reference_dimensions.0,
        reference_dimensions.1,
        bytes_per_sample,
        plane_index,
    )?;
    if plane_dimensions != reference_dimensions || data.len() != expanded_len {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "component plane {plane_index} data length mismatch: expected {expected_len} component-grid bytes or {expanded_len} expanded bytes, got {}",
                data.len()
            ),
        });
    }

    let mut compacted = budget.try_vec(expected_len, "HTJ2K recode compacted component plane")?;
    let source_width = dimension_usize(reference_dimensions.0, reference_dimensions)?;
    let component_height = dimension_usize(component_height, reference_dimensions)?;
    let component_width = dimension_usize(component_width, reference_dimensions)?;
    for component_y in 0..component_height {
        let source_y =
            component_y
                .checked_mul(usize::from(y_rsiz))
                .ok_or(J2kError::DimensionOverflow {
                    width: reference_dimensions.0,
                    height: reference_dimensions.1,
                })?;
        for component_x in 0..component_width {
            let source_x = component_x.checked_mul(usize::from(x_rsiz)).ok_or(
                J2kError::DimensionOverflow {
                    width: reference_dimensions.0,
                    height: reference_dimensions.1,
                },
            )?;
            let start = source_y
                .checked_mul(source_width)
                .and_then(|row| row.checked_add(source_x))
                .and_then(|sample| sample.checked_mul(bytes_per_sample))
                .ok_or(J2kError::DimensionOverflow {
                    width: reference_dimensions.0,
                    height: reference_dimensions.1,
                })?;
            let end = start
                .checked_add(bytes_per_sample)
                .ok_or(J2kError::DimensionOverflow {
                    width: reference_dimensions.0,
                    height: reference_dimensions.1,
                })?;
            let sample = data.get(start..end).ok_or(J2kError::InternalInvariant {
                what: "validated expanded recode component indexing escaped its source plane",
            })?;
            compacted.extend_from_slice(sample);
        }
    }
    if compacted.len() != expected_len {
        return Err(J2kError::InternalInvariant {
            what: "recode component compaction produced a non-planned length",
        });
    }
    Ok(Some(compacted))
}

fn checked_plane_bytes(
    width: u32,
    height: u32,
    bytes_per_sample: usize,
    plane_index: usize,
) -> Result<usize, J2kError> {
    let width = usize::try_from(width).map_err(|_| J2kError::InvalidSamples {
        what: format!("component plane {plane_index} width does not fit usize"),
    })?;
    let height = usize::try_from(height).map_err(|_| J2kError::InvalidSamples {
        what: format!("component plane {plane_index} height does not fit usize"),
    })?;
    width
        .checked_mul(height)
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or_else(|| J2kError::InvalidSamples {
            what: format!("component plane {plane_index} dimensions overflow"),
        })
}

fn dimension_usize(value: u32, reference_dimensions: (u32, u32)) -> Result<usize, J2kError> {
    usize::try_from(value).map_err(|_| J2kError::DimensionOverflow {
        width: reference_dimensions.0,
        height: reference_dimensions.1,
    })
}

fn bytes_per_sample(bit_depth: u8) -> Result<usize, J2kError> {
    match bit_depth {
        1..=8 => Ok(1),
        9..=16 => Ok(2),
        17..=24 => Ok(3),
        25..=32 => Ok(4),
        33..=38 => Ok(5),
        _ => Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 component planes support 1-38 bits per sample",
        })),
    }
}
