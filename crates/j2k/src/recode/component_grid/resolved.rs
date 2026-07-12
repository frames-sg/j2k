// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reference-grid expansion for resolved JP2 palette/mapping components.

use super::{bytes_per_sample, checked_plane_bytes, dimension_usize, RecodeAllocationBudget};
use crate::J2kError;
use alloc::vec::Vec;

/// Materialize a resolved component on the reference grid before dropping
/// JP2 palette/component-mapping metadata from pixel-preserving output.
pub(in crate::recode) fn resolved_plane_data(
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
    let full_len = checked_plane_bytes(
        reference_dimensions.0,
        reference_dimensions.1,
        bytes_per_sample,
        plane_index,
    )?;
    if plane_dimensions == reference_dimensions && data.len() == full_len {
        return Ok(None);
    }

    let sampled_dimensions = (
        reference_dimensions.0.div_ceil(u32::from(x_rsiz)),
        reference_dimensions.1.div_ceil(u32::from(y_rsiz)),
    );
    let sampled_len = checked_plane_bytes(
        sampled_dimensions.0,
        sampled_dimensions.1,
        bytes_per_sample,
        plane_index,
    )?;
    if plane_dimensions != sampled_dimensions || data.len() != sampled_len {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "resolved component plane {plane_index} data length mismatch: expected {sampled_len} sampled bytes or {full_len} expanded bytes, got {}",
                data.len()
            ),
        });
    }

    let mut expanded = budget.try_vec(full_len, "HTJ2K recode resolved component plane")?;
    let sampled_width = dimension_usize(sampled_dimensions.0, reference_dimensions)?;
    let reference_width = dimension_usize(reference_dimensions.0, reference_dimensions)?;
    let reference_height = dimension_usize(reference_dimensions.1, reference_dimensions)?;
    for y in 0..reference_height {
        let source_y = y / usize::from(y_rsiz);
        for x in 0..reference_width {
            let source_x = x / usize::from(x_rsiz);
            copy_sample(
                data,
                &mut expanded,
                source_y,
                source_x,
                sampled_width,
                bytes_per_sample,
                reference_dimensions,
            )?;
        }
    }
    if expanded.len() != full_len {
        return Err(J2kError::InternalInvariant {
            what: "resolved recode component expansion produced a non-planned length",
        });
    }
    Ok(Some(expanded))
}

fn copy_sample(
    data: &[u8],
    expanded: &mut Vec<u8>,
    source_y: usize,
    source_x: usize,
    sampled_width: usize,
    bytes_per_sample: usize,
    reference_dimensions: (u32, u32),
) -> Result<(), J2kError> {
    let start = source_y
        .checked_mul(sampled_width)
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
        what: "validated resolved recode component indexing escaped its source plane",
    })?;
    expanded.extend_from_slice(sample);
    Ok(())
}
