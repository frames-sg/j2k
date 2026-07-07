// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::bail;
use crate::{DecodingError, Result, ValidationError};

/// Maps an output coordinate within an IDWT step to the source sub-band index.
///
/// `origin` is the global coordinate of the IDWT output rectangle,
/// `local_coord` is the coordinate within that output rectangle, and
/// `low_pass` selects the low-pass (`LL`/`LH`) or high-pass (`HL`/`HH`) band
/// along one axis. This helper is exposed so backend adapters can compute
/// required input windows with the same odd-origin rounding as the native IDWT.
#[must_use]
pub fn idwt_band_index(origin: u32, local_coord: u32, low_pass: bool) -> u32 {
    let global = u64::from(origin) + u64::from(local_coord);
    let origin = u64::from(origin);
    let index = if low_pass {
        global.div_ceil(2).saturating_sub(origin.div_ceil(2))
    } else {
        (global / 2).saturating_sub(origin / 2)
    };
    u32::try_from(index).unwrap_or(u32::MAX)
}

pub(crate) fn add_roi_shift_to_bitplanes(
    bitplanes: u8,
    roi_shift: u8,
    max_bitplanes: u8,
) -> Result<u8> {
    let Some(coded_bitplanes) = bitplanes.checked_add(roi_shift) else {
        bail!(DecodingError::TooManyBitplanes);
    };
    if coded_bitplanes > max_bitplanes {
        bail!(DecodingError::TooManyBitplanes);
    }
    Ok(coded_bitplanes)
}

pub(crate) fn apply_roi_maxshift_inverse_i64(coefficient: i64, roi_shift: u8) -> i64 {
    if roi_shift == 0 || coefficient == 0 {
        return coefficient;
    }

    let magnitude = coefficient.unsigned_abs();
    let threshold = 1_u64.checked_shl(roi_shift as u32).unwrap_or(u64::MAX);
    if magnitude < threshold {
        return coefficient;
    }

    let shifted = magnitude >> roi_shift;
    let shifted = shifted.min(i64::MAX as u64) as i64;
    if coefficient < 0 {
        -shifted
    } else {
        shifted
    }
}

pub(crate) fn apply_roi_maxshift_inverse_i32(coefficient: i32, roi_shift: u8) -> i32 {
    apply_roi_maxshift_inverse_i64(i64::from(coefficient), roi_shift)
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

pub(crate) fn validate_roi(dims: (u32, u32), roi: (u32, u32, u32, u32)) -> Result<()> {
    let (image_width, image_height) = dims;
    let (x, y, width, height) = roi;
    let x_end = x
        .checked_add(width)
        .ok_or(ValidationError::InvalidDimensions)?;
    let y_end = y
        .checked_add(height)
        .ok_or(ValidationError::InvalidDimensions)?;
    if x_end > image_width || y_end > image_height {
        return Err(ValidationError::InvalidDimensions.into());
    }
    Ok(())
}
