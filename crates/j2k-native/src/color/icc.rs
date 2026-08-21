// SPDX-License-Identifier: MIT OR Apache-2.0

//! ICC profile validation, ownership, and color-space resolution.

use alloc::vec::Vec;

use super::ColorSpace;
use crate::jp2::icc::ICCMetadata;
use crate::{try_reserve_decode_elements, Result, ValidationError, DEFAULT_MAX_DECODE_BYTES};

pub(super) fn resolve_icc_color_space(
    profile: &[u8],
    retained_container_bytes: usize,
) -> Result<Option<ColorSpace>> {
    let Some(metadata) = ICCMetadata::from_data(profile) else {
        return Ok(None);
    };
    Ok(Some(ColorSpace::Icc {
        profile: try_clone_color_profile(profile, retained_container_bytes)?,
        num_channels: u16::from(metadata.color_space.num_components()),
    }))
}

pub(super) fn try_clone_color_profile(profile: &[u8], retained_bytes: usize) -> Result<Vec<u8>> {
    checked_color_profile_peak(retained_bytes, profile.len(), DEFAULT_MAX_DECODE_BYTES)?;
    let mut cloned = Vec::new();
    try_reserve_decode_elements(&mut cloned, profile.len())?;
    checked_color_profile_peak(retained_bytes, cloned.capacity(), DEFAULT_MAX_DECODE_BYTES)?;
    cloned.extend_from_slice(profile);
    Ok(cloned)
}

fn checked_color_profile_peak(
    retained_bytes: usize,
    profile_bytes: usize,
    cap: usize,
) -> Result<usize> {
    let peak = retained_bytes
        .checked_add(profile_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    if peak > cap {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(peak)
}

#[cfg(test)]
mod tests {
    use super::checked_color_profile_peak;

    #[test]
    fn retained_color_profile_peak_accepts_exact_cap_and_rejects_one_over() {
        assert_eq!(
            checked_color_profile_peak(7, 5, 12).expect("exact ICC clone peak"),
            12
        );
        assert!(checked_color_profile_peak(8, 5, 12).is_err());
    }
}
