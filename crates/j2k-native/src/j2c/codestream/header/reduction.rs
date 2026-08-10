// SPDX-License-Identifier: MIT OR Apache-2.0

//! Resolution-reduction validation and application for parsed main headers.

use super::super::super::DecodeSettings;
use super::super::validation::skipped_levels_to_reach_target;
use super::super::{ComponentInfo, SizeData};
use crate::error::{bail, DecodingError, Result, ValidationError};

pub(super) fn apply_resolution_reduction(
    size_data: &mut SizeData,
    component_infos: &[ComponentInfo],
    settings: &DecodeSettings,
    exact_reduction_levels: Option<u8>,
) -> Result<u8> {
    // Components can have different resolution ladders, so the shortest one
    // limits the number of levels that the complete image can discard.
    let min_num_resolution_levels = component_infos
        .iter()
        .map(ComponentInfo::num_resolution_levels)
        .min()
        .ok_or(ValidationError::InvalidComponentMetadata)?;
    let max_skipped_resolution_levels = min_num_resolution_levels
        .checked_sub(1)
        .ok_or(ValidationError::InvalidComponentMetadata)?;
    let skipped_resolution_levels = if let Some(requested) = exact_reduction_levels {
        if requested > max_skipped_resolution_levels {
            bail!(DecodingError::UnsupportedFeature(
                "requested reduction exceeds the codestream resolution ladder",
            ));
        }
        requested
    } else if let Some((target_width, target_height)) = settings.target_resolution {
        if target_width == 0 || target_height == 0 {
            bail!(ValidationError::InvalidDimensions);
        }
        let width_levels =
            skipped_levels_to_reach_target(size_data.checked_image_width()?, target_width);
        let height_levels =
            skipped_levels_to_reach_target(size_data.checked_image_height()?, target_height);
        width_levels
            .min(height_levels)
            .min(max_skipped_resolution_levels)
    } else {
        0
    };

    let Some(resolution_shrink_factor) = 1_u32.checked_shl(u32::from(skipped_resolution_levels))
    else {
        bail!(DecodingError::UnsupportedFeature(
            "requested reduction exceeds supported image geometry",
        ));
    };
    size_data.x_resolution_shrink_factor = size_data
        .x_resolution_shrink_factor
        .checked_mul(resolution_shrink_factor)
        .ok_or(ValidationError::InvalidDimensions)?;
    size_data.y_resolution_shrink_factor = size_data
        .y_resolution_shrink_factor
        .checked_mul(resolution_shrink_factor)
        .ok_or(ValidationError::InvalidDimensions)?;
    size_data.checked_image_width()?;
    size_data.checked_image_height()?;
    Ok(skipped_resolution_levels)
}
