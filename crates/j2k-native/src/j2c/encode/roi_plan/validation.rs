// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-allocating validation shared by routes that defer ROI plan materialization.

use super::{max_roi_coded_bitplanes, validate_roi_shift_for_max};
use crate::j2c::encode::{
    BlockCodingMode, EncodeOptions, EncodeRoiRegion, NativeEncodePipelineError,
    NativeEncodePipelineResult, MAX_J2K_SPEC_COMPONENTS,
};

pub(in crate::j2c::encode) fn validate_roi_encode_options_nonallocating(
    options: &EncodeOptions,
    regions: &[EncodeRoiRegion],
    num_components: u16,
    width: u32,
    height: u32,
    base_bitplanes: u8,
    block_coding_mode: BlockCodingMode,
) -> NativeEncodePipelineResult<()> {
    if num_components > MAX_J2K_SPEC_COMPONENTS {
        return Err(NativeEncodePipelineError::unsupported(
            "component count exceeds the JPEG 2000 Part 1 limit",
        ));
    }
    let max_bitplanes = max_roi_coded_bitplanes(block_coding_mode);
    if !options.roi_component_shifts.is_empty()
        && options.roi_component_shifts.len() != usize::from(num_components)
    {
        return Err(NativeEncodePipelineError::invalid_input(
            "ROI component shift count does not match component count",
        ));
    }
    for &shift in &options.roi_component_shifts {
        validate_roi_shift_for_max(shift, base_bitplanes, max_bitplanes)?;
    }

    let mut region_shifts = [0_u8; MAX_J2K_SPEC_COMPONENTS as usize];
    for region in regions {
        validate_region_geometry(region, num_components, width, height)?;
        let component_idx = usize::from(region.component);
        if options
            .roi_component_shifts
            .get(component_idx)
            .is_some_and(|&shift| shift != 0)
        {
            return Err(NativeEncodePipelineError::invalid_input(
                "ROI region cannot be combined with whole-component ROI shift",
            ));
        }
        if region.shift < base_bitplanes {
            return Err(NativeEncodePipelineError::invalid_input(
                "ROI region maxshift must cover background bitplanes",
            ));
        }
        validate_roi_shift_for_max(region.shift, base_bitplanes, max_bitplanes)?;
        if region_shifts[component_idx] != 0 && region_shifts[component_idx] != region.shift {
            return Err(NativeEncodePipelineError::invalid_input(
                "ROI regions for one component must use one maxshift",
            ));
        }
        region_shifts[component_idx] = region.shift;

        let (horizontal_sampling, vertical_sampling) = match &options.component_sampling {
            Some(sampling) => sampling.get(component_idx).copied().ok_or(
                NativeEncodePipelineError::invalid_input(
                    "component sampling count does not match component count",
                ),
            )?,
            None => (1, 1),
        };
        if horizontal_sampling == 0 || vertical_sampling == 0 {
            return Err(NativeEncodePipelineError::invalid_input(
                "component sampling factors must be non-zero",
            ));
        }
        let horizontal_region_end =
            region
                .x
                .checked_add(region.width)
                .ok_or(NativeEncodePipelineError::invalid_input(
                    "ROI region x bound overflows",
                ))?;
        let vertical_region_end =
            region
                .y
                .checked_add(region.height)
                .ok_or(NativeEncodePipelineError::invalid_input(
                    "ROI region y bound overflows",
                ))?;
        let component_horizontal_start = region.x / u32::from(horizontal_sampling);
        let component_vertical_start = region.y / u32::from(vertical_sampling);
        let component_horizontal_end =
            horizontal_region_end.div_ceil(u32::from(horizontal_sampling));
        let component_vertical_end = vertical_region_end.div_ceil(u32::from(vertical_sampling));
        if component_horizontal_start >= component_horizontal_end
            || component_vertical_start >= component_vertical_end
        {
            return Err(NativeEncodePipelineError::invalid_input(
                "ROI region does not intersect component grid",
            ));
        }
    }
    Ok(())
}

fn validate_region_geometry(
    region: &EncodeRoiRegion,
    num_components: u16,
    width: u32,
    height: u32,
) -> NativeEncodePipelineResult<()> {
    if region.component >= num_components {
        return Err(NativeEncodePipelineError::invalid_input(
            "ROI region component index out of range",
        ));
    }
    if region.width == 0 || region.height == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "ROI region dimensions must be non-zero",
        ));
    }
    if region.shift == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "ROI region maxshift must be non-zero",
        ));
    }
    let horizontal_end =
        region
            .x
            .checked_add(region.width)
            .ok_or(NativeEncodePipelineError::invalid_input(
                "ROI region x bound overflows",
            ))?;
    let vertical_end =
        region
            .y
            .checked_add(region.height)
            .ok_or(NativeEncodePipelineError::invalid_input(
                "ROI region y bound overflows",
            ))?;
    if region.x >= width || region.y >= height || horizontal_end > width || vertical_end > height {
        return Err(NativeEncodePipelineError::invalid_input(
            "ROI region must be inside image bounds",
        ));
    }
    Ok(())
}
