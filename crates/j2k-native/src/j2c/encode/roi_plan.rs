// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[derive(Clone, Debug, Default)]
pub(super) struct ComponentRoiEncodePlan {
    pub(super) shift: u8,
    pub(super) regions: Vec<ComponentRoiEncodeRegion>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ComponentRoiEncodeRegion {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

pub(super) fn component_sampling_for_options(
    options: &EncodeOptions,
    num_components: u16,
) -> Result<Vec<(u8, u8)>, &'static str> {
    match &options.component_sampling {
        Some(component_sampling) => {
            if component_sampling.len() != usize::from(num_components) {
                return Err("component sampling count does not match component count");
            }
            if component_sampling
                .iter()
                .any(|&(x_rsiz, y_rsiz)| x_rsiz == 0 || y_rsiz == 0)
            {
                return Err("component sampling factors must be non-zero");
            }
            Ok(component_sampling.clone())
        }
        None => Ok(vec![(1, 1); usize::from(num_components)]),
    }
}

pub(super) fn roi_encode_plans_for_options(
    options: &EncodeOptions,
    roi_regions: &[EncodeRoiRegion],
    num_components: u16,
    width: u32,
    height: u32,
    component_sampling: &[(u8, u8)],
    base_bitplanes: u8,
    block_coding_mode: BlockCodingMode,
) -> Result<Vec<ComponentRoiEncodePlan>, &'static str> {
    let whole_component_shifts = roi_component_shifts_for_options(
        options,
        num_components,
        base_bitplanes,
        block_coding_mode,
    )?;
    let mut plans = whole_component_shifts
        .iter()
        .map(|&shift| ComponentRoiEncodePlan {
            shift,
            regions: Vec::new(),
        })
        .collect::<Vec<_>>();

    for region in roi_regions {
        if region.component >= num_components {
            return Err("ROI region component index out of range");
        }
        if region.width == 0 || region.height == 0 {
            return Err("ROI region dimensions must be non-zero");
        }
        if region.shift == 0 {
            return Err("ROI region maxshift must be non-zero");
        }

        let x1 = region
            .x
            .checked_add(region.width)
            .ok_or("ROI region bounds overflow")?;
        let y1 = region
            .y
            .checked_add(region.height)
            .ok_or("ROI region bounds overflow")?;
        if region.x >= width || region.y >= height || x1 > width || y1 > height {
            return Err("ROI region must be inside image bounds");
        }

        let component_idx = usize::from(region.component);
        if whole_component_shifts[component_idx] != 0 {
            return Err("ROI region cannot be combined with whole-component ROI shift");
        }
        if region.shift < base_bitplanes {
            return Err("ROI region maxshift must cover background bitplanes");
        }
        validate_roi_shift(region.shift, base_bitplanes, block_coding_mode)?;

        let plan = &mut plans[component_idx];
        if plan.shift == 0 {
            plan.shift = region.shift;
        } else if plan.shift != region.shift {
            return Err("ROI regions for one component must use one maxshift");
        }

        let &(x_rsiz, y_rsiz) = component_sampling
            .get(component_idx)
            .ok_or("component sampling count does not match component count")?;
        let component_width = width.div_ceil(u32::from(x_rsiz));
        let component_height = height.div_ceil(u32::from(y_rsiz));
        let component_x0 = region.x / u32::from(x_rsiz);
        let component_y0 = region.y / u32::from(y_rsiz);
        let component_x1 = x1.div_ceil(u32::from(x_rsiz)).min(component_width);
        let component_y1 = y1.div_ceil(u32::from(y_rsiz)).min(component_height);
        if component_x0 >= component_x1 || component_y0 >= component_y1 {
            return Err("ROI region does not intersect component grid");
        }
        plan.regions.push(ComponentRoiEncodeRegion {
            x: component_x0,
            y: component_y0,
            width: component_x1 - component_x0,
            height: component_y1 - component_y0,
        });
    }

    Ok(plans)
}

fn roi_component_shifts_for_options(
    options: &EncodeOptions,
    num_components: u16,
    base_bitplanes: u8,
    block_coding_mode: BlockCodingMode,
) -> Result<Vec<u8>, &'static str> {
    if options.roi_component_shifts.is_empty() {
        return Ok(vec![0; usize::from(num_components)]);
    }
    if options.roi_component_shifts.len() != usize::from(num_components) {
        return Err("ROI component shift count does not match component count");
    }
    let max_bitplanes = max_roi_coded_bitplanes(block_coding_mode);
    for &shift in &options.roi_component_shifts {
        validate_roi_shift_for_max(shift, base_bitplanes, max_bitplanes)?;
    }
    Ok(options.roi_component_shifts.clone())
}

fn validate_roi_shift(
    shift: u8,
    base_bitplanes: u8,
    block_coding_mode: BlockCodingMode,
) -> Result<(), &'static str> {
    let max_bitplanes = max_roi_coded_bitplanes(block_coding_mode);
    validate_roi_shift_for_max(shift, base_bitplanes, max_bitplanes)
}

fn max_roi_coded_bitplanes(block_coding_mode: BlockCodingMode) -> u8 {
    match block_coding_mode {
        BlockCodingMode::Classic => MAX_CLASSIC_ROI_CODED_BITPLANES,
        BlockCodingMode::HighThroughput => MAX_HT_ROI_CODED_BITPLANES,
    }
}

fn validate_roi_shift_for_max(
    shift: u8,
    base_bitplanes: u8,
    max_bitplanes: u8,
) -> Result<(), &'static str> {
    if base_bitplanes
        .checked_add(shift)
        .is_none_or(|bitplanes| bitplanes > max_bitplanes)
    {
        return Err("ROI maxshift exceeds supported coded bitplane count");
    }
    Ok(())
}

pub(super) fn roi_subband_scale(
    num_levels: u8,
    level_idx: Option<usize>,
) -> Result<u32, &'static str> {
    let shift = match level_idx {
        Some(level_idx) => usize::from(num_levels)
            .checked_sub(level_idx)
            .ok_or("ROI subband level exceeds decomposition level count")?,
        None => usize::from(num_levels),
    };
    if shift >= u32::BITS as usize {
        return Err("ROI subband scale exceeds supported coordinate range");
    }
    Ok(1_u32 << shift)
}

fn max_total_bitplanes(step_sizes: &[QuantStepSize], guard_bits: u8) -> Result<u8, &'static str> {
    step_sizes
        .iter()
        .map(|step_size| {
            debug_assert!(step_size.exponent <= u16::from(u8::MAX));
            guard_bits
                .checked_add(
                    u8::try_from(step_size.exponent)
                        .map_err(|_| "quantization exponent exceeds supported bitplane count")?,
                )
                .and_then(|value| value.checked_sub(1))
                .ok_or("quantization bitplane count underflows")
        })
        .max()
        .unwrap_or(Ok(0))
}

pub(super) fn max_total_bitplanes_for_components(
    default_step_sizes: &[QuantStepSize],
    component_step_sizes: &[Vec<QuantStepSize>],
    guard_bits: u8,
) -> Result<u8, &'static str> {
    let default = max_total_bitplanes(default_step_sizes, guard_bits)?;
    component_step_sizes
        .iter()
        .try_fold(default, |max_bitplanes, step_sizes| {
            Ok(max_bitplanes.max(max_total_bitplanes(step_sizes, guard_bits)?))
        })
}
