// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    BlockCodingMode, NativeEncodePipelineError, NativeEncodePipelineResult, QuantStepSize, Vec,
    MAX_CLASSIC_ROI_CODED_BITPLANES, MAX_HT_ROI_CODED_BITPLANES,
};

mod validation;
pub(super) use validation::validate_roi_encode_options_nonallocating;

#[derive(Debug, Default)]
pub(super) struct ComponentRoiEncodePlan {
    pub(super) shift: u8,
    pub(super) planned_region_count: u32,
    pub(super) regions: Vec<ComponentRoiEncodeRegion>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ComponentRoiEncodeRegion {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
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
) -> NativeEncodePipelineResult<()> {
    if base_bitplanes
        .checked_add(shift)
        .is_none_or(|bitplanes| bitplanes > max_bitplanes)
    {
        return Err(NativeEncodePipelineError::unsupported(
            "ROI maxshift exceeds supported coded bitplane count",
        ));
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
            debug_assert!(u8::try_from(step_size.exponent).is_ok());
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
