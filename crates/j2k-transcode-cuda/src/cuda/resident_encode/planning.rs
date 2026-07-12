// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_element_product, checked_host_byte_sum, checked_host_bytes,
    htj2k97_subband_total_bitplanes, CudaHtj2k97DeviceCodeblockBands, CudaHtj2kEncodeCodeBlockJob,
    CudaHtj2kEncodeResidentTarget, CudaPooledDeviceBuffer, CudaTranscodeError, HostPhaseBudget,
    Htj2k97CodeBlockOptions, J2kSubBandType,
};
use super::{htj2k97_code_block_dim, to_u32};

pub(in crate::cuda) struct ResidentDeviceGroup<'a, J> {
    pub(in crate::cuda) group_index: usize,
    pub(in crate::cuda) bands: CudaHtj2k97DeviceCodeblockBands,
    pub(in crate::cuda) jobs: &'a [J],
}

pub(super) struct ResidentSubbandEncodePlan<'a> {
    pub(super) coefficients: &'a j2k_cuda_runtime::CudaDeviceBuffer,
    pub(super) coefficient_count: usize,
    pub(super) jobs: Vec<CudaHtj2kEncodeCodeBlockJob>,
    pub(super) shapes: Vec<(u32, u32)>,
    pub(super) sub_band_type: J2kSubBandType,
    pub(super) num_cbs_x: usize,
    pub(super) num_cbs_y: usize,
    pub(super) total_bitplanes: u8,
}

pub(super) struct ResidentSubbandGroupPlans<'a, J> {
    pub(super) group_index: usize,
    pub(super) jobs: &'a [J],
    pub(super) ll: ResidentSubbandEncodePlan<'a>,
    pub(super) hl: ResidentSubbandEncodePlan<'a>,
    pub(super) lh: ResidentSubbandEncodePlan<'a>,
    pub(super) hh: ResidentSubbandEncodePlan<'a>,
}

impl<'a, J> ResidentSubbandGroupPlans<'a, J> {
    fn plans(&self) -> [&ResidentSubbandEncodePlan<'a>; 4] {
        [&self.ll, &self.hl, &self.lh, &self.hh]
    }
}

pub(super) type ResidentMetadataBudget = HostPhaseBudget;

pub(super) fn reserve_component_assembly_budget<Component, Resolution, Subband>(
    budget: &mut ResidentMetadataBudget,
    item_count: usize,
    what: &'static str,
) -> Result<(), CudaTranscodeError> {
    let resolution_count = checked_element_product(&[item_count, 2], what)?;
    let destination_subband_count = checked_element_product(&[item_count, 4], what)?;
    let additional = checked_host_byte_sum(
        &[
            checked_host_bytes::<Component>(item_count, what)?,
            checked_host_bytes::<Resolution>(resolution_count, what)?,
            checked_host_bytes::<Subband>(destination_subband_count, what)?,
        ],
        what,
    )?;
    budget.preflight_bytes(additional)
}

pub(super) fn build_resident_subband_group_plans<'a, J>(
    groups: &'a [ResidentDeviceGroup<'a, J>],
    options: Htj2k97CodeBlockOptions,
    live_metadata_bytes: usize,
) -> Result<
    (
        Vec<ResidentSubbandGroupPlans<'a, J>>,
        ResidentMetadataBudget,
    ),
    CudaTranscodeError,
> {
    let mut budget = ResidentMetadataBudget::with_live_bytes(
        "CUDA resident aggregate metadata",
        live_metadata_bytes,
    )?;
    let mut group_plans =
        budget.try_vec_with_capacity(groups.len(), "CUDA resident subband group plans")?;
    for group in groups {
        if group.bands.item_count != group.jobs.len() {
            return Err(CudaTranscodeError::Kernel(
                "CUDA grouped resident 9/7 band item count mismatch",
            ));
        }
        group_plans.push(ResidentSubbandGroupPlans {
            group_index: group.group_index,
            jobs: group.jobs,
            ll: resident_subband_encode_plan(
                &group.bands.ll,
                group.bands.item_count,
                group.bands.low_width,
                group.bands.low_height,
                J2kSubBandType::LowLow,
                options,
                &mut budget,
            )?,
            hl: resident_subband_encode_plan(
                &group.bands.hl,
                group.bands.item_count,
                group.bands.high_width,
                group.bands.low_height,
                J2kSubBandType::HighLow,
                options,
                &mut budget,
            )?,
            lh: resident_subband_encode_plan(
                &group.bands.lh,
                group.bands.item_count,
                group.bands.low_width,
                group.bands.high_height,
                J2kSubBandType::LowHigh,
                options,
                &mut budget,
            )?,
            hh: resident_subband_encode_plan(
                &group.bands.hh,
                group.bands.item_count,
                group.bands.high_width,
                group.bands.high_height,
                J2kSubBandType::HighHigh,
                options,
                &mut budget,
            )?,
        });
    }
    Ok((group_plans, budget))
}

pub(super) fn resident_group_targets<'a, J>(
    group_plans: &'a [ResidentSubbandGroupPlans<'a, J>],
    budget: &mut ResidentMetadataBudget,
) -> Result<Vec<CudaHtj2kEncodeResidentTarget<'a>>, CudaTranscodeError> {
    let target_capacity = checked_element_product(
        &[group_plans.len(), 4],
        "CUDA resident grouped encode targets",
    )?;
    let mut targets =
        budget.try_vec_with_capacity(target_capacity, "CUDA resident grouped encode targets")?;
    for plan in group_plans
        .iter()
        .flat_map(ResidentSubbandGroupPlans::plans)
        .filter(|plan| !plan.jobs.is_empty())
    {
        targets.push(CudaHtj2kEncodeResidentTarget {
            coefficients: plan.coefficients,
            coefficient_count: plan.coefficient_count,
            jobs: &plan.jobs,
        });
    }
    Ok(targets)
}

pub(super) fn resident_targets<'a>(
    plans: &'a [ResidentSubbandEncodePlan<'a>],
    budget: &mut ResidentMetadataBudget,
) -> Result<Vec<CudaHtj2kEncodeResidentTarget<'a>>, CudaTranscodeError> {
    let mut targets = budget.try_vec_with_capacity(plans.len(), "CUDA resident encode targets")?;
    for plan in plans.iter().filter(|plan| !plan.jobs.is_empty()) {
        targets.push(CudaHtj2kEncodeResidentTarget {
            coefficients: plan.coefficients,
            coefficient_count: plan.coefficient_count,
            jobs: &plan.jobs,
        });
    }
    Ok(targets)
}

pub(super) fn resident_group_block_count<J>(
    group_plans: &[ResidentSubbandGroupPlans<'_, J>],
) -> Result<usize, CudaTranscodeError> {
    group_plans
        .iter()
        .flat_map(ResidentSubbandGroupPlans::plans)
        .try_fold(0usize, |count, plan| {
            count
                .checked_add(plan.jobs.len())
                .ok_or(CudaTranscodeError::HostAllocationTooLarge {
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                    what: "CUDA resident grouped code-block count",
                })
        })
}

pub(super) fn resident_subband_encode_plan<'a>(
    coefficients: &'a CudaPooledDeviceBuffer,
    item_count: usize,
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
    budget: &mut ResidentMetadataBudget,
) -> Result<ResidentSubbandEncodePlan<'a>, CudaTranscodeError> {
    let coefficient_buffer = coefficients
        .as_device_buffer()
        .ok_or(CudaTranscodeError::Kernel(
            "CUDA resident 9/7 pooled band checkout missing",
        ))?;
    let cb_width = htj2k97_code_block_dim(options.code_block_width_exp)?;
    let cb_height = htj2k97_code_block_dim(options.code_block_height_exp)?;
    let num_cbs_x = if width == 0 {
        0
    } else {
        width.div_ceil(cb_width)
    };
    let num_cbs_y = if height == 0 {
        0
    } else {
        height.div_ceil(cb_height)
    };
    let total_bitplanes = htj2k97_subband_total_bitplanes(options, sub_band_type);
    if width == 0 || height == 0 {
        return Ok(ResidentSubbandEncodePlan {
            coefficients: coefficient_buffer,
            coefficient_count: 0,
            jobs: Vec::new(),
            shapes: Vec::new(),
            sub_band_type,
            num_cbs_x: 0,
            num_cbs_y: 0,
            total_bitplanes,
        });
    }

    let item_stride = width.checked_mul(height).ok_or(CudaTranscodeError::Kernel(
        "CUDA resident 9/7 band dimensions overflow",
    ))?;
    let coefficient_count =
        item_stride
            .checked_mul(item_count)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident 9/7 band item count overflow",
            ))?;
    let job_count = checked_element_product(
        &[item_count, num_cbs_x, num_cbs_y],
        "CUDA resident 9/7 code-block metadata",
    )?;
    let mut encode_jobs =
        budget.try_vec_with_capacity(job_count, "CUDA resident 9/7 code-block jobs")?;
    let mut shapes =
        budget.try_vec_with_capacity(job_count, "CUDA resident 9/7 code-block shapes")?;
    for item in 0..item_count {
        let item_offset = item
            .checked_mul(item_stride)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident 9/7 band item offset overflow",
            ))?;
        for cby in 0..num_cbs_y {
            for cbx in 0..num_cbs_x {
                let block_width = (width - cbx * cb_width).min(cb_width);
                let block_height = (height - cby * cb_height).min(cb_height);
                let block_offset = cby
                    .checked_mul(cb_height)
                    .and_then(|value| value.checked_mul(width))
                    .and_then(|value| {
                        value.checked_add(cbx.checked_mul(cb_width)?.checked_mul(block_height)?)
                    })
                    .and_then(|value| value.checked_add(item_offset))
                    .ok_or(CudaTranscodeError::Kernel(
                        "CUDA resident 9/7 code-block offset overflow",
                    ))?;
                encode_jobs.push(CudaHtj2kEncodeCodeBlockJob {
                    coefficient_offset: to_u32(block_offset)?,
                    width: to_u32(block_width)?,
                    height: to_u32(block_height)?,
                    total_bitplanes,
                    target_coding_passes: 1,
                });
                shapes.push((to_u32(block_width)?, to_u32(block_height)?));
            }
        }
    }

    Ok(ResidentSubbandEncodePlan {
        coefficients: coefficient_buffer,
        coefficient_count,
        jobs: encode_jobs,
        shapes,
        sub_band_type,
        num_cbs_x,
        num_cbs_y,
        total_bitplanes,
    })
}

#[cfg(test)]
mod tests {
    use super::{reserve_component_assembly_budget, ResidentMetadataBudget};
    use crate::cuda::{
        CudaTranscodeError, PreencodedHtj2k97Component, PreencodedHtj2k97Resolution,
        PreencodedHtj2k97Subband,
    };
    use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

    #[test]
    fn resident_metadata_budget_rejects_aggregate_subcap_reservations() {
        let mut budget = ResidentMetadataBudget::with_cap(
            "CUDA resident aggregate metadata",
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        );
        let half_plus_one = DEFAULT_MAX_HOST_ALLOCATION_BYTES / 2 + 1;
        budget.account_bytes(half_plus_one).unwrap();
        assert!(matches!(
            budget.account_bytes(half_plus_one),
            Err(CudaTranscodeError::HostAllocationTooLarge {
                what: "CUDA resident aggregate metadata",
                ..
            })
        ));
    }

    #[test]
    fn resident_metadata_budget_counts_live_caller_bytes() {
        let half_plus_one = DEFAULT_MAX_HOST_ALLOCATION_BYTES / 2 + 1;
        let budget = ResidentMetadataBudget::with_live_bytes(
            "CUDA resident aggregate metadata",
            half_plus_one,
        )
        .unwrap();
        assert!(matches!(
            budget.preflight_bytes(half_plus_one),
            Err(CudaTranscodeError::HostAllocationTooLarge {
                what: "CUDA resident aggregate metadata",
                ..
            })
        ));
    }

    #[test]
    fn nested_component_metadata_is_preflighted_as_one_budget() {
        let mut budget = ResidentMetadataBudget::new("CUDA resident aggregate metadata");
        let item_count = DEFAULT_MAX_HOST_ALLOCATION_BYTES
            / core::mem::size_of::<PreencodedHtj2k97Component>()
            + 1;
        assert!(matches!(
            reserve_component_assembly_budget::<
                PreencodedHtj2k97Component,
                PreencodedHtj2k97Resolution,
                PreencodedHtj2k97Subband,
            >(&mut budget, item_count, "test nested resident metadata",),
            Err(CudaTranscodeError::HostAllocationTooLarge { .. })
        ));
    }
}
