// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTJ2K contiguous sub-band preparation.

use std::sync::Arc;

use super::{
    DirectTier1Mode, Error, J2kHtCleanupBatchJob, PreparedHtPayloadSource, PreparedHtSubBand,
};
use crate::compute::direct_plan_types::PreparedHtExecutionOwner;

mod grouped;
mod referenced;

pub(in crate::compute) use self::grouped::prepare_ht_sub_band_groups;
pub(super) use self::referenced::prepare_referenced_ht_sub_band;

#[cfg(target_os = "macos")]
pub(in crate::compute) fn prepare_ht_sub_band(
    job: &j2k_native::HtOwnedSubBandPlan,
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBand, Error> {
    let coded_len = crate::batch_allocation::checked_count_sum(
        job.jobs.iter().map(|block| block.data.len()),
        "HTJ2K MetalDirect coded payload",
    )?;
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("HTJ2K MetalDirect prepared sub-band");
    let mut jobs = budget.try_vec(job.jobs.len(), "HTJ2K MetalDirect jobs")?;
    let mut coded_data = budget.try_vec(coded_len, "HTJ2K MetalDirect coded payload")?;
    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        jobs.push(J2kHtCleanupBatchJob {
            coded_offset,
            width: block.width,
            height: block.height,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            missing_msbs: u32::from(block.missing_bit_planes),
            num_bitplanes: u32::from(block.num_bitplanes),
            roi_shift: u32::from(block.roi_shift),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
    }

    Ok(PreparedHtSubBand {
        band_id: job.band_id,
        width: job.width,
        height: job.height,
        payload_source: PreparedHtPayloadSource::Contiguous(coded_data),
        jobs,
        execution_owner: Arc::new(PreparedHtExecutionOwner),
    })
}
