// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checkpoint cadence, count, and aggregate phase preflight.

use super::allocation::checked_checkpoint_phase_bytes;
use super::eoi::ValidatedScanBytes;
use crate::entropy::sequential::PreparedDecodePlan;
use crate::error::JpegError;

pub(super) struct CheckpointBuildPlan<'a> {
    pub(super) total_mcus: u32,
    pub(super) cadence_mcus: u32,
    pub(super) restart_interval: Option<u32>,
    pub(super) expected_checkpoint_count: usize,
    pub(super) validated_scan: ValidatedScanBytes<'a>,
}

pub(super) fn plan_checkpoint_build_from_validated<'a, T>(
    plan: &PreparedDecodePlan,
    validated_scan: ValidatedScanBytes<'a>,
    cadence_mcus: u32,
    initial_live_bytes: usize,
    allocation_cap: usize,
) -> Result<CheckpointBuildPlan<'a>, JpegError> {
    let total_mcus = total_mcus(plan);
    let cadence_mcus = cadence_mcus.max(1);
    let restart_interval = plan
        .restart_interval
        .filter(|&interval| interval > 0)
        .map(u32::from);
    let expected_checkpoint_count =
        checkpoint_count_summary(total_mcus, cadence_mcus, restart_interval);
    let terminated_copy_bytes = validated_scan.terminated_copy_len(allocation_cap)?;
    checked_checkpoint_phase_bytes::<T>(
        initial_live_bytes,
        expected_checkpoint_count,
        terminated_copy_bytes,
        allocation_cap,
    )?;
    Ok(CheckpointBuildPlan {
        total_mcus,
        cadence_mcus,
        restart_interval,
        expected_checkpoint_count,
        validated_scan,
    })
}

pub(crate) fn checkpoint_count_summary(
    total_mcus: u32,
    cadence_mcus: u32,
    restart_interval: Option<u32>,
) -> usize {
    let interval = restart_interval.unwrap_or(cadence_mcus).max(1);
    total_mcus.div_ceil(interval).max(1) as usize
}

pub(crate) fn total_mcus(plan: &PreparedDecodePlan) -> u32 {
    let mcu_width = u32::from(plan.sampling.max_h) * 8;
    let mcu_height = u32::from(plan.sampling.max_v) * 8;
    let mcus_per_row = plan.dimensions.0.div_ceil(mcu_width);
    let mcu_rows = plan.dimensions.1.div_ceil(mcu_height);
    mcus_per_row.saturating_mul(mcu_rows)
}
