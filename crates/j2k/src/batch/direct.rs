// SPDX-License-Identifier: MIT OR Apache-2.0

//! Repeated-region direct HTJ2K planning and per-worker execution.

use j2k_core::{BatchInfrastructureError, Downscale, PixelFormat, Rect};
use j2k_native::{
    execute_direct_color_plan_rgb8_into, execute_direct_color_plan_rgba8_into, J2kDirectColorPlan,
    J2kDirectCpuScratch, J2kRect,
};

use crate::backend::DecodeSettings;
use crate::decode::{decode_warnings_for_settings, validate_buffer};
use crate::{J2kError, TileRegionScaledDecodeJob};

use super::admission::{BatchAllocationBudget, BatchAllocationClaim};
use super::allocation::GENERIC_WORKER_CLAIM_BYTES;
use super::BatchOutcome;
use std::sync::Arc;

mod planning;
use planning::build_direct_color_region_plan;
#[cfg(test)]
use planning::input_declares_htj2k;

pub(super) enum DirectDecodeAttemptError {
    Tile(J2kError),
    Infrastructure(BatchInfrastructureError),
}

impl From<J2kError> for DirectDecodeAttemptError {
    fn from(error: J2kError) -> Self {
        Self::Tile(error)
    }
}

/// Shared repeated-region plan retained only through the worker execution phase.
///
/// Its actual retained capacity is the admission budget's baseline. Each
/// worker initially claims the remainder of one authoritative native decode
/// cap, then reconciles that lease to its actual scratch plus maximum temporary
/// workspace peak after preparation.
pub(super) struct DirectColorRegionCache {
    input_ptr: usize,
    input_len: usize,
    roi: Rect,
    scale: Downscale,
    output_region: J2kRect,
    plan: J2kDirectColorPlan,
}

pub(super) struct DirectWorkerState {
    scratch: J2kDirectCpuScratch,
    cache: Option<DirectColorRegionCache>,
    allocation_claim: Option<BatchAllocationClaim>,
}

impl Default for DirectWorkerState {
    fn default() -> Self {
        Self {
            scratch: J2kDirectCpuScratch::new(),
            cache: None,
            allocation_claim: None,
        }
    }
}

impl DirectWorkerState {
    pub(super) fn try_decode(
        &mut self,
        job: &mut TileRegionScaledDecodeJob<'_, '_>,
        fmt: PixelFormat,
        shared_plan: Option<&DirectColorRegionCache>,
        allocation_budget: &Arc<BatchAllocationBudget>,
    ) -> Result<Option<BatchOutcome>, DirectDecodeAttemptError> {
        if let Some(outcome) = self.try_shared(job, fmt, shared_plan, allocation_budget)? {
            return Ok(Some(outcome));
        }
        self.try_cached(job, fmt, allocation_budget)
    }

    pub(super) fn release(&mut self) {
        self.cache = None;
        self.scratch.clear();
        self.allocation_claim = None;
    }

    fn try_shared(
        &mut self,
        job: &mut TileRegionScaledDecodeJob<'_, '_>,
        fmt: PixelFormat,
        shared_plan: Option<&DirectColorRegionCache>,
        allocation_budget: &Arc<BatchAllocationBudget>,
    ) -> Result<Option<BatchOutcome>, DirectDecodeAttemptError> {
        let Some(shared_plan) = shared_plan else {
            return Ok(None);
        };
        if !is_direct_color_u8_format(fmt)
            || !shared_plan.matches(DirectColorRegionKey::for_job(job))
        {
            return Ok(None);
        }

        let decoded = job.roi.scaled_covering(job.scale);
        validate_buffer((decoded.w, decoded.h), job.out.len(), job.stride, fmt)?;
        if self.allocation_claim.is_none() {
            let plan_bytes = shared_plan.retained_allocation_bytes()?;
            let initial_worker_bytes = GENERIC_WORKER_CLAIM_BYTES
                .checked_sub(plan_bytes)
                .ok_or_else(|| {
                    J2kError::internal_backend("shared direct plan exceeds one native claim")
                })?;
            let mut claim = allocation_budget
                .claim(initial_worker_bytes)
                .map_err(DirectDecodeAttemptError::Infrastructure)?;
            let execution_bytes = self
                .scratch
                .prepare_execution_allocation_bytes(&shared_plan.plan)
                .map_err(J2kError::from_native_decode_error)?;
            let worker_bytes = execution_bytes.checked_sub(plan_bytes).ok_or_else(|| {
                J2kError::internal_backend("direct execution allocation omitted its plan")
            })?;
            claim
                .reconcile(worker_bytes)
                .map_err(DirectDecodeAttemptError::Infrastructure)?;
            self.allocation_claim = Some(claim);
        }
        if let Err(error) = execute_direct_color_plan_u8_into(
            &shared_plan.plan,
            shared_plan.output_region,
            &mut self.scratch,
            job.out,
            job.stride,
            fmt,
        ) {
            self.release();
            return Err(DirectDecodeAttemptError::Tile(error));
        }
        Ok(Some(success_outcome(decoded)))
    }

    fn try_cached(
        &mut self,
        job: &mut TileRegionScaledDecodeJob<'_, '_>,
        fmt: PixelFormat,
        allocation_budget: &Arc<BatchAllocationBudget>,
    ) -> Result<Option<BatchOutcome>, DirectDecodeAttemptError> {
        if !is_direct_color_u8_format(fmt) || job.scale == Downscale::None {
            return Ok(None);
        }

        let decoded = job.roi.scaled_covering(job.scale);
        validate_buffer((decoded.w, decoded.h), job.out.len(), job.stride, fmt)?;
        let key = DirectColorRegionKey::for_job(job);
        if !self.cache.as_ref().is_some_and(|cache| cache.matches(key)) {
            // A prior cache and its scratch must not overlap construction of a
            // replacement plan under the single per-worker native claim.
            self.release();
            let mut claim = allocation_budget
                .claim(GENERIC_WORKER_CLAIM_BYTES)
                .map_err(DirectDecodeAttemptError::Infrastructure)?;
            let Some((plan, output_region)) =
                build_direct_color_region_plan(job.input, job.roi, job.scale)?
            else {
                return Ok(None);
            };
            let execution_bytes = self
                .scratch
                .prepare_execution_allocation_bytes(&plan)
                .map_err(J2kError::from_native_decode_error)?;
            claim
                .reconcile(execution_bytes)
                .map_err(DirectDecodeAttemptError::Infrastructure)?;
            self.cache = Some(DirectColorRegionCache {
                input_ptr: key.input_ptr,
                input_len: key.input_len,
                roi: key.roi,
                scale: key.scale,
                output_region,
                plan,
            });
            self.allocation_claim = Some(claim);
        }

        let cache = self.cache.as_ref().ok_or_else(|| {
            J2kError::internal_backend("internal direct color plan cache missing")
        })?;
        if let Err(error) = execute_direct_color_plan_u8_into(
            &cache.plan,
            cache.output_region,
            &mut self.scratch,
            job.out,
            job.stride,
            fmt,
        ) {
            self.release();
            return Err(DirectDecodeAttemptError::Tile(error));
        }
        Ok(Some(success_outcome(decoded)))
    }
}

pub(super) fn build_repeated_direct_color_region_plan(
    jobs: &[TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
) -> Result<Option<DirectColorRegionCache>, J2kError> {
    if !is_direct_color_u8_format(fmt) {
        return Ok(None);
    }
    let Some(first) = jobs.first() else {
        return Ok(None);
    };
    if first.scale == Downscale::None {
        return Ok(None);
    }
    let key = DirectColorRegionKey::for_job(first);
    if !jobs
        .iter()
        .all(|job| DirectColorRegionKey::for_job(job) == key)
    {
        return Ok(None);
    }

    let Some((plan, output_region)) =
        build_direct_color_region_plan(first.input, first.roi, first.scale)?
    else {
        return Ok(None);
    };
    Ok(Some(DirectColorRegionCache {
        input_ptr: key.input_ptr,
        input_len: key.input_len,
        roi: key.roi,
        scale: key.scale,
        output_region,
        plan,
    }))
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct DirectColorRegionKey {
    input_ptr: usize,
    input_len: usize,
    roi: Rect,
    scale: Downscale,
}

impl DirectColorRegionKey {
    fn for_job(job: &TileRegionScaledDecodeJob<'_, '_>) -> Self {
        Self {
            input_ptr: job.input.as_ptr() as usize,
            input_len: job.input.len(),
            roi: job.roi,
            scale: job.scale,
        }
    }
}

impl DirectColorRegionCache {
    fn matches(&self, key: DirectColorRegionKey) -> bool {
        self.input_ptr == key.input_ptr
            && self.input_len == key.input_len
            && self.roi == key.roi
            && self.scale == key.scale
    }

    pub(super) fn retained_allocation_bytes(&self) -> Result<usize, J2kError> {
        self.plan
            .retained_allocation_bytes()
            .map_err(J2kError::from_native_decode_error)
    }
}

fn success_outcome(decoded: Rect) -> BatchOutcome {
    BatchOutcome::new(
        decoded,
        decode_warnings_for_settings(DecodeSettings::default()),
    )
}

fn is_direct_color_u8_format(fmt: PixelFormat) -> bool {
    matches!(fmt, PixelFormat::Rgb8 | PixelFormat::Rgba8)
}

fn execute_direct_color_plan_u8_into(
    plan: &J2kDirectColorPlan,
    output_region: J2kRect,
    scratch: &mut J2kDirectCpuScratch,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let result = match fmt {
        PixelFormat::Rgb8 => {
            execute_direct_color_plan_rgb8_into(plan, output_region, scratch, out, stride)
        }
        PixelFormat::Rgba8 => {
            execute_direct_color_plan_rgba8_into(plan, output_region, scratch, out, stride)
        }
        _ => {
            return Err(J2kError::internal_backend(
                "direct color route received an unsupported output format",
            ));
        }
    };
    result.map_err(J2kError::from_native_decode_error)
}

#[cfg(test)]
mod tests;
