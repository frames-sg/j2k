// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    bytes::store_rgb8_mct_batch_jobs_as_bytes,
    context::CudaContext,
    error::CudaError,
    execution::{CudaExecutionStats, CudaKernelBatchOutput, CudaKernelContiguousBatchOutput},
    kernels::j2k_store_batch_launch_geometry,
    memory::{checked_image_words, CudaDeviceBufferRange},
};

use super::{
    destination::{validate_store_destination, zero_unwritten_store_output},
    validation::{validate_rgb8_mct_target_context, validate_store_plane},
};
use crate::j2k_decode::types::{CudaJ2kStoreRgb8MctBatchJob, CudaJ2kStoreRgb8MctTarget};

mod external;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Rgb8MctTargetPlan {
    output_bytes: usize,
    full_coverage: bool,
    active: bool,
}

struct Rgb8MctBatchPlan {
    targets: Vec<Rgb8MctTargetPlan>,
    total_bytes: usize,
    max_pixels: usize,
    active_job_count: usize,
    requires_zero_fill: bool,
}

pub(super) const STORE_BATCH_GEOMETRY_EXCEEDS_LAUNCH_LIMITS: &str =
    "J2K store batch exceeds static CUDA launch limits";

pub(super) fn validate_store_batch_launch(
    max_pixels: usize,
    active_job_count: usize,
) -> Result<(), CudaError> {
    if active_job_count != 0
        && j2k_store_batch_launch_geometry(max_pixels, active_job_count).is_none()
    {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "{STORE_BATCH_GEOMETRY_EXCEEDS_LAUNCH_LIMITS}: active jobs={active_job_count}, maximum pixels={max_pixels}"
            ),
        });
    }
    Ok(())
}

fn ensure_internal_count(
    actual: usize,
    expected: usize,
    what: &'static str,
) -> Result<(), CudaError> {
    if actual != expected {
        return Err(CudaError::InternalInvariant { what });
    }
    Ok(())
}

fn validate_rgb8_mct_targets(
    context: &CudaContext,
    targets: &[CudaJ2kStoreRgb8MctTarget<'_>],
    live_host_bytes: usize,
) -> Result<Rgb8MctBatchPlan, CudaError> {
    validate_rgb8_mct_target_context(context, targets)?;
    let mut host_budget =
        HostPhaseBudget::with_live_bytes("CUDA J2K store batch plan", live_host_bytes)?;
    let mut plans = host_budget.try_vec_with_capacity(targets.len())?;
    let mut total_bytes = 0usize;
    let mut max_pixels = 0usize;
    let mut active_job_count = 0usize;
    let mut requires_zero_fill = false;

    for target in targets {
        let store = target.job.store;
        let channels = if store.rgba == 0 { 3u8 } else { 4u8 };
        let output_bytes = checked_image_words(
            store.output_width,
            store.output_height,
            usize::from(channels),
        )?;
        let pixels = checked_image_words(store.copy_width, store.copy_height, 1)?;
        let full_coverage = validate_store_destination(
            store.output_width,
            store.output_height,
            store.output_x,
            store.output_y,
            store.copy_width,
            store.copy_height,
            u32::from(channels),
        )?;
        if pixels != 0 {
            for (plane, input_width, source_x, source_y) in [
                (
                    target.plane0,
                    store.input_width0,
                    store.source_x0,
                    store.source_y0,
                ),
                (
                    target.plane1,
                    store.input_width1,
                    store.source_x1,
                    store.source_y1,
                ),
                (
                    target.plane2,
                    store.input_width2,
                    store.source_x2,
                    store.source_y2,
                ),
            ] {
                validate_store_plane(
                    plane,
                    input_width,
                    source_x,
                    source_y,
                    store.copy_width,
                    store.copy_height,
                )?;
            }
            max_pixels = max_pixels.max(pixels);
            active_job_count = active_job_count
                .checked_add(1)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        }
        total_bytes = total_bytes
            .checked_add(output_bytes)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        requires_zero_fill |= output_bytes != 0 && !full_coverage;
        plans.push(Rgb8MctTargetPlan {
            output_bytes,
            full_coverage,
            active: pixels != 0,
        });
    }

    validate_store_batch_launch(max_pixels, active_job_count)?;

    Ok(Rgb8MctBatchPlan {
        targets: plans,
        total_bytes,
        max_pixels,
        active_job_count,
        requires_zero_fill,
    })
}

impl CudaContext {
    /// Apply inverse RCT/ICT and store multiple tightly packed RGB8/RGBA8 images
    /// in one dispatch.
    #[doc(hidden)]
    pub fn j2k_store_rgb8_mct_batch_device(
        &self,
        targets: &[CudaJ2kStoreRgb8MctTarget<'_>],
    ) -> Result<CudaKernelBatchOutput, CudaError> {
        let plan = validate_rgb8_mct_targets(self, targets, 0)?;
        if targets.is_empty() {
            return Ok(CudaKernelBatchOutput {
                outputs: Vec::new(),
                execution: CudaExecutionStats::default(),
            });
        }

        let mut host_budget = HostPhaseBudget::new("CUDA J2K store batch metadata");
        host_budget.account_vec(&plan.targets)?;
        let mut outputs = host_budget.try_vec_with_capacity(targets.len())?;
        let mut kernel_jobs = host_budget.try_vec_with_capacity(plan.active_job_count)?;
        for (target, target_plan) in targets.iter().zip(&plan.targets) {
            let output = self.allocate(target_plan.output_bytes)?;
            if target_plan.active {
                kernel_jobs.push(CudaJ2kStoreRgb8MctBatchJob {
                    plane0_ptr: target.plane0.device_ptr(),
                    plane1_ptr: target.plane1.device_ptr(),
                    plane2_ptr: target.plane2.device_ptr(),
                    output_ptr: output.device_ptr(),
                    job: target.job,
                    reserved_tail: 0,
                });
            }
            outputs.push(output);
        }
        let initialize_outputs = || -> Result<bool, CudaError> {
            let mut zero_fill_enqueued = false;
            for (output, target_plan) in outputs.iter().zip(&plan.targets) {
                zero_fill_enqueued |= zero_unwritten_store_output(
                    self,
                    output,
                    target_plan.output_bytes,
                    target_plan.full_coverage,
                )?;
            }
            Ok(zero_fill_enqueued)
        };
        if plan.max_pixels == 0 {
            if initialize_outputs()? {
                self.synchronize()?;
            }
            return Ok(CudaKernelBatchOutput {
                outputs,
                execution: CudaExecutionStats::default(),
            });
        }

        let jobs_buffer = self.upload(store_rgb8_mct_batch_jobs_as_bytes(&kernel_jobs))?;
        if initialize_outputs()? {
            self.synchronize()?;
        }
        ensure_internal_count(
            kernel_jobs.len(),
            plan.active_job_count,
            "J2K store batch active-job count mismatch",
        )?;
        // SAFETY: this owned path retains every plane, output, and uploaded
        // job buffer through the immediate context completion boundary.
        unsafe {
            self.launch_j2k_store_rgb8_mct_batch_enqueue(
                &jobs_buffer,
                plan.max_pixels,
                plan.active_job_count,
            )?;
        }
        self.synchronize()?;
        Ok(CudaKernelBatchOutput {
            outputs,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Apply inverse RCT/ICT and store multiple tightly packed RGB8/RGBA8 images
    /// into one contiguous device allocation in one dispatch.
    #[doc(hidden)]
    pub fn j2k_store_rgb8_mct_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreRgb8MctTarget<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        self.j2k_store_rgb8_mct_batch_contiguous_device_with_live_host_bytes(targets, 0)
    }

    /// Store a contiguous color batch while accounting caller-live host metadata.
    #[doc(hidden)]
    pub fn j2k_store_rgb8_mct_batch_contiguous_device_with_live_host_bytes(
        &self,
        targets: &[CudaJ2kStoreRgb8MctTarget<'_>],
        live_host_bytes: usize,
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        let plan = validate_rgb8_mct_targets(self, targets, live_host_bytes)?;
        let mut host_budget = HostPhaseBudget::with_live_bytes(
            "CUDA J2K contiguous store batch metadata",
            live_host_bytes,
        )?;
        host_budget.account_vec(&plan.targets)?;
        let mut ranges = host_budget.try_vec_with_capacity(targets.len())?;
        let mut offset = 0usize;
        for target_plan in &plan.targets {
            ranges.push(CudaDeviceBufferRange {
                offset,
                len: target_plan.output_bytes,
            });
            offset = offset
                .checked_add(target_plan.output_bytes)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        }
        ensure_internal_count(
            offset,
            plan.total_bytes,
            "J2K contiguous store output range mismatch",
        )?;

        let output = self.allocate(plan.total_bytes)?;
        let initialize_output = || {
            zero_unwritten_store_output(self, &output, plan.total_bytes, !plan.requires_zero_fill)
        };
        if targets.is_empty() || plan.max_pixels == 0 {
            if initialize_output()? {
                self.synchronize()?;
            }
            return Ok(CudaKernelContiguousBatchOutput {
                output,
                ranges,
                execution: CudaExecutionStats::default(),
            });
        }

        let base_ptr = output.device_ptr();
        let mut kernel_jobs = host_budget.try_vec_with_capacity(plan.active_job_count)?;
        for ((target, range), target_plan) in targets.iter().zip(&ranges).zip(&plan.targets) {
            if !target_plan.active {
                continue;
            }
            let range_offset = u64::try_from(range.offset)
                .map_err(|_| CudaError::LengthTooLarge { len: range.offset })?;
            let output_ptr = base_ptr
                .checked_add(range_offset)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            kernel_jobs.push(CudaJ2kStoreRgb8MctBatchJob {
                plane0_ptr: target.plane0.device_ptr(),
                plane1_ptr: target.plane1.device_ptr(),
                plane2_ptr: target.plane2.device_ptr(),
                output_ptr,
                job: target.job,
                reserved_tail: 0,
            });
        }
        ensure_internal_count(
            kernel_jobs.len(),
            plan.active_job_count,
            "J2K contiguous store active-job count mismatch",
        )?;
        let jobs_buffer = self.upload(store_rgb8_mct_batch_jobs_as_bytes(&kernel_jobs))?;
        if initialize_output()? {
            self.synchronize()?;
        }
        // SAFETY: this owned path retains every plane, output, and uploaded
        // job buffer through the immediate context completion boundary.
        unsafe {
            self.launch_j2k_store_rgb8_mct_batch_enqueue(
                &jobs_buffer,
                plan.max_pixels,
                plan.active_job_count,
            )?;
        }
        self.synchronize()?;
        Ok(CudaKernelContiguousBatchOutput {
            output,
            ranges,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }
}

#[cfg(test)]
mod tests;
