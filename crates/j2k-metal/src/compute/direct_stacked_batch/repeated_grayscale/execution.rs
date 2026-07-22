// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Buffer, CommandBufferRef, DirectScratchBuffer, DirectStatusCheck, Error, MetalRuntime,
    PixelFormat, RepeatedDirectGrayscalePlanRequest, Surface,
};
use crate::compute::direct_stacked_batch::DirectBandSlice;
use crate::compute::PreparedDirectGrayscaleStep;

mod final_store;
mod reconstruction;
mod tier1;

struct RepeatedGrayscaleExecution<'a> {
    runtime: &'a MetalRuntime,
    command_buffer: &'a CommandBufferRef,
    fmt: PixelFormat,
    dimensions: (u32, u32),
    bit_depth: u8,
    count: usize,
    retained_buffers: &'a mut Vec<Buffer>,
    status_checks: &'a mut Vec<DirectStatusCheck>,
    scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
    band_sets: Vec<Vec<DirectBandSlice>>,
    surfaces: Vec<Surface>,
    stacked_outputs: bool,
}

impl RepeatedGrayscaleExecution<'_> {
    fn encode_step(&mut self, step: &PreparedDirectGrayscaleStep) -> Result<(), Error> {
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                self.encode_classic_sub_band(sub_band)
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => self.encode_ht_sub_band(sub_band),
            PreparedDirectGrayscaleStep::Idwt(idwt) => self.encode_idwt(idwt),
            PreparedDirectGrayscaleStep::Store(store) => self.encode_store(store),
        }
    }

    fn finish(self) -> Result<Vec<Surface>, Error> {
        if self.surfaces.len() != self.count {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K MetalDirect repeated grayscale plan produced {} surfaces for count {}",
                    self.surfaces.len(),
                    self.count
                ),
            });
        }
        Ok(self.surfaces)
    }
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_repeated_direct_grayscale_plan_in_command_buffer(
    request: RepeatedDirectGrayscalePlanRequest<'_>,
) -> Result<Vec<Surface>, Error> {
    let RepeatedDirectGrayscalePlanRequest {
        runtime,
        command_buffer,
        plan,
        fmt,
        count,
        retained_buffers,
        status_checks,
        scratch_buffers,
    } = request;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal repeated grayscale execution metadata",
    );
    let total_band_capacity = crate::batch_allocation::checked_count_product(
        count,
        plan.steps.len(),
        "J2K Metal repeated grayscale band metadata",
    )?;
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<Vec<DirectBandSlice>>(count),
        crate::batch_allocation::BatchMetadataRequest::of::<DirectBandSlice>(total_band_capacity),
        crate::batch_allocation::BatchMetadataRequest::of::<Surface>(count),
    ])?;
    let band_sets = super::super::resources::allocate_preflighted_direct_band_sets(
        count,
        plan.steps.len(),
        &mut budget,
    )?;
    let surfaces = budget.try_vec(count, "J2K Metal repeated grayscale execution surfaces")?;
    let mut execution = RepeatedGrayscaleExecution {
        runtime,
        command_buffer,
        fmt,
        dimensions: plan.dimensions,
        bit_depth: plan.bit_depth,
        count,
        retained_buffers,
        status_checks,
        scratch_buffers,
        band_sets,
        surfaces,
        stacked_outputs: true,
    };
    let mut step_idx = 0;
    while step_idx < plan.steps.len() {
        if let Some(group) = plan.classic_group_starting_at(step_idx) {
            execution.encode_classic_group(group)?;
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            execution.encode_ht_group(group)?;
            step_idx = group.end_step;
            continue;
        }
        execution.encode_step(&plan.steps[step_idx])?;
        step_idx += 1;
    }
    execution.finish()
}
