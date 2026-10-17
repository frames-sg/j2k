// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::sync::Arc;

use super::super::{
    encode_prepared_direct_grayscale_plan_into_in_encoder,
    encode_stacked_direct_component_plane_batch, encode_stacked_direct_gray_batch,
    j2k_scalar_pack_params, supports_stacked_direct_component_plane_batch,
    DirectColorBatchCommandBuffers, DirectExecutionMetadata,
    DirectGrayscaleDestinationExecutionRequest, DirectHybridStageTimings, DirectTier1Mode, Error,
    MetalRuntime, PixelFormat, PreparedDirectGrayscalePlan,
    StackedDirectComponentPlaneBatchRequest, StackedGrayBatch, StackedGrayOutput,
};
use crate::engine::abi::J2kRepeatedGrayStoreParams;
use crate::engine::decode_dispatch::store::GrayStoreKind;
use j2k_metal_support::{dispatch_3d_pipeline, MetalImageDestination};

use super::super::destination_index_validation::validate_stacked_grayscale_destination_indices;

pub(super) struct GrayscaleGroupEncoder<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a crate::metal_types::CommandBufferRef,
    pub(super) compute_encoder: &'a crate::metal_types::ComputeCommandEncoderRef,
    pub(super) plans: &'a [Arc<PreparedDirectGrayscalePlan>],
    pub(super) fmt: PixelFormat,
    pub(super) destination: &'a MetalImageDestination,
    pub(super) source_indices: Option<&'a [usize]>,
    pub(super) metadata: &'a mut DirectExecutionMetadata,
}

impl GrayscaleGroupEncoder<'_> {
    pub(super) fn encode(&mut self) -> Result<(), Error> {
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal submitted stacked grayscale plan references",
        );
        let mut refs = budget.try_vec(
            self.plans.len(),
            "J2K Metal submitted stacked grayscale plan reference slots",
        )?;
        refs.extend(self.plans.iter().map(Arc::as_ref));
        if self.plans.len() < 2 || !supports_stacked_direct_component_plane_batch(&refs) {
            for index in 0..self.plans.len() {
                self.encode_individually(index)?;
            }
            return Ok(());
        }
        let first = &self.plans[0];
        if self.plans.iter().all(|plan| Arc::ptr_eq(plan, first)) {
            return self.encode_stacked(0, &refs);
        }
        // Each stacked submission keeps its images' coefficients in single
        // allocations, so a group too large for one allocation is decoded in
        // runs that each fit.
        let run = stacked_run_len(self.runtime, first.dimensions);
        for start in (0..refs.len()).step_by(run) {
            let end = (start + run).min(refs.len());
            if end - start == 1 {
                self.encode_individually(start)?;
            } else {
                self.encode_stacked(start, &refs[start..end])?;
            }
        }
        Ok(())
    }

    /// Encodes `plans`, the group's images from `start`, as one stacked
    /// submission into their destination ranges.
    fn encode_stacked(
        &mut self,
        start: usize,
        plans: &[&PreparedDirectGrayscalePlan],
    ) -> Result<(), Error> {
        let first = plans[0];
        let broadcast = plans.len() == self.plans.len()
            && self
                .plans
                .iter()
                .all(|plan| Arc::ptr_eq(plan, &self.plans[0]));
        let execution_plans = if broadcast { &plans[..1] } else { plans };
        let status_start = self.metadata.status_checks.len();
        let mut stage_timings = DirectHybridStageTimings::default();
        let request = StackedDirectComponentPlaneBatchRequest {
            runtime: self.runtime,
            command_buffers: DirectColorBatchCommandBuffers::single(self.command_buffer),
            compute_encoder: Some(self.compute_encoder),
            plans: execution_plans,
            component_idx: 0,
            flattened_cpu_tier1_cache: None,
            tier1_mode: DirectTier1Mode::Metal,
            stage_timings: &mut stage_timings,
            retained_buffers: &mut self.metadata.retained_buffers,
            status_checks: &mut self.metadata.status_checks,
            scratch_buffers: &mut self.metadata.scratch_buffers,
        };
        // Distinct images may be stored by the last wavelet level directly;
        // a broadcast plan decodes once and is stored into every image range.
        let stacked = if broadcast {
            StackedGrayBatch::Plane(encode_stacked_direct_component_plane_batch(request)?)
        } else {
            let target = stacked_destination_target(
                self.runtime,
                self.fmt,
                self.destination,
                first,
                plans.len(),
                start,
            )?;
            // SAFETY: the checked destination owns this exact dense group range
            // until the submitted command buffer has completed, and the handle
            // is only bound by the batch's final kernel.
            let buffer = unsafe { self.destination.raw_buffer() };
            encode_stacked_direct_gray_batch(
                request,
                StackedGrayOutput {
                    buffer,
                    offset_bytes: target.offset_bytes,
                    kind: target.kind,
                    max_value: target.params.max_value,
                    u8_scale: target.params.u8_scale,
                    u16_scale: target.params.u16_scale,
                },
            )?
        };
        let store_count = if broadcast { self.plans.len() } else { 1 };
        if let StackedGrayBatch::Plane(stacked) = stacked {
            if stacked.dimensions != first.dimensions || stacked.count != execution_plans.len() {
                return Err(Error::MetalStateInvariant {
                    state: "J2K Metal stacked grayscale destination",
                    reason: "stacked component output does not match prepared group",
                });
            }
            if broadcast {
                // Immutable repeated plans share one decoded plane. Store it
                // into each caller-owned image range without repeating Tier 1
                // or reconstruction.
                for destination_image_index in 0..store_count {
                    self.encode_stacked_destination(
                        &stacked.buffer,
                        first,
                        1,
                        destination_image_index,
                    )?;
                }
            } else {
                self.encode_stacked_destination(&stacked.buffer, first, plans.len(), start)?;
            }
        }
        if broadcast {
            let source = self.source_indices.map_or(0, |indices| indices[0]);
            for status in &mut self.metadata.status_checks[status_start..] {
                status.remap_source(source)?;
            }
        } else {
            let end = start + plans.len();
            let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal stacked grayscale run sources",
            );
            let mut sources =
                budget.try_vec(plans.len(), "J2K Metal stacked run source indices")?;
            match self.source_indices {
                Some(indices) => sources.extend_from_slice(&indices[start..end]),
                None => sources.extend(start..end),
            }
            for status in &mut self.metadata.status_checks[status_start..] {
                status.remap_sources(&sources)?;
            }
        }
        self.metadata.dispatch_report.idwt = self.metadata.dispatch_report.idwt.saturating_add(
            first
                .steps
                .iter()
                .filter(|step| matches!(step, super::super::PreparedDirectGrayscaleStep::Idwt(_)))
                .count(),
        );
        self.metadata.dispatch_report.color_output = self
            .metadata
            .dispatch_report
            .color_output
            .saturating_add(store_count);
        Ok(())
    }

    fn encode_individually(&mut self, index: usize) -> Result<(), Error> {
        let plan = &self.plans[index];
        let status_start = self.metadata.status_checks.len();
        encode_prepared_direct_grayscale_plan_into_in_encoder(
            DirectGrayscaleDestinationExecutionRequest {
                runtime: self.runtime,
                command_buffer: self.command_buffer,
                plan,
                fmt: self.fmt,
                destination: self.destination,
                destination_item_index: index,
                retained_buffers: &mut self.metadata.retained_buffers,
                status_checks: &mut self.metadata.status_checks,
                scratch_buffers: &mut self.metadata.scratch_buffers,
            },
            self.compute_encoder,
        )?;
        let source_index = self.source_indices.map_or(index, |indices| indices[index]);
        for status in &mut self.metadata.status_checks[status_start..] {
            status.remap_source(source_index)?;
        }
        self.metadata.dispatch_report.idwt = self.metadata.dispatch_report.idwt.saturating_add(
            plan.steps
                .iter()
                .filter(|step| matches!(step, super::super::PreparedDirectGrayscaleStep::Idwt(_)))
                .count(),
        );
        self.metadata.dispatch_report.color_output =
            self.metadata.dispatch_report.color_output.saturating_add(1);
        Ok(())
    }

    fn encode_stacked_destination(
        &self,
        plane: &crate::metal_types::Buffer,
        plan: &PreparedDirectGrayscalePlan,
        count: usize,
        destination_image_index: usize,
    ) -> Result<(), Error> {
        let target = stacked_destination_target(
            self.runtime,
            self.fmt,
            self.destination,
            plan,
            count,
            destination_image_index,
        )?;
        let encoder = self.compute_encoder;
        encoder.memory_barrier_with_resources(&[plane]);
        encoder.setComputePipelineState(target.pipeline);
        encoder.set_buffer(0, Some(plane), 0);
        // SAFETY: the checked destination owns this exact dense group range until
        // the submitted command buffer has completed.
        encoder.set_buffer(
            1,
            Some(unsafe { self.destination.raw_buffer() }),
            u64::try_from(target.offset_bytes).map_err(|_| Error::MetalKernel {
                message: "J2K Metal stacked grayscale destination offset exceeds u64".to_string(),
            })?,
        );
        encoder.set_bytes::<J2kRepeatedGrayStoreParams>(2, &target.params);
        dispatch_3d_pipeline(
            encoder,
            target.pipeline,
            (
                target.params.copy_width,
                target.params.copy_height,
                target.params.batch_count,
            ),
        );
        Ok(())
    }
}

/// Images per stacked submission of `dimensions`: its batched coefficient
/// planes hold four bytes per sample of every image, and each must fit one
/// allocation.
fn stacked_run_len(runtime: &MetalRuntime, dimensions: (u32, u32)) -> usize {
    #[cfg(test)]
    if let Some(len) = crate::engine::test_counters::stacked_run_len_for_test() {
        return len;
    }
    let limit = runtime
        .device
        .maxBufferLength()
        .min(j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES);
    let image_bytes = (dimensions.0 as usize)
        .saturating_mul(dimensions.1 as usize)
        .saturating_mul(core::mem::size_of::<f32>())
        .max(1);
    (limit / image_bytes).max(1)
}

/// Where and how a stacked grayscale group is packed into its destination.
struct StackedDestinationTarget<'a> {
    offset_bytes: usize,
    params: J2kRepeatedGrayStoreParams,
    kind: GrayStoreKind,
    pipeline: &'a crate::metal_types::ComputePipelineState,
}

/// Validates `destination` for `count` dense images of `plan` from
/// `destination_image_index`, and returns the group pack's binding and
/// parameters.
fn stacked_destination_target<'r>(
    runtime: &'r MetalRuntime,
    fmt: PixelFormat,
    destination: &MetalImageDestination,
    plan: &PreparedDirectGrayscalePlan,
    count: usize,
    destination_image_index: usize,
) -> Result<StackedDestinationTarget<'r>, Error> {
    let layout = destination.layout();
    let bytes_per_sample = fmt.bytes_per_sample();
    let tight_row_bytes = usize::try_from(plan.dimensions.0)
        .ok()
        .and_then(|width| width.checked_mul(bytes_per_sample))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal stacked grayscale row size overflow".to_string(),
        })?;
    let tight_image_bytes = tight_row_bytes
        .checked_mul(plan.dimensions.1 as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal stacked grayscale image size overflow".to_string(),
        })?;
    if layout.pitch_bytes() != tight_row_bytes || layout.image_stride_bytes() != tight_image_bytes {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::geometry_mismatch(
                "J2K Metal stacked grayscale destination must be one dense contiguous group",
            ),
        ));
    }
    validate_stacked_grayscale_destination_indices(plan.dimensions, count)?;
    let offset_bytes = destination_image_index
        .checked_mul(layout.image_stride_bytes())
        .and_then(|offset| layout.byte_offset().checked_add(offset))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal stacked grayscale destination offset overflow".to_string(),
        })?;
    let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
    let max_value = if fmt == PixelFormat::GrayI16 {
        let signed_bits = u32::from(plan.bit_depth).clamp(1, 16);
        f32::from(u16::try_from((1_u32 << (signed_bits - 1)) - 1).unwrap_or(u16::MAX))
    } else {
        scale.max_value
    };
    let params = J2kRepeatedGrayStoreParams {
        input_width: plan.dimensions.0,
        input_height: plan.dimensions.1,
        source_x: 0,
        source_y: 0,
        copy_width: plan.dimensions.0,
        copy_height: plan.dimensions.1,
        output_width: plan.dimensions.0,
        output_height: plan.dimensions.1,
        output_x: 0,
        output_y: 0,
        addend: 0.0,
        batch_count: u32::try_from(count).map_err(|_| Error::MetalKernel {
            message: "J2K Metal stacked grayscale batch count exceeds u32".to_string(),
        })?,
        max_value,
        u8_scale: 1.0,
        u16_scale: scale.u16_scale,
    };
    let decode = runtime.decode()?;
    let (kind, pipeline) = match fmt {
        PixelFormat::Gray8 => (GrayStoreKind::U8, &decode.store_component_repeated_gray_u8),
        PixelFormat::Gray16 => (
            GrayStoreKind::U16,
            &decode.store_component_repeated_gray_u16,
        ),
        PixelFormat::GrayI16 => (
            GrayStoreKind::I16,
            &decode.store_component_repeated_gray_i16,
        ),
        _ => {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::unsupported_format(
                    "J2K Metal stacked grayscale destination supports Gray8/Gray16/GrayI16",
                ),
            ));
        }
    };
    Ok(StackedDestinationTarget {
        offset_bytes,
        params,
        kind,
        pipeline,
    })
}
