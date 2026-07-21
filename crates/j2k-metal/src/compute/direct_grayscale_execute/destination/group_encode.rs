// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{mem::size_of, sync::Arc};

use super::super::{
    encode_prepared_direct_grayscale_plan_into_in_encoder,
    encode_stacked_direct_component_plane_batch, j2k_scalar_pack_params,
    supports_stacked_direct_component_plane_batch, DirectColorBatchCommandBuffers,
    DirectExecutionMetadata, DirectGrayscaleDestinationExecutionRequest, DirectHybridStageTimings,
    DirectTier1Mode, Error, MetalRuntime, PixelFormat, PreparedDirectGrayscalePlan,
    StackedDirectComponentPlaneBatchRequest,
};
use crate::compute::abi::J2kRepeatedGrayStoreParams;
use j2k_metal_support::{dispatch_3d_pipeline, MetalImageDestination};

use super::super::destination_index_validation::validate_stacked_grayscale_destination_indices;

pub(super) struct GrayscaleGroupEncoder<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a metal::CommandBufferRef,
    pub(super) compute_encoder: &'a metal::ComputeCommandEncoderRef,
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
        if self.plans.len() > 1 && supports_stacked_direct_component_plane_batch(&refs) {
            self.encode_stacked(&refs)
        } else {
            self.encode_individually()
        }
    }

    fn encode_stacked(&mut self, plans: &[&PreparedDirectGrayscalePlan]) -> Result<(), Error> {
        let first = &self.plans[0];
        let status_start = self.metadata.status_checks.len();
        let mut stage_timings = DirectHybridStageTimings::default();
        let stacked =
            encode_stacked_direct_component_plane_batch(StackedDirectComponentPlaneBatchRequest {
                runtime: self.runtime,
                command_buffers: DirectColorBatchCommandBuffers::single(self.command_buffer),
                compute_encoder: Some(self.compute_encoder),
                plans,
                component_idx: 0,
                flattened_cpu_tier1_cache: None,
                tier1_mode: DirectTier1Mode::Metal,
                stage_timings: &mut stage_timings,
                retained_buffers: &mut self.metadata.retained_buffers,
                status_checks: &mut self.metadata.status_checks,
                scratch_buffers: &mut self.metadata.scratch_buffers,
            })?;
        if stacked.dimensions != first.dimensions || stacked.count != self.plans.len() {
            return Err(Error::MetalStateInvariant {
                state: "J2K Metal stacked grayscale destination",
                reason: "stacked component output does not match prepared group",
            });
        }
        encode_stacked_grayscale_destination(
            self.runtime,
            self.compute_encoder,
            &stacked.buffer,
            first,
            self.fmt,
            self.plans.len(),
            self.destination,
        )?;
        if let Some(sources) = self.source_indices {
            for status in &mut self.metadata.status_checks[status_start..] {
                status.remap_sources(sources)?;
            }
        }
        Ok(())
    }

    fn encode_individually(&mut self) -> Result<(), Error> {
        for (index, plan) in self.plans.iter().enumerate() {
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
        }
        Ok(())
    }
}

fn encode_stacked_grayscale_destination(
    runtime: &MetalRuntime,
    encoder: &metal::ComputeCommandEncoderRef,
    plane: &metal::Buffer,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
    destination: &MetalImageDestination,
) -> Result<(), Error> {
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
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal stacked grayscale destination must be one dense contiguous group",
        });
    }
    validate_stacked_grayscale_destination_indices(plan.dimensions, count)?;
    let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
    let max_value = if fmt == PixelFormat::GrayI16 {
        let signed_bits = u32::from(plan.bit_depth).clamp(1, 16);
        f32::from(
            u16::try_from((1_u32 << (signed_bits - 1)) - 1)
                .expect("signed bit depth is clamped to 16 bits"),
        )
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
    let pipeline = match fmt {
        PixelFormat::Gray8 => &runtime.store_component_repeated_gray_u8,
        PixelFormat::Gray16 => &runtime.store_component_repeated_gray_u16,
        PixelFormat::GrayI16 => &runtime.store_component_repeated_gray_i16,
        _ => {
            return Err(Error::UnsupportedMetalRequest {
                reason: "J2K Metal stacked grayscale destination supports Gray8/Gray16/GrayI16",
            });
        }
    };
    encoder.memory_barrier_with_resources(&[plane]);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), 0);
    // SAFETY: the checked destination owns this exact dense group range until
    // the submitted command buffer has completed.
    encoder.set_buffer(
        1,
        Some(unsafe { destination.raw_buffer() }),
        u64::try_from(layout.byte_offset()).map_err(|_| Error::MetalKernel {
            message: "J2K Metal stacked grayscale destination offset exceeds u64".to_string(),
        })?,
    );
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedGrayStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        pipeline,
        (params.copy_width, params.copy_height, params.batch_count),
    );
    Ok(())
}
