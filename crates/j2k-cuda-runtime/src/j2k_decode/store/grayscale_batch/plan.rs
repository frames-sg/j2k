// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    destination_groups::{plan_dense_destination_regions, DenseDestinationRegion},
    validation::{validate_store_buffer_context, validate_store_plane},
};
use crate::{
    allocation::HostPhaseBudget,
    context::CudaContext,
    error::CudaError,
    j2k_decode::types::{
        CudaJ2kStoreGray16Target, CudaJ2kStoreGray8Target, CudaJ2kStoreGrayI16Target,
    },
    kernels::j2k_store_batch_launch_geometry,
    memory::{checked_image_words, CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut},
};

pub(super) struct GrayBatchItemPlan {
    pub(super) range_index: usize,
    pub(super) active: bool,
}

pub(super) struct GrayBatchPlan {
    pub(super) items: Vec<GrayBatchItemPlan>,
    pub(super) ranges: Vec<CudaDeviceBufferRange>,
    pub(super) total_bytes: usize,
    pub(super) max_pixels: usize,
    pub(super) active_count: usize,
    pub(super) requires_zero_fill: bool,
}

pub(super) fn validate_gray_targets<'a, T>(
    context: &CudaContext,
    targets: &'a [T],
    bytes_per_sample: usize,
    input: impl Fn(&'a T) -> &'a crate::memory::CudaDeviceBuffer,
    output_index: impl Fn(&T) -> usize,
    geometry: impl Fn(&T) -> (u32, u32, u32, u32, u32, u32, u32, u32, u32),
    live_host_bytes: usize,
) -> Result<GrayBatchPlan, CudaError> {
    validate_store_buffer_context(context, targets.iter().map(&input))?;
    let mut budget =
        HostPhaseBudget::with_live_bytes("CUDA J2K grayscale store batch plan", live_host_bytes)?;
    let mut regions = budget.try_vec_with_capacity(targets.len())?;
    let mut active = budget.try_vec_with_capacity(targets.len())?;
    let mut max_pixels = 0usize;
    let mut active_count = 0usize;

    for target in targets {
        let (
            input_width,
            source_x,
            source_y,
            copy_width,
            copy_height,
            output_width,
            output_height,
            output_x,
            output_y,
        ) = geometry(target);
        let pixels = checked_image_words(copy_width, copy_height, 1)?;
        if pixels != 0 {
            validate_store_plane(
                input(target),
                input_width,
                source_x,
                source_y,
                copy_width,
                copy_height,
            )?;
            active_count = active_count
                .checked_add(1)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            max_pixels = max_pixels.max(pixels);
        }
        regions.push(DenseDestinationRegion {
            output_index: output_index(target),
            output_width,
            output_height,
            output_x,
            output_y,
            copy_width,
            copy_height,
        });
        active.push(pixels != 0);
    }

    let destinations = plan_dense_destination_regions(&regions, 1, bytes_per_sample, &mut budget)?;
    let mut items = budget.try_vec_with_capacity(targets.len())?;
    items.extend(
        destinations
            .item_range_indices
            .iter()
            .copied()
            .zip(active)
            .map(|(range_index, active)| GrayBatchItemPlan {
                range_index,
                active,
            }),
    );

    if active_count != 0 && j2k_store_batch_launch_geometry(max_pixels, active_count).is_none() {
        return Err(CudaError::LengthTooLarge { len: active_count });
    }

    Ok(GrayBatchPlan {
        items,
        ranges: destinations.ranges,
        total_bytes: destinations.total_bytes,
        max_pixels,
        active_count,
        requires_zero_fill: destinations.requires_zero_fill,
    })
}

pub(super) fn gray8_geometry(
    target: &CudaJ2kStoreGray8Target<'_>,
) -> (u32, u32, u32, u32, u32, u32, u32, u32, u32) {
    let job = target.job;
    (
        job.input_width,
        job.source_x,
        job.source_y,
        job.copy_width,
        job.copy_height,
        job.output_width,
        job.output_height,
        job.output_x,
        job.output_y,
    )
}

pub(super) fn gray16_geometry(
    target: &CudaJ2kStoreGray16Target<'_>,
) -> (u32, u32, u32, u32, u32, u32, u32, u32, u32) {
    let job = target.job;
    (
        job.input_width,
        job.source_x,
        job.source_y,
        job.copy_width,
        job.copy_height,
        job.output_width,
        job.output_height,
        job.output_x,
        job.output_y,
    )
}

pub(super) fn grayi16_geometry(
    target: &CudaJ2kStoreGrayI16Target<'_>,
) -> (u32, u32, u32, u32, u32, u32, u32, u32, u32) {
    let job = target.job;
    (
        job.input_width,
        job.source_x,
        job.source_y,
        job.copy_width,
        job.copy_height,
        job.output_width,
        job.output_height,
        job.output_x,
        job.output_y,
    )
}

pub(super) fn external_destination_base(
    context: &CudaContext,
    destination: &CudaExternalDeviceBufferViewMut<'_>,
    plan: &GrayBatchPlan,
    alignment: usize,
) -> Result<u64, CudaError> {
    if !context.is_same_context(destination.context()) {
        return Err(CudaError::InvalidArgument {
            message: "external grayscale destination belongs to a different CUDA context"
                .to_string(),
        });
    }
    if destination.byte_len() < plan.total_bytes {
        return Err(CudaError::OutputTooSmall {
            required: plan.total_bytes,
            have: destination.byte_len(),
        });
    }
    if !destination.device_ptr().is_multiple_of(alignment as u64) {
        return Err(CudaError::InvalidArgument {
            message: format!("external grayscale destination is not {alignment}-byte aligned"),
        });
    }
    if plan.requires_zero_fill {
        return Err(CudaError::InvalidArgument {
            message: "external grayscale batch destination requires full output coverage"
                .to_string(),
        });
    }
    Ok(destination.device_ptr())
}
