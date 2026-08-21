// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    context::CudaContext,
    error::CudaError,
    j2k_decode::types::CudaJ2kStoreRgbNativeTarget,
    kernels::j2k_store_batch_launch_geometry,
    memory::{checked_image_words, CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut},
};

use super::super::{
    destination_groups::{plan_dense_destination_regions, DenseDestinationRegion},
    validation::{
        validate_rgb_tile_compatibility, validate_store_buffer_context, validate_store_plane,
    },
};

const RGB_CHANNELS: usize = 3;
pub(super) const RGB_LAYOUT_NHWC: u32 = 0;
pub(super) const RGB_LAYOUT_NCHW: u32 = 1;
const RGB_TRANSFORM_NONE: u32 = 0;
const RGB_TRANSFORM_ICT: u32 = 2;

#[derive(Clone, Copy)]
pub(super) enum NativeRgbStorage {
    U8,
    U16,
    I16,
}

impl NativeRgbStorage {
    const fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => std::mem::size_of::<u8>(),
            Self::U16 => std::mem::size_of::<u16>(),
            Self::I16 => std::mem::size_of::<i16>(),
        }
    }

    pub(super) const fn max_precision(self) -> u32 {
        match self {
            Self::U8 => 8,
            Self::U16 | Self::I16 => 16,
        }
    }

    const fn alignment(self) -> usize {
        self.bytes_per_sample()
    }
}

pub(super) struct NativeRgbBatchItemPlan {
    pub(super) range_index: usize,
    pub(super) active: bool,
}

pub(super) struct NativeRgbBatchPlan {
    pub(super) items: Vec<NativeRgbBatchItemPlan>,
    pub(super) ranges: Vec<CudaDeviceBufferRange>,
    pub(super) total_bytes: usize,
    pub(super) max_pixels: usize,
    pub(super) active_count: usize,
}

pub(super) fn validate_native_rgb_targets(
    context: &CudaContext,
    targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
    storage: NativeRgbStorage,
) -> Result<NativeRgbBatchPlan, CudaError> {
    validate_store_buffer_context(
        context,
        targets
            .iter()
            .flat_map(|target| [target.plane0, target.plane1, target.plane2]),
    )?;
    let mut budget = HostPhaseBudget::new("CUDA exact-native RGB store batch plan");
    let mut regions = budget.try_vec_with_capacity(targets.len())?;
    let mut active = budget.try_vec_with_capacity(targets.len())?;
    let mut max_pixels = 0usize;
    let mut active_count = 0usize;

    for (index, target) in targets.iter().enumerate() {
        let pixels = validate_native_rgb_target(target, storage)?;
        if pixels != 0 {
            active_count = active_count
                .checked_add(1)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            max_pixels = max_pixels.max(pixels);
        }
        if index != 0 && target.output_index == targets[index - 1].output_index {
            validate_rgb_tile_compatibility(&targets[index - 1].job, &target.job)?;
        }
        let job = target.job;
        regions.push(DenseDestinationRegion {
            output_index: target.output_index,
            output_width: job.output_width,
            output_height: job.output_height,
            output_x: job.output_x,
            output_y: job.output_y,
            copy_width: job.copy_width,
            copy_height: job.copy_height,
        });
        active.push(pixels != 0);
    }
    let destinations = plan_dense_destination_regions(
        &regions,
        RGB_CHANNELS,
        storage.bytes_per_sample(),
        &mut budget,
    )?;
    if destinations.requires_zero_fill {
        return Err(CudaError::InvalidArgument {
            message: "exact-native RGB tile stores must cover each dense destination exactly"
                .to_string(),
        });
    }
    let mut items = budget.try_vec_with_capacity(targets.len())?;
    items.extend(
        destinations
            .item_range_indices
            .iter()
            .copied()
            .zip(active)
            .map(|(range_index, active)| NativeRgbBatchItemPlan {
                range_index,
                active,
            }),
    );
    if active_count != 0 && j2k_store_batch_launch_geometry(max_pixels, active_count).is_none() {
        return Err(CudaError::LengthTooLarge { len: active_count });
    }
    Ok(NativeRgbBatchPlan {
        items,
        ranges: destinations.ranges,
        total_bytes: destinations.total_bytes,
        max_pixels,
        active_count,
    })
}

fn validate_native_rgb_target(
    target: &CudaJ2kStoreRgbNativeTarget<'_>,
    storage: NativeRgbStorage,
) -> Result<usize, CudaError> {
    let job = target.job;
    if !matches!(job.layout, RGB_LAYOUT_NHWC | RGB_LAYOUT_NCHW) {
        return Err(CudaError::InvalidArgument {
            message: "exact-native RGB layout must be NHWC or NCHW".to_string(),
        });
    }
    if !(RGB_TRANSFORM_NONE..=RGB_TRANSFORM_ICT).contains(&job.transform) {
        return Err(CudaError::InvalidArgument {
            message: "exact-native RGB transform selector is invalid".to_string(),
        });
    }
    if job.reserved != 0 {
        return Err(CudaError::InvalidArgument {
            message: "exact-native RGB reserved field must be zero".to_string(),
        });
    }
    let max_precision = storage.max_precision();
    if [job.bit_depth0, job.bit_depth1, job.bit_depth2]
        .into_iter()
        .any(|precision| precision == 0 || precision > max_precision)
    {
        return Err(CudaError::InvalidArgument {
            message: format!("exact-native RGB precision must fit in {max_precision}-bit storage"),
        });
    }
    let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
    if pixels != 0 {
        for (plane, input_width, source_x, source_y) in [
            (
                target.plane0,
                job.input_width0,
                job.source_x0,
                job.source_y0,
            ),
            (
                target.plane1,
                job.input_width1,
                job.source_x1,
                job.source_y1,
            ),
            (
                target.plane2,
                job.input_width2,
                job.source_x2,
                job.source_y2,
            ),
        ] {
            validate_store_plane(
                plane,
                input_width,
                source_x,
                source_y,
                job.copy_width,
                job.copy_height,
            )?;
        }
    }
    Ok(pixels)
}

pub(super) fn validate_external_destination(
    context: &CudaContext,
    destination: &CudaExternalDeviceBufferViewMut<'_>,
    plan: &NativeRgbBatchPlan,
    storage: NativeRgbStorage,
) -> Result<u64, CudaError> {
    if !context.is_same_context(destination.context()) {
        return Err(CudaError::InvalidArgument {
            message: "external exact-native RGB destination belongs to a different CUDA context"
                .to_string(),
        });
    }
    if destination.byte_len() < plan.total_bytes {
        return Err(CudaError::OutputTooSmall {
            required: plan.total_bytes,
            have: destination.byte_len(),
        });
    }
    if !destination
        .device_ptr()
        .is_multiple_of(storage.alignment() as u64)
    {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "external exact-native RGB destination is not {}-byte aligned",
                storage.alignment()
            ),
        });
    }
    Ok(destination.device_ptr())
}
