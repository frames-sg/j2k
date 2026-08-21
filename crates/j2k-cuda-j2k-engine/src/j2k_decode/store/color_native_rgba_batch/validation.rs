// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{NativeRgbaStorage, RGBA_CHANNELS};
use crate::{
    allocation::HostPhaseBudget,
    context::CudaContext,
    error::CudaError,
    j2k_decode::{
        store::{
            destination_groups::{plan_dense_destination_regions, DenseDestinationRegion},
            validation::{validate_store_buffer_context, validate_store_plane},
        },
        types::{CudaJ2kStoreRgbaNativeJob, CudaJ2kStoreRgbaNativeTarget},
    },
    kernels::j2k_store_batch_launch_geometry,
    memory::{checked_image_words, CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut},
};

pub(super) struct NativeRgbaBatchItemPlan {
    pub(super) range_index: usize,
    pub(super) active: bool,
}

pub(super) struct NativeRgbaBatchPlan {
    pub(super) items: Vec<NativeRgbaBatchItemPlan>,
    pub(super) ranges: Vec<CudaDeviceBufferRange>,
    pub(super) total_bytes: usize,
    pub(super) max_pixels: usize,
    pub(super) active_count: usize,
}

pub(super) fn validate_targets(
    context: &CudaContext,
    targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
    storage: NativeRgbaStorage,
) -> Result<NativeRgbaBatchPlan, CudaError> {
    validate_store_buffer_context(
        context,
        targets
            .iter()
            .flat_map(|target| [target.plane0, target.plane1, target.plane2, target.plane3]),
    )?;
    let mut budget = HostPhaseBudget::new("CUDA exact-native RGBA store batch plan");
    let mut regions = budget.try_vec_with_capacity(targets.len())?;
    let mut active = budget.try_vec_with_capacity(targets.len())?;
    let mut max_pixels = 0usize;
    let mut active_count = 0usize;
    for (index, target) in targets.iter().enumerate() {
        let pixels = validate_target(target, storage)?;
        if pixels != 0 {
            active_count = active_count
                .checked_add(1)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            max_pixels = max_pixels.max(pixels);
        }
        if index != 0 && target.output_index == targets[index - 1].output_index {
            validate_tile_compatibility(&targets[index - 1].job, &target.job)?;
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
        RGBA_CHANNELS,
        storage.bytes_per_sample(),
        &mut budget,
    )?;
    if destinations.requires_zero_fill {
        return Err(CudaError::InvalidArgument {
            message: "exact-native RGBA tile stores must cover each dense destination exactly"
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
            .map(|(range_index, active)| NativeRgbaBatchItemPlan {
                range_index,
                active,
            }),
    );
    if active_count != 0 && j2k_store_batch_launch_geometry(max_pixels, active_count).is_none() {
        return Err(CudaError::LengthTooLarge { len: active_count });
    }
    Ok(NativeRgbaBatchPlan {
        items,
        ranges: destinations.ranges,
        total_bytes: destinations.total_bytes,
        max_pixels,
        active_count,
    })
}

fn validate_target(
    target: &CudaJ2kStoreRgbaNativeTarget<'_>,
    storage: NativeRgbaStorage,
) -> Result<usize, CudaError> {
    let job = target.job;
    if !matches!(job.layout, 0 | 1) {
        return Err(CudaError::InvalidArgument {
            message: "exact-native RGBA layout must be NHWC or NCHW".to_string(),
        });
    }
    if job.transform > 2 || job.reserved != 0 {
        return Err(CudaError::InvalidArgument {
            message: "exact-native RGBA transform or reserved field is invalid".to_string(),
        });
    }
    if [
        job.bit_depth0,
        job.bit_depth1,
        job.bit_depth2,
        job.bit_depth3,
    ]
    .into_iter()
    .any(|precision| precision == 0 || precision > storage.max_precision())
    {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "exact-native RGBA precision must fit in {}-bit storage",
                storage.max_precision()
            ),
        });
    }
    let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
    if pixels != 0 {
        for (plane, width, x, y) in [
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
            (
                target.plane3,
                job.input_width3,
                job.source_x3,
                job.source_y3,
            ),
        ] {
            validate_store_plane(plane, width, x, y, job.copy_width, job.copy_height)?;
        }
    }
    Ok(pixels)
}

fn validate_tile_compatibility(
    previous: &CudaJ2kStoreRgbaNativeJob,
    current: &CudaJ2kStoreRgbaNativeJob,
) -> Result<(), CudaError> {
    if previous.output_width != current.output_width
        || previous.output_height != current.output_height
        || previous.bit_depth0 != current.bit_depth0
        || previous.bit_depth1 != current.bit_depth1
        || previous.bit_depth2 != current.bit_depth2
        || previous.bit_depth3 != current.bit_depth3
        || previous.layout != current.layout
        || previous.transform != current.transform
    {
        return Err(CudaError::InvalidArgument {
            message: "tile stores for one exact-native RGBA output have incompatible metadata"
                .to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_external(
    context: &CudaContext,
    destination: &CudaExternalDeviceBufferViewMut<'_>,
    plan: &NativeRgbaBatchPlan,
    storage: NativeRgbaStorage,
) -> Result<u64, CudaError> {
    if !context.is_same_context(destination.context()) {
        return Err(CudaError::InvalidArgument {
            message: "external exact-native RGBA destination has a different CUDA context"
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
        .is_multiple_of(storage.bytes_per_sample() as u64)
    {
        return Err(CudaError::InvalidArgument {
            message: "external exact-native RGBA destination is misaligned".to_string(),
        });
    }
    Ok(destination.device_ptr())
}
