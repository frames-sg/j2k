// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::{ensure_context_ownership, CudaContext},
    error::CudaError,
    j2k_decode::types::{CudaJ2kStoreRgb8MctTarget, CudaJ2kStoreRgbNativeJob},
    memory::CudaDeviceBuffer,
};

pub(super) const STORE_CONTEXT_MISMATCH: &str =
    "J2K MCT/store input buffers must belong to the launch context";

pub(super) fn validate_store_context_matches(
    matches_context: impl IntoIterator<Item = bool>,
) -> Result<(), CudaError> {
    // Empty inputs are valid no-ops, so the empty iterator intentionally passes.
    ensure_context_ownership(matches_context, STORE_CONTEXT_MISMATCH)
}

pub(super) fn validate_store_buffer_context<'a>(
    context: &CudaContext,
    buffers: impl IntoIterator<Item = &'a CudaDeviceBuffer>,
) -> Result<(), CudaError> {
    validate_store_context_matches(
        buffers
            .into_iter()
            .map(|buffer| buffer.is_owned_by(context)),
    )
}

pub(super) fn validate_rgb8_mct_target_context(
    context: &CudaContext,
    targets: &[CudaJ2kStoreRgb8MctTarget<'_>],
) -> Result<(), CudaError> {
    validate_store_buffer_context(
        context,
        targets
            .iter()
            .flat_map(|target| [target.plane0, target.plane1, target.plane2]),
    )
}

pub(super) const INVERSE_MCT_PLANE_OVERLAP: &str =
    "J2K inverse MCT planes must be pairwise disjoint";

pub(super) fn validate_inverse_mct_overlap_flags(overlaps: [bool; 3]) -> Result<(), CudaError> {
    if overlaps.into_iter().any(|overlaps| overlaps) {
        return Err(CudaError::InvalidArgument {
            message: INVERSE_MCT_PLANE_OVERLAP.to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_inverse_mct_planes_disjoint(
    planes: [&CudaDeviceBuffer; 3],
) -> Result<(), CudaError> {
    validate_inverse_mct_overlap_flags([
        planes[0].overlaps(planes[1])?,
        planes[0].overlaps(planes[2])?,
        planes[1].overlaps(planes[2])?,
    ])
}

pub(super) fn validate_rgb_tile_compatibility(
    previous: &CudaJ2kStoreRgbNativeJob,
    current: &CudaJ2kStoreRgbNativeJob,
) -> Result<(), CudaError> {
    if previous.output_width != current.output_width
        || previous.output_height != current.output_height
        || previous.bit_depth0 != current.bit_depth0
        || previous.bit_depth1 != current.bit_depth1
        || previous.bit_depth2 != current.bit_depth2
        || previous.layout != current.layout
        || previous.transform != current.transform
    {
        return Err(CudaError::InvalidArgument {
            message: "tile stores for one exact-native RGB output have incompatible metadata"
                .to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_store_plane(
    plane: &CudaDeviceBuffer,
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
) -> Result<(), CudaError> {
    validate_store_plane_layout(
        plane.byte_len(),
        input_width,
        source_x,
        source_y,
        copy_width,
        copy_height,
    )
}

pub(super) fn validate_store_plane_layout(
    plane_bytes: usize,
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
) -> Result<(), CudaError> {
    if source_x
        .checked_add(copy_width)
        .is_none_or(|end_x| end_x > input_width)
    {
        return Err(CudaError::LengthTooLarge { len: plane_bytes });
    }
    let required_samples = if copy_width == 0 || copy_height == 0 {
        0
    } else {
        let last_row = u64::from(source_y) + u64::from(copy_height) - 1;
        let last_column = u64::from(source_x) + u64::from(copy_width) - 1;
        let last_sample = last_row
            .checked_mul(u64::from(input_width))
            .and_then(|row| row.checked_add(last_column))
            .ok_or(CudaError::LengthTooLarge { len: plane_bytes })?;
        if last_sample > u64::from(u32::MAX) {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "J2K store source sample index {last_sample} exceeds the CUDA u32 kernel ABI"
                ),
            });
        }
        usize::try_from(last_sample + 1)
            .map_err(|_| CudaError::LengthTooLarge { len: plane_bytes })?
    };
    let required_bytes = required_samples
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CudaError::LengthTooLarge { len: plane_bytes })?;
    if required_bytes > plane_bytes {
        return Err(CudaError::LengthTooLarge {
            len: required_bytes,
        });
    }
    Ok(())
}
