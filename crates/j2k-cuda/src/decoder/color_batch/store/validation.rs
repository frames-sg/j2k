// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{CudaHtj2kStoreStep, Error, CUDA_HTJ2K_KERNELS_NOT_READY};

pub(super) fn bit_depth_addend(bit_depth: u8) -> f32 {
    let shift = bit_depth.saturating_sub(1).min(15);
    f32::from(1_u16 << shift)
}

pub(in crate::decoder::color_batch) fn validate_color_stores<const N: usize>(
    stores: [&CudaHtj2kStoreStep; N],
    dimensions: (u32, u32),
) -> Result<(), Error> {
    let first = stores.first().ok_or(Error::UnsupportedCudaRequest {
        reason: CUDA_HTJ2K_KERNELS_NOT_READY,
    })?;
    for store in stores {
        let input_width = store.input_rect.x1.saturating_sub(store.input_rect.x0);
        let input_height = store.input_rect.y1.saturating_sub(store.input_rect.y0);
        let source_end_x =
            store
                .source_x
                .checked_add(store.copy_width)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                })?;
        let source_end_y =
            store
                .source_y
                .checked_add(store.copy_height)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                })?;
        let output_end_x =
            store
                .output_x
                .checked_add(store.copy_width)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                })?;
        let output_end_y =
            store
                .output_y
                .checked_add(store.copy_height)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                })?;
        if store.output_width != dimensions.0
            || store.output_height != dimensions.1
            || output_end_x > dimensions.0
            || output_end_y > dimensions.1
            || source_end_x > input_width
            || source_end_y > input_height
            || store.source_x != first.source_x
            || store.source_y != first.source_y
            || store.copy_width != first.copy_width
            || store.copy_height != first.copy_height
            || store.output_x != first.output_x
            || store.output_y != first.output_y
        {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
    }
    Ok(())
}
