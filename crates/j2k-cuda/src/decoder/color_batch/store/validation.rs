// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{CudaHtj2kStoreStep, Error, CUDA_HTJ2K_KERNELS_NOT_READY};

pub(super) fn bit_depth_addend(bit_depth: u8) -> f32 {
    let shift = bit_depth.saturating_sub(1).min(15);
    f32::from(1_u16 << shift)
}

pub(in crate::decoder::color_batch) fn validate_color_stores(
    stores: [&CudaHtj2kStoreStep; 3],
    dimensions: (u32, u32),
) -> Result<(), Error> {
    let first = stores[0];
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
        if store.output_x != 0
            || store.output_y != 0
            || store.copy_width != dimensions.0
            || store.copy_height != dimensions.1
            || store.output_width != dimensions.0
            || store.output_height != dimensions.1
            || source_end_x > input_width
            || source_end_y > input_height
            || store.source_x != first.source_x
            || store.source_y != first.source_y
        {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
    }
    Ok(())
}
