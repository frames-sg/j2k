// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::BatchLayout;
use j2k_cuda_runtime::{CudaJ2kStoreRgbNativeJob, CudaJ2kStoreRgbaNativeJob};

use crate::decoder::CudaHtj2kTransform;

pub(super) fn transform_selector(mct: bool, transform: CudaHtj2kTransform) -> u32 {
    if !mct {
        return 0;
    }
    match transform {
        CudaHtj2kTransform::Reversible53 => 1,
        CudaHtj2kTransform::Irreversible97 => 2,
    }
}

pub(super) fn native_store_job(
    stores: [&crate::CudaHtj2kStoreStep; 3],
    bit_depths: [u8; 3],
    addends: [f32; 3],
    layout: BatchLayout,
    transform: u32,
) -> CudaJ2kStoreRgbNativeJob {
    let input_width =
        |store: &crate::CudaHtj2kStoreStep| store.input_rect.x1.saturating_sub(store.input_rect.x0);
    CudaJ2kStoreRgbNativeJob {
        input_width0: input_width(stores[0]),
        input_width1: input_width(stores[1]),
        input_width2: input_width(stores[2]),
        source_x0: stores[0].source_x,
        source_y0: stores[0].source_y,
        source_x1: stores[1].source_x,
        source_y1: stores[1].source_y,
        source_x2: stores[2].source_x,
        source_y2: stores[2].source_y,
        copy_width: stores[0].copy_width,
        copy_height: stores[0].copy_height,
        output_width: stores[0].output_width,
        output_height: stores[0].output_height,
        output_x: stores[0].output_x,
        output_y: stores[0].output_y,
        addend0: addends[0],
        addend1: addends[1],
        addend2: addends[2],
        bit_depth0: u32::from(bit_depths[0]),
        bit_depth1: u32::from(bit_depths[1]),
        bit_depth2: u32::from(bit_depths[2]),
        layout: match layout {
            BatchLayout::Nhwc => 0,
            BatchLayout::Nchw => 1,
            _ => u32::MAX,
        },
        transform,
        reserved: 0,
    }
}

pub(super) fn native_rgba_store_job(
    stores: [&crate::CudaHtj2kStoreStep; 4],
    bit_depths: [u8; 4],
    addends: [f32; 4],
    layout: BatchLayout,
    transform: u32,
) -> CudaJ2kStoreRgbaNativeJob {
    let input_width =
        |store: &crate::CudaHtj2kStoreStep| store.input_rect.x1.saturating_sub(store.input_rect.x0);
    CudaJ2kStoreRgbaNativeJob {
        input_width0: input_width(stores[0]),
        input_width1: input_width(stores[1]),
        input_width2: input_width(stores[2]),
        input_width3: input_width(stores[3]),
        source_x0: stores[0].source_x,
        source_y0: stores[0].source_y,
        source_x1: stores[1].source_x,
        source_y1: stores[1].source_y,
        source_x2: stores[2].source_x,
        source_y2: stores[2].source_y,
        source_x3: stores[3].source_x,
        source_y3: stores[3].source_y,
        copy_width: stores[0].copy_width,
        copy_height: stores[0].copy_height,
        output_width: stores[0].output_width,
        output_height: stores[0].output_height,
        output_x: stores[0].output_x,
        output_y: stores[0].output_y,
        addend0: addends[0],
        addend1: addends[1],
        addend2: addends[2],
        addend3: addends[3],
        bit_depth0: u32::from(bit_depths[0]),
        bit_depth1: u32::from(bit_depths[1]),
        bit_depth2: u32::from(bit_depths[2]),
        bit_depth3: u32::from(bit_depths[3]),
        layout: match layout {
            BatchLayout::Nhwc => 0,
            BatchLayout::Nchw => 1,
            _ => u32::MAX,
        },
        transform,
        reserved: 0,
    }
}

pub(super) fn native_level_shift(bit_depth: u8) -> f32 {
    f32::from(1_u16 << bit_depth.saturating_sub(1).min(15))
}
