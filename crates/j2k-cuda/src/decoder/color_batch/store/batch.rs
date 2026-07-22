// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;
use j2k_cuda_runtime::{CudaJ2kStoreRgb8MctJob, CudaJ2kStoreRgb8MctTarget};

use super::super::super::resident::{finish_cuda_component_decode, pooled_cuda_buffer};
use super::super::super::{
    CudaComponentDecodeWork, CudaDecodedComponent, CudaHtj2kColorDecodePlans, Error,
    CUDA_HTJ2K_KERNELS_NOT_READY,
};
use super::{
    bit_depth_addend, can_fuse_mct_store_for_stores, validate_color_stores, ColorStorePlan,
    ColorStoreRoute,
};

pub(in crate::decoder::color_batch) struct CudaPreparedRgb8MctBatchStore {
    pub(in crate::decoder::color_batch) color: CudaHtj2kColorDecodePlans,
    pub(in crate::decoder::color_batch) decoded_components: [CudaDecodedComponent; 3],
    pub(in crate::decoder::color_batch) dispatches: usize,
    pub(in crate::decoder::color_batch) decode_dispatches: usize,
    pub(in crate::decoder::color_batch) job: CudaJ2kStoreRgb8MctJob,
}

pub(in crate::decoder::color_batch) fn prepare_rgb8_mct_batch_store(
    fmt: PixelFormat,
    mut color: CudaHtj2kColorDecodePlans,
    component_work: Vec<CudaComponentDecodeWork>,
) -> Result<CudaPreparedRgb8MctBatchStore, Error> {
    let [work0, work1, work2]: [CudaComponentDecodeWork; 3] =
        component_work
            .try_into()
            .map_err(|_| Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            })?;
    let decoded_components = [
        finish_cuda_component_decode(work0)?,
        finish_cuda_component_decode(work1)?,
        finish_cuda_component_decode(work2)?,
    ];
    let [component0, component1, component2] = &decoded_components;
    let stores = [&component0.store, &component1.store, &component2.store];
    validate_color_stores(stores, color.dimensions)?;
    if !color.mct || !can_fuse_mct_store_for_stores(stores) {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }

    let dispatches = decoded_components
        .iter()
        .map(|component| component.dispatches)
        .sum::<usize>();
    let decode_dispatches = decoded_components
        .iter()
        .map(|component| component.decode_dispatches)
        .sum::<usize>();
    for component in &decoded_components {
        component.timings.add_to_report(&mut color.report);
    }

    let addends = [
        bit_depth_addend(color.bit_depths[0]),
        bit_depth_addend(color.bit_depths[1]),
        bit_depth_addend(color.bit_depths[2]),
    ];
    let store_plan = ColorStorePlan::new(
        stores,
        color.rgb_bit_depths(),
        addends,
        ColorStoreRoute::for_mct(true, color.transform),
    );
    let job = CudaJ2kStoreRgb8MctJob {
        store: store_plan.rgb8_job(fmt == PixelFormat::Rgba8),
        irreversible97: store_plan.irreversible97(),
    };

    Ok(CudaPreparedRgb8MctBatchStore {
        color,
        decoded_components,
        dispatches,
        decode_dispatches,
        job,
    })
}

pub(in crate::decoder::color_batch) fn rgb8_mct_batch_store_target(
    prepared: &CudaPreparedRgb8MctBatchStore,
) -> Result<CudaJ2kStoreRgb8MctTarget<'_>, Error> {
    let [component0, component1, component2] = &prepared.decoded_components;
    Ok(CudaJ2kStoreRgb8MctTarget {
        plane0: pooled_cuda_buffer(&component0.buffer)?,
        plane1: pooled_cuda_buffer(&component1.buffer)?,
        plane2: pooled_cuda_buffer(&component2.buffer)?,
        job: prepared.job,
    })
}
