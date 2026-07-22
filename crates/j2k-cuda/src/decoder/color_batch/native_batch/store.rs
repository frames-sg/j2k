// SPDX-License-Identifier: MIT OR Apache-2.0

mod jobs;
mod submission;
mod targets;

use j2k::BatchLayout;
use j2k_core::PixelFormat;
use j2k_cuda_runtime::CudaExternalDeviceBufferViewMut;

use submission::store_targets;
use targets::build_store_targets;

use super::StoredNativeColorBatch;
use crate::decoder::color_batch::{
    finish_cuda_component_decode, CudaComponentDecodeWork, CudaDecodedComponent,
    CudaHtj2kColorDecodePlans, CudaHtj2kProfileReport,
};
use crate::decoder::{Error, CUDA_HTJ2K_KERNELS_NOT_READY};

pub(super) fn finish_and_store_native_color(
    context: &j2k_cuda_runtime::CudaContext,
    colors: Vec<CudaHtj2kColorDecodePlans>,
    component_work: Vec<CudaComponentDecodeWork>,
    fmt: PixelFormat,
    layout: BatchLayout,
    external: Option<&mut CudaExternalDeviceBufferViewMut<'_>>,
    enqueue_external: bool,
) -> Result<
    (
        StoredNativeColorBatch,
        Vec<CudaHtj2kProfileReport>,
        Vec<CudaDecodedComponent>,
    ),
    Error,
> {
    let expected_components = colors.iter().try_fold(0usize, |count, color| {
        count.checked_add(color.components.len())
    });
    let decoded = finish_components(
        component_work,
        expected_components.ok_or(Error::HostAllocationFailed {
            bytes: usize::MAX,
            what: "j2k CUDA exact color decoded component count",
        })?,
    )?;
    let (targets, reports) = build_store_targets(colors, &decoded, fmt, layout)?;
    let stored = store_targets(context, fmt, external, enqueue_external, &targets)?;
    Ok((stored, reports, decoded))
}

fn finish_components(
    component_work: Vec<CudaComponentDecodeWork>,
    expected_components: usize,
) -> Result<Vec<CudaDecodedComponent>, Error> {
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(component_work.len())
        .map_err(|_| Error::HostAllocationFailed {
            bytes: component_work
                .len()
                .saturating_mul(std::mem::size_of::<CudaDecodedComponent>()),
            what: "j2k CUDA exact RGB decoded components",
        })?;
    for work in component_work {
        decoded.push(finish_cuda_component_decode(work)?);
    }
    if decoded.len() != expected_components {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }
    Ok(decoded)
}
