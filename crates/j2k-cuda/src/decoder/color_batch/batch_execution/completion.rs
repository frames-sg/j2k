// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) mod batch_store;
mod fallback;

use self::{
    batch_store::finish_color_cuda_resident_batch_surfaces_with_rgb8_mct_store,
    fallback::finish_color_cuda_resident_batch_surfaces_individually,
};
use super::super::{
    CudaBufferPool, CudaComponentDecodeWork, CudaHtj2kColorDecodePlans, CudaHtj2kProfileReport,
    CudaQueuedIdwtBatch, Error, PixelFormat, Surface,
};
use super::execution::EnqueuedColorCudaResidentBatch;
use crate::decoder::combine_cuda_cleanup_errors;
use j2k_cuda_runtime::CudaContext;

struct FinishColorBatchRequest<'a> {
    context: &'a CudaContext,
    pool: &'a CudaBufferPool,
    fmt: PixelFormat,
    colors: Vec<CudaHtj2kColorDecodePlans>,
    component_work: Vec<CudaComponentDecodeWork>,
    collect_stage_timings: bool,
    external_live_host_bytes: usize,
}

pub(super) fn complete_color_cuda_resident_batch(
    enqueued: EnqueuedColorCudaResidentBatch,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, Vec<CudaHtj2kProfileReport>, u128, u128), Error> {
    let EnqueuedColorCudaResidentBatch {
        context,
        pool,
        output_pool,
        colors,
        component_work,
        pending_idwt_batch,
        entropy_owners,
        entropy_live_host_bytes,
        use_batch_store,
        fused_final_vertical,
        table_upload_us,
        payload_upload_us,
    } = enqueued;
    let completion_result = (|| {
        let request = FinishColorBatchRequest {
            context: &context,
            pool: if use_batch_store { &output_pool } else { &pool },
            fmt,
            colors,
            component_work,
            collect_stage_timings,
            external_live_host_bytes: entropy_live_host_bytes,
        };
        let (surfaces, reports) = if use_batch_store {
            finish_color_cuda_resident_batch_surfaces_with_rgb8_mct_store(
                request,
                fused_final_vertical,
            )?
        } else {
            finish_color_cuda_resident_batch_surfaces_individually(request)?
        };
        // Runtime MCT/store launches synchronize before returning, so a
        // recorded dispatch is also a completion point for preceding IDWT.
        let completed = reports.iter().any(|report| {
            report.detail.mct_dispatch_count != 0 || report.detail.store_dispatch_count != 0
        });
        Ok(((surfaces, reports), completed))
    })();
    let completion_result = CudaQueuedIdwtBatch::resolve_optional_after_completed_work(
        pending_idwt_batch,
        completion_result,
    );
    // Final store completes the preceding entropy work. Validate its statuses
    // before exposing surfaces, retaining compressed input through every error path.
    let cleanup_result = entropy_owners.finish();
    let (surfaces, reports) = match (completion_result, cleanup_result) {
        (Ok(output), Ok(())) => output,
        (Err(error), Ok(())) | (Ok(_), Err(error)) => return Err(error),
        (Err(error), Err(cleanup)) => return Err(combine_cuda_cleanup_errors(error, cleanup)),
    };
    Ok((surfaces, reports, table_upload_us, payload_upload_us))
}
