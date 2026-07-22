// SPDX-License-Identifier: MIT OR Apache-2.0

mod batch_store;
mod fallback;

use self::{
    batch_store::{
        can_batch_rgb8_mct_color_store,
        finish_color_cuda_resident_batch_surfaces_with_rgb8_mct_store,
    },
    fallback::finish_color_cuda_resident_batch_surfaces_individually,
};
use super::super::{CudaHtj2kProfileReport, CudaQueuedIdwtBatch, Error, PixelFormat, Surface};
use super::execution::EnqueuedColorCudaResidentBatch;

pub(super) fn complete_color_cuda_resident_batch(
    enqueued: EnqueuedColorCudaResidentBatch,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, Vec<CudaHtj2kProfileReport>, u128, u128), Error> {
    let EnqueuedColorCudaResidentBatch {
        context,
        pool,
        colors,
        component_work,
        pending_idwt_batch,
        idwt_batched,
        table_upload_us,
        payload_upload_us,
    } = enqueued;
    let completion_result = (|| {
        let use_batch_store =
            idwt_batched && can_batch_rgb8_mct_color_store(fmt, &colors, &component_work)?;
        let (surfaces, reports) = if use_batch_store {
            finish_color_cuda_resident_batch_surfaces_with_rgb8_mct_store(
                &context,
                fmt,
                colors,
                component_work,
                collect_stage_timings,
            )?
        } else {
            finish_color_cuda_resident_batch_surfaces_individually(
                &context,
                &pool,
                fmt,
                colors,
                component_work,
                collect_stage_timings,
                idwt_batched,
            )?
        };
        // Runtime MCT/store launches synchronize before returning, so a
        // recorded dispatch is also a completion point for preceding IDWT.
        let completed = reports.iter().any(|report| {
            report.detail.mct_dispatch_count != 0 || report.detail.store_dispatch_count != 0
        });
        Ok(((surfaces, reports), completed))
    })();
    let (surfaces, reports) = CudaQueuedIdwtBatch::resolve_optional_after_completed_work(
        pending_idwt_batch,
        completion_result,
    )?;
    Ok((surfaces, reports, table_upload_us, payload_upload_us))
}
