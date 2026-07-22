// SPDX-License-Identifier: MIT OR Apache-2.0

mod completion;
mod execution;
mod preparation;

use self::{
    completion::complete_color_cuda_resident_batch, execution::enqueue_color_cuda_resident_batch,
    preparation::prepare_color_cuda_resident_batch,
};
use super::{
    aggregate_decode_reports, profile, CudaHtj2kProfileReport, CudaSession, Error, PixelFormat,
    Surface,
};

pub(in crate::decoder) fn decode_color_cuda_resident_batch_surfaces_with_profile(
    inputs: &[&[u8]],
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    let batch_wall_started = profile::profile_now(collect_stage_timings);
    let (colors, shared_payload) = prepare_color_cuda_resident_batch(inputs, fmt)?;
    let enqueued =
        enqueue_color_cuda_resident_batch(session, colors, &shared_payload, collect_stage_timings)?;
    let (surfaces, reports, table_upload_us, payload_upload_us) =
        complete_color_cuda_resident_batch(enqueued, fmt, collect_stage_timings)?;
    let aggregate = finalize_color_batch_decode_report(
        &reports,
        table_upload_us,
        payload_upload_us,
        batch_wall_started,
    );
    aggregate.emit("decode_batch");
    Ok((surfaces, aggregate))
}

pub(in crate::decoder) fn finalize_color_batch_decode_report(
    reports: &[CudaHtj2kProfileReport],
    table_upload_us: u128,
    payload_upload_us: u128,
    batch_wall_started: Option<profile::ProfileInstant>,
) -> CudaHtj2kProfileReport {
    let mut aggregate = aggregate_decode_reports(reports);
    aggregate.h2d_us = aggregate
        .h2d_us
        .saturating_add(table_upload_us)
        .saturating_add(payload_upload_us);
    aggregate.detail.table_upload_us = aggregate
        .detail
        .table_upload_us
        .saturating_add(table_upload_us);
    aggregate.detail.payload_upload_us = aggregate
        .detail
        .payload_upload_us
        .saturating_add(payload_upload_us);
    aggregate.detail.wall_total_us = profile::elapsed_us(batch_wall_started);
    profile::finalize_decode_total_us(&mut aggregate);
    aggregate
}
