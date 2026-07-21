// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::ChunkedHtj2kCleanup;
use super::{
    can_batch_color_idwt, cuda_error, decode_cuda_component_subbands_with_resources,
    enqueue_chunked_htj2k_cleanup_dequant, enqueue_component_classic_batches,
    finalize_color_batch_decode_report, finish_and_store_native_color, host_owners,
    prepare_native_color_batch, profile, run_color_component_idwt_batches, BatchLayout,
    CudaExternalDeviceBufferViewMut, CudaHtj2kProfileReport, CudaQueuedIdwtBatch, CudaSession,
    DeviceSubmitSession, Error, HostPhaseBudget, NativeColorBatchInput, NativeColorBatchOutput,
    NativeColorPendingCompletion, PixelFormat, PreparedNativeColorBatch,
};
use crate::decoder::pending_completion::{finish_decode_statuses, retire_decode_after_error};

#[expect(
    clippy::too_many_lines,
    reason = "exact RGB orchestration keeps shared arenas and asynchronous owner ordering atomic"
)]
pub(super) fn decode_native_color_batch(
    inputs: &[NativeColorBatchInput<'_>],
    session: &mut CudaSession,
    fmt: PixelFormat,
    layout: BatchLayout,
    external: Option<&mut CudaExternalDeviceBufferViewMut<'_>>,
    enqueue_external: bool,
) -> Result<
    (
        NativeColorBatchOutput,
        CudaHtj2kProfileReport,
        Option<NativeColorPendingCompletion>,
    ),
    Error,
> {
    if !matches!(layout, BatchLayout::Nhwc | BatchLayout::Nchw) {
        return Err(Error::UnsupportedCudaRequest {
            reason: "exact CUDA RGB batch layout must be NHWC or NCHW",
        });
    }
    let batch_wall_started = profile::profile_now(false);
    let PreparedNativeColorBatch {
        colors,
        shared_payload,
        source_indices,
    } = prepare_native_color_batch(inputs, fmt)?;
    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    let table_upload_start = profile::profile_now(false);
    let table_resources = if colors.iter().all(|color| {
        color
            .components
            .iter()
            .all(|plan| plan.subbands().is_empty())
    }) {
        None
    } else {
        Some(session.htj2k_decode_table_resources()?)
    };
    let table_upload_us = profile::elapsed_us(table_upload_start);
    let component_count = colors
        .iter()
        .try_fold(0usize, |count, color| {
            count.checked_add(color.components.len())
        })
        .ok_or(Error::HostAllocationFailed {
            bytes: usize::MAX,
            what: "j2k CUDA exact color component work",
        })?;
    let mut host_budget = HostPhaseBudget::new("j2k CUDA exact RGB execution graph");
    host_owners::account_colors(&mut host_budget, &colors)?;
    let mut component_work = host_budget.try_vec_with_capacity(component_count)?;
    let mut component_source_indices = host_budget.try_vec_with_capacity(component_count)?;
    for (color, source_index) in colors.iter().zip(source_indices.iter().copied()) {
        for plan in &color.components {
            component_work.push(decode_cuda_component_subbands_with_resources(
                &context,
                plan,
                &pool,
                false,
                &mut host_budget,
            )?);
            component_source_indices.push(source_index);
        }
    }
    let payload_upload_start = profile::profile_now(false);
    let (pending_classic, classic_resources) = if component_work
        .iter()
        .any(|work| !work.pending_classic_bands.is_empty())
    {
        let classic_resources = context
            .upload_j2k_decode_payload_with_pool(&shared_payload, &pool)
            .map_err(cuda_error)?;
        let classic_tables = session.classic_decode_table_resources()?;
        let pending = enqueue_component_classic_batches(
            &context,
            &classic_resources,
            &classic_tables,
            &mut component_work,
            &component_source_indices,
            &pool,
            host_budget.live_bytes(),
        )?;
        (pending, Some(classic_resources))
    } else {
        (None, None)
    };
    let pending_cleanup = Some(enqueue_chunked_htj2k_cleanup_dequant(
        &context,
        table_resources.as_ref(),
        &shared_payload,
        &mut component_work,
        &component_source_indices,
        &pool,
        session.htj2k_decode_chunk_limits(),
        host_budget.live_bytes(),
    )?);
    let payload_upload_us = profile::elapsed_us(payload_upload_start);
    #[cfg(test)]
    session.record_htj2k_decode_chunk_count_for_test(
        pending_cleanup
            .as_ref()
            .map_or(0, ChunkedHtj2kCleanup::chunk_count),
    );
    drop(component_source_indices);
    drop(shared_payload);
    let mut component_plans = host_budget.try_vec_with_capacity(component_count)?;
    for color in &colors {
        component_plans.extend(color.components.iter());
    }
    let idwt_batched = can_batch_color_idwt(&component_plans);
    let pending_idwt = if idwt_batched {
        run_color_component_idwt_batches(
            &context,
            &component_plans,
            &mut component_work,
            &pool,
            false,
            host_budget.live_bytes(),
        )?
    } else {
        let mut pending: Option<CudaQueuedIdwtBatch> = None;
        for (plan, work) in component_plans.iter().zip(component_work.iter_mut()) {
            let plans = [*plan];
            let next = run_color_component_idwt_batches(
                &context,
                &plans,
                std::slice::from_mut(work),
                &pool,
                false,
                host_budget.live_bytes(),
            )?;
            pending = match (pending, next) {
                (Some(current), Some(next)) => Some(current.merge(next)?),
                (Some(current), None) => Some(current),
                (None, next) => next,
            };
        }
        pending
    };
    drop(component_plans);

    let completion_result = finish_and_store_native_color(
        &context,
        colors,
        component_work,
        fmt,
        layout,
        external,
        enqueue_external,
    );
    if enqueue_external {
        let (stored, reports, decoded) = match completion_result {
            Ok(output) => output,
            Err(error) => {
                return Err(retire_decode_after_error(
                    error,
                    pending_idwt,
                    pending_cleanup,
                    pending_classic,
                ));
            }
        };
        let store = stored.queued.ok_or(Error::UnsupportedCudaRequest {
            reason: "CUDA exact RGB external store did not return a completion guard",
        })?;
        let pending = NativeColorPendingCompletion::new(
            Some(store),
            pending_idwt,
            pending_cleanup,
            pending_classic,
            decoded,
            classic_resources,
        );
        let aggregate = finalize_color_batch_decode_report(
            &reports,
            table_upload_us,
            payload_upload_us,
            batch_wall_started,
        );
        session.record_submit();
        return Ok((stored.output, aggregate, Some(pending)));
    }

    let completion_result = completion_result.and_then(|(stored, reports, decoded)| {
        if stored.queued.is_some() {
            return Err(Error::UnsupportedCudaRequest {
                reason: "synchronous exact CUDA RGB store unexpectedly returned pending work",
            });
        }
        Ok(((stored.output, reports, decoded), true))
    });
    let (output, reports, _decoded) = CudaQueuedIdwtBatch::resolve_optional_after_completed_work(
        pending_idwt,
        completion_result,
    )?;
    finish_decode_statuses(pending_cleanup, pending_classic)?;
    drop(classic_resources);
    let aggregate = finalize_color_batch_decode_report(
        &reports,
        table_upload_us,
        payload_upload_us,
        batch_wall_started,
    );
    session.record_submit();
    Ok((output, aggregate, None))
}
