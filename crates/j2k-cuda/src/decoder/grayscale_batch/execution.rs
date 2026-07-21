// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grayscale CUDA execution orchestration.

use super::preparation::{grayscale_owner_budget, prepare_grayscale_batch, PreparedGrayscaleBatch};
use super::store::{store_gray16_batch, store_gray8_batch, store_grayi16_batch};
use super::{
    can_batch_color_idwt, cuda_error, decode_cuda_component_subbands_with_resources,
    enqueue_component_classic_batches, enqueue_component_cleanup_dequant_batches,
    finalize_color_batch_decode_report, finish_cuda_component_decode,
    grayscale_htj2k_job_identities, profile, run_color_component_idwt_batches,
    run_component_cleanup_dequant_batches, run_cuda_component_idwt_steps, CudaDecodedComponent,
    CudaExternalDeviceBufferViewMut, CudaHtj2kProfileReport, CudaQueuedIdwtBatch, CudaSession,
    DecodeSettings, DeviceSubmitSession, Error, GrayscaleBatchInput, GrayscaleBatchOutput,
    GrayscaleHtj2kCleanup, GrayscalePendingCompletion, PixelFormat,
    CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
};
use crate::decoder::pending_completion::{finish_decode_statuses, retire_decode_after_error};

#[expect(
    clippy::too_many_lines,
    reason = "grayscale batch orchestration keeps shared resources and completion ordering atomic"
)]
pub(super) fn decode_grayscale_cuda_batch_with_profile(
    inputs: &[GrayscaleBatchInput<'_>],
    settings: DecodeSettings,
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
    external: Option<&mut CudaExternalDeviceBufferViewMut<'_>>,
    enqueue_external: bool,
) -> Result<
    (
        GrayscaleBatchOutput,
        CudaHtj2kProfileReport,
        Option<GrayscalePendingCompletion>,
    ),
    Error,
> {
    if !matches!(
        fmt,
        PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayI16
    ) {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    }
    let batch_wall_started = profile::profile_now(collect_stage_timings);
    let PreparedGrayscaleBatch {
        plans,
        mut reports,
        shared_payload,
        output_indices,
        output_dimensions,
        source_indices,
    } = prepare_grayscale_batch(inputs, fmt, settings)?;
    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    let table_upload_start = profile::profile_now(collect_stage_timings);
    let table_resources = if plans.iter().all(|plan| plan.subbands().is_empty()) {
        None
    } else {
        Some(session.htj2k_decode_table_resources()?)
    };
    let table_upload_us = profile::elapsed_us(table_upload_start);
    let payload_upload_start = profile::profile_now(collect_stage_timings);
    let decode_resources = match table_resources.as_ref() {
        Some(tables) => context.upload_htj2k_decode_resources_with_tables_and_pool(
            &shared_payload,
            tables,
            &pool,
        ),
        None => context.upload_j2k_decode_payload_with_pool(&shared_payload, &pool),
    }
    .map_err(cuda_error)?;
    let payload_upload_us = profile::elapsed_us(payload_upload_start);
    drop(shared_payload);

    let empty_payload = Vec::new();
    let mut host_budget = grayscale_owner_budget(
        &plans,
        &reports,
        &empty_payload,
        None,
        "j2k CUDA grayscale batch execution graph",
    )?;
    host_budget.account_vec(&output_indices)?;
    host_budget.account_vec(&output_dimensions)?;
    host_budget.account_vec(&source_indices)?;
    let mut component_work = host_budget.try_vec_with_capacity(plans.len())?;
    let mut component_source_indices = host_budget.try_vec_with_capacity(plans.len())?;
    for (plan, source_index) in plans.iter().zip(source_indices.iter().copied()) {
        component_work.push(decode_cuda_component_subbands_with_resources(
            &context,
            plan,
            &pool,
            collect_stage_timings,
            &mut host_budget,
        )?);
        component_source_indices.push(source_index);
    }
    let (pending_cleanup, pending_classic) = if collect_stage_timings {
        run_component_cleanup_dequant_batches(
            &context,
            &decode_resources,
            &mut component_work,
            &pool,
            true,
            host_budget.live_bytes(),
        )?;
        (None, None)
    } else {
        let pending_classic = if component_work
            .iter()
            .any(|work| !work.pending_classic_bands.is_empty())
        {
            let classic_tables = session.classic_decode_table_resources()?;
            enqueue_component_classic_batches(
                &context,
                &decode_resources,
                &classic_tables,
                &mut component_work,
                &component_source_indices,
                &pool,
                host_budget.live_bytes(),
            )?
        } else {
            None
        };
        let identities = grayscale_htj2k_job_identities(
            &component_work,
            &component_source_indices,
            host_budget.live_bytes(),
        )?;
        let pending_cleanup = enqueue_component_cleanup_dequant_batches(
            &context,
            &decode_resources,
            &mut component_work,
            &pool,
            host_budget.live_bytes(),
        )?
        .map(|queued| GrayscaleHtj2kCleanup::new(queued, identities));
        (pending_cleanup, pending_classic)
    };
    drop(component_source_indices);

    let plan_refs = plans.iter().collect::<Vec<_>>();
    let idwt_batched = can_batch_color_idwt(&plan_refs);
    let pending_idwt = if idwt_batched {
        run_color_component_idwt_batches(
            &context,
            &plan_refs,
            &mut component_work,
            &pool,
            collect_stage_timings,
            host_budget.live_bytes(),
        )?
    } else if collect_stage_timings {
        for (plan, work) in plans.iter().zip(component_work.iter_mut()) {
            run_cuda_component_idwt_steps(&context, plan.idwt_steps(), work, &pool, true)?;
        }
        None
    } else {
        let mut pending: Option<CudaQueuedIdwtBatch> = None;
        for (plan, work) in plans.iter().zip(component_work.iter_mut()) {
            let components = [plan];
            let next = run_color_component_idwt_batches(
                &context,
                &components,
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
    drop(plan_refs);

    let completion_result = (|| {
        let mut decoded = Vec::new();
        decoded
            .try_reserve_exact(component_work.len())
            .map_err(|_| Error::HostAllocationFailed {
                bytes: component_work
                    .len()
                    .saturating_mul(std::mem::size_of::<CudaDecodedComponent>()),
                what: "j2k CUDA grayscale decoded components",
            })?;
        for work in component_work {
            decoded.push(finish_cuda_component_decode(work)?);
        }

        let store_started = profile::profile_now(collect_stage_timings);
        let stored = match fmt {
            PixelFormat::Gray8 => store_gray8_batch(
                &context,
                &plans,
                &output_indices,
                &output_dimensions,
                &decoded,
                external,
                enqueue_external,
            )?,
            PixelFormat::Gray16 => store_gray16_batch(
                &context,
                &plans,
                &output_indices,
                &output_dimensions,
                &decoded,
                external,
                enqueue_external,
            )?,
            PixelFormat::GrayI16 => store_grayi16_batch(
                &context,
                &plans,
                &output_indices,
                &output_dimensions,
                &decoded,
                external,
                enqueue_external,
            )?,
            _ => {
                return Err(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
                })
            }
        };
        let store_us = profile::elapsed_us(store_started);
        let store_dispatches = usize::from(!decoded.is_empty());
        for (index, (component, report)) in decoded.iter().zip(reports.iter_mut()).enumerate() {
            let report_store_dispatches = usize::from(index == 0) * store_dispatches;
            report.dispatch_count = component.dispatches.saturating_add(report_store_dispatches);
            component.timings.add_to_report(report);
            if index == 0 {
                report.store_us = report.store_us.saturating_add(store_us);
                report.detail.store_dispatch_count = report
                    .detail
                    .store_dispatch_count
                    .saturating_add(store_dispatches);
            }
            profile::finalize_decode_total_us(report);
        }
        Ok((stored, decoded))
    })();

    if enqueue_external {
        let (stored, decoded) = match completion_result {
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
            reason: "CUDA external batch store did not return a completion guard",
        })?;
        let output = stored.output;
        let pending = GrayscalePendingCompletion::new(
            Some(store),
            pending_idwt,
            pending_cleanup,
            pending_classic,
            decoded,
            Some(decode_resources),
        );
        let aggregate = finalize_color_batch_decode_report(
            &reports,
            table_upload_us,
            payload_upload_us,
            batch_wall_started,
        );
        session.record_submit();
        aggregate.emit("submit_batch_gray");
        return Ok((output, aggregate, Some(pending)));
    }

    let completion_result = completion_result.and_then(|(stored, decoded)| {
        if stored.queued.is_some() {
            return Err(Error::UnsupportedCudaRequest {
                reason: "synchronous CUDA grayscale store unexpectedly returned pending work",
            });
        }
        Ok(((stored.output, decoded), true))
    });
    let resolved =
        CudaQueuedIdwtBatch::resolve_optional_after_completed_work(pending_idwt, completion_result);
    let (output, _decoded_owners) = match resolved {
        Ok(output) => {
            finish_decode_statuses(pending_cleanup, pending_classic)?;
            output
        }
        Err(error) => {
            return Err(retire_decode_after_error(
                error,
                None,
                pending_cleanup,
                pending_classic,
            ));
        }
    };

    let aggregate = finalize_color_batch_decode_report(
        &reports,
        table_upload_us,
        payload_upload_us,
        batch_wall_started,
    );
    session.record_submit();
    aggregate.emit("decode_batch_gray");
    Ok((output, aggregate, None))
}
