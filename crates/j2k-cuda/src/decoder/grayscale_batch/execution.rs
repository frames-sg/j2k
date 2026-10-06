// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grayscale CUDA execution orchestration.

use super::completion::{finish_submitted_grayscale_batch, finish_synchronous_grayscale_batch};
use super::preparation::{grayscale_owner_budget, prepare_grayscale_batch, PreparedGrayscaleBatch};
use super::store::{store_gray16_batch, store_gray8_batch, store_grayi16_batch};
use super::{
    cuda_error, decode_cuda_component_subbands_with_resources, enqueue_component_classic_batches,
    enqueue_component_cleanup_dequant_batches, finalize_color_batch_decode_report,
    finish_cuda_component_decode, grayscale_htj2k_job_identities, profile,
    run_color_component_idwt_batches, run_component_cleanup_dequant_batches,
    CudaComponentDecodeWork, CudaDecodedComponent, CudaExternalDeviceBufferViewMut,
    CudaHtj2kProfileReport, CudaQueuedIdwtBatch, CudaSession, DecodeSettings, DeviceSubmitSession,
    Error, GrayscaleBatchInput, GrayscaleBatchOutput, GrayscaleHtj2kCleanup,
    GrayscalePendingCompletion, HostPhaseBudget, PixelFormat, StoredGrayscaleBatch,
    CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
};
use j2k_cuda_j2k_engine::{CudaHtj2kDecodeResources, J2kCudaEngine};

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
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_format(CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED),
        ));
    }
    let batch_wall_started = profile::profile_now(collect_stage_timings);
    let mut prepared = prepare_grayscale_batch(inputs, fmt, settings)?;
    let (decode_resources, table_upload_us, payload_upload_us) =
        upload_grayscale_decode_resources(session, &prepared, collect_stage_timings)?;
    drop(std::mem::take(&mut prepared.payload_parts));
    let (host_budget, mut component_work) =
        build_grayscale_component_work(session, &prepared, collect_stage_timings)?;
    let (pending_cleanup, pending_classic) = enqueue_grayscale_entropy(
        session,
        &decode_resources,
        &prepared,
        &mut component_work,
        collect_stage_timings,
        host_budget.live_bytes(),
    )?;
    let pending_idwt = enqueue_grayscale_idwt(
        session,
        &prepared.plans,
        &mut component_work,
        collect_stage_timings,
        host_budget.live_bytes(),
    )?;
    let completion_result = finish_grayscale_components_and_store(
        session,
        &mut prepared,
        component_work,
        fmt,
        collect_stage_timings,
        external,
        enqueue_external,
    );

    if enqueue_external {
        let (output, pending) = finish_submitted_grayscale_batch(
            completion_result,
            pending_idwt,
            pending_cleanup,
            pending_classic,
            decode_resources,
        )?;
        let aggregate = finalize_color_batch_decode_report(
            &prepared.reports,
            table_upload_us,
            payload_upload_us,
            batch_wall_started,
        );
        session.record_submit();
        aggregate.emit("submit_batch_gray");
        return Ok((output, aggregate, Some(pending)));
    }

    let output = finish_synchronous_grayscale_batch(
        completion_result,
        pending_idwt,
        pending_cleanup,
        pending_classic,
    )?;
    let aggregate = finalize_color_batch_decode_report(
        &prepared.reports,
        table_upload_us,
        payload_upload_us,
        batch_wall_started,
    );
    session.record_submit();
    aggregate.emit("decode_batch_gray");
    Ok((output, aggregate, None))
}

fn upload_grayscale_decode_resources(
    session: &mut CudaSession,
    prepared: &PreparedGrayscaleBatch,
    collect_stage_timings: bool,
) -> Result<(CudaHtj2kDecodeResources, u128, u128), Error> {
    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    let table_upload_start = profile::profile_now(collect_stage_timings);
    let tables = if prepared.plans.iter().all(|plan| plan.subbands().is_empty()) {
        None
    } else {
        Some(session.htj2k_decode_table_resources()?)
    };
    let table_upload_us = profile::elapsed_us(table_upload_start);
    let payload_upload_start = profile::profile_now(collect_stage_timings);
    let mut payload_budget = grayscale_owner_budget(
        &prepared.plans,
        &prepared.reports,
        &prepared.payload_parts,
        "j2k CUDA grayscale batch payload upload",
    )?;
    let mut payload_parts = payload_budget.try_vec_with_capacity(prepared.payload_parts.len())?;
    payload_parts.extend(prepared.payload_parts.iter().map(Vec::as_slice));
    let engine = J2kCudaEngine::new(&context);
    let resources = match tables.as_ref() {
        Some(tables) => {
            engine.upload_htj2k_decode_resources_with_tables_and_pool(&payload_parts, tables, &pool)
        }
        None => engine.upload_j2k_decode_payload_with_pool(&payload_parts, &pool),
    }
    .map_err(cuda_error)?;
    Ok((
        resources,
        table_upload_us,
        profile::elapsed_us(payload_upload_start),
    ))
}

fn build_grayscale_component_work(
    session: &mut CudaSession,
    prepared: &PreparedGrayscaleBatch,
    collect_stage_timings: bool,
) -> Result<(HostPhaseBudget, Vec<CudaComponentDecodeWork>), Error> {
    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    let mut budget = grayscale_owner_budget(
        &prepared.plans,
        &prepared.reports,
        &prepared.payload_parts,
        "j2k CUDA grayscale batch execution graph",
    )?;
    budget.account_vec(&prepared.output_indices)?;
    budget.account_vec(&prepared.output_dimensions)?;
    budget.account_vec(&prepared.source_indices)?;
    let mut work = budget.try_vec_with_capacity(prepared.plans.len())?;
    for plan in &prepared.plans {
        work.push(decode_cuda_component_subbands_with_resources(
            &context,
            plan,
            &pool,
            collect_stage_timings,
            &mut budget,
        )?);
    }
    Ok((budget, work))
}

fn enqueue_grayscale_entropy(
    session: &mut CudaSession,
    resources: &CudaHtj2kDecodeResources,
    prepared: &PreparedGrayscaleBatch,
    work: &mut [CudaComponentDecodeWork],
    collect_stage_timings: bool,
    live_host_bytes: usize,
) -> Result<
    (
        Option<GrayscaleHtj2kCleanup>,
        Option<super::super::resident::QueuedComponentClassicDecode>,
    ),
    Error,
> {
    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    if collect_stage_timings {
        run_component_cleanup_dequant_batches(
            &context,
            resources,
            work,
            &pool,
            true,
            live_host_bytes,
        )?;
        return Ok((None, None));
    }

    let pending_classic = if work
        .iter()
        .any(|component| !component.pending_classic_bands.is_empty())
    {
        let classic_tables = session.classic_decode_table_resources()?;
        enqueue_component_classic_batches(
            &context,
            resources,
            &classic_tables,
            work,
            &prepared.source_indices,
            &pool,
            live_host_bytes,
        )?
    } else {
        None
    };
    let identities =
        grayscale_htj2k_job_identities(work, &prepared.source_indices, live_host_bytes)?;
    let pending_cleanup = enqueue_component_cleanup_dequant_batches(
        &context,
        resources,
        work,
        &pool,
        live_host_bytes,
    )?
    .map(|queued| GrayscaleHtj2kCleanup::new(queued, identities));
    Ok((pending_cleanup, pending_classic))
}

fn enqueue_grayscale_idwt(
    session: &mut CudaSession,
    plans: &[super::CudaHtj2kDecodePlan],
    work: &mut [CudaComponentDecodeWork],
    collect_stage_timings: bool,
    live_host_bytes: usize,
) -> Result<Option<CudaQueuedIdwtBatch>, Error> {
    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    let mut idwt_budget = HostPhaseBudget::with_live_bytes(
        "j2k CUDA grayscale IDWT plan references",
        live_host_bytes,
    )?;
    let plan_refs = idwt_budget.try_collect_exact(plans.iter())?;
    run_color_component_idwt_batches(
        &context,
        &plan_refs,
        work,
        &pool,
        collect_stage_timings,
        idwt_budget.live_bytes(),
        false,
    )
}

fn finish_grayscale_components_and_store(
    session: &mut CudaSession,
    prepared: &mut PreparedGrayscaleBatch,
    work: Vec<CudaComponentDecodeWork>,
    fmt: PixelFormat,
    collect_stage_timings: bool,
    external: Option<&mut CudaExternalDeviceBufferViewMut<'_>>,
    enqueue_external: bool,
) -> Result<(StoredGrayscaleBatch, Vec<CudaDecodedComponent>), Error> {
    let context = session.cuda_context()?;
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(work.len())
        .map_err(|_| Error::HostAllocationFailed {
            bytes: work
                .len()
                .saturating_mul(std::mem::size_of::<CudaDecodedComponent>()),
            what: "j2k CUDA grayscale decoded components",
        })?;
    for component in work {
        decoded.push(finish_cuda_component_decode(component)?);
    }

    let store_started = profile::profile_now(collect_stage_timings);
    let stored = match fmt {
        PixelFormat::Gray8 => store_gray8_batch(
            &context,
            &prepared.plans,
            &prepared.output_indices,
            &prepared.output_dimensions,
            &decoded,
            external,
            enqueue_external,
        )?,
        PixelFormat::Gray16 => store_gray16_batch(
            &context,
            &prepared.plans,
            &prepared.output_indices,
            &prepared.output_dimensions,
            &decoded,
            external,
            enqueue_external,
        )?,
        PixelFormat::GrayI16 => store_grayi16_batch(
            &context,
            &prepared.plans,
            &prepared.output_indices,
            &prepared.output_dimensions,
            &decoded,
            external,
            enqueue_external,
        )?,
        _ => {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::unsupported_format(
                    CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
                ),
            ));
        }
    };
    let store_us = profile::elapsed_us(store_started);
    let store_dispatches = usize::from(!decoded.is_empty());
    for (index, (component, report)) in decoded.iter().zip(prepared.reports.iter_mut()).enumerate()
    {
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
}
