// SPDX-License-Identifier: MIT OR Apache-2.0

mod lifecycle;

use self::lifecycle::{
    build_native_color_component_work, enqueue_native_color_entropy, enqueue_native_color_idwt,
    prepare_native_color_tables, SubmittedNativeColorEntropy,
};
use super::{
    finalize_color_batch_decode_report, finish_and_store_native_color, prepare_native_color_batch,
    profile, BatchLayout, CudaExternalDeviceBufferViewMut, CudaHtj2kProfileReport,
    CudaQueuedIdwtBatch, CudaSession, DeviceSubmitSession, Error, NativeColorBatchInput,
    NativeColorBatchOutput, NativeColorPendingCompletion, PixelFormat, PreparedNativeColorBatch,
    StoredNativeColorBatch,
};
use crate::decoder::color_batch::CudaDecodedComponent;
use crate::decoder::pending_completion::{finish_decode_statuses, retire_decode_after_error};

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
    validate_native_color_layout(layout)?;
    let batch_wall_started = profile::profile_now(false);
    let PreparedNativeColorBatch {
        colors,
        shared_payload,
        source_indices,
    } = prepare_native_color_batch(inputs, fmt)?;
    let (table_resources, table_upload_us) = prepare_native_color_tables(session, &colors)?;
    let (mut host_budget, mut component_work, component_source_indices) =
        build_native_color_component_work(session, &colors, &source_indices)?;
    let entropy = enqueue_native_color_entropy(
        session,
        table_resources.as_ref(),
        &shared_payload,
        &mut component_work,
        &component_source_indices,
        host_budget.live_bytes(),
    )?;
    drop(component_source_indices);
    drop(shared_payload);
    let pending_idwt =
        enqueue_native_color_idwt(session, &colors, &mut component_work, &mut host_budget)?;
    let context = session.cuda_context()?;
    let completion_result = finish_and_store_native_color(
        &context,
        colors,
        component_work,
        fmt,
        layout,
        external,
        enqueue_external,
    );
    let payload_upload_us = entropy.payload_upload_us;
    let (output, reports, pending) = if enqueue_external {
        let (output, reports, pending) =
            finish_submitted_native_color_execution(completion_result, pending_idwt, entropy)?;
        (output, reports, Some(pending))
    } else {
        let (output, reports) =
            finish_synchronous_native_color_execution(completion_result, pending_idwt, entropy)?;
        (output, reports, None)
    };
    let aggregate = finalize_color_batch_decode_report(
        &reports,
        table_upload_us,
        payload_upload_us,
        batch_wall_started,
    );
    session.record_submit();
    Ok((output, aggregate, pending))
}

fn validate_native_color_layout(layout: BatchLayout) -> Result<(), Error> {
    if matches!(layout, BatchLayout::Nhwc | BatchLayout::Nchw) {
        return Ok(());
    }
    Err(Error::UnsupportedCudaRequest {
        reason: "exact CUDA RGB batch layout must be NHWC or NCHW",
    })
}

fn finish_submitted_native_color_execution(
    completion_result: Result<
        (
            StoredNativeColorBatch,
            Vec<CudaHtj2kProfileReport>,
            Vec<CudaDecodedComponent>,
        ),
        Error,
    >,
    pending_idwt: Option<CudaQueuedIdwtBatch>,
    entropy: SubmittedNativeColorEntropy,
) -> Result<
    (
        NativeColorBatchOutput,
        Vec<CudaHtj2kProfileReport>,
        NativeColorPendingCompletion,
    ),
    Error,
> {
    let (stored, reports, decoded) = match completion_result {
        Ok(output) => output,
        Err(error) => {
            return Err(retire_decode_after_error(
                error,
                pending_idwt,
                entropy.pending_cleanup,
                entropy.pending_classic,
            ));
        }
    };
    let store = stored.queued.ok_or(Error::UnsupportedCudaRequest {
        reason: "CUDA exact RGB external store did not return a completion guard",
    })?;
    let pending = NativeColorPendingCompletion::new(
        Some(store),
        pending_idwt,
        entropy.pending_cleanup,
        entropy.pending_classic,
        decoded,
        entropy.classic_resources,
    );
    Ok((stored.output, reports, pending))
}

fn finish_synchronous_native_color_execution(
    completion_result: Result<
        (
            StoredNativeColorBatch,
            Vec<CudaHtj2kProfileReport>,
            Vec<CudaDecodedComponent>,
        ),
        Error,
    >,
    pending_idwt: Option<CudaQueuedIdwtBatch>,
    entropy: SubmittedNativeColorEntropy,
) -> Result<(NativeColorBatchOutput, Vec<CudaHtj2kProfileReport>), Error> {
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
    finish_decode_statuses(entropy.pending_cleanup, entropy.pending_classic)?;
    drop(entropy.classic_resources);
    Ok((output, reports))
}
