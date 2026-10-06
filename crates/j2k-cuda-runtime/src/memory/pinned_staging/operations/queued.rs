// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::retain_pinned_upload_staging_after_abandoned_checkout;
use super::gate::HeldPinnedUploadGate;
use crate::{
    context::{CudaContext, PinnedUploadStaging},
    execution::{CudaEvent, CudaStream},
    memory::CudaDeviceBuffer,
    CudaError,
};

/// Owns immutable host staging until its transfer event completes. The pooled
/// destination retains this owner and retires it before recycling its device allocation.
pub(crate) struct PendingPinnedUpload {
    context: CudaContext,
    staging: Option<PinnedUploadStaging>,
    len: usize,
    completion: CudaEvent,
    recorded: bool,
}

impl std::fmt::Debug for PendingPinnedUpload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingPinnedUpload")
            .field("len", &self.len)
            .field("recorded", &self.recorded)
            .finish_non_exhaustive()
    }
}

/// Why a queued upload could not be retired cleanly.
#[derive(Debug)]
pub(crate) enum PendingUploadRetireError {
    /// CUDA may still be writing the destination, so it must not be reused or freed.
    CompletionUncertain(CudaError),
    /// The transfer completed; only returning its staging to the pool failed.
    RecycleFailed(CudaError),
}

// SAFETY: staging is immutable once transferred into this owner. No shared
// method exposes host memory; enqueue, completion and retirement require &mut self.
unsafe impl Sync for PendingPinnedUpload {}

impl PendingPinnedUpload {
    pub(super) fn new(
        context: CudaContext,
        staging: PinnedUploadStaging,
        len: usize,
        completion: CudaEvent,
    ) -> Self {
        Self {
            context,
            staging: Some(staging),
            len,
            completion,
            recorded: false,
        }
    }

    pub(crate) fn enqueue(
        &mut self,
        destination: &CudaDeviceBuffer,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        if !self.context.is_same_context(&stream.context)
            || !self.context.is_same_context(&destination.context)
        {
            return Err(CudaError::InvalidArgument {
                message: "CUDA upload stream or destination belongs to a different context"
                    .to_string(),
            });
        }
        if destination.byte_len() < self.len {
            return Err(CudaError::OutputTooSmall {
                required: self.len,
                have: destination.byte_len(),
            });
        }
        let staging = self.staging.as_ref().ok_or(CudaError::InternalInvariant {
            what: "queued CUDA upload lost its staging allocation",
        })?;
        self.context.inner.with_current_resource_operation(|| {
            // SAFETY: `destination` is a live allocation of this context with at
            // least `len` bytes, and its pooled owner retains this checkout. The
            // filled, page-locked host span stays immutable until this event
            // proves transfer completion; the stream belongs to this context.
            self.context
                .inner
                .driver
                .check("cuMemcpyHtoDAsync_v2", unsafe {
                    (self.context.inner.driver.cu_memcpy_htod_async)(
                        destination.ptr,
                        staging.ptr.cast(),
                        self.len,
                        stream.stream,
                    )
                })
        })?;
        self.context.record_host_to_device_copy(self.len);
        self.completion.record_raw_stream(stream.stream)?;
        self.recorded = true;
        self.completion.wait_on_default_stream()
    }

    pub(crate) fn finish(&mut self) -> Result<(), PendingUploadRetireError> {
        if self.staging.is_none() {
            return Ok(());
        }
        let completion = if self.recorded {
            // A later ordered status readback often already established this
            // transfer's completion. Retire it without another host wait.
            self.completion.is_complete().and_then(|complete| {
                if complete {
                    Ok(())
                } else {
                    self.completion.synchronize()
                }
            })
        } else {
            // Recording the event may have failed after DMA submission.
            self.context
                .synchronize_for_resource_release()
                .into_result()
        };
        if let Err(error) = completion {
            self.quarantine();
            return Err(PendingUploadRetireError::CompletionUncertain(error));
        }
        let context = self.context.clone();
        // A thread that drops a queued buffer inside its own pinned-upload
        // transaction already serializes staging; locking the gate again would
        // deadlock, so recycle under that existing hold.
        if let Some(held) = HeldPinnedUploadGate::of_current_thread(&context) {
            let staging = self.take_completed_staging()?;
            return held
                .recycle_pinned_upload_staging(staging)
                .map_err(PendingUploadRetireError::RecycleFailed);
        }
        let operation = match context.begin_pinned_upload_operation() {
            Ok(operation) => operation,
            Err(error) => {
                self.quarantine();
                return Err(PendingUploadRetireError::RecycleFailed(error));
            }
        };
        let staging = self.take_completed_staging()?;
        operation
            .held()
            .recycle_pinned_upload_staging(staging)
            .map_err(PendingUploadRetireError::RecycleFailed)
    }

    fn take_completed_staging(&mut self) -> Result<PinnedUploadStaging, PendingUploadRetireError> {
        self.staging
            .take()
            .ok_or(PendingUploadRetireError::RecycleFailed(
                CudaError::InternalInvariant {
                    what: "completed CUDA upload lost its staging allocation",
                },
            ))
    }

    fn quarantine(&mut self) {
        if let Some(staging) = self.staging.take() {
            retain_pinned_upload_staging_after_abandoned_checkout(
                self.context.inner.pinned_upload_staging.lock(),
                staging,
            );
        }
    }
}

impl Drop for PendingPinnedUpload {
    fn drop(&mut self) {
        // Backstop for an owner dropped without retirement. Uncertain
        // completion quarantines the staging, which later operations report.
        let _ = self.finish();
    }
}
