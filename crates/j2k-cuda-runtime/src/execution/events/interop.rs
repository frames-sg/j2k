// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::{ContextOwnership, CudaContext},
    driver::CuStream,
    error::CudaError,
};

impl CudaContext {
    /// Order default-stream codec work between operations on a caller-owned
    /// CUDA stream without a host or context-wide synchronization.
    ///
    /// An event recorded on `raw_stream` is made a dependency of the codec's
    /// default stream before `work`. After `work` submits its codec graph, a
    /// second event makes the caller stream wait for the codec final store.
    /// The raw stream is confined to this guarded closure and is never stored
    /// or returned.
    ///
    /// # Safety
    ///
    /// `raw_stream` must be a live CUDA stream from this context's retained
    /// primary context and remain live for the complete call. `_managed_owner`
    /// must exclusively guard the external runtime resource that owns the
    /// stream. `work` must retain every resource reachable by asynchronously
    /// submitted codec work in its returned value or another typed guard.
    #[doc(hidden)]
    pub unsafe fn with_primary_stream_ordering<Owner, T, E>(
        &self,
        raw_stream: u64,
        _managed_owner: &mut Owner,
        work: impl FnOnce() -> Result<T, E>,
    ) -> Result<Result<T, E>, CudaError> {
        if !matches!(
            self.inner.ownership,
            ContextOwnership::RetainedPrimary { .. }
        ) {
            return Err(CudaError::InvalidArgument {
                message: "external CUDA stream interop requires a retained primary context"
                    .to_string(),
            });
        }
        if raw_stream == 0 {
            return Err(CudaError::InvalidArgument {
                message: "external CUDA stream handle must not be null".to_string(),
            });
        }
        let stream_address = usize::try_from(raw_stream)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        let stream = stream_address as CuStream;
        let producer_ready = self.create_event()?;
        producer_ready.record_raw_stream(stream)?;
        producer_ready.wait_on_default_stream()?;

        let output = work();
        let codec_ready = match self.create_event() {
            Ok(event) => event,
            Err(error) => {
                // `work` may already have enqueued kernels that reference an
                // external allocation. Keep `output` live while establishing
                // a context completion boundary before propagating failure.
                return self.synchronize_then_error(error);
            }
        };
        if let Err(error) = codec_ready
            .record_default_stream()
            .and_then(|()| codec_ready.wait_on_raw_stream(stream))
        {
            return self.synchronize_then_error(error);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::{CudaContext, CudaError};

    #[test]
    fn primary_stream_bridge_orders_both_streams_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }
        let context = CudaContext::retain_primary(0).expect("retained primary context");
        let mut stream = context.create_stream().expect("caller stream");
        let raw_stream = stream.stream as usize as u64;

        // SAFETY: the test stream was created by this retained primary context
        // and `stream` remains exclusively borrowed for the complete bridge.
        let output = unsafe {
            context.with_primary_stream_ordering(raw_stream, &mut stream, || {
                Ok::<_, CudaError>(17_u32)
            })
        }
        .expect("event ordering")
        .expect("guarded work");

        assert_eq!(output, 17);
    }

    #[test]
    fn primary_stream_bridge_reuses_captured_event_generations_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }
        let context = CudaContext::retain_primary(0).expect("retained primary context");
        let mut stream = context.create_stream().expect("caller stream");
        let raw_stream = stream.stream as usize as u64;

        // Warm the two-handle bridge cache, then establish a clean diagnostic
        // baseline. Each later cuStreamWaitEvent captures the current record,
        // so re-recording a recycled handle cannot retarget an earlier wait.
        // SAFETY: the test stream belongs to this retained primary context and
        // remains exclusively borrowed throughout each bridge call.
        unsafe {
            context.with_primary_stream_ordering(raw_stream, &mut stream, || Ok::<_, CudaError>(()))
        }
        .expect("warm bridge ordering")
        .expect("warm guarded work");
        context.synchronize().expect("warm bridge completion");
        let before = context.diagnostics().expect("diagnostics before reuse");

        for value in 0_u32..32 {
            // SAFETY: identical ownership and lifetime proof to the warm call.
            let output = unsafe {
                context.with_primary_stream_ordering(raw_stream, &mut stream, || {
                    Ok::<_, CudaError>(value)
                })
            }
            .expect("reused bridge ordering")
            .expect("reused guarded work");
            assert_eq!(output, value);
        }
        context.synchronize().expect("reused bridge completion");
        let after = context.diagnostics().expect("diagnostics after reuse");

        assert_eq!(
            after.event_driver_allocations,
            before.event_driver_allocations
        );
        assert_eq!(after.event_reuses - before.event_reuses, 64);
    }

    #[test]
    fn primary_stream_bridge_has_no_context_wide_sync_on_success_path() {
        let bridge = production_bridge_source();
        assert!(bridge.contains("wait_on_default_stream"));
        assert!(bridge.contains("wait_on_raw_stream"));
        assert!(!bridge.contains("cuCtxSynchronize"));
    }

    #[test]
    fn primary_stream_bridge_recovers_post_submit_event_creation_failure() {
        let bridge = production_bridge_source();
        let work = bridge.find("let output = work();").expect("submitted work");
        let recovery = bridge
            .find("return self.synchronize_then_error(error);")
            .expect("post-submit recovery");
        assert!(recovery > work);
    }

    #[test]
    fn completion_event_can_be_polled_without_a_host_wait_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }
        let context = CudaContext::system_default().expect("CUDA context");
        let completion = context.create_event().expect("completion event");
        completion
            .record_default_stream()
            .expect("record completion event");

        let _may_already_be_complete = completion.is_complete().expect("query completion event");
        completion.synchronize().expect("wait completion event");
        assert!(completion.is_complete().expect("query completed event"));
    }

    fn production_bridge_source() -> &'static str {
        include_str!("interop.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production primary stream bridge")
    }
}
