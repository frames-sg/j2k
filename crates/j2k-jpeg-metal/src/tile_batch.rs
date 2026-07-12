// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::{BackendRequest, DeviceSubmission, PixelFormat};
use j2k_metal_support::FallibleSubmissionQueue;

use crate::{batch, Error, MetalDecodeRequest, MetalSession, Surface};

enum TileRequestInput<'a> {
    Borrowed(&'a [u8]),
    Shared(Arc<[u8]>),
}

/// Convenience wrapper for submitting a group of JPEG tiles to one decoder
/// session.
///
/// The batch preserves submission order and lets compatible requests share a
/// Metal submission. Callers still own slide metadata, level selection, cache
/// policy, and viewport planning.
#[derive(Default)]
pub struct JpegTileBatch {
    session: MetalSession,
    queue: FallibleSubmissionQueue<batch::MetalSubmission>,
}

impl JpegTileBatch {
    /// Create an empty tile batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty tile batch with capacity for `capacity` submissions.
    pub fn with_capacity(capacity: usize) -> Self {
        // Capacity is a hint only: reserving is deferred to the fallible push
        // boundary because this constructor cannot report allocation failure.
        Self {
            queue: FallibleSubmissionQueue::with_capacity_hint(capacity),
            ..Self::default()
        }
    }

    /// Number of queued tile requests.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the batch has no queued tile requests.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Number of Metal session submissions already flushed.
    ///
    /// Queued requests normally do not increment this until `decode_all` waits
    /// on the first result.
    pub fn submissions(&self) -> Result<u64, Error> {
        self.session.submissions()
    }

    /// Queue a tile decode request, copying the compressed tile bytes into the batch.
    pub fn push_tile_request(
        &mut self,
        input: &[u8],
        request: MetalDecodeRequest,
    ) -> Result<usize, Error> {
        self.push_request(
            TileRequestInput::Borrowed(input),
            request.fmt,
            request.backend,
            request.op.batch_op(),
        )
    }

    /// Queue a tile decode request backed by shared compressed tile bytes.
    pub fn push_shared_tile_request(
        &mut self,
        input: Arc<[u8]>,
        request: MetalDecodeRequest,
    ) -> Result<usize, Error> {
        self.push_request(
            TileRequestInput::Shared(input),
            request.fmt,
            request.backend,
            request.op.batch_op(),
        )
    }

    /// Decode all queued tile requests and return surfaces in submission order.
    pub fn decode_all(self) -> Result<Vec<Surface>, Error> {
        self.queue.try_finish(
            "JPEG Metal tile batch surface collection",
            "JPEG Metal tile batch surface results",
            DeviceSubmission::wait,
        )
    }

    fn push_request(
        &mut self,
        input: TileRequestInput<'_>,
        fmt: PixelFormat,
        backend: BackendRequest,
        op: batch::BatchOp,
    ) -> Result<usize, Error> {
        let Self { session, queue } = self;
        queue.try_push_with(
            "JPEG Metal tile batch submissions",
            |_, retained_capacity| {
                let mut state = session.shared.lock()?;
                let submission_bytes =
                    crate::session::submission_capacity_bytes(retained_capacity)?;
                let resolved = match input {
                    TileRequestInput::Borrowed(input) => state
                        .resolve_jpeg_plan_with_external_live(input, backend, submission_bytes)?,
                    TileRequestInput::Shared(input) => state
                        .resolve_arc_jpeg_plan_with_external_live(
                            input,
                            backend,
                            submission_bytes,
                        )?,
                };
                let slot = state.queue_request_with_retained(
                    batch::QueuedRequest::new_shared(
                        resolved.input,
                        fmt,
                        backend,
                        op,
                        resolved.fast_packet,
                        resolved.shape,
                    ),
                    retained_capacity,
                )?;
                Ok(batch::MetalSubmission {
                    session: session.shared.clone(),
                    slot,
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SharedJpegInput;

    #[test]
    fn oversized_capacity_hint_fails_before_queue_mutation() {
        let mut batch = JpegTileBatch::with_capacity(usize::MAX);
        let error = batch
            .push_tile_request(
                &[0xff, 0xd8],
                MetalDecodeRequest::full(PixelFormat::Rgb8, BackendRequest::Cpu),
            )
            .expect_err("oversized capacity hint");

        assert!(matches!(
            error,
            Error::BatchInfrastructure(j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG Metal tile batch submissions",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })
        ));
        assert!(batch.is_empty());
    }

    #[test]
    fn shared_tile_request_reuses_caller_arc_payload_without_copying() {
        let input = Arc::<[u8]>::from(&b"shared non-Metal JPEG bytes"[..]);
        let expected = SharedJpegInput::try_from_arc(Arc::clone(&input))
            .expect("shared input within default cap");
        let mut batch = JpegTileBatch::new();

        batch
            .push_shared_tile_request(
                input,
                MetalDecodeRequest::full(PixelFormat::Rgb8, BackendRequest::Cpu),
            )
            .expect("queue shared CPU request");

        let state = batch.session.shared.lock().expect("session state");
        assert_eq!(state.queued.len(), 1);
        assert!(SharedJpegInput::ptr_eq(&expected, &state.queued[0].input));
    }
}
