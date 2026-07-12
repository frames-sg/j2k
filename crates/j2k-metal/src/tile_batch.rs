// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::DeviceSubmission;
use j2k_metal_support::FallibleSubmissionQueue;

use crate::{batch, Error, MetalDecodeRequest, MetalSession, Surface};

/// Convenience wrapper for submitting a group of J2K/HTJ2K tiles to one
/// decoder session.
///
/// This is intentionally codec-scoped: callers own slide metadata, tile
/// coordinates, cache policy, and viewport decisions. The batch only preserves
/// submission order and lets compatible tile requests share the Metal session.
#[derive(Default)]
pub struct MetalTileBatch {
    session: MetalSession,
    queue: FallibleSubmissionQueue<batch::MetalSubmission>,
}

impl MetalTileBatch {
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
        self.push_shared_tile_request(Arc::<[u8]>::from(input), request)
    }

    /// Queue a tile decode request backed by shared compressed tile bytes.
    pub fn push_shared_tile_request(
        &mut self,
        input: Arc<[u8]>,
        request: MetalDecodeRequest,
    ) -> Result<usize, Error> {
        let Self { session, queue } = self;
        queue.try_push_with(
            "J2K Metal tile batch submissions",
            |_, retained_capacity| {
                batch::queue_tile_request_shared_with_retained(
                    session,
                    input,
                    request.fmt,
                    request.backend,
                    request.op.batch_op(),
                    retained_capacity,
                )
            },
        )
    }

    /// Decode all queued tile requests and return surfaces in submission order.
    pub fn decode_all(self) -> Result<Vec<Surface>, Error> {
        self.queue.try_finish(
            "J2K Metal tile batch surface collection",
            "J2K Metal tile batch surface results",
            DeviceSubmission::wait,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};

    #[test]
    fn push_tile_request_preserves_submission_slots() {
        let mut batch = MetalTileBatch::with_capacity(2);
        let bytes = [0xff, 0x4f, 0xff, 0x51];
        let roi = Rect {
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        };

        let first = batch
            .push_tile_request(
                &bytes,
                MetalDecodeRequest::full(PixelFormat::Gray8, BackendRequest::Cpu),
            )
            .expect("queue full tile");
        let second = batch
            .push_shared_tile_request(
                Arc::<[u8]>::from(&bytes[..]),
                MetalDecodeRequest::region_scaled(
                    PixelFormat::Gray8,
                    roi,
                    Downscale::Half,
                    BackendRequest::Cpu,
                ),
            )
            .expect("queue region-scaled tile");

        assert_eq!(first, 0);
        assert_eq!(second, 1);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn oversized_capacity_hint_fails_before_queue_mutation() {
        let mut batch = MetalTileBatch::with_capacity(usize::MAX);
        let error = batch
            .push_tile_request(
                &[0xff, 0x4f],
                MetalDecodeRequest::full(PixelFormat::Gray8, BackendRequest::Cpu),
            )
            .expect_err("oversized capacity hint");

        assert!(matches!(
            error,
            Error::BatchInfrastructure(j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "J2K Metal tile batch submissions",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })
        ));
        assert!(batch.is_empty());
    }
}
