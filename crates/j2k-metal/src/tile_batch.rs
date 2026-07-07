// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::DeviceSubmission;

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
    submissions: Vec<batch::MetalSubmission>,
}

impl MetalTileBatch {
    /// Create an empty tile batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty tile batch with capacity for `capacity` submissions.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            submissions: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Number of queued tile requests.
    pub fn len(&self) -> usize {
        self.submissions.len()
    }

    /// Whether the batch has no queued tile requests.
    pub fn is_empty(&self) -> bool {
        self.submissions.is_empty()
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
        let slot = self.submissions.len();
        let submission = batch::queue_tile_request_shared(
            &mut self.session,
            input,
            request.fmt,
            request.backend,
            request.op.batch_op(),
        )?;
        self.submissions.push(submission);
        Ok(slot)
    }

    /// Decode all queued tile requests and return surfaces in submission order.
    pub fn decode_all(self) -> Result<Vec<Surface>, Error> {
        let mut surfaces = Vec::with_capacity(self.submissions.len());
        for submission in self.submissions {
            surfaces.push(submission.wait()?);
        }
        Ok(surfaces)
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
}
