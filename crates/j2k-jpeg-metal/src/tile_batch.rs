// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::{BackendRequest, DeviceSubmission, PixelFormat};

use crate::{batch, Error, MetalDecodeRequest, MetalSession, Surface};

/// Convenience wrapper for submitting a group of JPEG tiles to one decoder
/// session.
///
/// The batch preserves submission order and lets compatible requests share a
/// Metal submission. Callers still own slide metadata, level selection, cache
/// policy, and viewport planning.
#[derive(Default)]
pub struct JpegTileBatch {
    session: MetalSession,
    submissions: Vec<batch::MetalSubmission>,
}

impl JpegTileBatch {
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
        self.push_shared_request(input, request.fmt, request.backend, request.op.batch_op())
    }

    /// Decode all queued tile requests and return surfaces in submission order.
    pub fn decode_all(self) -> Result<Vec<Surface>, Error> {
        let mut surfaces = Vec::with_capacity(self.submissions.len());
        for submission in self.submissions {
            surfaces.push(submission.wait()?);
        }
        Ok(surfaces)
    }

    fn push_shared_request(
        &mut self,
        input: Arc<[u8]>,
        fmt: PixelFormat,
        backend: BackendRequest,
        op: batch::BatchOp,
    ) -> Result<usize, Error> {
        let slot = self.submissions.len();
        let submission = {
            let mut state = self.session.shared.lock()?;
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, backend);
            let slot = state.queue_request(batch::QueuedRequest::new_shared(
                input,
                fmt,
                backend,
                op,
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
            batch::MetalSubmission {
                session: self.session.shared.clone(),
                slot,
            }
        };
        self.submissions.push(submission);
        Ok(slot)
    }
}
