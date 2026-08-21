// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reusable-session tile submission boundary.

use crate::{batch, Codec, Error, MetalDecodeRequest, MetalSession};
use j2k_core::TileBatchDecodeSubmit;
use j2k_jpeg::{DecoderContext as CpuDecoderContext, ScratchPool as CpuScratchPool};

impl Codec {
    /// Submit a tile decode request into a reusable Metal session.
    #[doc(hidden)]
    pub fn submit_tile_request_to_device(
        ctx: &mut CpuDecoderContext,
        session: &mut MetalSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        request: MetalDecodeRequest,
    ) -> Result<<Self as TileBatchDecodeSubmit>::SubmittedSurface, Error> {
        let _ = (ctx, pool);
        let slot = {
            let mut state = session.shared.lock()?;
            let resolved = state.resolve_jpeg_plan(input, request.backend)?;
            state.queue_request(batch::QueuedRequest::new_shared(
                resolved.input,
                request.fmt,
                request.backend,
                request.op.batch_op(),
                resolved.fast_packet,
                resolved.shape,
            ))?
        };
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }
}
