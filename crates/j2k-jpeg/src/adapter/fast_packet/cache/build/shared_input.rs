// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoder and CPU-fallback operations over one shared JPEG input owner.

use alloc::vec::Vec;

use crate::adapter::fast_packet::cache::shared_allocation::checked_live_bytes;
use crate::adapter::fast_packet::cache::{JpegCachedPlanBuildError, SharedJpegInput};
use crate::decoder::{DecodeOutcome, DecodeRequest, Decoder, JpegView};

impl SharedJpegInput {
    /// Construct one decoder while charging this shared owner and adapter owners.
    ///
    /// # Errors
    ///
    /// Returns a typed aggregate-limit, parse, or decoder-construction error.
    #[doc(hidden)]
    pub fn decoder_with_external_live(
        &self,
        external_live_bytes: usize,
    ) -> Result<Decoder<'_>, JpegCachedPlanBuildError> {
        let owner_live_bytes = checked_live_bytes(
            "shared JPEG input, cache, and adapter owner graph",
            external_live_bytes,
            self.retained_cache_bytes()?,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let view = JpegView::parse_with_external_live(self.as_bytes(), owner_live_bytes)?;
        Decoder::from_view_with_external_live(view, owner_live_bytes).map_err(Into::into)
    }

    /// Construct and CPU-decode while charging this input and adapter owners.
    ///
    /// # Errors
    ///
    /// Returns a typed aggregate-limit, parse, construction, or decode error.
    #[doc(hidden)]
    pub fn decode_request_with_external_live(
        &self,
        request: DecodeRequest,
        external_live_bytes: usize,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegCachedPlanBuildError> {
        let decode_external_live = checked_live_bytes(
            "shared JPEG input and CPU fallback owner graph",
            external_live_bytes,
            self.retained_cache_bytes()?,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let decoder = self.decoder_with_external_live(external_live_bytes)?;
        decoder
            .decode_request_with_external_live(request, decode_external_live)
            .map_err(Into::into)
    }
}
