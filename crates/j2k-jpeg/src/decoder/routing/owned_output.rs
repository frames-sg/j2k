// SPDX-License-Identifier: MIT OR Apache-2.0

//! Freshly allocated decode-output routing with external-owner accounting.

use alloc::vec::Vec;

use crate::allocation::checked_add_allocation_bytes;

use super::super::{
    additional_decode_scratch_bytes, allocate_output_buffer_with_live_budget,
    checked_output_geometry, output_format_from_parts, scaled_dimensions, scaled_rect_covering,
    DecodeOutcome, DecodeRequest, Decoder, JpegError, Rect, ScratchPool, DEFAULT_SCRATCH,
};

impl Decoder<'_> {
    /// Decode into a freshly allocated tightly packed buffer using a request
    /// object instead of a method-name cross-product.
    ///
    /// # Errors
    ///
    /// Returns an output-geometry, unsupported-format, or scan decode error.
    pub fn decode_request(
        &self,
        request: DecodeRequest,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        self.decode_request_with_external_live(request, 0)
    }

    /// Decode into a new buffer while charging caller-owned host allocations.
    ///
    /// `external_live_bytes` excludes this decoder, its checkpoint cache,
    /// thread-local scratch, and the output allocated by this call.
    ///
    /// # Errors
    ///
    /// Returns [`JpegError::MemoryCapExceeded`] before a planned aggregate
    /// owner graph exceeds the decode cap, or another ordinary decode error.
    #[doc(hidden)]
    pub fn decode_request_with_external_live(
        &self,
        request: DecodeRequest,
        external_live_bytes: usize,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_request_with_scratch_and_external_live(
                &mut pool.borrow_mut(),
                request,
                external_live_bytes,
            )
        })
    }

    /// Decode into a new buffer with caller-owned scratch and external owners.
    ///
    /// # Errors
    ///
    /// Returns an aggregate host-cap or ordinary decode error.
    #[doc(hidden)]
    pub fn decode_request_with_scratch_and_external_live(
        &self,
        pool: &mut ScratchPool,
        request: DecodeRequest,
        external_live_bytes: usize,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        let legacy = output_format_from_parts(self.info.sof_kind, request.fmt, request.scale)?;
        let source_roi = request
            .region
            .unwrap_or_else(|| Rect::full(self.info.dimensions));
        if !source_roi.is_within(self.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: source_roi,
                width: self.info.dimensions.0,
                height: self.info.dimensions.1,
            });
        }
        let output_rect = if request.region.is_some() {
            scaled_rect_covering(source_roi, legacy.downscale())?
        } else {
            let (width, height) = scaled_dimensions(self.info.dimensions, legacy.downscale());
            Rect::full((width, height))
        };
        let (stride, len) =
            checked_output_geometry(output_rect.w, output_rect.h, legacy.bytes_per_pixel())?;
        let additional_scratch = additional_decode_scratch_bytes(
            self.info.sof_kind,
            self.info.dimensions,
            legacy,
            source_roi,
            output_rect,
            legacy.downscale(),
        )?;
        let planned_output_live = checked_add_allocation_bytes(external_live_bytes, len)?;
        self.prepare_decode_workspace_with_additional(
            pool,
            planned_output_live,
            additional_scratch,
        )?;
        let workspace_cap = self.decode_workspace_cap()?;
        let mut allocation_live =
            checked_add_allocation_bytes(external_live_bytes, pool.retained_bytes())?;
        let mut out =
            allocate_output_buffer_with_live_budget(len, &mut allocation_live, workspace_cap)?;
        let decode_external_live =
            checked_add_allocation_bytes(external_live_bytes, out.capacity())?;
        let outcome = if let Some(roi) = request.region {
            self.decode_region_into_output_format_with_scratch_and_external(
                pool,
                &mut out,
                stride,
                legacy,
                roi,
                decode_external_live,
            )?
        } else {
            self.decode_into_output_format_with_scratch_and_external(
                pool,
                &mut out,
                stride,
                legacy,
                decode_external_live,
            )?
        };
        Ok((out, outcome))
    }
}
