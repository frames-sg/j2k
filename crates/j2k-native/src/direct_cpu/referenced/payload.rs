// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{bail, DecodingError, Result};
use crate::{try_reserve_decode_elements, HtCodeBlockPayloadRanges};

pub(super) fn validate_payload_ranges(
    encoded_input: &[u8],
    payloads: &[HtCodeBlockPayloadRanges],
) -> Result<()> {
    for payload in payloads {
        payload_slice(encoded_input, payload.cleanup)?;
        if let Some(refinement) = payload.refinement {
            payload_slice(encoded_input, refinement)?;
        }
    }
    Ok(())
}

pub(super) struct ReferencedPayloadCursor<'input, 'scratch> {
    encoded_input: &'input [u8],
    payloads: &'input [HtCodeBlockPayloadRanges],
    combined: &'scratch mut alloc::vec::Vec<u8>,
    next: usize,
}

impl<'input, 'scratch> ReferencedPayloadCursor<'input, 'scratch> {
    pub(super) fn new(
        encoded_input: &'input [u8],
        payloads: &'input [HtCodeBlockPayloadRanges],
        combined: &'scratch mut alloc::vec::Vec<u8>,
    ) -> Self {
        Self {
            encoded_input,
            payloads,
            combined,
            next: 0,
        }
    }

    pub(super) fn next_data(
        &mut self,
        cleanup_length: u32,
        refinement_length: u32,
    ) -> Result<&[u8]> {
        let payload = self
            .payloads
            .get(self.next)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        self.next = self
            .next
            .checked_add(1)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        if payload.cleanup.length != cleanup_length as usize
            || payload.refinement.map_or(0, |range| range.length) != refinement_length as usize
        {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        let cleanup = payload_slice(self.encoded_input, payload.cleanup)?;
        let Some(refinement_range) = payload.refinement else {
            return Ok(cleanup);
        };
        let refinement = payload_slice(self.encoded_input, refinement_range)?;
        let combined_len = cleanup
            .len()
            .checked_add(refinement.len())
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        if payload.cleanup.end() == Some(refinement_range.offset) {
            return payload_slice(
                self.encoded_input,
                crate::J2kCodestreamRange {
                    offset: payload.cleanup.offset,
                    length: combined_len,
                },
            );
        }
        self.combined.clear();
        try_reserve_decode_elements(self.combined, combined_len)?;
        self.combined.extend_from_slice(cleanup);
        self.combined.extend_from_slice(refinement);
        Ok(self.combined)
    }

    pub(super) fn ensure_exhausted(&self) -> Result<()> {
        if self.next == self.payloads.len() {
            Ok(())
        } else {
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }
}

pub(in crate::direct_cpu) fn payload_slice(
    input: &[u8],
    range: crate::J2kCodestreamRange,
) -> Result<&[u8]> {
    let end = range
        .offset
        .checked_add(range.length)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    input
        .get(range.offset..end)
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}
