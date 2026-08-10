// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{bail, DecodingError, Result};
use crate::{try_reserve_decode_elements, HtCodeBlockPayloadRanges};

#[cfg(test)]
mod tests;

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

pub(super) struct ReferencedPayloadCursor<'input> {
    encoded_input: &'input [u8],
    payloads: &'input [HtCodeBlockPayloadRanges],
    next: usize,
}

impl<'input> ReferencedPayloadCursor<'input> {
    pub(super) fn new(
        encoded_input: &'input [u8],
        payloads: &'input [HtCodeBlockPayloadRanges],
    ) -> Self {
        Self {
            encoded_input,
            payloads,
            next: 0,
        }
    }

    pub(super) fn next_data<'scratch>(
        &'scratch mut self,
        cleanup_length: u32,
        refinement_length: u32,
        combined: &'scratch mut alloc::vec::Vec<u8>,
    ) -> Result<&'scratch [u8]> {
        let payload = *self
            .payloads
            .get(self.next)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        self.next = self
            .next
            .checked_add(1)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        if payload.cleanup.length != cleanup_length as usize {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        let cleanup = payload_slice(self.encoded_input, payload.cleanup)?;
        let expected_refinement_len = refinement_length as usize;
        let first_refinement = payload
            .refinement
            .map(|range| payload_slice(self.encoded_input, range))
            .transpose()?
            .unwrap_or(&[]);
        if first_refinement.len() > expected_refinement_len {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        if first_refinement.len() == expected_refinement_len {
            let Some(refinement_range) = payload.refinement else {
                return Ok(cleanup);
            };
            if payload.cleanup.end() == Some(refinement_range.offset) {
                return payload_slice(
                    self.encoded_input,
                    crate::J2kCodestreamRange {
                        offset: payload.cleanup.offset,
                        length: cleanup
                            .len()
                            .checked_add(first_refinement.len())
                            .ok_or(DecodingError::CodeBlockDecodeFailure)?,
                    },
                );
            }
        }

        let combined_len = cleanup
            .len()
            .checked_add(expected_refinement_len)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        combined.clear();
        try_reserve_decode_elements(combined, combined_len)?;
        combined.extend_from_slice(cleanup);
        combined.extend_from_slice(first_refinement);
        let mut collected_refinement_len = first_refinement.len();
        while collected_refinement_len < expected_refinement_len {
            let continuation = *self
                .payloads
                .get(self.next)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            self.next = self
                .next
                .checked_add(1)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            if continuation.cleanup.length != 0 {
                bail!(DecodingError::CodeBlockDecodeFailure);
            }
            payload_slice(self.encoded_input, continuation.cleanup)?;
            let refinement_range = continuation
                .refinement
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let refinement = payload_slice(self.encoded_input, refinement_range)?;
            if refinement.is_empty() {
                bail!(DecodingError::CodeBlockDecodeFailure);
            }
            collected_refinement_len = collected_refinement_len
                .checked_add(refinement.len())
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            if collected_refinement_len > expected_refinement_len {
                bail!(DecodingError::CodeBlockDecodeFailure);
            }
            combined.extend_from_slice(refinement);
        }
        Ok(combined)
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
