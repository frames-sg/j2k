// SPDX-License-Identifier: MIT OR Apache-2.0

//! Referenced classic JPEG 2000 payload traversal.

use super::super::{
    DirectTier1Mode, Error, J2kClassicCodeBlockPayload, J2kCodestreamRange,
    J2kReferencedClassicPlan, PreparedClassicSubBand,
};
use super::sub_band::prepare_classic_sub_band_with_payloads;

#[cfg(target_os = "macos")]
pub(in crate::compute::direct_prepare) struct ReferencedClassicPayloadCursor<'a> {
    input: &'a [u8],
    payloads: &'a [J2kClassicCodeBlockPayload],
    ranges: &'a [J2kCodestreamRange],
    pub(in crate::compute::direct_prepare) next_payload: usize,
    next_range: usize,
}

#[cfg(target_os = "macos")]
impl<'a> ReferencedClassicPayloadCursor<'a> {
    pub(in crate::compute::direct_prepare) fn new(
        input: &'a [u8],
        plan: &'a J2kReferencedClassicPlan,
    ) -> Self {
        Self {
            input,
            payloads: plan.payloads(),
            ranges: plan.ranges(),
            next_payload: 0,
            next_range: 0,
        }
    }

    fn expected_payload_bytes(&self, count: usize) -> Result<usize, Error> {
        let end = self
            .next_payload
            .checked_add(count)
            .ok_or(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "payload traversal count overflowed",
            })?;
        let payloads =
            self.payloads
                .get(self.next_payload..end)
                .ok_or(Error::MetalStateInvariant {
                    state: "classic J2K referenced payload cursor",
                    reason: "geometry contains more jobs than retained payload descriptors",
                })?;
        Ok(crate::batch_allocation::checked_count_sum(
            payloads.iter().map(|payload| payload.combined_length),
            "classic J2K referenced Metal coded payload",
        )?)
    }

    fn append_next(&mut self, coded_data: &mut Vec<u8>) -> Result<usize, Error> {
        let payload =
            self.payloads
                .get(self.next_payload)
                .copied()
                .ok_or(Error::MetalStateInvariant {
                    state: "classic J2K referenced payload cursor",
                    reason: "geometry contains more jobs than retained payload descriptors",
                })?;
        if payload.first_range != self.next_range {
            return Err(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "payload fragment ranges are not contiguous in traversal order",
            });
        }
        let end_range = payload.end_range().ok_or(Error::MetalStateInvariant {
            state: "classic J2K referenced payload cursor",
            reason: "payload fragment range overflowed",
        })?;
        let fragments =
            self.ranges
                .get(payload.first_range..end_range)
                .ok_or(Error::MetalStateInvariant {
                    state: "classic J2K referenced payload cursor",
                    reason: "payload fragment range exceeds the retained range table",
                })?;
        let before = coded_data.len();
        for range in fragments {
            let end = range.end().ok_or(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "encoded payload byte range overflowed",
            })?;
            let fragment = self
                .input
                .get(range.offset..end)
                .ok_or(Error::MetalStateInvariant {
                    state: "classic J2K referenced payload cursor",
                    reason: "encoded payload byte range exceeds the retained input",
                })?;
            coded_data.extend_from_slice(fragment);
        }
        let appended = coded_data
            .len()
            .checked_sub(before)
            .ok_or(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "coded payload length moved backwards",
            })?;
        if appended != payload.combined_length {
            return Err(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "concatenated fragments do not match their retained payload length",
            });
        }
        self.next_payload = self
            .next_payload
            .checked_add(1)
            .ok_or(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "payload cursor overflowed",
            })?;
        self.next_range = end_range;
        Ok(appended)
    }

    pub(in crate::compute::direct_prepare) fn ensure_exhausted(&self) -> Result<(), Error> {
        if self.next_payload == self.payloads.len() && self.next_range == self.ranges.len() {
            Ok(())
        } else {
            Err(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "retained payload descriptors or ranges were left unused",
            })
        }
    }
}

#[cfg(target_os = "macos")]
pub(in crate::compute::direct_prepare) fn prepare_referenced_classic_sub_band(
    job: &j2k_native::J2kOwnedSubBandPlan,
    payloads: &mut ReferencedClassicPayloadCursor<'_>,
) -> Result<PreparedClassicSubBand, Error> {
    if job.jobs.iter().any(|block| !block.data.is_empty()) {
        return Err(Error::MetalStateInvariant {
            state: "classic J2K referenced Metal sub-band",
            reason: "referenced geometry unexpectedly owns compressed payload bytes",
        });
    }
    let coded_len = payloads.expected_payload_bytes(job.jobs.len())?;
    prepare_classic_sub_band_with_payloads(
        job,
        DirectTier1Mode::Metal,
        coded_len,
        |_, coded_data| payloads.append_next(coded_data),
    )
}
