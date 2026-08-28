// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned candidate/location state retained only through budget assignment.

use super::super::super::allocation::checked_element_bytes;
use super::super::super::{
    bitplane_encode, ClassicSegmentAssignmentCandidate, ClassicSegmentLocation,
    HtSegmentAssignmentCandidate, HtSegmentLocation, NativeEncodePipelineError,
    NativeEncodePipelineResult, Vec,
};
use super::ownership::checked_sum;

#[derive(Default)]
pub(super) struct LayeredRateControlState {
    pub(super) classic_candidates: Vec<ClassicSegmentAssignmentCandidate>,
    pub(super) classic_candidate_bytes: usize,
    pub(super) classic_locations: Vec<ClassicSegmentLocation>,
    pub(super) classic_location_bytes: usize,
    pub(super) classic_block_index: usize,
    pub(super) ht_candidates: Vec<HtSegmentAssignmentCandidate>,
    pub(super) ht_candidate_bytes: usize,
    pub(super) ht_locations: Vec<HtSegmentLocation>,
    pub(super) ht_location_bytes: usize,
    pub(super) ht_block_index: usize,
    selected_ht_candidates: Vec<bitplane_encode::EncodedCodeBlock>,
    selected_ht_payload_bytes: usize,
}

impl LayeredRateControlState {
    pub(super) fn try_with_selected_ht_candidates(
        selected_ht_candidates: Vec<bitplane_encode::EncodedCodeBlock>,
    ) -> NativeEncodePipelineResult<Self> {
        let selected_ht_payload_bytes =
            selected_ht_candidates
                .iter()
                .try_fold(0usize, |total, candidate| {
                    total.checked_add(candidate.data.capacity()).ok_or(
                        crate::EncodeError::ArithmeticOverflow {
                            what: "selected HT candidate payload ownership",
                        },
                    )
                })?;
        Ok(Self {
            selected_ht_candidates,
            selected_ht_payload_bytes,
            ..Self::default()
        })
    }

    pub(super) fn take_selected_ht_candidate(
        &mut self,
    ) -> NativeEncodePipelineResult<bitplane_encode::EncodedCodeBlock> {
        let candidate = self.selected_ht_candidates.pop().ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "whole-tile selected HT candidate is missing",
            )
        })?;
        self.selected_ht_payload_bytes = self
            .selected_ht_payload_bytes
            .checked_sub(candidate.data.capacity())
            .ok_or(crate::EncodeError::InternalInvariant {
                what: "selected HT candidate payload ownership underflowed",
            })?;
        Ok(candidate)
    }

    pub(super) fn ensure_selected_ht_candidates_consumed(&self) -> NativeEncodePipelineResult<()> {
        if self.selected_ht_candidates.is_empty() && self.selected_ht_payload_bytes == 0 {
            Ok(())
        } else {
            Err(NativeEncodePipelineError::internal_invariant(
                "whole-tile selected HT candidates were not consumed",
            ))
        }
    }

    pub(super) fn owner_bytes(&self) -> Result<usize, crate::EncodeError> {
        let selected_structural = checked_element_bytes::<bitplane_encode::EncodedCodeBlock>(
            self.selected_ht_candidates.capacity(),
            "selected HT candidate owners",
        )?;
        checked_sum(
            [
                self.classic_candidate_bytes,
                self.classic_location_bytes,
                self.ht_candidate_bytes,
                self.ht_location_bytes,
                selected_structural,
                self.selected_ht_payload_bytes,
            ],
            "layered rate-control owners",
        )
    }

    pub(super) fn live_bytes(
        &self,
        source_bytes: usize,
        layered_bytes: usize,
    ) -> Result<usize, crate::EncodeError> {
        checked_sum(
            [source_bytes, layered_bytes, self.owner_bytes()?],
            "layered rate-control live owners",
        )
    }
}
