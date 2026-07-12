// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned candidate/location state retained only through budget assignment.

use super::super::super::{
    ClassicSegmentAssignmentCandidate, ClassicSegmentLocation, HtSegmentAssignmentCandidate,
    HtSegmentLocation, Vec,
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
}

impl LayeredRateControlState {
    pub(super) fn owner_bytes(&self) -> Result<usize, crate::EncodeError> {
        checked_sum(
            [
                self.classic_candidate_bytes,
                self.classic_location_bytes,
                self.ht_candidate_bytes,
                self.ht_location_bytes,
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
