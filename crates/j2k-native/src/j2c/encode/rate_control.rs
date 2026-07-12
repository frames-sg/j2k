// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-layer rate-control model and focused assignment/contribution modules.

use super::{bitplane_encode, CodeBlockStyle, Vec};

mod assignment;
#[cfg(test)]
pub(super) use assignment::{
    assign_classic_segment_layers_by_slope, assign_ht_segment_layers_by_budget,
};
pub(super) use assignment::{
    assign_classic_segment_layers_by_slope_accounted, assign_ht_segment_layers_by_budget_accounted,
    enforce_classic_segment_layer_monotonicity, enforce_ht_segment_layer_monotonicity,
};
mod contributions;
#[cfg(test)]
pub(super) use contributions::ht_layer_contributions;
pub(super) use contributions::{
    classic_layer_contributions_accounted, classic_unbudgeted_segment_layers_accounted,
    ht_layer_contributions_accounted, ht_segment_count, ht_segment_rate,
    ht_unbudgeted_segment_layers_accounted,
};

pub(super) fn classic_multilayer_code_block_style() -> CodeBlockStyle {
    CodeBlockStyle {
        termination_on_each_pass: true,
        ..CodeBlockStyle::default()
    }
}

pub(super) struct LayeredPreparedPacket {
    pub(super) component: u16,
    pub(super) resolution: u32,
    pub(super) precinct: u64,
    pub(super) subbands: Vec<LayeredPreparedSubband>,
}

pub(super) struct LayeredPreparedSubband {
    pub(super) num_cbs_x: u32,
    pub(super) num_cbs_y: u32,
    pub(super) blocks: Vec<LayeredPreparedBlock>,
}

pub(super) enum LayeredPreparedBlock {
    Classic {
        encoded: bitplane_encode::EncodedCodeBlockWithSegments,
        segment_layers: Vec<usize>,
    },
    HighThroughput {
        encoded: bitplane_encode::EncodedCodeBlock,
        segment_layers: Vec<usize>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ClassicSegmentAssignmentCandidate {
    pub(super) block_index: usize,
    pub(super) segment_index: usize,
    pub(super) rate: u64,
    pub(super) distortion_delta: f64,
}

#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::struct_field_names,
    reason = "the repeated _idx suffix distinguishes every nested packet location coordinate"
)]
pub(super) struct ClassicSegmentLocation {
    pub(super) packet_idx: usize,
    pub(super) subband_idx: usize,
    pub(super) block_idx: usize,
    pub(super) segment_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HtSegmentAssignmentCandidate {
    pub(super) block_index: usize,
    pub(super) segment_index: usize,
    pub(super) rate: u64,
}

#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::struct_field_names,
    reason = "the repeated _idx suffix distinguishes every nested packet location coordinate"
)]
pub(super) struct HtSegmentLocation {
    pub(super) packet_idx: usize,
    pub(super) subband_idx: usize,
    pub(super) block_idx: usize,
    pub(super) segment_idx: usize,
}

struct ClassicLayerBudgetAllocator {
    cumulative_targets: Vec<u64>,
    cumulative_used: Vec<u64>,
}

impl ClassicLayerBudgetAllocator {
    #[cfg(test)]
    fn new(cumulative_targets: &[u64], layer_count: usize) -> Result<Self, &'static str> {
        if cumulative_targets.is_empty() {
            return Ok(Self {
                cumulative_targets: Vec::new(),
                cumulative_used: Vec::new(),
            });
        }
        if cumulative_targets.len() != layer_count {
            return Err("quality layer byte target count must match quality layer count");
        }
        if cumulative_targets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err("quality layer byte targets must be cumulative and monotonic");
        }
        let mut targets = Vec::new();
        targets
            .try_reserve_exact(cumulative_targets.len())
            .map_err(|_| "classic PCRD target allocation failed")?;
        targets.extend(
            cumulative_targets
                .iter()
                .map(|&target| target.saturating_add(classic_rate_target_tolerance(target))),
        );
        let mut used = Vec::new();
        used.try_reserve_exact(layer_count)
            .map_err(|_| "classic PCRD usage allocation failed")?;
        used.resize(layer_count, 0);
        Ok(Self {
            cumulative_targets: targets,
            cumulative_used: used,
        })
    }

    fn is_budgeted(&self) -> bool {
        !self.cumulative_targets.is_empty()
    }

    fn assign_segment(
        &mut self,
        min_layer: usize,
        data_length: u64,
    ) -> Result<usize, &'static str> {
        if !self.is_budgeted() {
            return Ok(min_layer);
        }

        let rate = data_length;
        let last_layer = self
            .cumulative_targets
            .len()
            .checked_sub(1)
            .ok_or("quality layer target count underflow")?;
        for layer_idx in min_layer..last_layer {
            if self.layer_can_accept(layer_idx, rate)? {
                self.record_segment(layer_idx, rate)?;
                return Ok(layer_idx);
            }
        }
        self.record_segment(last_layer, rate)?;
        Ok(last_layer)
    }

    fn layer_can_accept(&self, layer_idx: usize, rate: u64) -> Result<bool, &'static str> {
        for cumulative_idx in layer_idx..self.cumulative_targets.len() {
            let used = self.cumulative_used[cumulative_idx]
                .checked_add(rate)
                .ok_or("quality layer byte budget overflow")?;
            if used > self.cumulative_targets[cumulative_idx] {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn record_segment(&mut self, layer_idx: usize, rate: u64) -> Result<(), &'static str> {
        for used in &mut self.cumulative_used[layer_idx..] {
            *used = used
                .checked_add(rate)
                .ok_or("quality layer byte budget overflow")?;
        }
        Ok(())
    }
}

fn classic_rate_target_tolerance(target: u64) -> u64 {
    (target / 100).max(512)
}
