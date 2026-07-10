// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    bitplane_encode, vec, BlockCodingMode, CodeBlockPacketData, CodeBlockStyle, Ordering, Vec,
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

pub(super) struct ClassicLayerBudgetAllocator {
    pub(super) cumulative_targets: Vec<u64>,
    pub(super) cumulative_used: Vec<u64>,
}

impl ClassicLayerBudgetAllocator {
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
        Ok(Self {
            cumulative_targets: cumulative_targets
                .iter()
                .map(|&target| target.saturating_add(classic_rate_target_tolerance(target)))
                .collect(),
            cumulative_used: vec![0; layer_count],
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

pub(super) fn classic_rate_target_tolerance(target: u64) -> u64 {
    (target / 100).max(512)
}

pub(super) fn assign_classic_segment_layers_by_slope(
    candidates: &[ClassicSegmentAssignmentCandidate],
    layer_count: usize,
    cumulative_targets: &[u64],
) -> Result<Vec<usize>, &'static str> {
    let mut allocator = ClassicLayerBudgetAllocator::new(cumulative_targets, layer_count)?;
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let block_count = candidates
        .iter()
        .map(|candidate| candidate.block_index)
        .max()
        .and_then(|max| max.checked_add(1))
        .ok_or("classic PCRD block count overflow")?;
    let mut block_candidates = vec![Vec::new(); block_count];
    for (candidate_idx, candidate) in candidates.iter().enumerate() {
        block_candidates
            .get_mut(candidate.block_index)
            .ok_or("classic PCRD block index mismatch")?
            .push(candidate_idx);
    }
    for block in &mut block_candidates {
        block.sort_by_key(|&idx| candidates[idx].segment_index);
    }

    let mut block_min_layers = vec![0usize; block_count];
    let mut assignments = vec![layer_count.saturating_sub(1); candidates.len()];
    let mut next_block_segment = vec![0usize; block_count];
    let mut remaining = candidates.len();
    while remaining > 0 {
        let candidate_idx = block_candidates
            .iter()
            .enumerate()
            .filter_map(|(block_idx, block)| block.get(next_block_segment[block_idx]).copied())
            .min_by(|&left, &right| compare_classic_segment_candidates(candidates, left, right))
            .ok_or("classic PCRD candidate queue underflow")?;
        let candidate = candidates[candidate_idx];
        let min_layer = *block_min_layers
            .get(candidate.block_index)
            .ok_or("classic PCRD block index mismatch")?;
        let layer = allocator.assign_segment(min_layer, candidate.rate)?;
        assignments[candidate_idx] = layer;
        if let Some(block_layer) = block_min_layers.get_mut(candidate.block_index) {
            *block_layer = layer;
        }
        if let Some(next) = next_block_segment.get_mut(candidate.block_index) {
            *next = next
                .checked_add(1)
                .ok_or("classic PCRD segment index overflow")?;
        }
        remaining -= 1;
    }

    enforce_classic_assignment_monotonicity(candidates, &mut assignments);
    Ok(assignments)
}

pub(super) fn compare_classic_segment_candidates(
    candidates: &[ClassicSegmentAssignmentCandidate],
    left: usize,
    right: usize,
) -> Ordering {
    let left_candidate = candidates[left];
    let right_candidate = candidates[right];
    pcrd_slope(right_candidate)
        .partial_cmp(&pcrd_slope(left_candidate))
        .unwrap_or(Ordering::Equal)
        .then_with(|| left_candidate.block_index.cmp(&right_candidate.block_index))
        .then_with(|| {
            left_candidate
                .segment_index
                .cmp(&right_candidate.segment_index)
        })
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the codec float domain intentionally receives bounded integer samples or metadata at this rounding boundary"
)]
pub(super) fn pcrd_slope(candidate: ClassicSegmentAssignmentCandidate) -> f64 {
    if candidate.rate == 0 {
        return f64::INFINITY;
    }
    candidate.distortion_delta / candidate.rate as f64
}

pub(super) fn enforce_classic_assignment_monotonicity(
    candidates: &[ClassicSegmentAssignmentCandidate],
    assignments: &mut [usize],
) {
    let mut order: Vec<_> = (0..candidates.len()).collect();
    order.sort_by_key(|&idx| (candidates[idx].block_index, candidates[idx].segment_index));
    let mut current_block = None;
    let mut min_layer = 0usize;
    for idx in order {
        if current_block != Some(candidates[idx].block_index) {
            current_block = Some(candidates[idx].block_index);
            min_layer = 0;
        }
        if assignments[idx] < min_layer {
            assignments[idx] = min_layer;
        }
        min_layer = assignments[idx];
    }
}

pub(super) fn enforce_classic_segment_layer_monotonicity(
    layered_packets: &mut [LayeredPreparedPacket],
) {
    for packet in layered_packets {
        for subband in &mut packet.subbands {
            for block in &mut subband.blocks {
                if let LayeredPreparedBlock::Classic { segment_layers, .. } = block {
                    let mut min_layer = 0usize;
                    for layer in segment_layers {
                        if *layer < min_layer {
                            *layer = min_layer;
                        }
                        min_layer = *layer;
                    }
                }
            }
        }
    }
}

pub(super) fn enforce_ht_segment_layer_monotonicity(layered_packets: &mut [LayeredPreparedPacket]) {
    for packet in layered_packets {
        for subband in &mut packet.subbands {
            for block in &mut subband.blocks {
                if let LayeredPreparedBlock::HighThroughput { segment_layers, .. } = block {
                    let mut min_layer = 0usize;
                    for layer in segment_layers {
                        if *layer < min_layer {
                            *layer = min_layer;
                        }
                        min_layer = *layer;
                    }
                }
            }
        }
    }
}

pub(super) fn assign_ht_segment_layers_by_budget(
    candidates: &[HtSegmentAssignmentCandidate],
    layer_count: usize,
    cumulative_targets: &[u64],
) -> Result<Vec<usize>, &'static str> {
    let mut allocator = ClassicLayerBudgetAllocator::new(cumulative_targets, layer_count)?;
    let mut assignments = vec![layer_count.saturating_sub(1); candidates.len()];
    let mut candidate_order: Vec<_> = (0..candidates.len()).collect();
    candidate_order
        .sort_by_key(|&idx| (candidates[idx].block_index, candidates[idx].segment_index));
    let mut block_min_layers = vec![
        0usize;
        candidates
            .iter()
            .map(|c| c.block_index)
            .max()
            .map_or(0, |idx| idx + 1)
    ];

    for candidate_idx in candidate_order {
        let candidate = candidates
            .get(candidate_idx)
            .ok_or("HTJ2K segment candidate index mismatch")?;
        let min_layer = *block_min_layers
            .get(candidate.block_index)
            .ok_or("HTJ2K segment candidate block index mismatch")?;
        let layer = allocator.assign_segment(min_layer, candidate.rate)?;
        assignments[candidate_idx] = layer;
        if let Some(block_layer) = block_min_layers.get_mut(candidate.block_index) {
            *block_layer = layer;
        }
    }

    Ok(assignments)
}

pub(super) fn ht_segment_count(encoded: &bitplane_encode::EncodedCodeBlock) -> usize {
    match encoded.num_coding_passes {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

pub(super) fn ht_segment_rate(
    encoded: &bitplane_encode::EncodedCodeBlock,
    segment_idx: usize,
) -> Result<u64, &'static str> {
    match segment_idx {
        0 if encoded.num_coding_passes > 0 => Ok(u64::from(encoded.ht_cleanup_length)),
        1 if encoded.num_coding_passes > 1 => Ok(u64::from(encoded.ht_refinement_length)),
        _ => Err("HTJ2K segment index out of range"),
    }
}

pub(super) fn ht_unbudgeted_segment_layers(
    encoded: &bitplane_encode::EncodedCodeBlock,
    num_layers: u8,
    block_idx: usize,
    block_count: usize,
) -> Result<Vec<usize>, &'static str> {
    let segment_count = ht_segment_count(encoded);
    if segment_count == 0 {
        return Ok(Vec::new());
    }
    let layer_count = usize::from(num_layers);
    if layer_count == 0 {
        return Err("HTJ2K layer allocation requires non-empty inputs");
    }
    if encoded.num_coding_passes == 1 {
        return Ok(vec![ht_target_layer(block_idx, block_count, layer_count)?]);
    }

    let mut segment_layers = Vec::with_capacity(segment_count);
    let mut min_layer = 0usize;
    for (_, end_pass) in [(0, 1), (1, encoded.num_coding_passes)] {
        let mut assigned = None;
        for layer_idx in min_layer..layer_count {
            let cumulative_passes = if layer_idx + 1 == layer_count {
                encoded.num_coding_passes
            } else {
                layer_pass_count(encoded.num_coding_passes, layer_idx + 1, num_layers)?
            };
            if end_pass <= cumulative_passes {
                assigned = Some(layer_idx);
                break;
            }
        }
        let assigned =
            assigned.ok_or("HTJ2K quality layer split must align to segment boundaries")?;
        segment_layers.push(assigned);
        min_layer = assigned;
    }
    Ok(segment_layers)
}

pub(super) fn classic_unbudgeted_segment_layers(
    encoded: &bitplane_encode::EncodedCodeBlockWithSegments,
    num_layers: u8,
) -> Result<Vec<usize>, &'static str> {
    let mut segment_layers = Vec::with_capacity(encoded.segments.len());
    for segment in &encoded.segments {
        let mut assigned = None;
        for layer_idx in 0..usize::from(num_layers) {
            let previous_pass =
                previous_layer_pass_count(encoded.num_coding_passes, layer_idx, num_layers)?;
            let cumulative_passes = if layer_idx + 1 == usize::from(num_layers) {
                encoded.num_coding_passes
            } else {
                layer_pass_count(encoded.num_coding_passes, layer_idx + 1, num_layers)?
            };
            if segment.start_coding_pass >= previous_pass
                && segment.end_coding_pass <= cumulative_passes
            {
                assigned = Some(layer_idx);
                break;
            }
        }
        segment_layers.push(
            assigned.ok_or("classic quality layer split must align to terminated coding passes")?,
        );
    }
    Ok(segment_layers)
}

pub(super) fn classic_layer_contributions(
    encoded: &bitplane_encode::EncodedCodeBlockWithSegments,
    num_layers: u8,
    segment_layers: &[usize],
) -> Result<Vec<CodeBlockPacketData>, &'static str> {
    let layer_count = usize::from(num_layers);
    if segment_layers.len() != encoded.segments.len() {
        return Err("classic PCRD segment assignment count mismatch");
    }
    if segment_layers.iter().any(|&layer| layer >= layer_count) {
        return Err("classic PCRD segment layer exceeds layer count");
    }
    let mut contributions = Vec::with_capacity(layer_count);

    for layer_idx in 0..layer_count {
        let mut data = Vec::new();
        let mut classic_segment_lengths = Vec::new();
        let mut contribution_passes = 0u8;

        for (segment_idx, segment) in encoded.segments.iter().enumerate() {
            if segment_layers[segment_idx] != layer_idx {
                continue;
            }
            let start = usize::try_from(segment.data_offset)
                .map_err(|_| "classic code-block segment offset overflow")?;
            let len = usize::try_from(segment.data_length)
                .map_err(|_| "classic code-block segment length overflow")?;
            let end = start
                .checked_add(len)
                .ok_or("classic code-block segment range overflow")?;
            data.extend_from_slice(
                encoded
                    .data
                    .get(start..end)
                    .ok_or("classic code-block segment range invalid")?,
            );
            classic_segment_lengths.push(segment.data_length);
            contribution_passes = contribution_passes
                .checked_add(segment.end_coding_pass - segment.start_coding_pass)
                .ok_or("classic code-block contribution pass count overflow")?;
        }

        contributions.push(CodeBlockPacketData {
            data,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: contribution_passes,
            classic_segment_lengths,
            num_zero_bitplanes: encoded.num_zero_bitplanes,
            previously_included: false,
            l_block: 3,
            block_coding_mode: BlockCodingMode::Classic,
        });
    }

    Ok(contributions)
}

pub(super) fn layer_pass_count(
    num_coding_passes: u8,
    layer_count: usize,
    num_layers: u8,
) -> Result<u8, &'static str> {
    let numerator = u32::from(num_coding_passes)
        .checked_mul(u32::try_from(layer_count).map_err(|_| "layer index overflow")?)
        .ok_or("quality layer pass allocation overflow")?;
    numerator
        .div_ceil(u32::from(num_layers))
        .try_into()
        .map_err(|_| "quality layer pass allocation overflow")
}

pub(super) fn previous_layer_pass_count(
    num_coding_passes: u8,
    layer_idx: usize,
    num_layers: u8,
) -> Result<u8, &'static str> {
    if layer_idx == 0 {
        Ok(0)
    } else {
        layer_pass_count(num_coding_passes, layer_idx, num_layers)
    }
}

pub(super) fn ht_target_layer(
    block_idx: usize,
    block_count: usize,
    layer_count: usize,
) -> Result<usize, &'static str> {
    if block_count == 0 || layer_count == 0 {
        return Err("HTJ2K layer allocation requires non-empty inputs");
    }
    Ok(block_idx
        .checked_mul(layer_count)
        .ok_or("HTJ2K layer allocation overflow")?
        / block_count)
}

pub(super) fn ht_layer_contributions(
    encoded: &bitplane_encode::EncodedCodeBlock,
    num_layers: u8,
    segment_layers: &[usize],
) -> Result<Vec<CodeBlockPacketData>, &'static str> {
    let layer_count = usize::from(num_layers);
    if segment_layers.len() != ht_segment_count(encoded) {
        return Err("HTJ2K segment assignment count mismatch");
    }
    if segment_layers.iter().any(|&layer| layer >= layer_count) {
        return Err("HTJ2K segment layer exceeds layer count");
    }
    if segment_layers
        .windows(2)
        .any(|layers| layers[1] < layers[0])
    {
        return Err("HTJ2K segment layers must be monotonic");
    }

    let cleanup_len = usize::try_from(encoded.ht_cleanup_length)
        .map_err(|_| "HTJ2K cleanup segment length overflow")?;
    let refinement_len = usize::try_from(encoded.ht_refinement_length)
        .map_err(|_| "HTJ2K refinement segment length overflow")?;
    let refinement_start = cleanup_len;
    let refinement_end = refinement_start
        .checked_add(refinement_len)
        .ok_or("HTJ2K refinement segment range overflow")?;
    if encoded.num_coding_passes > 0 && cleanup_len == 0 {
        return Err("HTJ2K cleanup segment is missing");
    }
    if encoded.num_coding_passes > 1 && refinement_len == 0 {
        return Err("HTJ2K refinement segment is missing");
    }
    if refinement_end > encoded.data.len() {
        return Err("HTJ2K segment range invalid");
    }

    let mut contributions = Vec::with_capacity(layer_count);
    for layer_idx in 0..layer_count {
        let mut data = Vec::new();
        let mut ht_cleanup_length = 0u32;
        let mut ht_refinement_length = 0u32;
        let mut num_coding_passes = 0u8;

        if segment_layers.first() == Some(&layer_idx) {
            data.extend_from_slice(
                encoded
                    .data
                    .get(..cleanup_len)
                    .ok_or("HTJ2K cleanup segment range invalid")?,
            );
            ht_cleanup_length = encoded.ht_cleanup_length;
            num_coding_passes = num_coding_passes
                .checked_add(1)
                .ok_or("HTJ2K packet contribution pass count overflow")?;
        }

        if encoded.num_coding_passes > 1 && segment_layers.get(1) == Some(&layer_idx) {
            data.extend_from_slice(
                encoded
                    .data
                    .get(refinement_start..refinement_end)
                    .ok_or("HTJ2K refinement segment range invalid")?,
            );
            ht_refinement_length = encoded.ht_refinement_length;
            num_coding_passes = num_coding_passes
                .checked_add(encoded.num_coding_passes - 1)
                .ok_or("HTJ2K packet contribution pass count overflow")?;
        }

        contributions.push(CodeBlockPacketData {
            data,
            ht_cleanup_length,
            ht_refinement_length,
            num_coding_passes,
            classic_segment_lengths: Vec::new(),
            num_zero_bitplanes: encoded.num_zero_bitplanes,
            previously_included: false,
            l_block: 3,
            block_coding_mode: BlockCodingMode::HighThroughput,
        });
    }

    Ok(contributions)
}
