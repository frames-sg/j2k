// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationEncodeJob,
    J2kPacketizationResolution,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{
    CudaHtj2kPacketizationBlock, CudaHtj2kPacketizationPacket, CudaHtj2kPacketizationSubband,
    CudaHtj2kPacketizationSubbandTagState, CudaHtj2kPacketizationTagNodeState,
};
use j2k_native::packet_math;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CudaHtj2kPacketizationPlan {
    pub(super) payload: Vec<u8>,
    pub(super) packets: Vec<CudaHtj2kPacketizationPlanPacket>,
    pub(super) subbands: Vec<CudaHtj2kPacketizationPlanSubband>,
    pub(super) blocks: Vec<CudaHtj2kPacketizationPlanBlock>,
    pub(super) tag_states: Vec<CudaHtj2kPacketizationPlanSubbandTagState>,
    pub(super) tag_nodes: Vec<CudaHtj2kPacketizationPlanTagNodeState>,
}

struct CudaHtj2kPacketizationPlanSink<'a> {
    payload: &'a mut Vec<u8>,
    packets: &'a mut Vec<CudaHtj2kPacketizationPlanPacket>,
    subbands: &'a mut Vec<CudaHtj2kPacketizationPlanSubband>,
    blocks: &'a mut Vec<CudaHtj2kPacketizationPlanBlock>,
    tag_states: &'a mut Vec<CudaHtj2kPacketizationPlanSubbandTagState>,
    tag_nodes: &'a mut Vec<CudaHtj2kPacketizationPlanTagNodeState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaHtj2kPacketizationPlanPacket {
    pub(super) block_start: u32,
    pub(super) block_count: u32,
    pub(super) subband_start: u32,
    pub(super) subband_count: u32,
    pub(super) output_capacity: u32,
    pub(super) layer: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaHtj2kPacketizationPlanSubband {
    pub(super) block_start: u32,
    pub(super) block_count: u32,
    pub(super) num_cbs_x: u32,
    pub(super) num_cbs_y: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaHtj2kPacketizationPlanBlock {
    pub(super) data_offset: u32,
    pub(super) data_len: u32,
    pub(super) cleanup_length: u32,
    pub(super) refinement_length: u32,
    pub(super) num_coding_passes: u32,
    pub(super) num_zero_bitplanes: u32,
    pub(super) l_block: u32,
    pub(super) previously_included: u32,
    pub(super) inclusion_layer: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaHtj2kPacketizationPlanSubbandTagState {
    pub(super) inclusion_node_start: u32,
    pub(super) zero_bitplane_node_start: u32,
    pub(super) node_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaHtj2kPacketizationPlanTagNodeState {
    pub(super) current: u32,
    pub(super) known: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaHtj2kPacketizationTagTreeState {
    values: Vec<u32>,
    current: Vec<u32>,
    known: Vec<u32>,
    widths: Vec<u32>,
    heights: Vec<u32>,
    offsets: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaHtj2kPacketizationBlockState {
    previously_included: bool,
    l_block: u32,
    inclusion_layer: u32,
    first_inclusion_zero_bitplanes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaHtj2kPacketizationSubbandState {
    num_cbs_x: u32,
    num_cbs_y: u32,
    inclusion_tree: CudaHtj2kPacketizationTagTreeState,
    zero_bitplane_tree: CudaHtj2kPacketizationTagTreeState,
    blocks: Vec<CudaHtj2kPacketizationBlockState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaHtj2kPacketizationState {
    subbands: Vec<CudaHtj2kPacketizationSubbandState>,
}

pub(super) fn flatten_cuda_htj2k_packetization_job(
    job: J2kPacketizationEncodeJob<'_>,
) -> core::result::Result<CudaHtj2kPacketizationPlan, &'static str> {
    if job.resolution_count as usize != job.resolutions.len() {
        return Err("CUDA HTJ2K packetization resolution count mismatch");
    }

    let mut payload = Vec::new();
    let mut packets = Vec::new();
    let mut subbands = Vec::new();
    let mut blocks = Vec::new();
    let mut tag_states = Vec::new();
    let mut tag_nodes = Vec::new();

    {
        let mut sink = CudaHtj2kPacketizationPlanSink {
            payload: &mut payload,
            packets: &mut packets,
            subbands: &mut subbands,
            blocks: &mut blocks,
            tag_states: &mut tag_states,
            tag_nodes: &mut tag_nodes,
        };
        if job.packet_descriptors.is_empty() {
            if job.num_layers != 1 {
                return Err(
                    "CUDA HTJ2K packetization requires explicit descriptors for multiple layers",
                );
            }
            for packet_index in 0..job.resolutions.len() {
                flatten_cuda_htj2k_packet(
                    job.resolutions
                        .get(packet_index)
                        .ok_or("CUDA HTJ2K packet descriptor index out of range")?,
                    &mut sink,
                )?;
            }
        } else {
            let state_count = job
                .packet_descriptors
                .iter()
                .map(|descriptor| descriptor.state_index as usize)
                .max()
                .map_or(0usize, |max_state| max_state + 1);
            let mut states: Vec<Option<CudaHtj2kPacketizationState>> =
                core::iter::repeat_with(|| None).take(state_count).collect();
            for descriptor in job.packet_descriptors {
                if descriptor.layer >= job.num_layers {
                    return Err("CUDA HTJ2K packetization descriptor layer exceeds layer count");
                }
                let resolution = job
                    .resolutions
                    .get(descriptor.packet_index as usize)
                    .ok_or("CUDA HTJ2K packet descriptor index out of range")?;
                let state = states
                    .get_mut(descriptor.state_index as usize)
                    .ok_or("CUDA HTJ2K packet descriptor state index out of range")?;
                if let Some(existing) = state {
                    validate_cuda_htj2k_packetization_state_layout(existing, resolution)?;
                } else {
                    *state = Some(seed_cuda_htj2k_packetization_state(resolution)?);
                }
                let state = state
                    .as_mut()
                    .ok_or("CUDA HTJ2K packetization state initialization failed")?;
                record_cuda_htj2k_packetization_first_inclusion_layers(
                    state,
                    resolution,
                    descriptor.layer,
                )?;
            }
            for state in states.iter_mut().flatten() {
                finalize_cuda_htj2k_packetization_tag_trees(state);
            }
            for descriptor in job.packet_descriptors {
                if descriptor.layer >= job.num_layers {
                    return Err("CUDA HTJ2K packetization descriptor layer exceeds layer count");
                }
                let resolution = job
                    .resolutions
                    .get(descriptor.packet_index as usize)
                    .ok_or("CUDA HTJ2K packet descriptor index out of range")?;
                let state = states
                    .get_mut(descriptor.state_index as usize)
                    .ok_or("CUDA HTJ2K packet descriptor state index out of range")?;
                if let Some(existing) = state {
                    validate_cuda_htj2k_packetization_state_layout(existing, resolution)?;
                } else {
                    *state = Some(seed_cuda_htj2k_packetization_state(resolution)?);
                }
                let state = state
                    .as_mut()
                    .ok_or("CUDA HTJ2K packetization state initialization failed")?;
                flatten_cuda_htj2k_packet_with_state(
                    resolution,
                    descriptor.layer,
                    state,
                    &mut sink,
                )?;
            }
        }
    }

    if job.code_block_count as usize != blocks.len() {
        return Err("CUDA HTJ2K packetization code-block count mismatch");
    }

    Ok(CudaHtj2kPacketizationPlan {
        payload,
        packets,
        subbands,
        blocks,
        tag_states,
        tag_nodes,
    })
}

fn seed_cuda_htj2k_packetization_state(
    resolution: &J2kPacketizationResolution<'_>,
) -> core::result::Result<CudaHtj2kPacketizationState, &'static str> {
    let mut subbands = Vec::with_capacity(resolution.subbands.len());
    for subband in &resolution.subbands {
        let block_count = u32::try_from(subband.code_blocks.len())
            .map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?;
        if subband.num_cbs_x == 0
            || subband.num_cbs_y == 0
            || subband.num_cbs_x.saturating_mul(subband.num_cbs_y) != block_count
        {
            return Err("CUDA HTJ2K packetization subband code-block layout mismatch");
        }
        let mut inclusion_tree =
            CudaHtj2kPacketizationTagTreeState::new(subband.num_cbs_x, subband.num_cbs_y)?;
        let zero_bitplane_tree =
            CudaHtj2kPacketizationTagTreeState::new(subband.num_cbs_x, subband.num_cbs_y)?;
        for idx in 0..subband.code_blocks.len() {
            let (x, y) = cuda_htj2k_packetization_block_xy(idx, subband.num_cbs_x)?;
            inclusion_tree.set_leaf_value(x, y, CUDA_HTJ2K_PACKET_TAG_INF);
        }
        subbands.push(CudaHtj2kPacketizationSubbandState {
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
            inclusion_tree,
            zero_bitplane_tree,
            blocks: subband
                .code_blocks
                .iter()
                .map(|block| CudaHtj2kPacketizationBlockState {
                    previously_included: block.previously_included,
                    l_block: block.l_block,
                    inclusion_layer: CUDA_HTJ2K_PACKET_TAG_INF,
                    first_inclusion_zero_bitplanes: 0,
                })
                .collect(),
        });
    }
    Ok(CudaHtj2kPacketizationState { subbands })
}

fn validate_cuda_htj2k_packetization_state_layout(
    state: &CudaHtj2kPacketizationState,
    resolution: &J2kPacketizationResolution<'_>,
) -> core::result::Result<(), &'static str> {
    if state.subbands.len() != resolution.subbands.len() {
        return Err("CUDA HTJ2K packetization state layout mismatch");
    }
    for (state_subband, packet_subband) in state.subbands.iter().zip(&resolution.subbands) {
        if state_subband.num_cbs_x != packet_subband.num_cbs_x
            || state_subband.num_cbs_y != packet_subband.num_cbs_y
            || state_subband.blocks.len() != packet_subband.code_blocks.len()
        {
            return Err("CUDA HTJ2K packetization state layout mismatch");
        }
    }
    Ok(())
}

const CUDA_HTJ2K_PACKET_TAG_INF: u32 = 0x7FFF_FFFF;
const CUDA_HTJ2K_PACKET_MAX_TAG_NODES: usize = 2048;
const CUDA_HTJ2K_PACKET_MAX_TAG_LEVELS: usize = 16;

fn cuda_htj2k_packetization_block_xy(
    index: usize,
    num_cbs_x: u32,
) -> core::result::Result<(u32, u32), &'static str> {
    let index =
        u32::try_from(index).map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?;
    Ok((index % num_cbs_x, index / num_cbs_x))
}

impl CudaHtj2kPacketizationTagTreeState {
    fn new(width: u32, height: u32) -> core::result::Result<Self, &'static str> {
        if width == 0 || height == 0 {
            return Err("CUDA HTJ2K packetization subband code-block layout mismatch");
        }

        let mut widths = Vec::new();
        let mut heights = Vec::new();
        let mut offsets = Vec::new();
        let mut total_nodes = 0usize;
        let mut w = width;
        let mut h = height;
        loop {
            if widths.len() >= CUDA_HTJ2K_PACKET_MAX_TAG_LEVELS {
                return Err("CUDA HTJ2K packetization tag-tree exceeds kernel bounds");
            }
            let nodes = (w as usize)
                .checked_mul(h as usize)
                .ok_or("CUDA HTJ2K packetization tag-tree exceeds kernel bounds")?;
            let next_total = total_nodes
                .checked_add(nodes)
                .ok_or("CUDA HTJ2K packetization tag-tree exceeds kernel bounds")?;
            if next_total > CUDA_HTJ2K_PACKET_MAX_TAG_NODES {
                return Err("CUDA HTJ2K packetization tag-tree exceeds kernel bounds");
            }
            offsets.push(total_nodes);
            widths.push(w);
            heights.push(h);
            total_nodes = next_total;
            if w <= 1 && h <= 1 {
                break;
            }
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }

        Ok(Self {
            values: vec![0; total_nodes],
            current: vec![0; total_nodes],
            known: vec![0; total_nodes],
            widths,
            heights,
            offsets,
        })
    }

    fn set_leaf_value(&mut self, x: u32, y: u32, value: u32) {
        let idx = self.offsets[0] + (y * self.widths[0] + x) as usize;
        self.values[idx] = value;
    }

    #[allow(clippy::similar_names)]
    fn propagate(&mut self) {
        for level in 1..self.widths.len() {
            let prev_w = self.widths[level - 1];
            let prev_h = self.heights[level - 1];
            let curr_w = self.widths[level];
            let curr_h = self.heights[level];
            for cy in 0..curr_h {
                for cx in 0..curr_w {
                    let child_x_start = cx * 2;
                    let child_y_start = cy * 2;
                    let child_x_end = ((cx + 1) * 2).min(prev_w);
                    let child_y_end = ((cy + 1) * 2).min(prev_h);
                    let mut min_value = u32::MAX;
                    for child_y in child_y_start..child_y_end {
                        for child_x in child_x_start..child_x_end {
                            let child_idx =
                                self.offsets[level - 1] + (child_y * prev_w + child_x) as usize;
                            min_value = min_value.min(self.values[child_idx]);
                        }
                    }
                    let parent_idx = self.offsets[level] + (cy * curr_w + cx) as usize;
                    self.values[parent_idx] = min_value;
                }
            }
        }
    }

    fn encode_state_only(&mut self, x: u32, y: u32, max_value: u32) {
        let mut path = Vec::with_capacity(self.widths.len());
        let mut cx = x;
        let mut cy = y;
        for level in 0..self.widths.len() {
            path.push(self.offsets[level] + (cy * self.widths[level] + cx) as usize);
            cx /= 2;
            cy /= 2;
        }

        for node_idx in path.into_iter().rev() {
            if self.known[node_idx] == 0 {
                let target = self.values[node_idx].min(max_value);
                if self.values[node_idx] < max_value {
                    self.known[node_idx] = 1;
                }
                self.current[node_idx] = target;
            }
        }
    }

    fn append_snapshot(
        &self,
        out: &mut Vec<CudaHtj2kPacketizationPlanTagNodeState>,
    ) -> core::result::Result<u32, &'static str> {
        let start = u32::try_from(out.len())
            .map_err(|_| "CUDA HTJ2K packetization tag-state exceeds u32")?;
        out.extend(
            self.current
                .iter()
                .copied()
                .zip(self.known.iter().copied())
                .map(|(current, known)| CudaHtj2kPacketizationPlanTagNodeState { current, known }),
        );
        Ok(start)
    }

    fn node_count(&self) -> core::result::Result<u32, &'static str> {
        u32::try_from(self.current.len())
            .map_err(|_| "CUDA HTJ2K packetization tag-tree node count exceeds u32")
    }
}

fn record_cuda_htj2k_packetization_first_inclusion_layers(
    state: &mut CudaHtj2kPacketizationState,
    resolution: &J2kPacketizationResolution<'_>,
    layer: u8,
) -> core::result::Result<(), &'static str> {
    validate_cuda_htj2k_packetization_state_layout(state, resolution)?;
    for (state_subband, packet_subband) in state.subbands.iter_mut().zip(&resolution.subbands) {
        for (idx, (state_block, packet_block)) in state_subband
            .blocks
            .iter_mut()
            .zip(&packet_subband.code_blocks)
            .enumerate()
        {
            if packet_block.num_coding_passes == 0 {
                continue;
            }
            let layer = u32::from(layer);
            if layer < state_block.inclusion_layer {
                state_block.inclusion_layer = layer;
                state_block.first_inclusion_zero_bitplanes =
                    u32::from(packet_block.num_zero_bitplanes);
                let (x, y) = cuda_htj2k_packetization_block_xy(idx, state_subband.num_cbs_x)?;
                state_subband.inclusion_tree.set_leaf_value(x, y, layer);
                state_subband.zero_bitplane_tree.set_leaf_value(
                    x,
                    y,
                    state_block.first_inclusion_zero_bitplanes,
                );
            }
        }
    }
    Ok(())
}

fn finalize_cuda_htj2k_packetization_tag_trees(state: &mut CudaHtj2kPacketizationState) {
    for subband in &mut state.subbands {
        subband.inclusion_tree.propagate();
        subband.zero_bitplane_tree.propagate();
    }
}

fn append_cuda_htj2k_packetization_tag_state(
    state_subband: Option<&CudaHtj2kPacketizationSubbandState>,
    num_cbs_x: u32,
    num_cbs_y: u32,
    tag_states: &mut Vec<CudaHtj2kPacketizationPlanSubbandTagState>,
    tag_nodes: &mut Vec<CudaHtj2kPacketizationPlanTagNodeState>,
) -> core::result::Result<(), &'static str> {
    let (inclusion_node_start, zero_bitplane_node_start, node_count) =
        if let Some(state_subband) = state_subband {
            let inclusion_start = state_subband.inclusion_tree.append_snapshot(tag_nodes)?;
            let zero_bitplane_start = state_subband
                .zero_bitplane_tree
                .append_snapshot(tag_nodes)?;
            (
                inclusion_start,
                zero_bitplane_start,
                state_subband.inclusion_tree.node_count()?,
            )
        } else {
            let zero_tree = CudaHtj2kPacketizationTagTreeState::new(num_cbs_x, num_cbs_y)?;
            let inclusion_start = zero_tree.append_snapshot(tag_nodes)?;
            let zero_bitplane_start = zero_tree.append_snapshot(tag_nodes)?;
            (
                inclusion_start,
                zero_bitplane_start,
                zero_tree.node_count()?,
            )
        };
    tag_states.push(CudaHtj2kPacketizationPlanSubbandTagState {
        inclusion_node_start,
        zero_bitplane_node_start,
        node_count,
    });
    Ok(())
}

fn update_cuda_htj2k_packetization_state_after_block(
    state: &mut CudaHtj2kPacketizationState,
    subband_index: usize,
    block_index: usize,
    layer: u8,
    code_block: &J2kPacketizationCodeBlock<'_>,
    l_block: u32,
) -> core::result::Result<(), &'static str> {
    let state_subband = state
        .subbands
        .get_mut(subband_index)
        .ok_or("CUDA HTJ2K packetization state layout mismatch")?;
    let (x, y) = cuda_htj2k_packetization_block_xy(block_index, state_subband.num_cbs_x)?;
    let previously_included = state_subband
        .blocks
        .get(block_index)
        .ok_or("CUDA HTJ2K packetization state layout mismatch")?
        .previously_included;

    if !previously_included {
        state_subband
            .inclusion_tree
            .encode_state_only(x, y, u32::from(layer) + 1);
        if code_block.num_coding_passes == 0 {
            return Ok(());
        }
        state_subband.zero_bitplane_tree.encode_state_only(
            x,
            y,
            u32::from(code_block.num_zero_bitplanes) + 1,
        );
    }

    if code_block.num_coding_passes > 0 {
        let state_block = state_subband
            .blocks
            .get_mut(block_index)
            .ok_or("CUDA HTJ2K packetization state layout mismatch")?;
        let (cleanup_length, refinement_length) = cuda_ht_segment_lengths(code_block)?;
        state_block.l_block = updated_ht_l_block(
            l_block,
            code_block.num_coding_passes,
            cleanup_length,
            refinement_length,
        )?;
        state_block.previously_included = true;
    }
    Ok(())
}

fn flatten_cuda_htj2k_packet(
    resolution: &J2kPacketizationResolution<'_>,
    sink: &mut CudaHtj2kPacketizationPlanSink<'_>,
) -> core::result::Result<(), &'static str> {
    flatten_cuda_htj2k_packet_inner(resolution, 0, None, sink)
}

fn flatten_cuda_htj2k_packet_with_state(
    resolution: &J2kPacketizationResolution<'_>,
    layer: u8,
    state: &mut CudaHtj2kPacketizationState,
    sink: &mut CudaHtj2kPacketizationPlanSink<'_>,
) -> core::result::Result<(), &'static str> {
    flatten_cuda_htj2k_packet_inner(resolution, layer, Some(state), sink)
}

fn flatten_cuda_htj2k_packet_inner(
    resolution: &J2kPacketizationResolution<'_>,
    layer: u8,
    mut state: Option<&mut CudaHtj2kPacketizationState>,
    sink: &mut CudaHtj2kPacketizationPlanSink<'_>,
) -> core::result::Result<(), &'static str> {
    let block_start = u32::try_from(sink.blocks.len())
        .map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?;
    let subband_start = u32::try_from(sink.subbands.len())
        .map_err(|_| "CUDA HTJ2K packetization subband count exceeds u32")?;
    let mut body_len = 0usize;
    let mut block_count = 0usize;
    let packet_has_data = resolution.subbands.iter().any(|subband| {
        subband
            .code_blocks
            .iter()
            .any(|block| block.num_coding_passes > 0)
    });

    for (subband_index, subband) in resolution.subbands.iter().enumerate() {
        let subband_code_blocks = u32::try_from(subband.code_blocks.len())
            .map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?;
        if subband.num_cbs_x == 0
            || subband.num_cbs_y == 0
            || subband.num_cbs_x.saturating_mul(subband.num_cbs_y) != subband_code_blocks
        {
            return Err("CUDA HTJ2K packetization subband code-block layout mismatch");
        }

        let subband_block_start = u32::try_from(sink.blocks.len())
            .map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?;
        let state_subband = state
            .as_deref()
            .and_then(|state| state.subbands.get(subband_index));
        append_cuda_htj2k_packetization_tag_state(
            state_subband,
            subband.num_cbs_x,
            subband.num_cbs_y,
            sink.tag_states,
            sink.tag_nodes,
        )?;
        for (block_index, code_block) in subband.code_blocks.iter().enumerate() {
            if code_block.block_coding_mode != J2kPacketizationBlockCodingMode::HighThroughput {
                return Err("CUDA packetization only supports HTJ2K block-coded packets");
            }
            if code_block.num_coding_passes > 164 {
                return Err("CUDA HTJ2K packetization coding pass count exceeds JPEG 2000 bounds");
            }
            let (previously_included, l_block, inclusion_layer, zero_bitplanes) =
                if let Some(state) = state.as_deref() {
                    let state_block = state
                        .subbands
                        .get(subband_index)
                        .and_then(|state_subband| state_subband.blocks.get(block_index))
                        .ok_or("CUDA HTJ2K packetization state layout mismatch")?;
                    (
                        state_block.previously_included,
                        state_block.l_block,
                        state_block.inclusion_layer,
                        state_block.first_inclusion_zero_bitplanes,
                    )
                } else {
                    (
                        code_block.previously_included,
                        code_block.l_block,
                        if code_block.num_coding_passes > 0 {
                            0
                        } else {
                            CUDA_HTJ2K_PACKET_TAG_INF
                        },
                        u32::from(code_block.num_zero_bitplanes),
                    )
                };
            if code_block.num_coding_passes > 0
                && !previously_included
                && inclusion_layer != u32::from(layer)
            {
                return Err(
                    "CUDA HTJ2K packetization descriptor order does not match first inclusion layer",
                );
            }
            if state.is_none() && previously_included {
                return Err("CUDA HTJ2K packetization requires first-inclusion packets");
            }
            if code_block.num_coding_passes == 0 && !code_block.data.is_empty() {
                return Err("CUDA HTJ2K packetization empty contributions must not carry payload");
            }
            if zero_bitplanes > 31 || l_block > 31 {
                return Err("CUDA HTJ2K packetization header fields exceed kernel bounds");
            }

            let data_offset = u32::try_from(sink.payload.len())
                .map_err(|_| "CUDA HTJ2K packetization payload exceeds u32")?;
            let data_len = if code_block.num_coding_passes == 0 {
                0
            } else {
                u32::try_from(code_block.data.len())
                    .map_err(|_| "CUDA HTJ2K packetization code-block payload exceeds u32")?
            };
            let (cleanup_length, refinement_length) = cuda_ht_segment_lengths(code_block)?;
            if code_block.num_coding_passes > 0 {
                sink.payload.extend_from_slice(code_block.data);
                body_len = body_len
                    .checked_add(code_block.data.len())
                    .ok_or("CUDA HTJ2K packetization body length overflow")?;
            }
            sink.blocks.push(CudaHtj2kPacketizationPlanBlock {
                data_offset,
                data_len,
                cleanup_length,
                refinement_length,
                num_coding_passes: u32::from(code_block.num_coding_passes),
                num_zero_bitplanes: zero_bitplanes,
                l_block,
                previously_included: u32::from(previously_included),
                inclusion_layer,
            });
            if packet_has_data {
                if let Some(state) = state.as_deref_mut() {
                    update_cuda_htj2k_packetization_state_after_block(
                        state,
                        subband_index,
                        block_index,
                        layer,
                        code_block,
                        l_block,
                    )?;
                }
            }
            block_count = block_count
                .checked_add(1)
                .ok_or("CUDA HTJ2K packetization block count overflow")?;
        }
        sink.subbands.push(CudaHtj2kPacketizationPlanSubband {
            block_start: subband_block_start,
            block_count: subband_code_blocks,
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
        });
    }

    let header_capacity = 256usize
        .checked_add(
            block_count
                .checked_mul(64)
                .ok_or("CUDA HTJ2K packetization capacity overflow")?,
        )
        .ok_or("CUDA HTJ2K packetization capacity overflow")?;
    let output_capacity = body_len
        .checked_add(header_capacity)
        .ok_or("CUDA HTJ2K packetization capacity overflow")?;
    sink.packets.push(CudaHtj2kPacketizationPlanPacket {
        block_start,
        block_count: u32::try_from(block_count)
            .map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?,
        subband_start,
        subband_count: u32::try_from(resolution.subbands.len())
            .map_err(|_| "CUDA HTJ2K packetization subband count exceeds u32")?,
        output_capacity: u32::try_from(output_capacity)
            .map_err(|_| "CUDA HTJ2K packetization packet capacity exceeds u32")?,
        layer: u32::from(layer),
    });
    Ok(())
}

fn updated_ht_l_block(
    mut l_block: u32,
    num_coding_passes: u8,
    cleanup_length: u32,
    refinement_length: u32,
) -> core::result::Result<u32, &'static str> {
    let mut num_bits = packet_math::bits_for_ht_cleanup_length(l_block, num_coding_passes);
    let refinement_extra_bits = u32::from(num_coding_passes > 2);
    while !packet_math::value_fits_in_bits(cleanup_length, num_bits)
        || (num_coding_passes > 1
            && !packet_math::value_fits_in_bits(refinement_length, l_block + refinement_extra_bits))
    {
        l_block = l_block
            .checked_add(1)
            .ok_or("CUDA HTJ2K packetization L-block overflow")?;
        num_bits = num_bits
            .checked_add(1)
            .ok_or("CUDA HTJ2K packetization L-block overflow")?;
    }
    Ok(l_block)
}

pub(super) fn cuda_ht_segment_lengths(
    code_block: &J2kPacketizationCodeBlock<'_>,
) -> core::result::Result<(u32, u32), &'static str> {
    packet_math::ht_segment_lengths(
        code_block.num_coding_passes,
        code_block.data.len(),
        code_block.ht_cleanup_length,
        code_block.ht_refinement_length,
    )
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_packetization_packets(
    plan: &CudaHtj2kPacketizationPlan,
) -> Vec<CudaHtj2kPacketizationPacket> {
    plan.packets
        .iter()
        .map(|packet| CudaHtj2kPacketizationPacket {
            block_start: packet.block_start,
            block_count: packet.block_count,
            subband_start: packet.subband_start,
            subband_count: packet.subband_count,
            output_capacity: packet.output_capacity,
            layer: packet.layer,
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_packetization_subbands(
    plan: &CudaHtj2kPacketizationPlan,
) -> Vec<CudaHtj2kPacketizationSubband> {
    plan.subbands
        .iter()
        .map(|subband| CudaHtj2kPacketizationSubband {
            block_start: subband.block_start,
            block_count: subband.block_count,
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_packetization_blocks(
    plan: &CudaHtj2kPacketizationPlan,
) -> Vec<CudaHtj2kPacketizationBlock> {
    plan.blocks
        .iter()
        .map(|block| CudaHtj2kPacketizationBlock {
            data_offset: block.data_offset,
            data_len: block.data_len,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            num_coding_passes: block.num_coding_passes,
            num_zero_bitplanes: block.num_zero_bitplanes,
            l_block: block.l_block,
            previously_included: block.previously_included,
            inclusion_layer: block.inclusion_layer,
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_packetization_tag_states(
    plan: &CudaHtj2kPacketizationPlan,
) -> Vec<CudaHtj2kPacketizationSubbandTagState> {
    plan.tag_states
        .iter()
        .map(|state| CudaHtj2kPacketizationSubbandTagState {
            inclusion_node_start: state.inclusion_node_start,
            zero_bitplane_node_start: state.zero_bitplane_node_start,
            node_count: state.node_count,
            reserved0: 0,
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_packetization_tag_nodes(
    plan: &CudaHtj2kPacketizationPlan,
) -> Vec<CudaHtj2kPacketizationTagNodeState> {
    plan.tag_nodes
        .iter()
        .map(|node| CudaHtj2kPacketizationTagNodeState {
            current: node.current,
            known: node.known,
        })
        .collect()
}
