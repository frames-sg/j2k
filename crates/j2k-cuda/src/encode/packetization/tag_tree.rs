// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::allocation::HostPhaseBudget;

mod allocation;
use self::allocation::{
    checked_tag_tree_retained_bytes, try_tag_tree_vec_filled, try_tag_tree_vec_with_capacity,
};

use super::types::{
    packetization_plan_allocation_error, CudaHtj2kPacketizationPlanError,
    CudaHtj2kPacketizationPlanTagNodeState, PacketizationPlanResult,
};

/// Move-only packet tag-tree state.
///
/// Cloning would infallibly duplicate six allocator-backed state arrays.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct CudaHtj2kPacketizationTagTreeState {
    values: Vec<u32>,
    current: Vec<u32>,
    known: Vec<u32>,
    widths: Vec<u32>,
    heights: Vec<u32>,
    offsets: Vec<usize>,
}

const CUDA_HTJ2K_PACKET_MAX_TAG_NODES: usize = 2048;
const CUDA_HTJ2K_PACKET_MAX_TAG_LEVELS: usize = 16;

pub(super) fn cuda_htj2k_packetization_block_xy(
    index: usize,
    num_cbs_x: u32,
) -> PacketizationPlanResult<(u32, u32)> {
    let index = u32::try_from(index).map_err(|_| {
        CudaHtj2kPacketizationPlanError::Invalid("CUDA HTJ2K packetization block count exceeds u32")
    })?;
    Ok((index % num_cbs_x, index / num_cbs_x))
}

impl CudaHtj2kPacketizationTagTreeState {
    pub(super) fn new(
        width: u32,
        height: u32,
        host_budget: &mut HostPhaseBudget,
    ) -> PacketizationPlanResult<Self> {
        if width == 0 || height == 0 {
            return Err(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization subband code-block layout mismatch",
            ));
        }

        let mut widths =
            try_tag_tree_vec_with_capacity(host_budget, CUDA_HTJ2K_PACKET_MAX_TAG_LEVELS)?;
        let mut heights =
            try_tag_tree_vec_with_capacity(host_budget, CUDA_HTJ2K_PACKET_MAX_TAG_LEVELS)?;
        let mut offsets =
            try_tag_tree_vec_with_capacity(host_budget, CUDA_HTJ2K_PACKET_MAX_TAG_LEVELS)?;
        let mut total_nodes = 0usize;
        let mut w = width;
        let mut h = height;
        loop {
            if widths.len() >= CUDA_HTJ2K_PACKET_MAX_TAG_LEVELS {
                return Err(CudaHtj2kPacketizationPlanError::Invalid(
                    "CUDA HTJ2K packetization tag-tree exceeds kernel bounds",
                ));
            }
            let nodes = (w as usize).checked_mul(h as usize).ok_or(
                CudaHtj2kPacketizationPlanError::Invalid(
                    "CUDA HTJ2K packetization tag-tree exceeds kernel bounds",
                ),
            )?;
            let next_total =
                total_nodes
                    .checked_add(nodes)
                    .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
                        "CUDA HTJ2K packetization tag-tree exceeds kernel bounds",
                    ))?;
            if next_total > CUDA_HTJ2K_PACKET_MAX_TAG_NODES {
                return Err(CudaHtj2kPacketizationPlanError::Invalid(
                    "CUDA HTJ2K packetization tag-tree exceeds kernel bounds",
                ));
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

        let values = try_tag_tree_vec_filled(host_budget, total_nodes, 0)?;
        let current = try_tag_tree_vec_filled(host_budget, total_nodes, 0)?;
        let known = try_tag_tree_vec_filled(host_budget, total_nodes, 0)?;
        checked_tag_tree_retained_bytes(
            [widths.capacity(), heights.capacity()],
            offsets.capacity(),
            [values.capacity(), current.capacity(), known.capacity()],
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;

        Ok(Self {
            values,
            current,
            known,
            widths,
            heights,
            offsets,
        })
    }

    pub(super) fn set_leaf_value(&mut self, x: u32, y: u32, value: u32) {
        let idx = self.offsets[0] + (y * self.widths[0] + x) as usize;
        self.values[idx] = value;
    }

    #[expect(
        clippy::similar_names,
        reason = "prev/curr dimensions identify adjacent tag-tree levels"
    )]
    pub(super) fn propagate(&mut self) {
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

    pub(super) fn encode_state_only(&mut self, x: u32, y: u32, max_value: u32) {
        let mut path = [0usize; CUDA_HTJ2K_PACKET_MAX_TAG_LEVELS];
        let mut cx = x;
        let mut cy = y;
        for ((path_entry, &offset), &width) in path.iter_mut().zip(&self.offsets).zip(&self.widths)
        {
            *path_entry = offset + (cy * width + cx) as usize;
            cx /= 2;
            cy /= 2;
        }

        for &node_idx in path[..self.widths.len()].iter().rev() {
            if self.known[node_idx] == 0 {
                let target = self.values[node_idx].min(max_value);
                if self.values[node_idx] < max_value {
                    self.known[node_idx] = 1;
                }
                self.current[node_idx] = target;
            }
        }
    }

    pub(super) fn append_snapshot(
        &self,
        out: &mut Vec<CudaHtj2kPacketizationPlanTagNodeState>,
        host_budget: &mut HostPhaseBudget,
    ) -> PacketizationPlanResult<u32> {
        let start = u32::try_from(out.len()).map_err(|_| {
            CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization tag-state exceeds u32",
            )
        })?;
        host_budget
            .try_vec_reserve(out, self.current.len())
            .map_err(packetization_plan_allocation_error)?;
        out.extend(
            self.current
                .iter()
                .copied()
                .zip(self.known.iter().copied())
                .map(|(current, known)| CudaHtj2kPacketizationPlanTagNodeState { current, known }),
        );
        Ok(start)
    }

    pub(super) fn node_count(&self) -> PacketizationPlanResult<u32> {
        u32::try_from(self.current.len()).map_err(|_| {
            CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization tag-tree node count exceeds u32",
            )
        })
    }
}
