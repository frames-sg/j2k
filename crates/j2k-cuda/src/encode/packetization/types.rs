// SPDX-License-Identifier: MIT OR Apache-2.0

pub(in crate::encode) use super::error::CudaHtj2kPacketizationPlanError;
pub(super) use super::error::{
    packetization_plan_allocation_error, PacketizationPlanResult,
    CUDA_PACKETIZATION_TAG_TREE_ALLOCATION,
};

/// Move-only flattened packetization plan owning payload and metadata buffers.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::encode) struct CudaHtj2kPacketizationPlan {
    pub(in crate::encode) payload: Vec<u8>,
    pub(in crate::encode) packets: Vec<CudaHtj2kPacketizationPlanPacket>,
    pub(in crate::encode) subbands: Vec<CudaHtj2kPacketizationPlanSubband>,
    pub(in crate::encode) blocks: Vec<CudaHtj2kPacketizationPlanBlock>,
    pub(in crate::encode) tag_states: Vec<CudaHtj2kPacketizationPlanSubbandTagState>,
    pub(in crate::encode) tag_nodes: Vec<CudaHtj2kPacketizationPlanTagNodeState>,
}

pub(super) struct CudaHtj2kPacketizationPlanSink<'a> {
    pub(super) host_budget: &'a mut crate::allocation::HostPhaseBudget,
    pub(super) payload: &'a mut Vec<u8>,
    pub(super) packets: &'a mut Vec<CudaHtj2kPacketizationPlanPacket>,
    pub(super) subbands: &'a mut Vec<CudaHtj2kPacketizationPlanSubband>,
    pub(super) blocks: &'a mut Vec<CudaHtj2kPacketizationPlanBlock>,
    pub(super) tag_states: &'a mut Vec<CudaHtj2kPacketizationPlanSubbandTagState>,
    pub(super) tag_nodes: &'a mut Vec<CudaHtj2kPacketizationPlanTagNodeState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::encode) struct CudaHtj2kPacketizationPlanPacket {
    pub(in crate::encode) block_start: u32,
    pub(in crate::encode) block_count: u32,
    pub(in crate::encode) subband_start: u32,
    pub(in crate::encode) subband_count: u32,
    pub(in crate::encode) output_capacity: u32,
    pub(in crate::encode) layer: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::encode) struct CudaHtj2kPacketizationPlanSubband {
    pub(in crate::encode) block_start: u32,
    pub(in crate::encode) block_count: u32,
    pub(in crate::encode) num_cbs_x: u32,
    pub(in crate::encode) num_cbs_y: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::encode) struct CudaHtj2kPacketizationPlanBlock {
    pub(in crate::encode) data_offset: u32,
    pub(in crate::encode) data_len: u32,
    pub(in crate::encode) cleanup_length: u32,
    pub(in crate::encode) refinement_length: u32,
    pub(in crate::encode) num_coding_passes: u32,
    pub(in crate::encode) num_zero_bitplanes: u32,
    pub(in crate::encode) l_block: u32,
    pub(in crate::encode) previously_included: u32,
    pub(in crate::encode) inclusion_layer: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::encode) struct CudaHtj2kPacketizationPlanSubbandTagState {
    pub(in crate::encode) inclusion_node_start: u32,
    pub(in crate::encode) zero_bitplane_node_start: u32,
    pub(in crate::encode) node_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::encode) struct CudaHtj2kPacketizationPlanTagNodeState {
    pub(in crate::encode) current: u32,
    pub(in crate::encode) known: u32,
}
