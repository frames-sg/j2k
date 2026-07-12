// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_cuda_runtime::{
    CudaHtj2kPacketizationBlock, CudaHtj2kPacketizationPacket, CudaHtj2kPacketizationSubband,
    CudaHtj2kPacketizationSubbandTagState, CudaHtj2kPacketizationTagNodeState,
};

use crate::allocation::HostPhaseBudget;
use crate::encode::stage_error::{adapter_error, CudaStageResult};

use super::types::CudaHtj2kPacketizationPlan;

fn packetization_allocation_error(error: crate::Error) -> j2k::J2kEncodeStageError {
    adapter_error("allocate CUDA packetization runtime descriptors", error)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_packetization_packets(
    plan: &CudaHtj2kPacketizationPlan,
    host_budget: &mut HostPhaseBudget,
) -> CudaStageResult<Vec<CudaHtj2kPacketizationPacket>> {
    let mut packets = host_budget
        .try_vec_with_capacity(plan.packets.len())
        .map_err(packetization_allocation_error)?;
    packets.extend(
        plan.packets
            .iter()
            .map(|packet| CudaHtj2kPacketizationPacket {
                block_start: packet.block_start,
                block_count: packet.block_count,
                subband_start: packet.subband_start,
                subband_count: packet.subband_count,
                output_capacity: packet.output_capacity,
                layer: packet.layer,
            }),
    );
    Ok(packets)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_packetization_subbands(
    plan: &CudaHtj2kPacketizationPlan,
    host_budget: &mut HostPhaseBudget,
) -> CudaStageResult<Vec<CudaHtj2kPacketizationSubband>> {
    let mut subbands = host_budget
        .try_vec_with_capacity(plan.subbands.len())
        .map_err(packetization_allocation_error)?;
    subbands.extend(
        plan.subbands
            .iter()
            .map(|subband| CudaHtj2kPacketizationSubband {
                block_start: subband.block_start,
                block_count: subband.block_count,
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
            }),
    );
    Ok(subbands)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_packetization_blocks(
    plan: &CudaHtj2kPacketizationPlan,
    host_budget: &mut HostPhaseBudget,
) -> CudaStageResult<Vec<CudaHtj2kPacketizationBlock>> {
    let mut blocks = host_budget
        .try_vec_with_capacity(plan.blocks.len())
        .map_err(packetization_allocation_error)?;
    blocks.extend(plan.blocks.iter().map(|block| CudaHtj2kPacketizationBlock {
        data_offset: block.data_offset,
        data_len: block.data_len,
        cleanup_length: block.cleanup_length,
        refinement_length: block.refinement_length,
        num_coding_passes: block.num_coding_passes,
        num_zero_bitplanes: block.num_zero_bitplanes,
        l_block: block.l_block,
        previously_included: block.previously_included,
        inclusion_layer: block.inclusion_layer,
    }));
    Ok(blocks)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_packetization_tag_states(
    plan: &CudaHtj2kPacketizationPlan,
    host_budget: &mut HostPhaseBudget,
) -> CudaStageResult<Vec<CudaHtj2kPacketizationSubbandTagState>> {
    let mut states = host_budget
        .try_vec_with_capacity(plan.tag_states.len())
        .map_err(packetization_allocation_error)?;
    states.extend(
        plan.tag_states
            .iter()
            .map(|state| CudaHtj2kPacketizationSubbandTagState {
                inclusion_node_start: state.inclusion_node_start,
                zero_bitplane_node_start: state.zero_bitplane_node_start,
                node_count: state.node_count,
                reserved0: 0,
            }),
    );
    Ok(states)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_packetization_tag_nodes(
    plan: &CudaHtj2kPacketizationPlan,
    host_budget: &mut HostPhaseBudget,
) -> CudaStageResult<Vec<CudaHtj2kPacketizationTagNodeState>> {
    let mut nodes = host_budget
        .try_vec_with_capacity(plan.tag_nodes.len())
        .map_err(packetization_allocation_error)?;
    nodes.extend(
        plan.tag_nodes
            .iter()
            .map(|node| CudaHtj2kPacketizationTagNodeState {
                current: node.current,
                known: node.known,
            }),
    );
    Ok(nodes)
}
