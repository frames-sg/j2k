// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{J2kPacketizationCodeBlock, J2kPacketizationResolution};
use j2k_native::packet_math;

use crate::allocation::HostPhaseBudget;

use super::tag_tree::{cuda_htj2k_packetization_block_xy, CudaHtj2kPacketizationTagTreeState};
use super::types::{
    packetization_plan_allocation_error, CudaHtj2kPacketizationPlanError,
    CudaHtj2kPacketizationPlanSubbandTagState, CudaHtj2kPacketizationPlanTagNodeState,
    PacketizationPlanResult,
};

mod count;
#[cfg(test)]
pub(super) use self::count::checked_cuda_packetization_state_count;
pub(super) use self::count::cuda_packetization_state_count;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaHtj2kPacketizationBlockState {
    pub(super) previously_included: bool,
    pub(super) l_block: u32,
    pub(super) inclusion_layer: u32,
    pub(super) first_inclusion_zero_bitplanes: u32,
}

/// Move-only subband state owning tag trees and code-block state storage.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct CudaHtj2kPacketizationSubbandState {
    pub(super) num_cbs_x: u32,
    pub(super) num_cbs_y: u32,
    pub(super) inclusion_tree: CudaHtj2kPacketizationTagTreeState,
    pub(super) zero_bitplane_tree: CudaHtj2kPacketizationTagTreeState,
    pub(super) blocks: Vec<CudaHtj2kPacketizationBlockState>,
}

/// Move-only packetization state owning all subband state graphs.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct CudaHtj2kPacketizationState {
    pub(super) subbands: Vec<CudaHtj2kPacketizationSubbandState>,
}

pub(super) fn seed_cuda_htj2k_packetization_state(
    resolution: &J2kPacketizationResolution<'_>,
    host_budget: &mut HostPhaseBudget,
) -> PacketizationPlanResult<CudaHtj2kPacketizationState> {
    let mut subbands = host_budget
        .try_vec_with_capacity(resolution.subbands.len())
        .map_err(packetization_plan_allocation_error)?;
    for subband in &resolution.subbands {
        let block_count = u32::try_from(subband.code_blocks.len()).map_err(|_| {
            CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization block count exceeds u32",
            )
        })?;
        if subband.num_cbs_x == 0
            || subband.num_cbs_y == 0
            || subband.num_cbs_x.saturating_mul(subband.num_cbs_y) != block_count
        {
            return Err(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization subband code-block layout mismatch",
            ));
        }
        let mut inclusion_tree = CudaHtj2kPacketizationTagTreeState::new(
            subband.num_cbs_x,
            subband.num_cbs_y,
            host_budget,
        )?;
        let zero_bitplane_tree = CudaHtj2kPacketizationTagTreeState::new(
            subband.num_cbs_x,
            subband.num_cbs_y,
            host_budget,
        )?;
        for idx in 0..subband.code_blocks.len() {
            let (x, y) = cuda_htj2k_packetization_block_xy(idx, subband.num_cbs_x)?;
            inclusion_tree.set_leaf_value(x, y, CUDA_HTJ2K_PACKET_TAG_INF);
        }
        let mut blocks = host_budget
            .try_vec_with_capacity(subband.code_blocks.len())
            .map_err(packetization_plan_allocation_error)?;
        blocks.extend(
            subband
                .code_blocks
                .iter()
                .map(|block| CudaHtj2kPacketizationBlockState {
                    previously_included: block.previously_included,
                    l_block: block.l_block,
                    inclusion_layer: CUDA_HTJ2K_PACKET_TAG_INF,
                    first_inclusion_zero_bitplanes: 0,
                }),
        );
        host_budget
            .try_vec_push(
                &mut subbands,
                CudaHtj2kPacketizationSubbandState {
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                    inclusion_tree,
                    zero_bitplane_tree,
                    blocks,
                },
            )
            .map_err(packetization_plan_allocation_error)?;
    }
    Ok(CudaHtj2kPacketizationState { subbands })
}

pub(super) fn validate_cuda_htj2k_packetization_state_layout(
    state: &CudaHtj2kPacketizationState,
    resolution: &J2kPacketizationResolution<'_>,
) -> PacketizationPlanResult<()> {
    if state.subbands.len() != resolution.subbands.len() {
        return Err(CudaHtj2kPacketizationPlanError::Invalid(
            "CUDA HTJ2K packetization state layout mismatch",
        ));
    }
    for (state_subband, packet_subband) in state.subbands.iter().zip(&resolution.subbands) {
        if state_subband.num_cbs_x != packet_subband.num_cbs_x
            || state_subband.num_cbs_y != packet_subband.num_cbs_y
            || state_subband.blocks.len() != packet_subband.code_blocks.len()
        {
            return Err(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization state layout mismatch",
            ));
        }
    }
    Ok(())
}

pub(super) const CUDA_HTJ2K_PACKET_TAG_INF: u32 = 0x7FFF_FFFF;

pub(super) fn record_cuda_htj2k_packetization_first_inclusion_layers(
    state: &mut CudaHtj2kPacketizationState,
    resolution: &J2kPacketizationResolution<'_>,
    layer: u8,
) -> PacketizationPlanResult<()> {
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

pub(super) fn finalize_cuda_htj2k_packetization_tag_trees(state: &mut CudaHtj2kPacketizationState) {
    for subband in &mut state.subbands {
        subband.inclusion_tree.propagate();
        subband.zero_bitplane_tree.propagate();
    }
}

pub(super) fn append_cuda_htj2k_packetization_tag_state(
    state_subband: Option<&CudaHtj2kPacketizationSubbandState>,
    num_cbs_x: u32,
    num_cbs_y: u32,
    tag_states: &mut Vec<CudaHtj2kPacketizationPlanSubbandTagState>,
    tag_nodes: &mut Vec<CudaHtj2kPacketizationPlanTagNodeState>,
    host_budget: &mut HostPhaseBudget,
) -> PacketizationPlanResult<()> {
    let (inclusion_node_start, zero_bitplane_node_start, node_count) = if let Some(state_subband) =
        state_subband
    {
        let inclusion_start = state_subband
            .inclusion_tree
            .append_snapshot(tag_nodes, host_budget)?;
        let zero_bitplane_start = state_subband
            .zero_bitplane_tree
            .append_snapshot(tag_nodes, host_budget)?;
        (
            inclusion_start,
            zero_bitplane_start,
            state_subband.inclusion_tree.node_count()?,
        )
    } else {
        let previous_tag_node_capacity = tag_nodes.capacity();
        let mut temporary_budget = HostPhaseBudget::with_live_bytes(
            "j2k CUDA packetization temporary zero tag tree",
            host_budget.live_bytes(),
        )
        .map_err(packetization_plan_allocation_error)?;
        let zero_tree =
            CudaHtj2kPacketizationTagTreeState::new(num_cbs_x, num_cbs_y, &mut temporary_budget)?;
        let inclusion_start = zero_tree.append_snapshot(tag_nodes, &mut temporary_budget)?;
        let zero_bitplane_start = zero_tree.append_snapshot(tag_nodes, &mut temporary_budget)?;
        host_budget
            .account_capacity::<CudaHtj2kPacketizationPlanTagNodeState>(
                tag_nodes
                    .capacity()
                    .saturating_sub(previous_tag_node_capacity),
            )
            .map_err(packetization_plan_allocation_error)?;
        (
            inclusion_start,
            zero_bitplane_start,
            zero_tree.node_count()?,
        )
    };
    host_budget
        .try_vec_push(
            tag_states,
            CudaHtj2kPacketizationPlanSubbandTagState {
                inclusion_node_start,
                zero_bitplane_node_start,
                node_count,
            },
        )
        .map_err(packetization_plan_allocation_error)?;
    Ok(())
}

pub(super) fn update_cuda_htj2k_packetization_state_after_block(
    state: &mut CudaHtj2kPacketizationState,
    subband_index: usize,
    block_index: usize,
    layer: u8,
    code_block: &J2kPacketizationCodeBlock<'_>,
    l_block: u32,
) -> PacketizationPlanResult<()> {
    let state_subband =
        state
            .subbands
            .get_mut(subband_index)
            .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization state layout mismatch",
            ))?;
    let (x, y) = cuda_htj2k_packetization_block_xy(block_index, state_subband.num_cbs_x)?;
    let previously_included = state_subband
        .blocks
        .get(block_index)
        .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
            "CUDA HTJ2K packetization state layout mismatch",
        ))?
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
        let state_block = state_subband.blocks.get_mut(block_index).ok_or(
            CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization state layout mismatch",
            ),
        )?;
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

fn updated_ht_l_block(
    mut l_block: u32,
    num_coding_passes: u8,
    cleanup_length: u32,
    refinement_length: u32,
) -> PacketizationPlanResult<u32> {
    let mut num_bits = packet_math::bits_for_ht_cleanup_length(l_block, num_coding_passes);
    let refinement_extra_bits = u32::from(num_coding_passes > 2);
    while !packet_math::value_fits_in_bits(cleanup_length, num_bits)
        || (num_coding_passes > 1
            && !packet_math::value_fits_in_bits(refinement_length, l_block + refinement_extra_bits))
    {
        l_block =
            l_block
                .checked_add(1)
                .ok_or(CudaHtj2kPacketizationPlanError::ArithmeticOverflow(
                    "CUDA HTJ2K packetization L-block overflow",
                ))?;
        num_bits =
            num_bits
                .checked_add(1)
                .ok_or(CudaHtj2kPacketizationPlanError::ArithmeticOverflow(
                    "CUDA HTJ2K packetization L-block overflow",
                ))?;
    }
    Ok(l_block)
}

pub(in crate::encode) fn cuda_ht_segment_lengths(
    code_block: &J2kPacketizationCodeBlock<'_>,
) -> PacketizationPlanResult<(u32, u32)> {
    packet_math::ht_segment_lengths(
        code_block.num_coding_passes,
        code_block.data.len(),
        code_block.ht_cleanup_length,
        code_block.ht_refinement_length,
    )
    .map_err(cuda_ht_segment_length_error)
}

pub(super) fn cuda_ht_segment_length_error(
    error: packet_math::HtSegmentLengthError,
) -> CudaHtj2kPacketizationPlanError {
    match error {
        packet_math::HtSegmentLengthError::ContributionLengthExceedsU32 { .. }
        | packet_math::HtSegmentLengthError::MultiPassLengthOverflow { .. } => {
            CudaHtj2kPacketizationPlanError::ArithmeticOverflow(error.reason())
        }
        packet_math::HtSegmentLengthError::EmptyContributionHasSegments
        | packet_math::HtSegmentLengthError::RefinementOnlyLengthMismatch { .. }
        | packet_math::HtSegmentLengthError::RefinementLengthOutOfRange { .. }
        | packet_math::HtSegmentLengthError::SinglePassHasRefinement { .. }
        | packet_math::HtSegmentLengthError::SinglePassLengthMismatch { .. }
        | packet_math::HtSegmentLengthError::MultiPassRequiresSegments { .. }
        | packet_math::HtSegmentLengthError::MultiPassLengthMismatch { .. }
        | packet_math::HtSegmentLengthError::CleanupLengthOutOfRange { .. } => {
            CudaHtj2kPacketizationPlanError::Invalid(error.reason())
        }
        _ => CudaHtj2kPacketizationPlanError::Invalid(error.reason()),
    }
}
