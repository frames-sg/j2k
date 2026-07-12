// SPDX-License-Identifier: MIT OR Apache-2.0
use j2k::{
    J2kEncodeStageError, J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock,
    J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor, J2kPacketizationResolution,
    J2kPacketizationSubband, J2kResidentHtj2kTileEncodeJob,
};
use j2k_cuda_runtime::CudaContext;

use crate::allocation::HostPhaseBudget;
use crate::encode::stage_error::{arithmetic_overflow, runtime_error, CudaStageResult};

use super::super::packetization::{
    cuda_packetization_blocks, cuda_packetization_packets, cuda_packetization_subbands,
    cuda_packetization_tag_nodes, cuda_packetization_tag_states,
    flatten_cuda_htj2k_packetization_job_classified_with_live_host_bytes,
};
use super::host_budget::account_encoded_resolution_owners;
use super::htj2k_allocation_error;
use super::types::CudaEncodedHtj2kResolution;

pub(super) fn cuda_packetize_tile_body(
    context: &CudaContext,
    job: J2kResidentHtj2kTileEncodeJob<'_>,
    resolution_packets: &[CudaEncodedHtj2kResolution],
    resolution_packet_capacity: usize,
    code_block_count: usize,
) -> CudaStageResult<(Vec<u8>, usize, u128)> {
    let mut host_budget = HostPhaseBudget::new("j2k CUDA HTJ2K tile packetization");
    account_encoded_resolution_owners(
        &mut host_budget,
        resolution_packets,
        resolution_packet_capacity,
    )?;
    let packet_descriptors = cuda_tile_packet_descriptors(
        resolution_packets.len(),
        1,
        job.num_components(),
        &mut host_budget,
    )?;
    let resolutions = cuda_tile_packetization_resolutions(resolution_packets, &mut host_budget)?;

    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: u32::try_from(resolutions.len())
            .map_err(|_| arithmetic_overflow("CUDA HTJ2K tile resolution count exceeds u32"))?,
        num_layers: 1,
        num_components: job.num_components(),
        code_block_count: u32::try_from(code_block_count)
            .map_err(|_| arithmetic_overflow("CUDA HTJ2K tile code-block count exceeds u32"))?,
        progression_order: job.progression_order,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };
    let plan = flatten_cuda_htj2k_packetization_job_classified_with_live_host_bytes(
        packetization_job,
        host_budget.live_bytes(),
    )
    .map_err(super::super::packetization::CudaHtj2kPacketizationPlanError::into_stage_error)?;
    host_budget
        .account_vec(&plan.payload)
        .map_err(htj2k_allocation_error)?;
    host_budget
        .account_vec(&plan.packets)
        .map_err(htj2k_allocation_error)?;
    host_budget
        .account_vec(&plan.subbands)
        .map_err(htj2k_allocation_error)?;
    host_budget
        .account_vec(&plan.blocks)
        .map_err(htj2k_allocation_error)?;
    host_budget
        .account_vec(&plan.tag_states)
        .map_err(htj2k_allocation_error)?;
    host_budget
        .account_vec(&plan.tag_nodes)
        .map_err(htj2k_allocation_error)?;
    let packets = cuda_packetization_packets(&plan, &mut host_budget)?;
    let subbands = cuda_packetization_subbands(&plan, &mut host_budget)?;
    let blocks = cuda_packetization_blocks(&plan, &mut host_budget)?;
    let tag_states = cuda_packetization_tag_states(&plan, &mut host_budget)?;
    let tag_nodes = cuda_packetization_tag_nodes(&plan, &mut host_budget)?;
    let packetized = context
        .packetize_htj2k_cleanup_packets_with_tag_state_and_live_host_bytes(
            &plan.payload,
            &packets,
            &subbands,
            &blocks,
            &tag_states,
            &tag_nodes,
            host_budget.live_bytes(),
        )
        .map_err(|error| runtime_error("packetize CUDA HTJ2K tile", error))?;
    let dispatches = packetized.execution().kernel_dispatches();
    let packetize_us = packetized.stage_timings().packetize_us;
    Ok((packetized.into_data(), dispatches, packetize_us))
}

fn cuda_tile_packetization_resolutions<'a>(
    resolution_packets: &'a [CudaEncodedHtj2kResolution],
    host_budget: &mut HostPhaseBudget,
) -> CudaStageResult<Vec<J2kPacketizationResolution<'a>>> {
    let mut resolutions = host_budget
        .try_vec_with_capacity(resolution_packets.len())
        .map_err(htj2k_allocation_error)?;
    for resolution in resolution_packets {
        let mut packet_subbands = host_budget
            .try_vec_with_capacity(resolution.subbands.len())
            .map_err(htj2k_allocation_error)?;
        for subband in &resolution.subbands {
            let mut code_blocks = host_budget
                .try_vec_with_capacity(subband.code_blocks.len())
                .map_err(htj2k_allocation_error)?;
            code_blocks.extend(
                subband
                    .code_blocks
                    .iter()
                    .map(|block| J2kPacketizationCodeBlock {
                        data: block.data.as_slice(),
                        ht_cleanup_length: block.cleanup_length,
                        ht_refinement_length: block.refinement_length,
                        num_coding_passes: block.num_coding_passes,
                        num_zero_bitplanes: block.num_zero_bitplanes,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    }),
            );
            host_budget
                .try_vec_push(
                    &mut packet_subbands,
                    J2kPacketizationSubband {
                        code_blocks,
                        num_cbs_x: subband.num_cbs_x,
                        num_cbs_y: subband.num_cbs_y,
                    },
                )
                .map_err(htj2k_allocation_error)?;
        }
        host_budget
            .try_vec_push(
                &mut resolutions,
                J2kPacketizationResolution {
                    subbands: packet_subbands,
                },
            )
            .map_err(htj2k_allocation_error)?;
    }
    Ok(resolutions)
}

fn cuda_tile_packet_descriptors(
    packet_count: usize,
    num_layers: u8,
    num_components: u16,
    host_budget: &mut HostPhaseBudget,
) -> CudaStageResult<Vec<J2kPacketizationPacketDescriptor>> {
    if num_layers != 1 {
        return Err(J2kEncodeStageError::unsupported(
            "CUDA HTJ2K tile encode currently prepares one packet layer",
        ));
    }
    let component_count = usize::from(num_components).max(1);
    let mut descriptors = host_budget
        .try_vec_with_capacity(packet_count)
        .map_err(htj2k_allocation_error)?;
    for packet_index in 0..packet_count {
        host_budget
            .try_vec_push(
                &mut descriptors,
                J2kPacketizationPacketDescriptor {
                    packet_index: u32::try_from(packet_index).map_err(|_| {
                        arithmetic_overflow("CUDA HTJ2K tile packet index exceeds u32")
                    })?,
                    state_index: u32::try_from(packet_index).map_err(|_| {
                        arithmetic_overflow("CUDA HTJ2K tile packet state index exceeds u32")
                    })?,
                    layer: 0,
                    resolution: u32::try_from(packet_index / component_count).map_err(|_| {
                        arithmetic_overflow("CUDA HTJ2K tile packet resolution exceeds u32")
                    })?,
                    component: u16::try_from(packet_index % component_count).map_err(|_| {
                        arithmetic_overflow("CUDA HTJ2K tile packet component exceeds u16")
                    })?,
                    precinct: 0,
                },
            )
            .map_err(htj2k_allocation_error)?;
    }
    Ok(descriptors)
}
