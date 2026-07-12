// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{J2kPacketizationBlockCodingMode, J2kPacketizationEncodeJob, J2kPacketizationResolution};

use crate::allocation::HostPhaseBudget;

use super::state::{
    append_cuda_htj2k_packetization_tag_state, cuda_ht_segment_lengths,
    cuda_packetization_state_count, finalize_cuda_htj2k_packetization_tag_trees,
    record_cuda_htj2k_packetization_first_inclusion_layers, seed_cuda_htj2k_packetization_state,
    update_cuda_htj2k_packetization_state_after_block,
    validate_cuda_htj2k_packetization_state_layout, CudaHtj2kPacketizationState,
    CUDA_HTJ2K_PACKET_TAG_INF,
};
use super::types::{
    packetization_plan_allocation_error, CudaHtj2kPacketizationPlan,
    CudaHtj2kPacketizationPlanBlock, CudaHtj2kPacketizationPlanError,
    CudaHtj2kPacketizationPlanPacket, CudaHtj2kPacketizationPlanSink,
    CudaHtj2kPacketizationPlanSubband, PacketizationPlanResult,
};

#[cfg(test)]
pub(in crate::encode) fn flatten_cuda_htj2k_packetization_job(
    job: J2kPacketizationEncodeJob<'_>,
) -> PacketizationPlanResult<CudaHtj2kPacketizationPlan> {
    flatten_cuda_htj2k_packetization_job_classified(job)
}

pub(in crate::encode) fn flatten_cuda_htj2k_packetization_job_classified(
    job: J2kPacketizationEncodeJob<'_>,
) -> PacketizationPlanResult<CudaHtj2kPacketizationPlan> {
    flatten_cuda_htj2k_packetization_job_classified_with_live_host_bytes(job, 0)
}

pub(in crate::encode) fn flatten_cuda_htj2k_packetization_job_classified_with_live_host_bytes(
    job: J2kPacketizationEncodeJob<'_>,
    live_host_bytes: usize,
) -> PacketizationPlanResult<CudaHtj2kPacketizationPlan> {
    if job.resolution_count as usize != job.resolutions.len() {
        return Err(CudaHtj2kPacketizationPlanError::Invalid(
            "CUDA HTJ2K packetization resolution count mismatch",
        ));
    }

    let mut payload = Vec::new();
    let mut packets = Vec::new();
    let mut subbands = Vec::new();
    let mut blocks = Vec::new();
    let mut tag_states = Vec::new();
    let mut tag_nodes = Vec::new();
    let mut host_budget =
        HostPhaseBudget::with_live_bytes("j2k CUDA packetization owner graph", live_host_bytes)
            .map_err(packetization_plan_allocation_error)?;

    {
        let mut sink = CudaHtj2kPacketizationPlanSink {
            host_budget: &mut host_budget,
            payload: &mut payload,
            packets: &mut packets,
            subbands: &mut subbands,
            blocks: &mut blocks,
            tag_states: &mut tag_states,
            tag_nodes: &mut tag_nodes,
        };
        flatten_cuda_htj2k_job_packets(&job, &mut sink)?;
    }

    if job.code_block_count as usize != blocks.len() {
        return Err(CudaHtj2kPacketizationPlanError::Invalid(
            "CUDA HTJ2K packetization code-block count mismatch",
        ));
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

fn flatten_cuda_htj2k_job_packets(
    job: &J2kPacketizationEncodeJob<'_>,
    sink: &mut CudaHtj2kPacketizationPlanSink<'_>,
) -> PacketizationPlanResult<()> {
    if job.packet_descriptors.is_empty() {
        if job.num_layers != 1 {
            return Err(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization requires explicit descriptors for multiple layers",
            ));
        }
        for packet_index in 0..job.resolutions.len() {
            flatten_cuda_htj2k_packet(
                job.resolutions.get(packet_index).ok_or(
                    CudaHtj2kPacketizationPlanError::Invalid(
                        "CUDA HTJ2K packet descriptor index out of range",
                    ),
                )?,
                sink,
            )?;
        }
        return Ok(());
    }

    let state_count = cuda_packetization_state_count(job.packet_descriptors)?;
    let mut states: Vec<Option<CudaHtj2kPacketizationState>> = sink
        .host_budget
        .try_vec_with_capacity(state_count)
        .map_err(packetization_plan_allocation_error)?;
    states.resize_with(state_count, || None);
    for descriptor in job.packet_descriptors {
        if descriptor.layer >= job.num_layers {
            return Err(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization descriptor layer exceeds layer count",
            ));
        }
        let resolution = job
            .resolutions
            .get(descriptor.packet_index as usize)
            .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packet descriptor index out of range",
            ))?;
        let state = states.get_mut(descriptor.state_index as usize).ok_or(
            CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packet descriptor state index out of range",
            ),
        )?;
        if let Some(existing) = state {
            validate_cuda_htj2k_packetization_state_layout(existing, resolution)?;
        } else {
            *state = Some(seed_cuda_htj2k_packetization_state(
                resolution,
                sink.host_budget,
            )?);
        }
        let state = state
            .as_mut()
            .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization state initialization failed",
            ))?;
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
            return Err(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization descriptor layer exceeds layer count",
            ));
        }
        let resolution = job
            .resolutions
            .get(descriptor.packet_index as usize)
            .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packet descriptor index out of range",
            ))?;
        let state = states.get_mut(descriptor.state_index as usize).ok_or(
            CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packet descriptor state index out of range",
            ),
        )?;
        if let Some(existing) = state {
            validate_cuda_htj2k_packetization_state_layout(existing, resolution)?;
        } else {
            *state = Some(seed_cuda_htj2k_packetization_state(
                resolution,
                sink.host_budget,
            )?);
        }
        let state = state
            .as_mut()
            .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization state initialization failed",
            ))?;
        flatten_cuda_htj2k_packet_with_state(resolution, descriptor.layer, state, sink)?;
    }
    Ok(())
}

fn flatten_cuda_htj2k_packet(
    resolution: &J2kPacketizationResolution<'_>,
    sink: &mut CudaHtj2kPacketizationPlanSink<'_>,
) -> PacketizationPlanResult<()> {
    flatten_cuda_htj2k_packet_inner(resolution, 0, None, sink)
}

fn flatten_cuda_htj2k_packet_with_state(
    resolution: &J2kPacketizationResolution<'_>,
    layer: u8,
    state: &mut CudaHtj2kPacketizationState,
    sink: &mut CudaHtj2kPacketizationPlanSink<'_>,
) -> PacketizationPlanResult<()> {
    flatten_cuda_htj2k_packet_inner(resolution, layer, Some(state), sink)
}

#[expect(
    clippy::too_many_lines,
    reason = "single-packet flattening keeps tag-tree state and emitted descriptor offsets coupled"
)]
fn flatten_cuda_htj2k_packet_inner(
    resolution: &J2kPacketizationResolution<'_>,
    layer: u8,
    mut state: Option<&mut CudaHtj2kPacketizationState>,
    sink: &mut CudaHtj2kPacketizationPlanSink<'_>,
) -> PacketizationPlanResult<()> {
    let block_start = u32::try_from(sink.blocks.len()).map_err(|_| {
        CudaHtj2kPacketizationPlanError::Invalid("CUDA HTJ2K packetization block count exceeds u32")
    })?;
    let subband_start = u32::try_from(sink.subbands.len()).map_err(|_| {
        CudaHtj2kPacketizationPlanError::Invalid(
            "CUDA HTJ2K packetization subband count exceeds u32",
        )
    })?;
    let mut body_len = 0usize;
    let mut block_count = 0usize;
    let packet_has_data = resolution.subbands.iter().any(|subband| {
        subband
            .code_blocks
            .iter()
            .any(|block| block.num_coding_passes > 0)
    });

    for (subband_index, subband) in resolution.subbands.iter().enumerate() {
        let subband_code_blocks = u32::try_from(subband.code_blocks.len()).map_err(|_| {
            CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization block count exceeds u32",
            )
        })?;
        if subband.num_cbs_x == 0
            || subband.num_cbs_y == 0
            || subband.num_cbs_x.saturating_mul(subband.num_cbs_y) != subband_code_blocks
        {
            return Err(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization subband code-block layout mismatch",
            ));
        }

        let subband_block_start = u32::try_from(sink.blocks.len()).map_err(|_| {
            CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization block count exceeds u32",
            )
        })?;
        let state_subband = state
            .as_deref()
            .and_then(|state| state.subbands.get(subband_index));
        append_cuda_htj2k_packetization_tag_state(
            state_subband,
            subband.num_cbs_x,
            subband.num_cbs_y,
            sink.tag_states,
            sink.tag_nodes,
            sink.host_budget,
        )?;
        for (block_index, code_block) in subband.code_blocks.iter().enumerate() {
            if code_block.block_coding_mode != J2kPacketizationBlockCodingMode::HighThroughput {
                return Err(CudaHtj2kPacketizationPlanError::Invalid(
                    "CUDA packetization only supports HTJ2K block-coded packets",
                ));
            }
            if code_block.num_coding_passes > 164 {
                return Err(CudaHtj2kPacketizationPlanError::Invalid(
                    "CUDA HTJ2K packetization coding pass count exceeds JPEG 2000 bounds",
                ));
            }
            let (previously_included, l_block, inclusion_layer, zero_bitplanes) =
                if let Some(state) = state.as_deref() {
                    let state_block = state
                        .subbands
                        .get(subband_index)
                        .and_then(|state_subband| state_subband.blocks.get(block_index))
                        .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
                            "CUDA HTJ2K packetization state layout mismatch",
                        ))?;
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
                return Err(CudaHtj2kPacketizationPlanError::Invalid(
                    "CUDA HTJ2K packetization descriptor order does not match first inclusion layer",
                ));
            }
            if state.is_none() && previously_included {
                return Err(CudaHtj2kPacketizationPlanError::Invalid(
                    "CUDA HTJ2K packetization requires first-inclusion packets",
                ));
            }
            if code_block.num_coding_passes == 0 && !code_block.data.is_empty() {
                return Err(CudaHtj2kPacketizationPlanError::Invalid(
                    "CUDA HTJ2K packetization empty contributions must not carry payload",
                ));
            }
            if zero_bitplanes > 31 || l_block > 31 {
                return Err(CudaHtj2kPacketizationPlanError::Invalid(
                    "CUDA HTJ2K packetization header fields exceed kernel bounds",
                ));
            }

            let data_offset = u32::try_from(sink.payload.len()).map_err(|_| {
                CudaHtj2kPacketizationPlanError::Invalid(
                    "CUDA HTJ2K packetization payload exceeds u32",
                )
            })?;
            let data_len = if code_block.num_coding_passes == 0 {
                0
            } else {
                u32::try_from(code_block.data.len()).map_err(|_| {
                    CudaHtj2kPacketizationPlanError::Invalid(
                        "CUDA HTJ2K packetization code-block payload exceeds u32",
                    )
                })?
            };
            let (cleanup_length, refinement_length) = cuda_ht_segment_lengths(code_block)?;
            if code_block.num_coding_passes > 0 {
                sink.host_budget
                    .try_vec_extend_from_slice(sink.payload, code_block.data)
                    .map_err(packetization_plan_allocation_error)?;
                body_len = body_len.checked_add(code_block.data.len()).ok_or(
                    CudaHtj2kPacketizationPlanError::Invalid(
                        "CUDA HTJ2K packetization body length overflow",
                    ),
                )?;
            }
            sink.host_budget
                .try_vec_push(
                    sink.blocks,
                    CudaHtj2kPacketizationPlanBlock {
                        data_offset,
                        data_len,
                        cleanup_length,
                        refinement_length,
                        num_coding_passes: u32::from(code_block.num_coding_passes),
                        num_zero_bitplanes: zero_bitplanes,
                        l_block,
                        previously_included: u32::from(previously_included),
                        inclusion_layer,
                    },
                )
                .map_err(packetization_plan_allocation_error)?;
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
            block_count =
                block_count
                    .checked_add(1)
                    .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
                        "CUDA HTJ2K packetization block count overflow",
                    ))?;
        }
        sink.host_budget
            .try_vec_push(
                sink.subbands,
                CudaHtj2kPacketizationPlanSubband {
                    block_start: subband_block_start,
                    block_count: subband_code_blocks,
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                },
            )
            .map_err(packetization_plan_allocation_error)?;
    }

    let header_capacity = 256usize
        .checked_add(block_count.checked_mul(64).ok_or(
            CudaHtj2kPacketizationPlanError::Invalid("CUDA HTJ2K packetization capacity overflow"),
        )?)
        .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
            "CUDA HTJ2K packetization capacity overflow",
        ))?;
    let output_capacity =
        body_len
            .checked_add(header_capacity)
            .ok_or(CudaHtj2kPacketizationPlanError::Invalid(
                "CUDA HTJ2K packetization capacity overflow",
            ))?;
    sink.host_budget
        .try_vec_push(
            sink.packets,
            CudaHtj2kPacketizationPlanPacket {
                block_start,
                block_count: u32::try_from(block_count).map_err(|_| {
                    CudaHtj2kPacketizationPlanError::Invalid(
                        "CUDA HTJ2K packetization block count exceeds u32",
                    )
                })?,
                subband_start,
                subband_count: u32::try_from(resolution.subbands.len()).map_err(|_| {
                    CudaHtj2kPacketizationPlanError::Invalid(
                        "CUDA HTJ2K packetization subband count exceeds u32",
                    )
                })?,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    CudaHtj2kPacketizationPlanError::Invalid(
                        "CUDA HTJ2K packetization packet capacity exceeds u32",
                    )
                })?,
                layer: u32::from(layer),
            },
        )
        .map_err(packetization_plan_allocation_error)?;
    Ok(())
}
