// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_buffer_read, checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer,
    dispatch_single_thread, encode_status_error, new_command_buffer, new_compute_command_encoder,
    new_private_buffer, new_shared_buffer, packet_tree_node_count, size_of, with_runtime,
    zeroed_shared_buffer, Error, J2kPacketBlock, J2kPacketDescriptor, J2kPacketEncodeParams,
    J2kPacketEncodeStatus, J2kPacketResolution, J2kPacketStateBlock, J2kPacketSubband,
    J2kPacketizationBlockCodingMode, J2kPacketizationEncodeJob, MetalRuntime, J2K_ENCODE_STATUS_OK,
};

#[cfg(target_os = "macos")]
pub(crate) fn encode_tier2_packetization(
    job: J2kPacketizationEncodeJob<'_>,
) -> Result<Vec<u8>, Error> {
    with_runtime(|runtime| {
        let plan = plan_tier2_packetization(job)?;
        execute_tier2_packetization(runtime, plan)
    })
}

struct Tier2PacketizationPlan {
    resolutions: Vec<J2kPacketResolution>,
    subbands: Vec<J2kPacketSubband>,
    blocks: Vec<J2kPacketBlock>,
    payload: Vec<u8>,
    descriptors: Vec<J2kPacketDescriptor>,
    state_blocks: Vec<J2kPacketStateBlock>,
    max_tree_nodes: usize,
    header_capacity: usize,
    output_capacity: usize,
    params: J2kPacketEncodeParams,
}

struct Tier2PacketAllocationCounts {
    resolutions: usize,
    subbands: usize,
    blocks: usize,
    payload_bytes: usize,
    unique_states: usize,
    state_blocks: usize,
    descriptors: usize,
}

#[cfg(target_os = "macos")]
fn tier2_packet_block_count(
    job: J2kPacketizationEncodeJob<'_>,
    packet_index: u32,
) -> Result<usize, Error> {
    let packet_index = usize::try_from(packet_index).map_err(|_| Error::MetalKernel {
        message: "Tier-2 Metal packet descriptor packet index exceeds usize".to_string(),
    })?;
    let resolution = job
        .resolutions
        .get(packet_index)
        .ok_or_else(|| Error::MetalKernel {
            message: "Tier-2 Metal packet descriptor packet index out of range".to_string(),
        })?;
    crate::batch_allocation::checked_count_sum(
        resolution
            .subbands
            .iter()
            .map(|subband| subband.code_blocks.len()),
        "J2K Metal Tier-2 packet state blocks",
    )
    .map_err(Error::from)
}

#[cfg(target_os = "macos")]
fn tier2_packet_allocation_counts(
    job: J2kPacketizationEncodeJob<'_>,
) -> Result<Tier2PacketAllocationCounts, Error> {
    let subbands = crate::batch_allocation::checked_count_sum(
        job.resolutions
            .iter()
            .map(|resolution| resolution.subbands.len()),
        "J2K Metal Tier-2 packet subbands",
    )?;
    let blocks = crate::batch_allocation::checked_count_sum(
        job.resolutions
            .iter()
            .flat_map(|resolution| &resolution.subbands)
            .map(|subband| subband.code_blocks.len()),
        "J2K Metal Tier-2 packet blocks",
    )?;
    let payload_bytes = crate::batch_allocation::checked_count_sum(
        job.resolutions
            .iter()
            .flat_map(|resolution| &resolution.subbands)
            .flat_map(|subband| &subband.code_blocks)
            .map(|block| block.data.len()),
        "J2K Metal Tier-2 packet payload",
    )?;
    let mut unique_states = 0usize;
    let mut state_blocks = 0usize;
    for (index, descriptor) in job.packet_descriptors.iter().enumerate() {
        if job.packet_descriptors[..index]
            .iter()
            .any(|previous| previous.state_index == descriptor.state_index)
        {
            continue;
        }
        unique_states = crate::batch_allocation::checked_count_sum(
            [unique_states, 1],
            "J2K Metal Tier-2 packet states",
        )?;
        state_blocks = crate::batch_allocation::checked_count_sum(
            [
                state_blocks,
                tier2_packet_block_count(job, descriptor.packet_index)?,
            ],
            "J2K Metal Tier-2 packet state blocks",
        )?;
    }
    Ok(Tier2PacketAllocationCounts {
        resolutions: job.resolutions.len(),
        subbands,
        blocks,
        payload_bytes,
        unique_states,
        state_blocks,
        descriptors: job.packet_descriptors.len(),
    })
}

#[cfg(target_os = "macos")]
fn tier2_packet_allocation_requests(
    counts: &Tier2PacketAllocationCounts,
) -> [crate::batch_allocation::BatchMetadataRequest; 7] {
    [
        crate::batch_allocation::BatchMetadataRequest::of::<J2kPacketResolution>(
            counts.resolutions,
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kPacketSubband>(counts.subbands),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kPacketBlock>(counts.blocks),
        crate::batch_allocation::BatchMetadataRequest::of::<u8>(counts.payload_bytes),
        crate::batch_allocation::BatchMetadataRequest::of::<(u32, u32, usize)>(
            counts.unique_states,
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kPacketStateBlock>(
            counts.state_blocks,
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kPacketDescriptor>(
            counts.descriptors,
        ),
    ]
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "single-pass Tier-2 planning keeps packet topology and capacities consistent"
)]
fn plan_tier2_packetization(
    job: J2kPacketizationEncodeJob<'_>,
) -> Result<Tier2PacketizationPlan, Error> {
    let counts = tier2_packet_allocation_counts(job)?;
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal Tier-2 packet metadata");
    budget.preflight(&tier2_packet_allocation_requests(&counts))?;
    let mut resolutions =
        budget.try_vec(counts.resolutions, "J2K Metal Tier-2 packet resolutions")?;
    let mut subbands = budget.try_vec(counts.subbands, "J2K Metal Tier-2 packet subbands")?;
    let mut blocks = budget.try_vec(counts.blocks, "J2K Metal Tier-2 packet blocks")?;
    let mut payload = budget.try_vec(counts.payload_bytes, "J2K Metal Tier-2 packet payload")?;
    let mut max_tree_nodes = 1usize;

    for resolution in job.resolutions {
        let subband_offset = u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
            message: "Tier-2 Metal packet subband table exceeds u32".to_string(),
        })?;
        for subband in &resolution.subbands {
            let block_offset = u32::try_from(blocks.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet block table exceeds u32".to_string(),
            })?;
            max_tree_nodes = max_tree_nodes.max(packet_tree_node_count(
                subband.num_cbs_x,
                subband.num_cbs_y,
            )?);
            for code_block in &subband.code_blocks {
                let data_offset = u32::try_from(payload.len()).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet payload exceeds u32".to_string(),
                })?;
                let data_len =
                    u32::try_from(code_block.data.len()).map_err(|_| Error::MetalKernel {
                        message: "Tier-2 Metal packet code-block payload exceeds u32".to_string(),
                    })?;
                payload.extend_from_slice(code_block.data);
                blocks.push(J2kPacketBlock {
                    data_offset,
                    data_len,
                    num_coding_passes: u32::from(code_block.num_coding_passes),
                    num_zero_bitplanes: u32::from(code_block.num_zero_bitplanes),
                    previously_included: u32::from(code_block.previously_included),
                    l_block: code_block.l_block,
                    block_coding_mode: match code_block.block_coding_mode {
                        J2kPacketizationBlockCodingMode::Classic => 0,
                        J2kPacketizationBlockCodingMode::HighThroughput => 1,
                    },
                    reserved0: 0,
                });
            }
            subbands.push(J2kPacketSubband {
                block_offset,
                block_count: u32::try_from(subband.code_blocks.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "Tier-2 Metal packet subband block count exceeds u32".to_string(),
                    }
                })?,
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
            });
        }
        resolutions.push(J2kPacketResolution {
            subband_offset,
            subband_count: u32::try_from(resolution.subbands.len()).map_err(|_| {
                Error::MetalKernel {
                    message: "Tier-2 Metal packet resolution subband count exceeds u32".to_string(),
                }
            })?,
        });
    }

    let mut state_block_offsets = budget.try_vec(
        counts.unique_states,
        "J2K Metal Tier-2 packet state offsets",
    )?;
    let mut state_blocks =
        budget.try_vec(counts.state_blocks, "J2K Metal Tier-2 packet state blocks")?;
    let mut descriptors =
        budget.try_vec(counts.descriptors, "J2K Metal Tier-2 packet descriptors")?;
    for descriptor in job.packet_descriptors {
        let packet_index =
            usize::try_from(descriptor.packet_index).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet descriptor packet index exceeds usize".to_string(),
            })?;
        let resolution = resolutions
            .get(packet_index)
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet descriptor packet index out of range".to_string(),
            })?;
        let subband_start =
            usize::try_from(resolution.subband_offset).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet descriptor subband offset exceeds usize".to_string(),
            })?;
        let subband_count =
            usize::try_from(resolution.subband_count).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet descriptor subband count exceeds usize".to_string(),
            })?;
        let subband_end =
            subband_start
                .checked_add(subband_count)
                .ok_or_else(|| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor subband range overflow".to_string(),
                })?;
        if subband_end > subbands.len() {
            return Err(Error::MetalKernel {
                message: "Tier-2 Metal packet descriptor subband range out of bounds".to_string(),
            });
        }
        let mut packet_block_count = 0usize;
        for subband in &subbands[subband_start..subband_end] {
            packet_block_count = packet_block_count
                .checked_add(usize::try_from(subband.block_count).map_err(|_| {
                    Error::MetalKernel {
                        message: "Tier-2 Metal packet descriptor block count exceeds usize"
                            .to_string(),
                    }
                })?)
                .ok_or_else(|| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor block count overflow".to_string(),
                })?;
        }

        let (state_block_offset, existing_count) = if let Some(&(_, offset, count)) =
            state_block_offsets
                .iter()
                .find(|(state_index, _, _)| *state_index == descriptor.state_index)
        {
            (offset, count)
        } else {
            let offset = u32::try_from(state_blocks.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet state block offset exceeds u32".to_string(),
            })?;
            for subband in &subbands[subband_start..subband_end] {
                let block_start =
                    usize::try_from(subband.block_offset).map_err(|_| Error::MetalKernel {
                        message: "Tier-2 Metal packet state block offset exceeds usize".to_string(),
                    })?;
                let block_count =
                    usize::try_from(subband.block_count).map_err(|_| Error::MetalKernel {
                        message: "Tier-2 Metal packet state block count exceeds usize".to_string(),
                    })?;
                let block_end =
                    block_start
                        .checked_add(block_count)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "Tier-2 Metal packet state block range overflow".to_string(),
                        })?;
                if block_end > blocks.len() {
                    return Err(Error::MetalKernel {
                        message: "Tier-2 Metal packet state block range out of bounds".to_string(),
                    });
                }
                for block in &blocks[block_start..block_end] {
                    state_blocks.push(J2kPacketStateBlock {
                        previously_included: block.previously_included,
                        l_block: block.l_block,
                    });
                }
            }
            state_block_offsets.push((descriptor.state_index, offset, packet_block_count));
            (offset, packet_block_count)
        };
        if existing_count != packet_block_count {
            return Err(Error::MetalKernel {
                message: "Tier-2 Metal packet descriptor state layout mismatch".to_string(),
            });
        }

        descriptors.push(J2kPacketDescriptor {
            packet_index: descriptor.packet_index,
            state_index: descriptor.state_index,
            layer: u32::from(descriptor.layer),
            resolution: descriptor.resolution,
            component: u32::from(descriptor.component),
            precinct_lo: u32::try_from(descriptor.precinct & u64::from(u32::MAX))
                .expect("masked precinct low word fits u32"),
            precinct_hi: u32::try_from(descriptor.precinct >> 32)
                .expect("precinct high word fits u32"),
            state_block_offset,
        });
    }

    let header_capacity = blocks
        .len()
        .checked_mul(256)
        .and_then(|bytes| bytes.checked_add(4096))
        .map(|bytes| bytes.max(4096))
        .ok_or_else(|| Error::MetalKernel {
            message: "Tier-2 Metal packet header capacity overflow".to_string(),
        })?;
    let output_capacity = payload
        .len()
        .checked_add(
            header_capacity
                .checked_mul(descriptors.len().max(resolutions.len()).max(1))
                .ok_or_else(|| Error::MetalKernel {
                    message: "Tier-2 Metal packet output capacity overflow".to_string(),
                })?,
        )
        .and_then(|bytes| bytes.checked_add(1024))
        .ok_or_else(|| Error::MetalKernel {
            message: "Tier-2 Metal packet output capacity overflow".to_string(),
        })?;

    let params = J2kPacketEncodeParams {
        resolution_count: u32::try_from(resolutions.len()).map_err(|_| Error::MetalKernel {
            message: "Tier-2 Metal packet resolution count exceeds u32".to_string(),
        })?,
        num_layers: u32::from(job.num_layers),
        num_components: u32::from(job.num_components),
        code_block_count: u32::try_from(blocks.len()).map_err(|_| Error::MetalKernel {
            message: "Tier-2 Metal packet code-block count exceeds u32".to_string(),
        })?,
        subband_count: u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
            message: "Tier-2 Metal packet subband count exceeds u32".to_string(),
        })?,
        descriptor_count: u32::try_from(descriptors.len()).map_err(|_| Error::MetalKernel {
            message: "Tier-2 Metal packet descriptor count exceeds u32".to_string(),
        })?,
        output_capacity: u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
            message: "Tier-2 Metal packet output capacity exceeds u32".to_string(),
        })?,
        header_capacity: u32::try_from(header_capacity).map_err(|_| Error::MetalKernel {
            message: "Tier-2 Metal packet header capacity exceeds u32".to_string(),
        })?,
        scratch_node_capacity: u32::try_from(max_tree_nodes).map_err(|_| Error::MetalKernel {
            message: "Tier-2 Metal packet scratch node capacity exceeds u32".to_string(),
        })?,
    };

    Ok(Tier2PacketizationPlan {
        resolutions,
        subbands,
        blocks,
        payload,
        descriptors,
        state_blocks,
        max_tree_nodes,
        header_capacity,
        output_capacity,
        params,
    })
}

#[cfg(target_os = "macos")]
fn execute_tier2_packetization(
    runtime: &MetalRuntime,
    plan: Tier2PacketizationPlan,
) -> Result<Vec<u8>, Error> {
    let Tier2PacketizationPlan {
        resolutions,
        subbands,
        blocks,
        payload,
        descriptors,
        state_blocks,
        max_tree_nodes,
        header_capacity,
        output_capacity,
        params,
    } = plan;
    let resolution_buffer = copied_slice_buffer(&runtime.device, &resolutions)?;
    let subband_buffer = copied_slice_buffer(&runtime.device, &subbands)?;
    let block_buffer = copied_slice_buffer(&runtime.device, &blocks)?;
    let payload_buffer = copied_slice_buffer(&runtime.device, &payload)?;
    let descriptor_buffer = copied_slice_buffer(&runtime.device, &descriptors)?;
    let state_block_buffer = copied_slice_buffer(&runtime.device, &state_blocks)?;
    let output_buffer = new_shared_buffer(&runtime.device, output_capacity)?;
    let header_buffer = new_private_buffer(&runtime.device, header_capacity)?;
    let scratch_words = max_tree_nodes
        .checked_mul(6)
        .ok_or_else(|| Error::MetalKernel {
            message: "Tier-2 Metal packet scratch size overflow".to_string(),
        })?;
    let scratch_bytes =
        scratch_words
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet scratch byte size overflow".to_string(),
            })?;
    let scratch_buffer = new_private_buffer(&runtime.device, scratch_bytes)?;
    let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kPacketEncodeStatus>())?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let encoder = new_compute_command_encoder(&command_buffer)?;
    encoder.set_compute_pipeline_state(&runtime.packet_encode);
    encoder.set_buffer(0, Some(&resolution_buffer), 0);
    encoder.set_buffer(1, Some(&subband_buffer), 0);
    encoder.set_buffer(2, Some(&block_buffer), 0);
    encoder.set_buffer(3, Some(&payload_buffer), 0);
    encoder.set_buffer(4, Some(&output_buffer), 0);
    encoder.set_buffer(5, Some(&header_buffer), 0);
    encoder.set_buffer(6, Some(&scratch_buffer), 0);
    encoder.set_bytes(
        7,
        size_of::<J2kPacketEncodeParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(8, Some(&status_buffer), 0);
    encoder.set_buffer(9, Some(&descriptor_buffer), 0);
    encoder.set_buffer(10, Some(&state_block_buffer), 0);
    dispatch_single_thread(&encoder);
    encoder.end_encoding();
    commit_and_wait_metal(&command_buffer)?;

    let status =
        checked_buffer_read::<J2kPacketEncodeStatus>(&status_buffer, "Tier-2 packet status")?;
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "Tier-2 packetization",
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: "Tier-2 Metal packet output length exceeds usize".to_string(),
    })?;
    if data_len > output_capacity {
        return Err(Error::MetalKernel {
            message: "Tier-2 Metal packet output length exceeds buffer".to_string(),
        });
    }
    Ok(if data_len == 0 {
        Vec::new()
    } else {
        checked_buffer_slice::<u8>(&output_buffer, data_len, "Tier-2 packet payload")?
    })
}

#[cfg(test)]
mod tests;
