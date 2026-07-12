// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    codestream_progression_order_code, copied_slice_buffer, dispatch_single_thread,
    lossless_codestream_assembly_capacity, new_command_buffer, new_compute_command_encoder,
    new_private_buffer, new_shared_buffer, packet_tree_node_count, size_of,
    wait_resident_lossless_codestream, with_runtime_for_session, zeroed_shared_buffer, Error,
    J2kCodestreamAssemblyStatus, J2kLosslessCodestreamAssemblyJob,
    J2kLosslessCodestreamAssemblyParams, J2kLosslessCodestreamBlockCodingMode, J2kPacketBlock,
    J2kPacketDescriptor, J2kPacketEncodeParams, J2kPacketEncodeStatus, J2kPacketResolution,
    J2kPacketStateBlock, J2kPacketSubband, J2kPendingResidentLosslessCodestream,
    J2kResidentLosslessCodestream, J2kResidentPacketBlock, J2kResidentPacketBlockParams,
    J2kResidentPacketizationEncodeJob, MTLSize, MetalRuntime, ResidentLosslessTier1Metal,
};

#[cfg(target_os = "macos")]
pub(crate) fn encode_lossless_codestream_buffer_from_resident_tier1<
    T: ResidentLosslessTier1Metal,
>(
    session: &crate::MetalBackendSession,
    tier1: &T,
    job: J2kResidentPacketizationEncodeJob<'_>,
    codestream_job: J2kLosslessCodestreamAssemblyJob,
) -> Result<J2kResidentLosslessCodestream, Error> {
    wait_resident_lossless_codestream(submit_lossless_codestream_buffer_from_resident_tier1(
        session,
        tier1,
        job,
        codestream_job,
    )?)
}

#[cfg(target_os = "macos")]
pub(crate) fn submit_lossless_codestream_buffer_from_resident_tier1<
    T: ResidentLosslessTier1Metal,
>(
    session: &crate::MetalBackendSession,
    tier1: &T,
    job: J2kResidentPacketizationEncodeJob<'_>,
    codestream_job: J2kLosslessCodestreamAssemblyJob,
) -> Result<J2kPendingResidentLosslessCodestream, Error> {
    with_runtime_for_session(session, |runtime| {
        let plan = plan_resident_single(tier1, job, codestream_job)?;
        submit_resident_single_plan(runtime, tier1, plan)
    })
}

struct ResidentSinglePlan {
    resolutions: Vec<J2kPacketResolution>,
    subbands: Vec<J2kPacketSubband>,
    resident_blocks: Vec<J2kResidentPacketBlock>,
    descriptors: Vec<J2kPacketDescriptor>,
    state_blocks: Vec<J2kPacketStateBlock>,
    max_tree_nodes: usize,
    header_capacity: usize,
    output_capacity: usize,
    codestream_capacity: usize,
    packet_params: J2kPacketEncodeParams,
    codestream_params: J2kLosslessCodestreamAssemblyParams,
    resident_block_params: J2kResidentPacketBlockParams,
}

struct ResidentPacketTopology {
    resolutions: Vec<J2kPacketResolution>,
    subbands: Vec<J2kPacketSubband>,
    resident_blocks: Vec<J2kResidentPacketBlock>,
    max_tree_nodes: usize,
}

struct ResidentPacketDescriptors {
    descriptors: Vec<J2kPacketDescriptor>,
    state_blocks: Vec<J2kPacketStateBlock>,
}

struct ResidentPacketAllocationCounts {
    resolutions: usize,
    subbands: usize,
    blocks: usize,
    unique_states: usize,
    state_blocks: usize,
    descriptors: usize,
}

#[cfg(target_os = "macos")]
fn resident_job_packet_block_count(
    tier2_prefix: &str,
    job: J2kResidentPacketizationEncodeJob<'_>,
    packet_index: u32,
) -> Result<usize, Error> {
    let packet_index = usize::try_from(packet_index).map_err(|_| Error::MetalKernel {
        message: format!(
            "{tier2_prefix}Tier-2 Metal resident packet descriptor packet index exceeds usize"
        ),
    })?;
    let resolution = job
        .resolutions
        .get(packet_index)
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "{tier2_prefix}Tier-2 Metal resident packet descriptor packet index out of range"
            ),
        })?;
    resolution
        .subbands
        .iter()
        .try_fold(0usize, |total, subband| {
            let count =
                usize::try_from(subband.code_block_count).map_err(|_| Error::MetalKernel {
                    message: format!(
                        "{tier2_prefix}Tier-2 Metal resident packet descriptor block count exceeds usize"
                    ),
                })?;
            crate::batch_allocation::checked_count_sum(
                [total, count],
                "J2K Metal resident single state blocks",
            )
            .map_err(Error::from)
        })
}

#[cfg(target_os = "macos")]
fn resident_packet_allocation_counts(
    tier2_prefix: &str,
    job: J2kResidentPacketizationEncodeJob<'_>,
) -> Result<ResidentPacketAllocationCounts, Error> {
    let subbands = crate::batch_allocation::checked_count_sum(
        job.resolutions
            .iter()
            .map(|resolution| resolution.subbands.len()),
        "J2K Metal resident single subbands",
    )?;
    let blocks = job
        .resolutions
        .iter()
        .flat_map(|resolution| &resolution.subbands)
        .try_fold(0usize, |total, subband| {
            let count =
                usize::try_from(subband.code_block_count).map_err(|_| Error::MetalKernel {
                    message: format!(
                        "{tier2_prefix}Tier-2 Metal resident packet code-block count exceeds usize"
                    ),
                })?;
            crate::batch_allocation::checked_count_sum(
                [total, count],
                "J2K Metal resident single blocks",
            )
            .map_err(Error::from)
        })?;
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
            "J2K Metal resident single packet states",
        )?;
        state_blocks = crate::batch_allocation::checked_count_sum(
            [
                state_blocks,
                resident_job_packet_block_count(tier2_prefix, job, descriptor.packet_index)?,
            ],
            "J2K Metal resident single state blocks",
        )?;
    }
    Ok(ResidentPacketAllocationCounts {
        resolutions: job.resolutions.len(),
        subbands,
        blocks,
        unique_states,
        state_blocks,
        descriptors: job.packet_descriptors.len(),
    })
}

#[cfg(target_os = "macos")]
fn resident_packet_allocation_requests(
    counts: &ResidentPacketAllocationCounts,
) -> [crate::batch_allocation::BatchMetadataRequest; 6] {
    [
        crate::batch_allocation::BatchMetadataRequest::of::<J2kPacketResolution>(
            counts.resolutions,
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kPacketSubband>(counts.subbands),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kResidentPacketBlock>(counts.blocks),
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
    reason = "single-pass topology construction keeps packet indices and offsets consistent"
)]
fn build_resident_packet_topology<T: ResidentLosslessTier1Metal>(
    tier1: &T,
    job: J2kResidentPacketizationEncodeJob<'_>,
    budget: &mut crate::batch_allocation::BatchMetadataBudget,
    counts: &ResidentPacketAllocationCounts,
) -> Result<ResidentPacketTopology, Error> {
    if tier1.batch_job_count() != tier1.code_block_count() {
        return Err(Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packetization Tier-1 table mismatch",
                T::TIER2_PREFIX
            ),
        });
    }

    let mut resolutions = budget.try_vec(
        counts.resolutions,
        "J2K Metal resident single packet resolutions",
    )?;
    let mut subbands =
        budget.try_vec(counts.subbands, "J2K Metal resident single packet subbands")?;
    let mut resident_blocks =
        budget.try_vec(counts.blocks, "J2K Metal resident single packet blocks")?;
    let mut max_tree_nodes = 1usize;

    for resolution in job.resolutions {
        let subband_offset = u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet subband table exceeds u32",
                T::TIER2_PREFIX
            ),
        })?;
        for subband in &resolution.subbands {
            let block_offset =
                u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
                    message: format!(
                        "{}Tier-2 Metal resident packet block table exceeds u32",
                        T::TIER2_PREFIX
                    ),
                })?;
            max_tree_nodes = max_tree_nodes.max(packet_tree_node_count(
                subband.num_cbs_x,
                subband.num_cbs_y,
            )?);
            let code_block_start =
                usize::try_from(subband.code_block_start).map_err(|_| Error::MetalKernel {
                    message: format!(
                        "{}Tier-2 Metal resident packet code-block offset exceeds usize",
                        T::TIER2_PREFIX
                    ),
                })?;
            let code_block_count =
                usize::try_from(subband.code_block_count).map_err(|_| Error::MetalKernel {
                    message: format!(
                        "{}Tier-2 Metal resident packet code-block count exceeds usize",
                        T::TIER2_PREFIX
                    ),
                })?;
            let code_block_end =
                code_block_start
                    .checked_add(code_block_count)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "{}Tier-2 Metal resident packet code-block range overflow",
                            T::TIER2_PREFIX
                        ),
                    })?;
            if code_block_end > tier1.batch_job_count() {
                return Err(Error::MetalKernel {
                    message: format!(
                        "{}Tier-2 Metal resident packet code-block range out of bounds",
                        T::TIER2_PREFIX
                    ),
                });
            }
            for tier1_job_index in code_block_start..code_block_end {
                resident_blocks.push(J2kResidentPacketBlock {
                    tier1_job_index: u32::try_from(tier1_job_index).map_err(|_| {
                        Error::MetalKernel {
                            message: format!(
                                "{}Tier-2 Metal resident packet Tier-1 index exceeds u32",
                                T::TIER2_PREFIX
                            ),
                        }
                    })?,
                    previously_included: 0,
                    l_block: 3,
                    block_coding_mode: T::BLOCK_CODING_MODE,
                });
            }
            subbands.push(J2kPacketSubband {
                block_offset,
                block_count: subband.code_block_count,
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
            });
        }
        resolutions.push(J2kPacketResolution {
            subband_offset,
            subband_count: u32::try_from(resolution.subbands.len()).map_err(|_| {
                Error::MetalKernel {
                    message: format!(
                        "{}Tier-2 Metal resident packet resolution subband count exceeds u32",
                        T::TIER2_PREFIX
                    ),
                }
            })?,
        });
    }

    if resolutions.len()
        != usize::try_from(job.resolution_count).map_err(|_| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet resolution count exceeds usize",
                T::TIER2_PREFIX
            ),
        })?
    {
        return Err(Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet resolution count mismatch",
                T::TIER2_PREFIX
            ),
        });
    }
    if resident_blocks.len()
        != usize::try_from(job.code_block_count).map_err(|_| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet code-block count exceeds usize",
                T::TIER2_PREFIX
            ),
        })?
    {
        return Err(Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet code-block count mismatch",
                T::TIER2_PREFIX
            ),
        });
    }

    Ok(ResidentPacketTopology {
        resolutions,
        subbands,
        resident_blocks,
        max_tree_nodes,
    })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "single-pass descriptor construction keeps packet state offsets consistent"
)]
fn build_resident_packet_descriptors<T: ResidentLosslessTier1Metal>(
    job: J2kResidentPacketizationEncodeJob<'_>,
    resolutions: &[J2kPacketResolution],
    subbands: &[J2kPacketSubband],
    resident_blocks: &[J2kResidentPacketBlock],
    budget: &mut crate::batch_allocation::BatchMetadataBudget,
    counts: &ResidentPacketAllocationCounts,
) -> Result<ResidentPacketDescriptors, Error> {
    let mut state_block_offsets = budget.try_vec(
        counts.unique_states,
        "J2K Metal resident single packet state offsets",
    )?;
    let mut state_blocks = budget.try_vec(
        counts.state_blocks,
        "J2K Metal resident single packet state blocks",
    )?;
    let mut descriptors = budget.try_vec(
        counts.descriptors,
        "J2K Metal resident single packet descriptors",
    )?;
    for descriptor in job.packet_descriptors {
        let packet_index =
            usize::try_from(descriptor.packet_index).map_err(|_| Error::MetalKernel {
                message: format!(
                    "{}Tier-2 Metal resident packet descriptor packet index exceeds usize",
                    T::TIER2_PREFIX
                ),
            })?;
        let resolution = resolutions
            .get(packet_index)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "{}Tier-2 Metal resident packet descriptor packet index out of range",
                    T::TIER2_PREFIX
                ),
            })?;
        let subband_start =
            usize::try_from(resolution.subband_offset).map_err(|_| Error::MetalKernel {
                message: format!(
                    "{}Tier-2 Metal resident packet descriptor subband offset exceeds usize",
                    T::TIER2_PREFIX
                ),
            })?;
        let subband_count =
            usize::try_from(resolution.subband_count).map_err(|_| Error::MetalKernel {
                message: format!(
                    "{}Tier-2 Metal resident packet descriptor subband count exceeds usize",
                    T::TIER2_PREFIX
                ),
            })?;
        let subband_end =
            subband_start
                .checked_add(subband_count)
                .ok_or_else(|| Error::MetalKernel {
                    message: format!(
                        "{}Tier-2 Metal resident packet descriptor subband range overflow",
                        T::TIER2_PREFIX
                    ),
                })?;
        if subband_end > subbands.len() {
            return Err(Error::MetalKernel {
                message: format!(
                    "{}Tier-2 Metal resident packet descriptor subband range out of bounds",
                    T::TIER2_PREFIX
                ),
            });
        }
        let mut packet_block_count = 0usize;
        for subband in &subbands[subband_start..subband_end] {
            packet_block_count = packet_block_count
                .checked_add(usize::try_from(subband.block_count).map_err(|_| {
                    Error::MetalKernel {
                        message: format!(
                            "{}Tier-2 Metal resident packet descriptor block count exceeds usize",
                            T::TIER2_PREFIX
                        ),
                    }
                })?)
                .ok_or_else(|| Error::MetalKernel {
                    message: format!(
                        "{}Tier-2 Metal resident packet descriptor block count overflow",
                        T::TIER2_PREFIX
                    ),
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
                message: format!(
                    "{}Tier-2 Metal resident packet state block offset exceeds u32",
                    T::TIER2_PREFIX
                ),
            })?;
            for subband in &subbands[subband_start..subband_end] {
                let block_start =
                    usize::try_from(subband.block_offset).map_err(|_| Error::MetalKernel {
                        message: format!(
                            "{}Tier-2 Metal resident packet state block offset exceeds usize",
                            T::TIER2_PREFIX
                        ),
                    })?;
                let block_count =
                    usize::try_from(subband.block_count).map_err(|_| Error::MetalKernel {
                        message: format!(
                            "{}Tier-2 Metal resident packet state block count exceeds usize",
                            T::TIER2_PREFIX
                        ),
                    })?;
                let block_end =
                    block_start
                        .checked_add(block_count)
                        .ok_or_else(|| Error::MetalKernel {
                            message: format!(
                                "{}Tier-2 Metal resident packet state block range overflow",
                                T::TIER2_PREFIX
                            ),
                        })?;
                if block_end > resident_blocks.len() {
                    return Err(Error::MetalKernel {
                        message: format!(
                            "{}Tier-2 Metal resident packet state block range out of bounds",
                            T::TIER2_PREFIX
                        ),
                    });
                }
                for block in &resident_blocks[block_start..block_end] {
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
                message: format!(
                    "{}Tier-2 Metal resident packet descriptor state layout mismatch",
                    T::TIER2_PREFIX
                ),
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

    Ok(ResidentPacketDescriptors {
        descriptors,
        state_blocks,
    })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "resident planning keeps derived capacities and buffer layouts co-located"
)]
fn plan_resident_single<T: ResidentLosslessTier1Metal>(
    tier1: &T,
    job: J2kResidentPacketizationEncodeJob<'_>,
    codestream_job: J2kLosslessCodestreamAssemblyJob,
) -> Result<ResidentSinglePlan, Error> {
    let counts = resident_packet_allocation_counts(T::TIER2_PREFIX, job)?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal resident single packet metadata",
    );
    budget.preflight(&resident_packet_allocation_requests(&counts))?;
    let ResidentPacketTopology {
        resolutions,
        subbands,
        resident_blocks,
        max_tree_nodes,
    } = build_resident_packet_topology(tier1, job, &mut budget, &counts)?;
    let ResidentPacketDescriptors {
        descriptors,
        state_blocks,
    } = build_resident_packet_descriptors::<T>(
        job,
        &resolutions,
        &subbands,
        &resident_blocks,
        &mut budget,
        &counts,
    )?;

    let header_capacity = resident_blocks
        .len()
        .checked_mul(256)
        .and_then(|bytes| bytes.checked_add(4096))
        .map(|bytes| bytes.max(4096))
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet header capacity overflow",
                T::TIER2_PREFIX
            ),
        })?;
    let output_capacity = tier1
        .output_capacity_total()
        .checked_add(
            header_capacity
                .checked_mul(descriptors.len().max(resolutions.len()).max(1))
                .ok_or_else(|| Error::MetalKernel {
                    message: format!(
                        "{}Tier-2 Metal resident packet output capacity overflow",
                        T::TIER2_PREFIX
                    ),
                })?,
        )
        .and_then(|bytes| bytes.checked_add(1024))
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet output capacity overflow",
                T::TIER2_PREFIX
            ),
        })?;
    let codestream_capacity =
        lossless_codestream_assembly_capacity(output_capacity, codestream_job)?;

    let params = J2kPacketEncodeParams {
        resolution_count: u32::try_from(resolutions.len()).map_err(|_| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet resolution count exceeds u32",
                T::TIER2_PREFIX
            ),
        })?,
        num_layers: u32::from(job.num_layers),
        num_components: u32::from(job.component_count),
        code_block_count: u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet code-block count exceeds u32",
                T::TIER2_PREFIX
            ),
        })?,
        subband_count: u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet subband count exceeds u32",
                T::TIER2_PREFIX
            ),
        })?,
        descriptor_count: u32::try_from(descriptors.len()).map_err(|_| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet descriptor count exceeds u32",
                T::TIER2_PREFIX
            ),
        })?,
        output_capacity: u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet output capacity exceeds u32",
                T::TIER2_PREFIX
            ),
        })?,
        header_capacity: u32::try_from(header_capacity).map_err(|_| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet header capacity exceeds u32",
                T::TIER2_PREFIX
            ),
        })?,
        scratch_node_capacity: u32::try_from(max_tree_nodes).map_err(|_| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet scratch node capacity exceeds u32",
                T::TIER2_PREFIX
            ),
        })?,
    };
    let codestream_params = J2kLosslessCodestreamAssemblyParams {
        width: codestream_job.width,
        height: codestream_job.height,
        num_components: u32::from(codestream_job.component_count),
        bit_depth: u32::from(codestream_job.bit_depth),
        signed_samples: u32::from(codestream_job.signed),
        num_decomposition_levels: u32::from(codestream_job.num_decomposition_levels),
        use_mct: u32::from(codestream_job.use_mct),
        guard_bits: u32::from(codestream_job.guard_bits),
        progression_order: codestream_progression_order_code(codestream_job.progression_order),
        write_tlm: u32::from(codestream_job.write_tlm),
        high_throughput: u32::from(
            codestream_job.block_coding_mode
                == J2kLosslessCodestreamBlockCodingMode::HighThroughput,
        ),
        code_block_style: match codestream_job.block_coding_mode {
            J2kLosslessCodestreamBlockCodingMode::Classic => 0,
            J2kLosslessCodestreamBlockCodingMode::HighThroughput => 0x40,
        },
        code_block_width_exp: u32::from(codestream_job.code_block_width_exp),
        code_block_height_exp: u32::from(codestream_job.code_block_height_exp),
        output_capacity: u32::try_from(codestream_capacity).map_err(|_| Error::MetalKernel {
            message: format!(
                "{} Metal codestream assembly capacity exceeds u32",
                T::FAMILY_NAME
            ),
        })?,
    };

    let resident_block_params = J2kResidentPacketBlockParams {
        block_count: u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet block count exceeds u32",
                T::TIER2_PREFIX
            ),
        })?,
        tier1_job_count: u32::try_from(tier1.batch_job_count()).map_err(|_| {
            Error::MetalKernel {
                message: format!(
                    "{}Tier-2 Metal resident packet Tier-1 job count exceeds u32",
                    T::TIER2_PREFIX
                ),
            }
        })?,
    };

    Ok(ResidentSinglePlan {
        resolutions,
        subbands,
        resident_blocks,
        descriptors,
        state_blocks,
        max_tree_nodes,
        header_capacity,
        output_capacity,
        codestream_capacity,
        packet_params: params,
        codestream_params,
        resident_block_params,
    })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "resident submission preserves Metal ABI binding and command ordering"
)]
fn submit_resident_single_plan<T: ResidentLosslessTier1Metal>(
    runtime: &MetalRuntime,
    tier1: &T,
    plan: ResidentSinglePlan,
) -> Result<J2kPendingResidentLosslessCodestream, Error> {
    let ResidentSinglePlan {
        resolutions,
        subbands,
        resident_blocks,
        descriptors,
        state_blocks,
        max_tree_nodes,
        header_capacity,
        output_capacity,
        codestream_capacity,
        packet_params: params,
        codestream_params,
        resident_block_params,
    } = plan;
    let resolution_buffer = copied_slice_buffer(&runtime.device, &resolutions)?;
    let subband_buffer = copied_slice_buffer(&runtime.device, &subbands)?;
    let resident_block_buffer = copied_slice_buffer(&runtime.device, &resident_blocks)?;
    let packet_block_bytes = resident_blocks
        .len()
        .max(1)
        .checked_mul(size_of::<J2kPacketBlock>())
        .ok_or_else(|| Error::MetalKernel {
            message: "Tier-2 Metal resident packet block size overflow".to_string(),
        })?;
    let packet_block_buffer = new_private_buffer(&runtime.device, packet_block_bytes)?;
    let descriptor_buffer = copied_slice_buffer(&runtime.device, &descriptors)?;
    let state_block_buffer = copied_slice_buffer(&runtime.device, &state_blocks)?;
    let output_buffer = new_private_buffer(&runtime.device, output_capacity)?;
    let codestream_buffer = new_shared_buffer(&runtime.device, codestream_capacity)?;
    let header_buffer = new_private_buffer(&runtime.device, header_capacity)?;
    let scratch_words = max_tree_nodes
        .checked_mul(6)
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "{}Tier-2 Metal resident packet scratch size overflow",
                T::TIER2_PREFIX
            ),
        })?;
    let scratch_bytes =
        scratch_words
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "{}Tier-2 Metal resident packet scratch byte size overflow",
                    T::TIER2_PREFIX
                ),
            })?;
    let scratch_buffer = new_private_buffer(&runtime.device, scratch_bytes)?;
    let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kPacketEncodeStatus>())?;
    let codestream_status_buffer =
        zeroed_shared_buffer(&runtime.device, size_of::<J2kCodestreamAssemblyStatus>())?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    if !resident_blocks.is_empty() {
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(T::packet_block_prepare_pipeline(runtime));
        encoder.set_buffer(0, Some(&resident_block_buffer), 0);
        encoder.set_buffer(1, Some(tier1.job_buffer()), 0);
        encoder.set_buffer(2, Some(tier1.status_buffer()), 0);
        encoder.set_buffer(3, Some(&packet_block_buffer), 0);
        encoder.set_bytes(
            4,
            size_of::<J2kResidentPacketBlockParams>() as u64,
            (&raw const resident_block_params).cast(),
        );
        encoder.dispatch_threads(
            MTLSize {
                width: resident_blocks.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: T::packet_block_prepare_pipeline(runtime)
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
    }

    let encoder = new_compute_command_encoder(&command_buffer)?;
    encoder.set_compute_pipeline_state(&runtime.packet_encode);
    encoder.set_buffer(0, Some(&resolution_buffer), 0);
    encoder.set_buffer(1, Some(&subband_buffer), 0);
    encoder.set_buffer(2, Some(&packet_block_buffer), 0);
    encoder.set_buffer(3, Some(tier1.output_buffer()), 0);
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

    let encoder = new_compute_command_encoder(&command_buffer)?;
    encoder.set_compute_pipeline_state(&runtime.lossless_codestream_assemble);
    encoder.set_buffer(0, Some(&output_buffer), 0);
    encoder.set_buffer(1, Some(&status_buffer), 0);
    encoder.set_buffer(2, Some(&codestream_buffer), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kLosslessCodestreamAssemblyParams>() as u64,
        (&raw const codestream_params).cast(),
    );
    encoder.set_buffer(4, Some(&codestream_status_buffer), 0);
    dispatch_single_thread(&encoder);
    encoder.end_encoding();
    command_buffer.commit();

    Ok(J2kPendingResidentLosslessCodestream {
        buffer: codestream_buffer,
        capacity: codestream_capacity,
        status_buffer: codestream_status_buffer,
        command_buffer,
        retained_command_buffers: vec![
            tier1.prepare_command_buffer().clone(),
            tier1.tier1_command_buffer().clone(),
        ],
        _retained_buffers: vec![
            resolution_buffer,
            subband_buffer,
            resident_block_buffer,
            packet_block_buffer,
            descriptor_buffer,
            state_block_buffer,
            output_buffer,
            header_buffer,
            scratch_buffer,
            status_buffer,
            tier1.output_buffer().clone(),
            tier1.status_buffer().clone(),
            tier1.job_buffer().clone(),
        ],
        status_stage: T::CODESTREAM_STATUS_STAGE,
        length_error: T::CODESTREAM_LENGTH_ERROR,
        capacity_error: T::CODESTREAM_CAPACITY_ERROR,
    })
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use super::*;
    use crate::compute::{J2kResidentPacketizationResolution, J2kResidentPacketizationSubband};

    #[test]
    fn resident_single_metadata_plan_honors_exact_aggregate_cap() {
        let resolutions = [J2kResidentPacketizationResolution {
            subbands: vec![J2kResidentPacketizationSubband {
                code_block_start: 0,
                code_block_count: 1,
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        }];
        let descriptors = [
            j2k_native::J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 7,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
            j2k_native::J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 7,
                layer: 1,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ];
        let job = J2kResidentPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 2,
            component_count: 1,
            code_block_count: 1,
            packet_descriptors: &descriptors,
            resolutions: &resolutions,
        };
        let counts = resident_packet_allocation_counts("test ", job).expect("metadata counts");
        assert_eq!(counts.unique_states, 1);
        assert_eq!(counts.state_blocks, 1);
        let exact_cap = size_of::<J2kPacketResolution>()
            + size_of::<J2kPacketSubband>()
            + size_of::<J2kResidentPacketBlock>()
            + size_of::<(u32, u32, usize)>()
            + size_of::<J2kPacketStateBlock>()
            + 2 * size_of::<J2kPacketDescriptor>();
        let requests = resident_packet_allocation_requests(&counts);
        crate::batch_allocation::BatchMetadataBudget::with_cap(
            "J2K Metal resident single packet metadata",
            exact_cap,
        )
        .preflight(&requests)
        .expect("exact aggregate cap");
        assert!(matches!(
            crate::batch_allocation::BatchMetadataBudget::with_cap(
                "J2K Metal resident single packet metadata",
                exact_cap - 1,
            )
            .preflight(&requests),
            Err(j2k_core::BatchInfrastructureError::AllocationTooLarge {
                requested,
                cap,
                ..
            }) if requested == exact_cap && cap == exact_cap - 1
        ));
    }
}
