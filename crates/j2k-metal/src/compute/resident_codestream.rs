// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::test_counters;
use super::{
    borrow_slice_buffer, build_resident_batch_packet_plan, checked_buffer_read,
    checked_buffer_slice, classic_cod_block_style_from_flags, classic_encode_code_blocks_pipeline,
    classic_encode_output_capacity_for_mode, classic_encode_segment_capacity,
    classic_encode_sub_band_code, classic_packet_output_capacity,
    classic_resident_style_flags_from_env, classic_tier1_gpu_token_pack_requested,
    classic_tier1_gpu_token_pack_supported, classic_tier1_split_gpu_token_pack_requested,
    classic_tier1_split_mq_byte_gpu_token_pack_disabled,
    classic_tier1_split_mq_byte_gpu_token_pack_requested, codestream_progression_order_code,
    commit_and_wait_metal, copied_recyclable_shared_slice_buffer, copied_slice_buffer,
    decode_ht_status_error, dispatch_1d_pipeline, dispatch_batched_packet_payload_copy,
    dispatch_classic_tier1_arithmetic_pack_profile, dispatch_classic_tier1_density_profile,
    dispatch_classic_tier1_pass_plan_profile, dispatch_classic_tier1_raw_pack_profile,
    dispatch_classic_tier1_split_token_emit_for_gpu_pack,
    dispatch_classic_tier1_split_token_emit_profile,
    dispatch_classic_tier1_split_token_pack_from_gpu_tokens,
    dispatch_classic_tier1_symbol_plan_profile, dispatch_classic_tier1_token_emit_for_gpu_pack,
    dispatch_classic_tier1_token_emit_profile, dispatch_classic_tier1_token_pack_from_gpu_tokens,
    dispatch_single_thread, dispatch_zero_u32_buffer_in_encoder, encode_status_error,
    finish_resident_encode_split_command_buffer, finish_resident_encode_split_command_buffer_timed,
    ht_batch_output_word_count, ht_encode_output_capacity, ht_output_word_count,
    ht_packet_output_capacity_for_mode, hybrid_stage_signpost, label_command_buffer,
    label_compute_encoder, lossless_codestream_assembly_capacity,
    metal_profile_classic_tier1_arithmetic_pack_enabled,
    metal_profile_classic_tier1_density_enabled, metal_profile_classic_tier1_pass_plan_enabled,
    metal_profile_classic_tier1_raw_pack_enabled,
    metal_profile_classic_tier1_split_token_emit_enabled,
    metal_profile_classic_tier1_symbol_plan_enabled,
    metal_profile_classic_tier1_token_emit_enabled, metal_profile_stages_enabled,
    new_resident_encode_command_buffer, owned_slice_buffer, packet_tree_node_count,
    prepared_lossless_batch_tiles, schedule_classic_tier1_gpu_token_pack_readback,
    schedule_resident_tier1_status_readback, size_of, take_recyclable_private_buffer,
    wait_resident_lossless_codestream, with_runtime, with_runtime_for_session,
    zeroed_recyclable_shared_buffer, zeroed_shared_buffer, Buffer, CommandBufferRef,
    ComputeCommandEncoderRef, DirectStatusCheck, Duration, Error, ForeignType, HashMap, Instant,
    J2kBatchedPacketPayloadCopyDispatch, J2kClassicEncodeBatchJob,
    J2kClassicEncodeOutputCapacityMode, J2kClassicEncodeStatus, J2kClassicSegment,
    J2kCodestreamAssemblyStatus, J2kHtCleanupBatchJob, J2kHtCleanupParams, J2kHtEncodeBatchJob,
    J2kHtEncodeStatus, J2kHtPacketOutputCapacityMode, J2kHtRepeatedBatchParams, J2kHtStatus,
    J2kLosslessCodestreamAssemblyJob, J2kLosslessCodestreamAssemblyParams,
    J2kLosslessCodestreamBlockCodingMode, J2kPacketBlock, J2kPacketDescriptor,
    J2kPacketEncodeParams, J2kPacketEncodeStatus, J2kPacketPayloadCopyJob, J2kPacketResolution,
    J2kPacketStateBlock, J2kPacketSubband, J2kPacketizationBlockCodingMode,
    J2kPacketizationEncodeJob, J2kPendingResidentLosslessCodestream,
    J2kPendingResidentLosslessCodestreamBatch, J2kResidentBatchEncodeItem,
    J2kResidentEncodeGpuStage, J2kResidentEncodeGpuStageCommandBuffer, J2kResidentEncodeStageStats,
    J2kResidentLosslessCodestream, J2kResidentPacketBlock, J2kResidentPacketBlockParams,
    J2kResidentPacketizationEncodeJob, MTLResourceOptions, MTLSize, MetalRuntime,
    ResidentBatchPacketPlan, ResidentBatchPacketPlanParams, ResidentLosslessTier1Metal,
    ResidentTier1StatusReadbackRequest, J2K_ENCODE_STATUS_OK, J2K_HT_STATUS_OK,
    PACKET_PAYLOAD_COPY_STRIPES_PER_JOB,
    SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP, SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN,
    SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP,
    SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP, SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN,
    SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE, SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP,
};

mod classic_labels;
mod ht_cleanup;
use self::classic_labels::{
    next_enabled_classic_stage_label, CLASSIC_TIER1_ARITHMETIC_PACK_LABEL,
    CLASSIC_TIER1_DENSITY_LABEL, CLASSIC_TIER1_PASS_PLAN_LABEL, CLASSIC_TIER1_RAW_PACK_LABEL,
    CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL, CLASSIC_TIER1_SYMBOL_PLAN_LABEL,
    CLASSIC_TIER1_TOKEN_EMIT_LABEL,
};
pub(super) use self::ht_cleanup::*;

#[cfg(target_os = "macos")]
pub(crate) fn encode_tier2_packetization(
    job: J2kPacketizationEncodeJob<'_>,
) -> Result<Vec<u8>, Error> {
    with_runtime(|runtime| {
        let mut resolutions = Vec::<J2kPacketResolution>::new();
        let mut subbands = Vec::<J2kPacketSubband>::new();
        let mut blocks = Vec::<J2kPacketBlock>::new();
        let mut payload = Vec::<u8>::new();
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
                    let data_offset =
                        u32::try_from(payload.len()).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet payload exceeds u32".to_string(),
                        })?;
                    let data_len =
                        u32::try_from(code_block.data.len()).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet code-block payload exceeds u32"
                                .to_string(),
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
                            message: "Tier-2 Metal packet subband block count exceeds u32"
                                .to_string(),
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
                        message: "Tier-2 Metal packet resolution subband count exceeds u32"
                            .to_string(),
                    }
                })?,
            });
        }

        let mut state_block_offsets = HashMap::<u32, (u32, usize)>::new();
        let mut state_blocks = Vec::<J2kPacketStateBlock>::new();
        let mut descriptors =
            Vec::<J2kPacketDescriptor>::with_capacity(job.packet_descriptors.len());
        for descriptor in job.packet_descriptors {
            let packet_index =
                usize::try_from(descriptor.packet_index).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor packet index exceeds usize"
                        .to_string(),
                })?;
            let resolution = resolutions
                .get(packet_index)
                .ok_or_else(|| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor packet index out of range".to_string(),
                })?;
            let subband_start =
                usize::try_from(resolution.subband_offset).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor subband offset exceeds usize"
                        .to_string(),
                })?;
            let subband_count =
                usize::try_from(resolution.subband_count).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor subband count exceeds usize"
                        .to_string(),
                })?;
            let subband_end =
                subband_start
                    .checked_add(subband_count)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal packet descriptor subband range overflow"
                            .to_string(),
                    })?;
            if subband_end > subbands.len() {
                return Err(Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor subband range out of bounds"
                        .to_string(),
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

            let (state_block_offset, existing_count) = if let Some(&(offset, count)) =
                state_block_offsets.get(&descriptor.state_index)
            {
                (offset, count)
            } else {
                let offset = u32::try_from(state_blocks.len()).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet state block offset exceeds u32".to_string(),
                })?;
                for subband in &subbands[subband_start..subband_end] {
                    let block_start =
                        usize::try_from(subband.block_offset).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet state block offset exceeds usize"
                                .to_string(),
                        })?;
                    let block_count =
                        usize::try_from(subband.block_count).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet state block count exceeds usize"
                                .to_string(),
                        })?;
                    let block_end =
                        block_start
                            .checked_add(block_count)
                            .ok_or_else(|| Error::MetalKernel {
                                message: "Tier-2 Metal packet state block range overflow"
                                    .to_string(),
                            })?;
                    if block_end > blocks.len() {
                        return Err(Error::MetalKernel {
                            message: "Tier-2 Metal packet state block range out of bounds"
                                .to_string(),
                        });
                    }
                    for block in &blocks[block_start..block_end] {
                        state_blocks.push(J2kPacketStateBlock {
                            previously_included: block.previously_included,
                            l_block: block.l_block,
                        });
                    }
                }
                state_block_offsets.insert(descriptor.state_index, (offset, packet_block_count));
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
                precinct_lo: descriptor.precinct as u32,
                precinct_hi: (descriptor.precinct >> 32) as u32,
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
            scratch_node_capacity: u32::try_from(max_tree_nodes).map_err(|_| {
                Error::MetalKernel {
                    message: "Tier-2 Metal packet scratch node capacity exceeds u32".to_string(),
                }
            })?,
        };

        let resolution_buffer = copied_slice_buffer(&runtime.device, &resolutions);
        let subband_buffer = copied_slice_buffer(&runtime.device, &subbands);
        let block_buffer = copied_slice_buffer(&runtime.device, &blocks);
        let payload_buffer = copied_slice_buffer(&runtime.device, &payload);
        let descriptor_buffer = copied_slice_buffer(&runtime.device, &descriptors);
        let state_block_buffer = copied_slice_buffer(&runtime.device, &state_blocks);
        let output_buffer = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let header_buffer = runtime.device.new_buffer(
            header_capacity as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let scratch_words = max_tree_nodes
            .checked_mul(6)
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet scratch size overflow".to_string(),
            })?;
        let scratch_buffer = runtime.device.new_buffer(
            (scratch_words * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer =
            zeroed_shared_buffer(&runtime.device, size_of::<J2kPacketEncodeStatus>());

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
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
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

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
            checked_buffer_slice::<u8>(&output_buffer, data_len, "Tier-2 packet payload")?.to_vec()
        })
    })
}

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
        if tier1.batch_job_count() != tier1.code_block_count() {
            return Err(Error::MetalKernel {
                message: format!(
                    "{}Tier-2 Metal resident packetization Tier-1 table mismatch",
                    T::TIER2_PREFIX
                ),
            });
        }

        let mut resolutions = Vec::<J2kPacketResolution>::new();
        let mut subbands = Vec::<J2kPacketSubband>::new();
        let mut resident_blocks = Vec::<J2kResidentPacketBlock>::new();
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

        let mut state_block_offsets = HashMap::<u32, (u32, usize)>::new();
        let mut state_blocks = Vec::<J2kPacketStateBlock>::new();
        let mut descriptors =
            Vec::<J2kPacketDescriptor>::with_capacity(job.packet_descriptors.len());
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
                            message: format!("{}Tier-2 Metal resident packet descriptor block count exceeds usize", T::TIER2_PREFIX),
                        }
                    })?)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!("{}Tier-2 Metal resident packet descriptor block count overflow", T::TIER2_PREFIX),
                    })?;
            }

            let (state_block_offset, existing_count) = if let Some(&(offset, count)) =
                state_block_offsets.get(&descriptor.state_index)
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
                state_block_offsets.insert(descriptor.state_index, (offset, packet_block_count));
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
                precinct_lo: descriptor.precinct as u32,
                precinct_hi: (descriptor.precinct >> 32) as u32,
                state_block_offset,
            });
        }

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
            code_block_count: u32::try_from(resident_blocks.len()).map_err(|_| {
                Error::MetalKernel {
                    message: format!(
                        "{}Tier-2 Metal resident packet code-block count exceeds u32",
                        T::TIER2_PREFIX
                    ),
                }
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
            scratch_node_capacity: u32::try_from(max_tree_nodes).map_err(|_| {
                Error::MetalKernel {
                    message: format!(
                        "{}Tier-2 Metal resident packet scratch node capacity exceeds u32",
                        T::TIER2_PREFIX
                    ),
                }
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
            output_capacity: u32::try_from(codestream_capacity).map_err(|_| {
                Error::MetalKernel {
                    message: format!(
                        "{} Metal codestream assembly capacity exceeds u32",
                        T::FAMILY_NAME
                    ),
                }
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

        let resolution_buffer = copied_slice_buffer(&runtime.device, &resolutions);
        let subband_buffer = copied_slice_buffer(&runtime.device, &subbands);
        let resident_block_buffer = copied_slice_buffer(&runtime.device, &resident_blocks);
        let packet_block_buffer = runtime.device.new_buffer(
            (resident_blocks.len().max(1) * size_of::<J2kPacketBlock>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let descriptor_buffer = copied_slice_buffer(&runtime.device, &descriptors);
        let state_block_buffer = copied_slice_buffer(&runtime.device, &state_blocks);
        let output_buffer = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let codestream_buffer = runtime.device.new_buffer(
            codestream_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let header_buffer = runtime.device.new_buffer(
            header_capacity as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let scratch_words = max_tree_nodes
            .checked_mul(6)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "{}Tier-2 Metal resident packet scratch size overflow",
                    T::TIER2_PREFIX
                ),
            })?;
        let scratch_buffer = runtime.device.new_buffer(
            (scratch_words * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer =
            zeroed_shared_buffer(&runtime.device, size_of::<J2kPacketEncodeStatus>());
        let codestream_status_buffer =
            zeroed_shared_buffer(&runtime.device, size_of::<J2kCodestreamAssemblyStatus>());

        let command_buffer = runtime.queue.new_command_buffer();
        if !resident_blocks.is_empty() {
            let encoder = command_buffer.new_compute_command_encoder();
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

        let encoder = command_buffer.new_compute_command_encoder();
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
        dispatch_single_thread(encoder);
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
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
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        command_buffer.commit();

        Ok(J2kPendingResidentLosslessCodestream {
            buffer: codestream_buffer,
            capacity: codestream_capacity,
            status_buffer: codestream_status_buffer,
            command_buffer: command_buffer.to_owned(),
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
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn submit_lossless_codestream_buffers_from_prepared_ht_batch(
    session: &crate::MetalBackendSession,
    items: Vec<J2kResidentBatchEncodeItem>,
    packet_capacity_mode: J2kHtPacketOutputCapacityMode,
) -> Result<J2kPendingResidentLosslessCodestreamBatch, Error> {
    if items.is_empty() {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal resident batch encode requires at least one tile".to_string(),
        });
    }

    let prepared_tiles = prepared_lossless_batch_tiles(items);

    with_runtime_for_session(session, |runtime| {
        let profile_stages = metal_profile_stages_enabled();
        let mut stage_stats = J2kResidentEncodeStageStats::default();
        let mut ht_table_build_duration = Duration::ZERO;
        let mut ht_block_encode_duration = Duration::ZERO;
        let mut packet_block_prep_duration = Duration::ZERO;
        let mut packetization_duration = Duration::ZERO;
        let mut codestream_assembly_duration = Duration::ZERO;
        let mut ht_table_build_started = profile_stages.then(Instant::now);
        let ht_tier1_setup_signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP);
        let split_profile_commands = true;
        let mut retained_command_buffers = Vec::with_capacity(prepared_tiles.len());
        let mut gpu_stage_command_buffers = Vec::new();
        let mut retained_buffers = Vec::<Buffer>::new();
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let mut recyclable_shared_buffers = Vec::<(usize, Buffer)>::new();
        let shared_coefficient_buffer = prepared_tiles.first().and_then(|first| {
            let ptr = first.coefficient_buffer.as_ptr();
            prepared_tiles
                .iter()
                .all(|tile| {
                    tile.coefficient_buffer_is_batch_shared
                        && tile.coefficient_buffer.as_ptr() == ptr
                })
                .then(|| first.coefficient_buffer.clone())
        });
        let needs_coefficient_copy = shared_coefficient_buffer.is_none();
        let initial_command_buffer_label = if split_profile_commands && needs_coefficient_copy {
            "j2k htj2k resident coefficient copy"
        } else if split_profile_commands {
            "j2k htj2k resident tier1 encode"
        } else {
            "j2k htj2k resident encode batch"
        };
        let mut command_buffer =
            new_resident_encode_command_buffer(runtime, initial_command_buffer_label);
        let (coefficient_buffer, coefficient_offsets) = if let Some(coefficient_buffer) =
            shared_coefficient_buffer
        {
            (
                coefficient_buffer,
                prepared_tiles
                    .iter()
                    .map(|tile| tile.coefficient_byte_offset)
                    .collect::<Vec<_>>(),
            )
        } else {
            let mut coefficient_offsets = Vec::<usize>::with_capacity(prepared_tiles.len());
            let mut total_coefficient_bytes = 0usize;
            for tile in &prepared_tiles {
                coefficient_offsets.push(total_coefficient_bytes);
                total_coefficient_bytes = total_coefficient_bytes
                    .checked_add(tile.coefficient_byte_len)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch coefficient buffer size overflow".to_string(),
                    })?;
            }
            let coefficient_buffer = take_recyclable_private_buffer(
                runtime,
                total_coefficient_bytes.max(1),
                &mut recyclable_private_buffers,
            )?;
            let blit = command_buffer.new_blit_command_encoder();
            if metal_profile_stages_enabled() {
                blit.set_label("HTJ2K coefficient prep");
            }
            for (tile, &dst_offset) in prepared_tiles.iter().zip(coefficient_offsets.iter()) {
                if tile.coefficient_byte_len > 0 {
                    #[cfg(test)]
                    test_counters::record_ht_batch_coefficient_copy_blit();
                    blit.copy_from_buffer(
                        &tile.coefficient_buffer,
                        tile.coefficient_byte_offset as u64,
                        &coefficient_buffer,
                        dst_offset as u64,
                        tile.coefficient_byte_len as u64,
                    );
                }
            }
            blit.end_encoding();
            if split_profile_commands {
                command_buffer = finish_resident_encode_split_command_buffer(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::CoefficientCopy,
                    "j2k htj2k resident tier1 encode",
                    &mut gpu_stage_command_buffers,
                );
            }
            (coefficient_buffer, coefficient_offsets)
        };

        let mut tier1_jobs = Vec::<J2kHtEncodeBatchJob>::new();
        let mut tier1_output_capacity_total = 0usize;
        let mut max_tier1_output_capacity = 0usize;
        let mut tile_tier1_job_bases = Vec::<usize>::with_capacity(prepared_tiles.len());
        for (tile, &coefficient_byte_offset) in
            prepared_tiles.iter().zip(coefficient_offsets.iter())
        {
            tile_tier1_job_bases.push(tier1_jobs.len());
            let coefficient_word_offset = coefficient_byte_offset
                .checked_div(size_of::<i32>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch coefficient offset division failed".to_string(),
                })?;
            let coefficient_word_offset_u32 =
                u32::try_from(coefficient_word_offset).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal batch coefficient offset exceeds u32".to_string(),
                })?;
            for block in &tile.code_blocks {
                let output_capacity_per_job = ht_encode_output_capacity(block.width, block.height)?;
                max_tier1_output_capacity = max_tier1_output_capacity.max(output_capacity_per_job);
                let output_capacity_per_job_u32 =
                    u32::try_from(output_capacity_per_job).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal batch output capacity exceeds u32".to_string(),
                    })?;
                let output_offset =
                    u32::try_from(tier1_output_capacity_total).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal batch Tier-1 output offset exceeds u32".to_string(),
                    })?;
                let coefficient_offset = block
                    .coefficient_offset
                    .checked_add(coefficient_word_offset_u32)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch coefficient offset overflow".to_string(),
                    })?;
                tier1_jobs.push(J2kHtEncodeBatchJob {
                    coefficient_offset,
                    output_offset,
                    width: block.width,
                    height: block.height,
                    total_bitplanes: u32::from(block.total_bitplanes),
                    output_capacity: output_capacity_per_job_u32,
                });
                tier1_output_capacity_total = tier1_output_capacity_total
                    .checked_add(output_capacity_per_job)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch Tier-1 output buffer overflow".to_string(),
                    })?;
            }
        }

        let tier1_job_buffer = owned_slice_buffer(&runtime.device, &tier1_jobs);
        let tier1_output_buffer = take_recyclable_private_buffer(
            runtime,
            tier1_output_capacity_total.max(1),
            &mut recyclable_private_buffers,
        )?;
        let tier1_status_buffer = take_recyclable_private_buffer(
            runtime,
            tier1_jobs.len().max(1) * size_of::<J2kHtEncodeStatus>(),
            &mut recyclable_private_buffers,
        )?;
        let tier1_job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal batch Tier-1 job count exceeds u32".to_string(),
        })?;
        drop(ht_tier1_setup_signpost);
        if let Some(started) = ht_table_build_started.take() {
            ht_table_build_duration = ht_table_build_duration.saturating_add(started.elapsed());
        }
        if tier1_job_count > 0 {
            let command_encode_started = profile_stages.then(Instant::now);
            let signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE);
            let encoder = command_buffer.new_compute_command_encoder();
            label_compute_encoder(encoder, "HTJ2K Tier-1 encode");
            let pipeline = &runtime.ht_encode_code_blocks;
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&coefficient_buffer), 0);
            encoder.set_buffer(1, Some(&tier1_output_buffer), 0);
            encoder.set_buffer(2, Some(&tier1_job_buffer), 0);
            encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
            encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
            encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
            encoder.set_buffer(6, Some(&tier1_status_buffer), 0);
            encoder.set_bytes(
                7,
                size_of::<u32>() as u64,
                (&raw const tier1_job_count).cast(),
            );
            dispatch_1d_pipeline(encoder, pipeline, u64::from(tier1_job_count));
            encoder.end_encoding();
            drop(signpost);
            if let Some(started) = command_encode_started {
                ht_block_encode_duration =
                    ht_block_encode_duration.saturating_add(started.elapsed());
            }
            if split_profile_commands {
                command_buffer = finish_resident_encode_split_command_buffer(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::HtBlock,
                    "j2k htj2k resident packetization",
                    &mut gpu_stage_command_buffers,
                );
            }
        } else if split_profile_commands {
            label_command_buffer(&command_buffer, "j2k htj2k resident packetization");
        }

        ht_table_build_started = profile_stages.then(Instant::now);
        let ht_packet_plan_signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN);
        let ResidentBatchPacketPlan {
            packet_resolutions,
            packet_subbands,
            resident_blocks,
            packet_descriptors,
            state_blocks,
            packet_jobs,
            assembly_jobs,
            packet_output_capacity_total,
            packet_payload_copy_job_capacity_total,
            max_payload_copy_jobs_per_tile,
            header_capacity_total,
            scratch_words_total,
            codestream_capacity_total,
            codestream_offsets,
            codestream_capacities,
        } = build_resident_batch_packet_plan(
            &prepared_tiles,
            &tile_tier1_job_bases,
            ResidentBatchPacketPlanParams {
                family_name: "HTJ2K",
                block_coding_mode: 1,
                high_throughput: 1,
                code_block_style: 0x40,
            },
            |_tile_index, tile, header_capacity| {
                ht_packet_output_capacity_for_mode(
                    tile.code_blocks.len(),
                    header_capacity,
                    tile.packet_descriptors.len().max(tile.resolutions.len()),
                    tile.codestream,
                    packet_capacity_mode,
                )
            },
        )?;

        drop(ht_packet_plan_signpost);
        if let Some(started) = ht_table_build_started.take() {
            ht_table_build_duration = ht_table_build_duration.saturating_add(started.elapsed());
        }
        let ht_buffer_allocation_started = profile_stages.then(Instant::now);
        let ht_packet_buffer_setup_signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP);
        let packet_resolution_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &packet_resolutions,
            &mut recyclable_shared_buffers,
        )?;
        let packet_subband_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &packet_subbands,
            &mut recyclable_shared_buffers,
        )?;
        let resident_block_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &resident_blocks,
            &mut recyclable_shared_buffers,
        )?;
        let packet_block_buffer = take_recyclable_private_buffer(
            runtime,
            resident_blocks.len().max(1) * size_of::<J2kPacketBlock>(),
            &mut recyclable_private_buffers,
        )?;
        let packet_descriptor_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &packet_descriptors,
            &mut recyclable_shared_buffers,
        )?;
        let state_block_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &state_blocks,
            &mut recyclable_shared_buffers,
        )?;
        let packet_payload_copy_job_buffer = take_recyclable_private_buffer(
            runtime,
            packet_payload_copy_job_capacity_total
                .max(1)
                .checked_mul(size_of::<J2kPacketPayloadCopyJob>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch packet payload-copy buffer size overflow"
                        .to_string(),
                })?,
            &mut recyclable_private_buffers,
        )?;
        let header_buffer = take_recyclable_private_buffer(
            runtime,
            header_capacity_total.max(1),
            &mut recyclable_private_buffers,
        )?;
        let scratch_buffer = take_recyclable_private_buffer(
            runtime,
            scratch_words_total.max(1) * size_of::<u32>(),
            &mut recyclable_private_buffers,
        )?;
        let packet_job_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &packet_jobs,
            &mut recyclable_shared_buffers,
        )?;
        let packet_status_buffer = zeroed_recyclable_shared_buffer(
            runtime,
            packet_jobs.len().max(1) * size_of::<J2kPacketEncodeStatus>(),
            &mut recyclable_shared_buffers,
        )?;
        let codestream_job_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &assembly_jobs,
            &mut recyclable_shared_buffers,
        )?;
        let codestream_buffer = runtime.device.new_buffer(
            codestream_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let codestream_status_buffer = zeroed_recyclable_shared_buffer(
            runtime,
            assembly_jobs.len() * size_of::<J2kCodestreamAssemblyStatus>(),
            &mut recyclable_shared_buffers,
        )?;
        drop(ht_packet_buffer_setup_signpost);
        if let Some(started) = ht_buffer_allocation_started {
            stage_stats.ht_buffer_allocation_duration = started.elapsed();
        }

        let resident_block_params = J2kResidentPacketBlockParams {
            block_count: u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal batch resident block count exceeds u32".to_string(),
            })?,
            tier1_job_count,
        };

        let tile_count = u64::try_from(packet_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal batch tile count exceeds u64".to_string(),
        })?;
        if !resident_blocks.is_empty() {
            let command_encode_started = profile_stages.then(Instant::now);
            let signpost =
                hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE);
            let encoder = command_buffer.new_compute_command_encoder();
            label_compute_encoder(encoder, "HTJ2K packet block prep");
            encoder.set_compute_pipeline_state(&runtime.packet_block_prepare_resident_ht);
            encoder.set_buffer(0, Some(&resident_block_buffer), 0);
            encoder.set_buffer(1, Some(&tier1_job_buffer), 0);
            encoder.set_buffer(2, Some(&tier1_status_buffer), 0);
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
                    width: runtime
                        .packet_block_prepare_resident_ht
                        .thread_execution_width()
                        .max(1),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            drop(signpost);
            if let Some(started) = command_encode_started {
                packet_block_prep_duration =
                    packet_block_prep_duration.saturating_add(started.elapsed());
            }
            if split_profile_commands {
                command_buffer = finish_resident_encode_split_command_buffer(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::PacketBlockPrep,
                    "j2k htj2k resident packetization",
                    &mut gpu_stage_command_buffers,
                );
            }
        } else if split_profile_commands {
            label_command_buffer(&command_buffer, "j2k htj2k resident packetization");
        }
        let command_encode_started = profile_stages.then(Instant::now);
        let signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE);
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "HTJ2K packetization");
        encoder.set_compute_pipeline_state(&runtime.packet_encode_batched);
        encoder.set_buffer(0, Some(&packet_resolution_buffer), 0);
        encoder.set_buffer(1, Some(&packet_subband_buffer), 0);
        encoder.set_buffer(2, Some(&packet_block_buffer), 0);
        encoder.set_buffer(3, Some(&tier1_output_buffer), 0);
        encoder.set_buffer(4, Some(&codestream_buffer), 0);
        encoder.set_buffer(5, Some(&header_buffer), 0);
        encoder.set_buffer(6, Some(&scratch_buffer), 0);
        encoder.set_buffer(7, Some(&packet_job_buffer), 0);
        encoder.set_buffer(8, Some(&packet_status_buffer), 0);
        encoder.set_buffer(9, Some(&packet_descriptor_buffer), 0);
        encoder.set_buffer(10, Some(&state_block_buffer), 0);
        encoder.set_buffer(11, Some(&packet_payload_copy_job_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: tile_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: runtime
                    .packet_encode_batched
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        drop(signpost);
        if let Some(started) = command_encode_started {
            packetization_duration = packetization_duration.saturating_add(started.elapsed());
        }
        if split_profile_commands {
            command_buffer = finish_resident_encode_split_command_buffer(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::Packetization,
                "j2k htj2k resident packet payload copy",
                &mut gpu_stage_command_buffers,
            );
        }
        let packet_payload_copy_dispatched = dispatch_batched_packet_payload_copy(
            runtime,
            &command_buffer,
            J2kBatchedPacketPayloadCopyDispatch {
                payload_buffer: &tier1_output_buffer,
                packet_output_buffer: &codestream_buffer,
                packet_job_buffer: &packet_job_buffer,
                packet_status_buffer: &packet_status_buffer,
                packet_payload_copy_job_buffer: &packet_payload_copy_job_buffer,
                tile_count,
                max_payload_copy_jobs_per_tile: max_payload_copy_jobs_per_tile as u64,
                label: "HTJ2K packetization payload copy",
                signpost_name: SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE,
            },
        );
        if split_profile_commands {
            if packet_payload_copy_dispatched {
                command_buffer = finish_resident_encode_split_command_buffer(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::PacketPayloadCopy,
                    "j2k htj2k resident codestream assembly",
                    &mut gpu_stage_command_buffers,
                );
            } else {
                label_command_buffer(&command_buffer, "j2k htj2k resident codestream assembly");
            }
        }

        let command_encode_started = profile_stages.then(Instant::now);
        let signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE);
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "HTJ2K codestream assembly");
        encoder.set_compute_pipeline_state(&runtime.lossless_codestream_assemble_batched);
        encoder.set_buffer(0, Some(&codestream_buffer), 0);
        encoder.set_buffer(1, Some(&packet_status_buffer), 0);
        encoder.set_buffer(2, Some(&codestream_buffer), 0);
        encoder.set_buffer(3, Some(&codestream_job_buffer), 0);
        encoder.set_buffer(4, Some(&codestream_status_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: tile_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: runtime
                    .lossless_codestream_assemble_batched
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        drop(signpost);
        let max_packet_output_capacity = packet_jobs
            .iter()
            .map(|job| job.output_capacity)
            .max()
            .unwrap_or(0);
        let max_packet_output_capacity_usize = usize::try_from(max_packet_output_capacity)
            .map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal batch max packet output capacity exceeds usize".to_string(),
            })?;
        if split_profile_commands {
            command_buffer = finish_resident_encode_split_command_buffer(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::CodestreamAssembly,
                "j2k htj2k resident result readback",
                &mut gpu_stage_command_buffers,
            );
        }
        let codestream_payload_copy_dispatched = false;
        if let Some(started) = command_encode_started {
            codestream_assembly_duration =
                codestream_assembly_duration.saturating_add(started.elapsed());
        }
        let tier1_status_readback = schedule_resident_tier1_status_readback(
            ResidentTier1StatusReadbackRequest::high_throughput(
                runtime,
                &command_buffer,
                &tier1_status_buffer,
                tier1_jobs.len(),
                profile_stages,
            ),
        )?;
        command_buffer.commit();
        if split_profile_commands && codestream_payload_copy_dispatched {
            gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                stage: J2kResidentEncodeGpuStage::CodestreamPayloadCopy,
                command_buffer: command_buffer.clone(),
            });
        }

        if profile_stages {
            let mut prepare_command_buffer_ptrs = Vec::new();
            for tile in &prepared_tiles {
                let mut pushed_split_prepare = false;
                for (stage, command_buffer) in [
                    (
                        J2kResidentEncodeGpuStage::CoefficientDeinterleaveRct,
                        tile.prepare_deinterleave_rct_command_buffer.as_ref(),
                    ),
                    (
                        J2kResidentEncodeGpuStage::CoefficientDwt53,
                        tile.prepare_dwt53_command_buffer.as_ref(),
                    ),
                    (
                        J2kResidentEncodeGpuStage::CoefficientExtract,
                        tile.prepare_coefficient_extract_command_buffer.as_ref(),
                    ),
                ] {
                    if let Some(command_buffer) = command_buffer {
                        let ptr = command_buffer.as_ptr();
                        if prepare_command_buffer_ptrs.contains(&ptr) {
                            continue;
                        }
                        prepare_command_buffer_ptrs.push(ptr);
                        pushed_split_prepare = true;
                        gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                            stage,
                            command_buffer: command_buffer.clone(),
                        });
                    }
                }
                for command_buffer in &tile.prepare_dwt53_vertical_command_buffers {
                    let ptr = command_buffer.as_ptr();
                    if prepare_command_buffer_ptrs.contains(&ptr) {
                        continue;
                    }
                    prepare_command_buffer_ptrs.push(ptr);
                    pushed_split_prepare = true;
                    gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                        stage: J2kResidentEncodeGpuStage::CoefficientDwt53Vertical,
                        command_buffer: command_buffer.clone(),
                    });
                }
                for command_buffer in &tile.prepare_dwt53_horizontal_command_buffers {
                    let ptr = command_buffer.as_ptr();
                    if prepare_command_buffer_ptrs.contains(&ptr) {
                        continue;
                    }
                    prepare_command_buffer_ptrs.push(ptr);
                    pushed_split_prepare = true;
                    gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                        stage: J2kResidentEncodeGpuStage::CoefficientDwt53Horizontal,
                        command_buffer: command_buffer.clone(),
                    });
                }
                if pushed_split_prepare {
                    continue;
                }
                let ptr = tile.prepare_command_buffer.as_ptr();
                if prepare_command_buffer_ptrs.contains(&ptr) {
                    continue;
                }
                prepare_command_buffer_ptrs.push(ptr);
                gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                    stage: J2kResidentEncodeGpuStage::CoefficientPrep,
                    command_buffer: tile.prepare_command_buffer.clone(),
                });
            }
        }

        retained_command_buffers.extend(
            gpu_stage_command_buffers
                .iter()
                .map(|stage_command_buffer| stage_command_buffer.command_buffer.clone()),
        );
        for tile in prepared_tiles {
            if let Some(command_buffer) = tile.prepare_deinterleave_rct_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            if let Some(command_buffer) = tile.prepare_dwt53_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            retained_command_buffers.extend(tile.prepare_dwt53_vertical_command_buffers);
            retained_command_buffers.extend(tile.prepare_dwt53_horizontal_command_buffers);
            if let Some(command_buffer) = tile.prepare_coefficient_extract_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            retained_command_buffers.push(tile.prepare_command_buffer);
            retained_buffers.push(tile.coefficient_buffer);
            retained_buffers.push(tile.deinterleave_status_buffer);
            retained_buffers.extend(tile.plane_buffers);
            retained_buffers.extend(tile.scratch_buffers);
            retained_buffers.push(tile.coefficient_job_buffer);
            recyclable_private_buffers.extend(tile.recyclable_private_buffers);
        }
        retained_buffers.push(coefficient_buffer);
        retained_buffers.push(tier1_job_buffer);
        retained_buffers.push(tier1_output_buffer);
        retained_buffers.push(tier1_status_buffer);
        retained_buffers.push(packet_resolution_buffer);
        retained_buffers.push(packet_subband_buffer);
        retained_buffers.push(resident_block_buffer);
        retained_buffers.push(packet_block_buffer);
        retained_buffers.push(packet_descriptor_buffer);
        retained_buffers.push(state_block_buffer);
        retained_buffers.push(packet_payload_copy_job_buffer);
        retained_buffers.push(header_buffer);
        retained_buffers.push(scratch_buffer);
        retained_buffers.push(packet_job_buffer);
        retained_buffers.push(packet_status_buffer.clone());
        retained_buffers.push(codestream_job_buffer);

        stage_stats.ht_table_build_duration = ht_table_build_duration;
        stage_stats.ht_block_encode_duration = ht_block_encode_duration;
        stage_stats.packet_block_prep_duration = packet_block_prep_duration;
        stage_stats.packetization_duration = packetization_duration;
        stage_stats.codestream_assembly_duration = codestream_assembly_duration;
        stage_stats.ht_command_encode_duration = ht_block_encode_duration
            .saturating_add(packet_block_prep_duration)
            .saturating_add(packetization_duration)
            .saturating_add(codestream_assembly_duration);
        stage_stats.packet_payload_copy_job_capacity_total = packet_payload_copy_job_capacity_total;
        stage_stats.max_packet_payload_copy_jobs_per_tile = max_payload_copy_jobs_per_tile;
        stage_stats.packet_payload_copy_launched_stripe_count_total = packet_jobs
            .len()
            .saturating_mul(max_payload_copy_jobs_per_tile)
            .saturating_mul(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB as usize);
        stage_stats.tier1_output_capacity_total = tier1_output_capacity_total;
        stage_stats.max_tier1_output_capacity = max_tier1_output_capacity;
        stage_stats.packet_output_capacity_total = packet_output_capacity_total;
        stage_stats.max_packet_output_capacity = max_packet_output_capacity_usize;
        stage_stats.codestream_payload_copy_launched_thread_count_total = 0;
        stage_stats.code_block_count = tier1_jobs.len();

        Ok(J2kPendingResidentLosslessCodestreamBatch {
            runtime: session.runtime()?,
            buffer: codestream_buffer,
            byte_offsets: codestream_offsets,
            capacities: codestream_capacities,
            status_buffer: codestream_status_buffer,
            packet_status_buffer,
            tier1_status_readback,
            classic_tier1_density_readback: None,
            classic_tier1_symbol_plan_readback: None,
            classic_tier1_pass_plan_readback: None,
            classic_tier1_token_emit_readback: None,
            classic_tier1_split_token_emit_readback: None,
            classic_gpu_token_pack_used: false,
            command_buffer,
            retained_command_buffers,
            _retained_buffers: retained_buffers,
            recyclable_private_buffers,
            recyclable_shared_buffers,
            gpu_stage_command_buffers,
            stage_stats,
            codestream_payload_copy_dispatched,
            status_stage: "HTJ2K batched codestream assembly",
            length_error: "HTJ2K Metal batched codestream output length exceeds usize",
            capacity_error: "HTJ2K Metal batched codestream output length exceeds buffer",
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn submit_lossless_codestream_buffers_from_prepared_classic_batch(
    session: &crate::MetalBackendSession,
    items: Vec<J2kResidentBatchEncodeItem>,
    output_capacity_mode: J2kClassicEncodeOutputCapacityMode,
) -> Result<J2kPendingResidentLosslessCodestreamBatch, Error> {
    if items.is_empty() {
        return Err(Error::MetalKernel {
            message: "J2K Metal resident batch encode requires at least one tile".to_string(),
        });
    }

    let prepared_tiles = prepared_lossless_batch_tiles(items);

    with_runtime_for_session(session, |runtime| {
        let profile_stages = metal_profile_stages_enabled();
        // Commit classic stages independently so the long Tier-1 kernel can run
        // while CPU packet metadata for the following stages is built.
        let split_command_buffers = true;
        let mut stage_stats = J2kResidentEncodeStageStats::default();
        let mut classic_tier1_setup_duration = Duration::ZERO;
        let mut classic_block_encode_duration = Duration::ZERO;
        let mut classic_packet_plan_duration = Duration::ZERO;
        let mut classic_packet_buffer_setup_duration = Duration::ZERO;
        let mut classic_command_buffer_commit_duration = Duration::ZERO;
        let packet_block_prep_duration = Duration::ZERO;
        let mut packetization_duration = Duration::ZERO;
        let mut codestream_assembly_duration = Duration::ZERO;
        let mut retained_command_buffers = Vec::with_capacity(prepared_tiles.len());
        let mut gpu_stage_command_buffers = Vec::new();
        let mut retained_buffers = Vec::<Buffer>::new();
        let profile_classic_tier1_density = metal_profile_classic_tier1_density_enabled();
        let profile_classic_tier1_raw_pack = metal_profile_classic_tier1_raw_pack_enabled();
        let profile_classic_tier1_arithmetic_pack =
            metal_profile_classic_tier1_arithmetic_pack_enabled();
        let profile_classic_tier1_pass_plan = metal_profile_classic_tier1_pass_plan_enabled();
        let profile_classic_tier1_symbol_plan = metal_profile_classic_tier1_symbol_plan_enabled();
        let profile_classic_tier1_token_emit = metal_profile_classic_tier1_token_emit_enabled();
        let profile_classic_tier1_split_token_emit =
            metal_profile_classic_tier1_split_token_emit_enabled();
        let classic_token_pack_next_label = next_enabled_classic_stage_label(&[
            (profile_classic_tier1_density, CLASSIC_TIER1_DENSITY_LABEL),
            (profile_classic_tier1_raw_pack, CLASSIC_TIER1_RAW_PACK_LABEL),
            (
                profile_classic_tier1_arithmetic_pack,
                CLASSIC_TIER1_ARITHMETIC_PACK_LABEL,
            ),
            (
                profile_classic_tier1_symbol_plan,
                CLASSIC_TIER1_SYMBOL_PLAN_LABEL,
            ),
            (
                profile_classic_tier1_token_emit,
                CLASSIC_TIER1_TOKEN_EMIT_LABEL,
            ),
            (
                profile_classic_tier1_split_token_emit,
                CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
            ),
        ]);
        let shared_coefficient_buffer = prepared_tiles.first().and_then(|first| {
            let ptr = first.coefficient_buffer.as_ptr();
            prepared_tiles
                .iter()
                .all(|tile| {
                    tile.coefficient_buffer_is_batch_shared
                        && tile.coefficient_buffer.as_ptr() == ptr
                })
                .then(|| first.coefficient_buffer.clone())
        });
        let needs_coefficient_copy = shared_coefficient_buffer.is_none();
        let initial_command_buffer_label = if split_command_buffers && needs_coefficient_copy {
            "j2k classic resident coefficient copy"
        } else if split_command_buffers {
            "j2k classic resident Tier-1 encode"
        } else {
            "j2k classic resident encode batch"
        };
        let mut command_buffer =
            new_resident_encode_command_buffer(runtime, initial_command_buffer_label);
        let (coefficient_buffer, coefficient_offsets) =
            if let Some(coefficient_buffer) = shared_coefficient_buffer {
                (
                    coefficient_buffer,
                    prepared_tiles
                        .iter()
                        .map(|tile| tile.coefficient_byte_offset)
                        .collect::<Vec<_>>(),
                )
            } else {
                let mut coefficient_offsets = Vec::<usize>::with_capacity(prepared_tiles.len());
                let mut total_coefficient_bytes = 0usize;
                for tile in &prepared_tiles {
                    coefficient_offsets.push(total_coefficient_bytes);
                    total_coefficient_bytes = total_coefficient_bytes
                        .checked_add(tile.coefficient_byte_len)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "J2K Metal batch coefficient buffer size overflow".to_string(),
                        })?;
                }
                let coefficient_buffer = runtime.device.new_buffer(
                    total_coefficient_bytes.max(1) as u64,
                    MTLResourceOptions::StorageModePrivate,
                );
                let blit = command_buffer.new_blit_command_encoder();
                if profile_stages {
                    blit.set_label("J2K coefficient prep");
                }
                for (tile, &dst_offset) in prepared_tiles.iter().zip(coefficient_offsets.iter()) {
                    if tile.coefficient_byte_len > 0 {
                        blit.copy_from_buffer(
                            &tile.coefficient_buffer,
                            tile.coefficient_byte_offset as u64,
                            &coefficient_buffer,
                            dst_offset as u64,
                            tile.coefficient_byte_len as u64,
                        );
                    }
                }
                blit.end_encoding();
                if split_command_buffers {
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::CoefficientCopy,
                        "j2k classic resident Tier-1 encode",
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
                (coefficient_buffer, coefficient_offsets)
            };

        let classic_tier1_setup_started = profile_stages.then(Instant::now);
        let classic_tier1_setup_signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP);
        let classic_resident_style_flags = classic_resident_style_flags_from_env();
        let mut tier1_jobs = Vec::<J2kClassicEncodeBatchJob>::new();
        let mut tier1_output_capacity_total = 0usize;
        let mut max_tier1_output_capacity = 0usize;
        let mut tier1_segment_capacity_total = 0usize;
        let mut tile_tier1_job_bases = Vec::<usize>::with_capacity(prepared_tiles.len());
        let mut tile_tier1_output_capacities = Vec::<usize>::with_capacity(prepared_tiles.len());
        for (tile, &coefficient_byte_offset) in
            prepared_tiles.iter().zip(coefficient_offsets.iter())
        {
            tile_tier1_job_bases.push(tier1_jobs.len());
            let tile_output_start = tier1_output_capacity_total;
            let coefficient_word_offset = coefficient_byte_offset
                .checked_div(size_of::<i32>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch coefficient offset division failed".to_string(),
                })?;
            let coefficient_word_offset_u32 =
                u32::try_from(coefficient_word_offset).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal batch coefficient offset exceeds u32".to_string(),
                })?;
            for block in &tile.code_blocks {
                let output_capacity = classic_encode_output_capacity_for_mode(
                    block.width,
                    block.height,
                    block.total_bitplanes,
                    output_capacity_mode,
                )?;
                max_tier1_output_capacity = max_tier1_output_capacity.max(output_capacity);
                let output_offset =
                    u32::try_from(tier1_output_capacity_total).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal batch Tier-1 output offset exceeds u32".to_string(),
                    })?;
                let segment_offset = u32::try_from(tier1_segment_capacity_total).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch Tier-1 segment offset exceeds u32".to_string(),
                    }
                })?;
                let coefficient_offset = block
                    .coefficient_offset
                    .checked_add(coefficient_word_offset_u32)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch coefficient offset overflow".to_string(),
                    })?;
                let segment_capacity = classic_encode_segment_capacity(
                    classic_resident_style_flags,
                    block.total_bitplanes,
                );
                tier1_jobs.push(J2kClassicEncodeBatchJob {
                    coefficient_offset,
                    output_offset,
                    segment_offset,
                    width: block.width,
                    height: block.height,
                    sub_band_type: classic_encode_sub_band_code(block.sub_band_type),
                    total_bitplanes: u32::from(block.total_bitplanes),
                    style_flags: classic_resident_style_flags,
                    output_capacity: u32::try_from(output_capacity).map_err(|_| {
                        Error::MetalKernel {
                            message: "J2K Metal batch Tier-1 output capacity exceeds u32"
                                .to_string(),
                        }
                    })?,
                    segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                        Error::MetalKernel {
                            message: "J2K Metal batch Tier-1 segment capacity exceeds u32"
                                .to_string(),
                        }
                    })?,
                });
                tier1_output_capacity_total = tier1_output_capacity_total
                    .checked_add(output_capacity)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch Tier-1 output buffer overflow".to_string(),
                    })?;
                tier1_segment_capacity_total = tier1_segment_capacity_total
                    .checked_add(segment_capacity)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch Tier-1 segment buffer overflow".to_string(),
                    })?;
            }
            tile_tier1_output_capacities.push(
                tier1_output_capacity_total
                    .checked_sub(tile_output_start)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch tile Tier-1 capacity underflow".to_string(),
                    })?,
            );
        }

        let tier1_job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "J2K Metal batch Tier-1 job count exceeds u32".to_string(),
        })?;
        let tier1_job_buffer = owned_slice_buffer(&runtime.device, &tier1_jobs);
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let mut recyclable_shared_buffers = Vec::<(usize, Buffer)>::new();
        let tier1_output_buffer = take_recyclable_private_buffer(
            runtime,
            tier1_output_capacity_total.max(1),
            &mut recyclable_private_buffers,
        )?;
        let tier1_status_buffer = take_recyclable_private_buffer(
            runtime,
            tier1_jobs.len().max(1) * size_of::<J2kClassicEncodeStatus>(),
            &mut recyclable_private_buffers,
        )?;
        let tier1_segment_buffer = take_recyclable_private_buffer(
            runtime,
            tier1_segment_capacity_total.max(1) * size_of::<J2kClassicSegment>(),
            &mut recyclable_private_buffers,
        )?;
        drop(classic_tier1_setup_signpost);
        if let Some(started) = classic_tier1_setup_started {
            classic_tier1_setup_duration = started.elapsed();
        }
        let classic_split_mq_byte_gpu_token_pack_requested =
            classic_tier1_split_mq_byte_gpu_token_pack_requested();
        let classic_split_mq_byte_gpu_token_pack_disabled =
            classic_tier1_split_mq_byte_gpu_token_pack_disabled();
        let classic_split_gpu_token_pack_requested = classic_tier1_split_gpu_token_pack_requested();
        let classic_gpu_token_pack_requested = classic_tier1_gpu_token_pack_requested();
        let use_classic_split_mq_byte_gpu_token_pack = if tier1_job_count > 0 {
            if classic_split_mq_byte_gpu_token_pack_requested {
                if !classic_tier1_gpu_token_pack_supported(&tier1_jobs) {
                    return Err(Error::MetalKernel {
                        message: "J2K Metal classic split MQ-byte GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
                    });
                }
                true
            } else {
                !classic_split_mq_byte_gpu_token_pack_disabled
                    && !classic_split_gpu_token_pack_requested
                    && !classic_gpu_token_pack_requested
                    && classic_tier1_gpu_token_pack_supported(&tier1_jobs)
            }
        } else {
            false
        };
        let use_classic_split_gpu_token_pack = if classic_split_gpu_token_pack_requested
            && !use_classic_split_mq_byte_gpu_token_pack
            && tier1_job_count > 0
        {
            if !classic_tier1_gpu_token_pack_supported(&tier1_jobs) {
                return Err(Error::MetalKernel {
                    message: "J2K Metal classic split GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
                });
            }
            true
        } else {
            false
        };
        let use_classic_gpu_token_pack = if !use_classic_split_mq_byte_gpu_token_pack
            && !use_classic_split_gpu_token_pack
            && classic_gpu_token_pack_requested
            && tier1_job_count > 0
        {
            if !classic_tier1_gpu_token_pack_supported(&tier1_jobs) {
                return Err(Error::MetalKernel {
                    message: "J2K Metal classic GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
                });
            }
            true
        } else {
            false
        };
        let mut classic_gpu_token_pack_readback = None;
        if tier1_job_count > 0 {
            let command_encode_started = profile_stages.then(Instant::now);
            let signpost =
                hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE);
            if use_classic_split_mq_byte_gpu_token_pack || use_classic_split_gpu_token_pack {
                let token_buffers = dispatch_classic_tier1_split_token_emit_for_gpu_pack(
                    runtime,
                    &command_buffer,
                    &coefficient_buffer,
                    &tier1_job_buffer,
                    &tier1_jobs,
                    &mut recyclable_private_buffers,
                    use_classic_split_mq_byte_gpu_token_pack,
                )?;
                if split_command_buffers {
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::ClassicTier1SplitTokenEmit,
                        "j2k classic resident Tier-1 split token pack",
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
                dispatch_classic_tier1_split_token_pack_from_gpu_tokens(
                    runtime,
                    &command_buffer,
                    &tier1_job_buffer,
                    &token_buffers,
                    &tier1_output_buffer,
                    &tier1_status_buffer,
                    &tier1_segment_buffer,
                );
                drop(signpost);
                if let Some(started) = command_encode_started {
                    classic_block_encode_duration =
                        classic_block_encode_duration.saturating_add(started.elapsed());
                }
                if split_command_buffers {
                    let next_label = classic_token_pack_next_label;
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::ClassicTier1TokenPack,
                        next_label,
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
            } else if use_classic_gpu_token_pack {
                let token_buffers = dispatch_classic_tier1_token_emit_for_gpu_pack(
                    runtime,
                    &command_buffer,
                    &coefficient_buffer,
                    &tier1_job_buffer,
                    &tier1_jobs,
                    &mut recyclable_private_buffers,
                )?;
                if split_command_buffers {
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::ClassicTier1TokenEmit,
                        "j2k classic resident Tier-1 token pack",
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
                dispatch_classic_tier1_token_pack_from_gpu_tokens(
                    runtime,
                    &command_buffer,
                    &tier1_job_buffer,
                    &token_buffers,
                    &tier1_output_buffer,
                    &tier1_status_buffer,
                    &tier1_segment_buffer,
                );
                classic_gpu_token_pack_readback = schedule_classic_tier1_gpu_token_pack_readback(
                    runtime,
                    &command_buffer,
                    &token_buffers,
                    profile_stages,
                )?;
                drop(signpost);
                if let Some(started) = command_encode_started {
                    classic_block_encode_duration =
                        classic_block_encode_duration.saturating_add(started.elapsed());
                }
                if split_command_buffers {
                    let next_label = classic_token_pack_next_label;
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::ClassicTier1TokenPack,
                        next_label,
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
            } else {
                let encoder = command_buffer.new_compute_command_encoder();
                label_compute_encoder(encoder, "J2K Tier-1 encode");
                let classic_encode_pipeline =
                    classic_encode_code_blocks_pipeline(runtime, &tier1_jobs);
                encoder.set_compute_pipeline_state(classic_encode_pipeline);
                encoder.set_buffer(0, Some(&coefficient_buffer), 0);
                encoder.set_buffer(1, Some(&tier1_output_buffer), 0);
                encoder.set_buffer(2, Some(&tier1_job_buffer), 0);
                encoder.set_buffer(3, Some(&tier1_status_buffer), 0);
                encoder.set_buffer(4, Some(&tier1_segment_buffer), 0);
                encoder.set_bytes(
                    5,
                    size_of::<u32>() as u64,
                    (&raw const tier1_job_count).cast(),
                );
                dispatch_1d_pipeline(encoder, classic_encode_pipeline, u64::from(tier1_job_count));
                encoder.end_encoding();
                drop(signpost);
                if let Some(started) = command_encode_started {
                    classic_block_encode_duration =
                        classic_block_encode_duration.saturating_add(started.elapsed());
                }
                if split_command_buffers {
                    let next_label = classic_token_pack_next_label;
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::ClassicBlock,
                        next_label,
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
            }
        } else if split_command_buffers {
            label_command_buffer(&command_buffer, "j2k classic resident packetization");
        }
        let classic_tier1_density_readback = if tier1_job_count > 0 {
            let readback = dispatch_classic_tier1_density_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
            )?;
            if readback.is_some() && split_command_buffers {
                let next_label = next_enabled_classic_stage_label(&[
                    (profile_classic_tier1_raw_pack, CLASSIC_TIER1_RAW_PACK_LABEL),
                    (
                        profile_classic_tier1_arithmetic_pack,
                        CLASSIC_TIER1_ARITHMETIC_PACK_LABEL,
                    ),
                    (
                        profile_classic_tier1_symbol_plan,
                        CLASSIC_TIER1_SYMBOL_PLAN_LABEL,
                    ),
                    (
                        profile_classic_tier1_token_emit,
                        CLASSIC_TIER1_TOKEN_EMIT_LABEL,
                    ),
                    (
                        profile_classic_tier1_split_token_emit,
                        CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
                    ),
                ]);
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1Density,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            readback
        } else {
            None
        };
        let classic_tier1_raw_pack_buffer = if tier1_job_count > 0 {
            let buffer = dispatch_classic_tier1_raw_pack_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
                tier1_output_capacity_total,
            )?;
            if buffer.is_some() && split_command_buffers {
                let next_label = next_enabled_classic_stage_label(&[
                    (
                        profile_classic_tier1_arithmetic_pack,
                        CLASSIC_TIER1_ARITHMETIC_PACK_LABEL,
                    ),
                    (
                        profile_classic_tier1_symbol_plan,
                        CLASSIC_TIER1_SYMBOL_PLAN_LABEL,
                    ),
                    (
                        profile_classic_tier1_token_emit,
                        CLASSIC_TIER1_TOKEN_EMIT_LABEL,
                    ),
                    (
                        profile_classic_tier1_split_token_emit,
                        CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
                    ),
                ]);
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1RawPack,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            buffer
        } else {
            None
        };
        let classic_tier1_arithmetic_pack_buffer = if tier1_job_count > 0 {
            let buffer = dispatch_classic_tier1_arithmetic_pack_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
                tier1_output_capacity_total,
            )?;
            if buffer.is_some() && split_command_buffers {
                let next_label = next_enabled_classic_stage_label(&[
                    (
                        profile_classic_tier1_symbol_plan,
                        CLASSIC_TIER1_SYMBOL_PLAN_LABEL,
                    ),
                    (
                        profile_classic_tier1_token_emit,
                        CLASSIC_TIER1_TOKEN_EMIT_LABEL,
                    ),
                    (
                        profile_classic_tier1_split_token_emit,
                        CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
                    ),
                ]);
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1ArithmeticPack,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            buffer
        } else {
            None
        };
        let classic_tier1_symbol_plan_readback = if tier1_job_count > 0 {
            let readback = dispatch_classic_tier1_symbol_plan_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
            )?;
            if readback.is_some() && split_command_buffers {
                let next_label = next_enabled_classic_stage_label(&[
                    (
                        profile_classic_tier1_pass_plan,
                        CLASSIC_TIER1_PASS_PLAN_LABEL,
                    ),
                    (
                        profile_classic_tier1_token_emit,
                        CLASSIC_TIER1_TOKEN_EMIT_LABEL,
                    ),
                    (
                        profile_classic_tier1_split_token_emit,
                        CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
                    ),
                ]);
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1SymbolPlan,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            readback
        } else {
            None
        };
        let classic_tier1_pass_plan_readback = if tier1_job_count > 0 {
            let readback = dispatch_classic_tier1_pass_plan_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
            )?;
            if readback.is_some() && split_command_buffers {
                let next_label = next_enabled_classic_stage_label(&[
                    (
                        profile_classic_tier1_token_emit,
                        CLASSIC_TIER1_TOKEN_EMIT_LABEL,
                    ),
                    (
                        profile_classic_tier1_split_token_emit,
                        CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
                    ),
                ]);
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1PassPlan,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            readback
        } else {
            None
        };
        let classic_tier1_token_emit_readback = if classic_gpu_token_pack_readback.is_some() {
            classic_gpu_token_pack_readback
        } else if tier1_job_count > 0 {
            let readback = dispatch_classic_tier1_token_emit_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
            )?;
            if readback.is_some() && split_command_buffers {
                let next_label = next_enabled_classic_stage_label(&[(
                    profile_classic_tier1_split_token_emit,
                    CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
                )]);
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1TokenEmit,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            readback
        } else {
            None
        };
        let classic_tier1_split_token_emit_readback = if tier1_job_count > 0 {
            let readback = dispatch_classic_tier1_split_token_emit_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
            )?;
            if readback.is_some() && split_command_buffers {
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1SplitTokenEmit,
                    "j2k classic resident packetization",
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            readback
        } else {
            None
        };

        let classic_packet_plan_started = profile_stages.then(Instant::now);
        let classic_packet_plan_signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN);
        let ResidentBatchPacketPlan {
            packet_resolutions,
            packet_subbands,
            resident_blocks,
            packet_descriptors,
            state_blocks,
            packet_jobs,
            assembly_jobs,
            packet_output_capacity_total,
            packet_payload_copy_job_capacity_total,
            max_payload_copy_jobs_per_tile,
            header_capacity_total,
            scratch_words_total,
            codestream_capacity_total,
            codestream_offsets,
            codestream_capacities,
        } = build_resident_batch_packet_plan(
            &prepared_tiles,
            &tile_tier1_job_bases,
            ResidentBatchPacketPlanParams {
                family_name: "J2K",
                block_coding_mode: 0,
                high_throughput: 0,
                code_block_style: classic_cod_block_style_from_flags(classic_resident_style_flags),
            },
            |tile_index, tile, header_capacity| {
                classic_packet_output_capacity(
                    tile_tier1_output_capacities[tile_index],
                    header_capacity,
                    tile.packet_descriptors.len().max(tile.resolutions.len()),
                    tile.codestream,
                )
            },
        )?;
        drop(classic_packet_plan_signpost);
        if let Some(started) = classic_packet_plan_started {
            classic_packet_plan_duration = started.elapsed();
        }

        let classic_packet_buffer_setup_started = profile_stages.then(Instant::now);
        let classic_packet_buffer_setup_signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP);
        let packet_resolution_buffer = copied_slice_buffer(&runtime.device, &packet_resolutions);
        let packet_subband_buffer = copied_slice_buffer(&runtime.device, &packet_subbands);
        let resident_block_buffer = copied_slice_buffer(&runtime.device, &resident_blocks);
        let packet_descriptor_buffer = copied_slice_buffer(&runtime.device, &packet_descriptors);
        let state_block_buffer = copied_slice_buffer(&runtime.device, &state_blocks);
        let packet_payload_copy_job_buffer = take_recyclable_private_buffer(
            runtime,
            packet_payload_copy_job_capacity_total
                .max(1)
                .checked_mul(size_of::<J2kPacketPayloadCopyJob>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch packet payload-copy buffer size overflow".to_string(),
                })?,
            &mut recyclable_private_buffers,
        )?;
        let header_buffer = take_recyclable_private_buffer(
            runtime,
            header_capacity_total.max(1),
            &mut recyclable_private_buffers,
        )?;
        let scratch_buffer = take_recyclable_private_buffer(
            runtime,
            scratch_words_total.max(1) * size_of::<u32>(),
            &mut recyclable_private_buffers,
        )?;
        let packet_job_buffer = copied_slice_buffer(&runtime.device, &packet_jobs);
        let packet_status_buffer = zeroed_recyclable_shared_buffer(
            runtime,
            packet_jobs.len().max(1) * size_of::<J2kPacketEncodeStatus>(),
            &mut recyclable_shared_buffers,
        )?;
        let codestream_job_buffer = copied_slice_buffer(&runtime.device, &assembly_jobs);
        let codestream_buffer = runtime.device.new_buffer(
            codestream_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let codestream_status_buffer = zeroed_shared_buffer(
            &runtime.device,
            assembly_jobs.len() * size_of::<J2kCodestreamAssemblyStatus>(),
        );
        drop(classic_packet_buffer_setup_signpost);
        if let Some(started) = classic_packet_buffer_setup_started {
            classic_packet_buffer_setup_duration = started.elapsed();
        }

        let resident_block_params = J2kResidentPacketBlockParams {
            block_count: u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
                message: "J2K Metal batch resident block count exceeds u32".to_string(),
            })?,
            tier1_job_count,
        };

        let tile_count = u64::try_from(packet_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "J2K Metal batch tile count exceeds u64".to_string(),
        })?;
        let command_encode_started = profile_stages.then(Instant::now);
        let signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE);
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "J2K packetization");
        encoder.set_compute_pipeline_state(&runtime.packet_encode_resident_classic_batched);
        encoder.set_buffer(0, Some(&packet_resolution_buffer), 0);
        encoder.set_buffer(1, Some(&packet_subband_buffer), 0);
        encoder.set_buffer(2, Some(&resident_block_buffer), 0);
        encoder.set_buffer(3, Some(&tier1_output_buffer), 0);
        encoder.set_buffer(4, Some(&codestream_buffer), 0);
        encoder.set_buffer(5, Some(&header_buffer), 0);
        encoder.set_buffer(6, Some(&scratch_buffer), 0);
        encoder.set_buffer(7, Some(&packet_job_buffer), 0);
        encoder.set_buffer(8, Some(&packet_status_buffer), 0);
        encoder.set_buffer(9, Some(&packet_descriptor_buffer), 0);
        encoder.set_buffer(10, Some(&state_block_buffer), 0);
        encoder.set_buffer(11, Some(&packet_payload_copy_job_buffer), 0);
        encoder.set_buffer(12, Some(&tier1_job_buffer), 0);
        encoder.set_buffer(13, Some(&tier1_status_buffer), 0);
        encoder.set_buffer(14, Some(&tier1_segment_buffer), 0);
        encoder.set_bytes(
            15,
            size_of::<J2kResidentPacketBlockParams>() as u64,
            (&raw const resident_block_params).cast(),
        );
        encoder.dispatch_threads(
            MTLSize {
                width: tile_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: runtime
                    .packet_encode_resident_classic_batched
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        drop(signpost);
        if let Some(started) = command_encode_started {
            packetization_duration = packetization_duration.saturating_add(started.elapsed());
        }
        if split_command_buffers {
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::Packetization,
                "j2k classic resident packet payload copy",
                &mut gpu_stage_command_buffers,
                profile_stages,
                &mut classic_command_buffer_commit_duration,
            );
        }
        let packet_payload_copy_dispatched = dispatch_batched_packet_payload_copy(
            runtime,
            &command_buffer,
            J2kBatchedPacketPayloadCopyDispatch {
                payload_buffer: &tier1_output_buffer,
                packet_output_buffer: &codestream_buffer,
                packet_job_buffer: &packet_job_buffer,
                packet_status_buffer: &packet_status_buffer,
                packet_payload_copy_job_buffer: &packet_payload_copy_job_buffer,
                tile_count,
                max_payload_copy_jobs_per_tile: max_payload_copy_jobs_per_tile as u64,
                label: "J2K packetization payload copy",
                signpost_name: SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE,
            },
        );
        if split_command_buffers {
            if packet_payload_copy_dispatched {
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::PacketPayloadCopy,
                    "j2k classic resident codestream assembly",
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            } else {
                label_command_buffer(&command_buffer, "j2k classic resident codestream assembly");
            }
        }

        let max_packet_output_capacity = packet_jobs
            .iter()
            .map(|job| job.output_capacity)
            .max()
            .unwrap_or(0);
        let max_packet_output_capacity_usize = usize::try_from(max_packet_output_capacity)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal batch max packet output capacity exceeds usize".to_string(),
            })?;
        let command_encode_started = profile_stages.then(Instant::now);
        let signpost = hybrid_stage_signpost(
            SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
        );
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "J2K codestream assembly");
        encoder.set_compute_pipeline_state(&runtime.lossless_codestream_assemble_batched);
        encoder.set_buffer(0, Some(&codestream_buffer), 0);
        encoder.set_buffer(1, Some(&packet_status_buffer), 0);
        encoder.set_buffer(2, Some(&codestream_buffer), 0);
        encoder.set_buffer(3, Some(&codestream_job_buffer), 0);
        encoder.set_buffer(4, Some(&codestream_status_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: tile_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: runtime
                    .lossless_codestream_assemble_batched
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        drop(signpost);
        if split_command_buffers {
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::CodestreamAssembly,
                "j2k classic resident result readback",
                &mut gpu_stage_command_buffers,
                profile_stages,
                &mut classic_command_buffer_commit_duration,
            );
        }
        let codestream_payload_copy_dispatched = false;
        if let Some(started) = command_encode_started {
            codestream_assembly_duration =
                codestream_assembly_duration.saturating_add(started.elapsed());
        }
        let tier1_status_readback =
            schedule_resident_tier1_status_readback(ResidentTier1StatusReadbackRequest::classic(
                runtime,
                &command_buffer,
                &tier1_status_buffer,
                classic_resident_style_flags,
                &tier1_jobs,
                profile_stages,
            ))?;
        let final_commit_started = profile_stages.then(Instant::now);
        command_buffer.commit();
        if let Some(started) = final_commit_started {
            classic_command_buffer_commit_duration =
                classic_command_buffer_commit_duration.saturating_add(started.elapsed());
        }
        if split_command_buffers && codestream_payload_copy_dispatched {
            gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                stage: J2kResidentEncodeGpuStage::CodestreamPayloadCopy,
                command_buffer: command_buffer.clone(),
            });
        }

        if profile_stages {
            let mut prepare_command_buffer_ptrs = Vec::new();
            for tile in &prepared_tiles {
                let mut pushed_split_prepare = false;
                for (stage, command_buffer) in [
                    (
                        J2kResidentEncodeGpuStage::CoefficientDeinterleaveRct,
                        tile.prepare_deinterleave_rct_command_buffer.as_ref(),
                    ),
                    (
                        J2kResidentEncodeGpuStage::CoefficientDwt53,
                        tile.prepare_dwt53_command_buffer.as_ref(),
                    ),
                    (
                        J2kResidentEncodeGpuStage::CoefficientExtract,
                        tile.prepare_coefficient_extract_command_buffer.as_ref(),
                    ),
                ] {
                    if let Some(command_buffer) = command_buffer {
                        let ptr = command_buffer.as_ptr();
                        if prepare_command_buffer_ptrs.contains(&ptr) {
                            continue;
                        }
                        prepare_command_buffer_ptrs.push(ptr);
                        pushed_split_prepare = true;
                        gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                            stage,
                            command_buffer: command_buffer.clone(),
                        });
                    }
                }
                for command_buffer in &tile.prepare_dwt53_vertical_command_buffers {
                    let ptr = command_buffer.as_ptr();
                    if prepare_command_buffer_ptrs.contains(&ptr) {
                        continue;
                    }
                    prepare_command_buffer_ptrs.push(ptr);
                    pushed_split_prepare = true;
                    gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                        stage: J2kResidentEncodeGpuStage::CoefficientDwt53Vertical,
                        command_buffer: command_buffer.clone(),
                    });
                }
                for command_buffer in &tile.prepare_dwt53_horizontal_command_buffers {
                    let ptr = command_buffer.as_ptr();
                    if prepare_command_buffer_ptrs.contains(&ptr) {
                        continue;
                    }
                    prepare_command_buffer_ptrs.push(ptr);
                    pushed_split_prepare = true;
                    gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                        stage: J2kResidentEncodeGpuStage::CoefficientDwt53Horizontal,
                        command_buffer: command_buffer.clone(),
                    });
                }
                if pushed_split_prepare {
                    continue;
                }
                let ptr = tile.prepare_command_buffer.as_ptr();
                if prepare_command_buffer_ptrs.contains(&ptr) {
                    continue;
                }
                prepare_command_buffer_ptrs.push(ptr);
                gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                    stage: J2kResidentEncodeGpuStage::CoefficientPrep,
                    command_buffer: tile.prepare_command_buffer.clone(),
                });
            }
        }

        retained_command_buffers.extend(
            gpu_stage_command_buffers
                .iter()
                .map(|stage_command_buffer| stage_command_buffer.command_buffer.clone()),
        );
        for tile in prepared_tiles {
            if let Some(command_buffer) = tile.prepare_deinterleave_rct_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            if let Some(command_buffer) = tile.prepare_dwt53_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            retained_command_buffers.extend(tile.prepare_dwt53_vertical_command_buffers);
            retained_command_buffers.extend(tile.prepare_dwt53_horizontal_command_buffers);
            if let Some(command_buffer) = tile.prepare_coefficient_extract_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            retained_command_buffers.push(tile.prepare_command_buffer);
            retained_buffers.push(tile.coefficient_buffer);
            retained_buffers.push(tile.deinterleave_status_buffer);
            retained_buffers.extend(tile.plane_buffers);
            retained_buffers.extend(tile.scratch_buffers);
            retained_buffers.push(tile.coefficient_job_buffer);
            recyclable_private_buffers.extend(tile.recyclable_private_buffers);
        }
        retained_buffers.push(coefficient_buffer);
        retained_buffers.push(tier1_job_buffer);
        retained_buffers.push(tier1_output_buffer);
        retained_buffers.push(tier1_status_buffer);
        retained_buffers.push(tier1_segment_buffer);
        if let Some(buffer) = classic_tier1_raw_pack_buffer {
            retained_buffers.push(buffer);
        }
        if let Some(buffer) = classic_tier1_arithmetic_pack_buffer {
            retained_buffers.push(buffer);
        }
        retained_buffers.push(packet_resolution_buffer);
        retained_buffers.push(packet_subband_buffer);
        retained_buffers.push(resident_block_buffer);
        retained_buffers.push(packet_descriptor_buffer);
        retained_buffers.push(state_block_buffer);
        retained_buffers.push(packet_payload_copy_job_buffer);
        retained_buffers.push(header_buffer);
        retained_buffers.push(scratch_buffer);
        retained_buffers.push(packet_job_buffer);
        retained_buffers.push(packet_status_buffer.clone());
        retained_buffers.push(codestream_job_buffer);

        stage_stats.classic_tier1_setup_duration = classic_tier1_setup_duration;
        stage_stats.classic_block_encode_duration = classic_block_encode_duration;
        stage_stats.classic_packet_plan_duration = classic_packet_plan_duration;
        stage_stats.classic_packet_buffer_setup_duration = classic_packet_buffer_setup_duration;
        stage_stats.classic_command_buffer_commit_duration = classic_command_buffer_commit_duration;
        stage_stats.packet_block_prep_duration = packet_block_prep_duration;
        stage_stats.packetization_duration = packetization_duration;
        stage_stats.codestream_assembly_duration = codestream_assembly_duration;
        stage_stats.packet_payload_copy_job_capacity_total = packet_payload_copy_job_capacity_total;
        stage_stats.max_packet_payload_copy_jobs_per_tile = max_payload_copy_jobs_per_tile;
        stage_stats.packet_payload_copy_launched_stripe_count_total = packet_jobs
            .len()
            .saturating_mul(max_payload_copy_jobs_per_tile)
            .saturating_mul(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB as usize);
        stage_stats.tier1_output_capacity_total = tier1_output_capacity_total;
        stage_stats.max_tier1_output_capacity = max_tier1_output_capacity;
        stage_stats.tier1_segment_capacity_total = tier1_segment_capacity_total;
        stage_stats.max_tier1_segment_capacity_per_block = tier1_jobs
            .iter()
            .map(|job| job.segment_capacity as usize)
            .max()
            .unwrap_or(0);
        stage_stats.packet_output_capacity_total = packet_output_capacity_total;
        stage_stats.max_packet_output_capacity = max_packet_output_capacity_usize;
        stage_stats.codestream_payload_copy_launched_thread_count_total = 0;
        stage_stats.code_block_count = tier1_jobs.len();

        Ok(J2kPendingResidentLosslessCodestreamBatch {
            runtime: session.runtime()?,
            buffer: codestream_buffer,
            byte_offsets: codestream_offsets,
            capacities: codestream_capacities,
            status_buffer: codestream_status_buffer,
            packet_status_buffer,
            tier1_status_readback,
            classic_tier1_density_readback,
            classic_tier1_symbol_plan_readback,
            classic_tier1_pass_plan_readback,
            classic_tier1_token_emit_readback,
            classic_tier1_split_token_emit_readback,
            classic_gpu_token_pack_used: use_classic_gpu_token_pack
                || use_classic_split_gpu_token_pack
                || use_classic_split_mq_byte_gpu_token_pack,
            command_buffer: command_buffer.clone(),
            retained_command_buffers,
            _retained_buffers: retained_buffers,
            recyclable_private_buffers,
            recyclable_shared_buffers,
            gpu_stage_command_buffers,
            stage_stats,
            codestream_payload_copy_dispatched,
            status_stage: "J2K batched codestream assembly",
            length_error: "J2K Metal batched codestream output length exceeds usize",
            capacity_error: "J2K Metal batched codestream output length exceeds buffer",
        })
    })
}
