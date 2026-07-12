// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    build_resident_batch_packet_plan, checked_buffer_read, checked_buffer_slice,
    classic_cod_block_style_from_flags, classic_encode_code_blocks_pipeline,
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
    new_blit_command_encoder, new_command_buffer, new_compute_command_encoder, new_private_buffer,
    new_resident_encode_command_buffer, new_shared_buffer, packet_tree_node_count,
    prepared_lossless_batch_tiles, schedule_classic_tier1_gpu_token_pack_readback,
    schedule_resident_tier1_status_readback, size_of, take_recyclable_private_buffer,
    wait_resident_lossless_codestream, with_runtime, with_runtime_for_session,
    zeroed_recyclable_shared_buffer, zeroed_shared_buffer, Buffer, CommandBufferRef,
    ComputeCommandEncoderRef, DirectStatusCheck, Duration, Error, ForeignType, Instant,
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
    J2kResidentPacketizationEncodeJob, MTLSize, MetalRuntime, ResidentBatchPacketPlan,
    ResidentBatchPacketPlanParams, ResidentLosslessTier1Metal, ResidentTier1StatusReadbackRequest,
    J2K_ENCODE_STATUS_OK, J2K_HT_STATUS_OK, PACKET_PAYLOAD_COPY_STRIPES_PER_JOB,
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

mod batch_reporting;
mod classic_labels;
mod ht_cleanup;
use self::batch_reporting::collect_prepared_batch_retention;
use self::classic_labels::{
    next_enabled_classic_stage_label, CLASSIC_TIER1_ARITHMETIC_PACK_LABEL,
    CLASSIC_TIER1_DENSITY_LABEL, CLASSIC_TIER1_PASS_PLAN_LABEL, CLASSIC_TIER1_RAW_PACK_LABEL,
    CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL, CLASSIC_TIER1_SYMBOL_PLAN_LABEL,
    CLASSIC_TIER1_TOKEN_EMIT_LABEL,
};
pub(super) use self::ht_cleanup::{
    dispatch_ht_cleanup, dispatch_ht_cleanup_batched,
    dispatch_ht_cleanup_batched_in_command_buffer, dispatch_ht_cleanup_batched_in_encoder,
    dispatch_ht_cleanup_repeated_batched_in_command_buffer, HtRepeatedCleanupDispatch,
};

mod classic_packet;
mod classic_profile;
mod classic_tier1;
mod ht_packet;
mod ht_tier1;
mod resident_single;
mod tier2_packetization;

pub(crate) use self::classic_packet::submit_lossless_codestream_buffers_from_prepared_classic_batch;
use self::classic_profile::{
    classic_profile_stages_from_env, dispatch_classic_tier1_profiles, ClassicTier1ProfileRequest,
    ClassicTier1ProfileResult,
};
use self::classic_tier1::{prepare_classic_tier1, ClassicTier1Prepared};
pub(crate) use self::ht_packet::submit_lossless_codestream_buffers_from_prepared_ht_batch;
use self::ht_tier1::{prepare_ht_tier1, HtTier1Prepared};
pub(crate) use self::resident_single::encode_lossless_codestream_buffer_from_resident_tier1;
pub(crate) use self::tier2_packetization::encode_tier2_packetization;
