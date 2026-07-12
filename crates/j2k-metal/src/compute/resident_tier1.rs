// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::test_counters;
use super::{
    accumulate_classic_tier1_scan_estimates, checked_buffer_read, checked_buffer_slice,
    classic_encode_code_blocks_pipeline_kind, classic_tier1_gpu_token_pack_supported,
    classic_tier1_pass_class_counts, completed_command_buffers_gpu_duration,
    completed_command_buffers_gpu_duration_and_elapsed_window, dispatch_1d_pipeline,
    duration_share, encode_status_error, hybrid_stage_signpost, label_compute_encoder,
    metal_profile_classic_tier1_arithmetic_pack_enabled,
    metal_profile_classic_tier1_density_enabled, metal_profile_classic_tier1_pass_plan_enabled,
    metal_profile_classic_tier1_raw_pack_enabled,
    metal_profile_classic_tier1_split_token_emit_enabled,
    metal_profile_classic_tier1_symbol_plan_enabled,
    metal_profile_classic_tier1_token_emit_enabled, metal_profile_classic_tier1_token_pack_enabled,
    metal_profile_stages_enabled, new_blit_command_encoder, new_compute_command_encoder,
    new_private_buffer, new_shared_buffer, pack_j2k_code_block_scalar_from_tier1_tokens,
    packet_encode_status_error, record_completed_resident_encode_gpu_stages,
    recycle_private_buffers, recycle_shared_buffers, size_of, take_recyclable_private_buffer,
    wait_for_completion_metal, Arc, Buffer, CommandBuffer, CommandBufferRef, ComputePipelineState,
    EncodeProgressionOrder, Error, HybridSignpostName, Instant, IntoParallelIterator,
    J2kClassicEncodeBatchJob, J2kClassicEncodePipelineKind, J2kClassicEncodeStatus,
    J2kClassicTier1DensityCounters, J2kClassicTier1PassPlanCounters,
    J2kClassicTier1SymbolPlanCounters, J2kClassicTier1TokenSegment, J2kCodestreamAssemblyStatus,
    J2kHtEncodeBatchJob, J2kHtEncodeStatus, J2kPacketEncodeStatus,
    J2kPacketizationPacketDescriptor, J2kPendingResidentLosslessCodestream,
    J2kResidentEncodeGpuStageCommandBuffer, J2kResidentEncodeStageStats,
    J2kResidentLosslessCodestream, J2kResidentLosslessCodestreamBatchResult, J2kTier1TokenSegment,
    MTLSize, MetalRuntime, ParallelIterator, CLASSIC_TIER1_MQ_BYTE_TOKEN_ARENA_BYTES,
    CLASSIC_TIER1_TOKEN_ARENA_BYTES, CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY, J2K_ENCODE_STATUS_OK,
    PACKET_PAYLOAD_COPY_STRIPES_PER_JOB, SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT,
    SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST,
};

mod counter_validation;
mod profile_dispatch;
mod readback;
mod result_harvest;
mod types;

pub(in crate::compute) use self::counter_validation::{
    compare_classic_tier1_symbol_plan_and_pass_plan_counters,
    compare_classic_tier1_symbol_plan_and_split_token_emit_counters,
    compare_classic_tier1_symbol_plan_and_token_emit_counters, profile_classic_tier1_token_pack,
    record_classic_tier1_density_counters, record_classic_tier1_pass_plan_counters,
    record_classic_tier1_symbol_plan_counters, record_classic_tier1_token_emit_counters,
    validate_classic_tier1_split_token_emit_counters,
};
#[cfg(test)]
pub(in crate::compute) use self::profile_dispatch::dispatch_classic_tier1_split_token_emit_for_cpu_pack;
pub(in crate::compute) use self::profile_dispatch::{
    dispatch_classic_tier1_arithmetic_pack_profile, dispatch_classic_tier1_density_profile,
    dispatch_classic_tier1_pass_plan_profile, dispatch_classic_tier1_raw_pack_profile,
    dispatch_classic_tier1_split_token_emit_for_gpu_pack,
    dispatch_classic_tier1_split_token_emit_profile,
    dispatch_classic_tier1_split_token_pack_from_gpu_tokens,
    dispatch_classic_tier1_symbol_plan_profile, dispatch_classic_tier1_token_emit_for_gpu_pack,
    dispatch_classic_tier1_token_emit_profile, dispatch_classic_tier1_token_pack_from_gpu_tokens,
    schedule_classic_tier1_gpu_token_pack_readback,
};
pub(in crate::compute) use self::readback::{
    schedule_resident_tier1_status_readback, ResidentTier1StatusReadbackRequest,
};
pub(crate) use self::readback::{
    wait_resident_lossless_codestream, wait_resident_lossless_codestream_batch,
};
pub(in crate::compute) use self::result_harvest::{
    finish_completed_resident_lossless_codestream_batch, wait_resident_codestream_command_buffer,
};
pub(in crate::compute) use self::types::{
    J2kBatchedPacketPayloadCopyDispatch, J2kResidentClassicTier1DensityReadback,
    J2kResidentClassicTier1GpuTokenBuffers, J2kResidentClassicTier1PassPlanReadback,
    J2kResidentClassicTier1SplitTokenBuffers, J2kResidentClassicTier1SymbolPlanReadback,
    J2kResidentClassicTier1TokenEmitReadback, J2kResidentTier1StatusKind,
    J2kResidentTier1StatusReadback,
};
pub(crate) use self::types::{
    J2kLosslessCodestreamAssemblyJob, J2kLosslessCodestreamBlockCodingMode,
    J2kLosslessDeviceBatchPrepareItem, J2kLosslessDeviceCodeBlock, J2kLosslessDevicePrepareJob,
    J2kPendingResidentLosslessCodestreamBatch, J2kPreparedLosslessDeviceCodeBlocks,
    J2kResidentLosslessHtCodeBlocks, J2kResidentLosslessTier1CodeBlocks,
    J2kResidentPacketizationEncodeJob, J2kResidentPacketizationResolution,
    J2kResidentPacketizationSubband, ResidentLosslessTier1Metal,
};
