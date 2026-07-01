// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::{
    cell::RefCell,
    collections::HashMap,
    mem::{size_of, size_of_val},
    sync::Arc,
    time::{Duration, Instant},
};

use j2k_core::{PixelFormat, Rect};
#[cfg(test)]
use j2k_metal_support::system_default_device;
#[cfg(target_os = "macos")]
use j2k_metal_support::{
    checked_command_queue, commit_and_wait, dispatch_1d_pipeline, dispatch_2d_pipeline,
    dispatch_3d_pipeline, dispatch_single_thread, wait_for_completion, MetalPipelineLoader,
    MetalSupportError,
};
#[cfg(target_os = "macos")]
use j2k_native::{
    ht_uvlc_encode_table, ht_uvlc_table0, ht_uvlc_table1, ht_vlc_encode_table0,
    ht_vlc_encode_table1, ht_vlc_table0, ht_vlc_table1,
    pack_j2k_code_block_scalar_from_tier1_tokens, ColorSpace as NativeColorSpace,
    DecodedComponents as NativeDecodedComponents, EncodeProgressionOrder, EncodedHtJ2kCodeBlock,
    EncodedJ2kCodeBlock, HtCodeBlockDecodeJob, HtSubBandDecodeJob, J2kCodeBlockDecodeJob,
    J2kCodeBlockSegment, J2kDeinterleaveToF32Job, J2kDirectBandId, J2kDirectColorPlan,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep,
    J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Level, J2kForwardDwt97Output,
    J2kHtCodeBlockEncodeJob, J2kInverseMctJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor, J2kQuantizeSubbandJob,
    J2kSingleDecompositionIdwtJob, J2kStoreComponentJob, J2kSubBandDecodeJob,
    J2kTier1CodeBlockEncodeJob, J2kTier1TokenSegment, J2kWaveletTransform,
};
#[cfg(target_os = "macos")]
use metal::{
    foreign_types::ForeignType, Buffer, CommandBuffer, CommandBufferRef, CommandQueue,
    ComputeCommandEncoderRef, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
#[cfg(target_os = "macos")]
use rayon::prelude::*;

#[cfg(target_os = "macos")]
use crate::{
    buffer_pool::MetalBufferPools,
    profile_env::{
        classic_tier1_gpu_token_pack_requested, classic_tier1_split_gpu_token_pack_requested,
        classic_tier1_split_mq_byte_gpu_token_pack_disabled,
        classic_tier1_split_mq_byte_gpu_token_pack_requested, hybrid_stage_signpost,
        label_command_buffer, label_compute_encoder,
        metal_profile_classic_tier1_arithmetic_pack_enabled,
        metal_profile_classic_tier1_density_enabled, metal_profile_classic_tier1_pass_plan_enabled,
        metal_profile_classic_tier1_raw_pack_enabled,
        metal_profile_classic_tier1_split_token_emit_enabled,
        metal_profile_classic_tier1_symbol_plan_enabled,
        metal_profile_classic_tier1_token_emit_enabled,
        metal_profile_classic_tier1_token_pack_enabled,
        metal_profile_coefficient_prep_split_commands_enabled,
        metal_profile_decode_split_commands_enabled, HybridSignpostName,
        SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD, SIGNPOST_DECODE_HYBRID_COMMAND_WAIT,
        SIGNPOST_DECODE_HYBRID_CPU_TIER1, SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
        SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE,
        SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE,
        SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
        SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE,
        SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP,
        SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN,
        SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE,
        SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE,
        SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP, SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT,
        SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
        SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE,
        SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE,
        SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP, SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN,
        SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE,
        SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE, SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP,
        SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST,
    },
};
use crate::{Error, Surface};

mod abi;
pub(crate) use self::abi::*;
#[cfg(target_os = "macos")]
mod buffer_validation;
#[cfg(target_os = "macos")]
pub(crate) use self::buffer_validation::{
    copy_interleaved_padded_to_shared_buffer, validate_metal_buffer_matches_bytes,
    validate_metal_buffers_match,
};
#[cfg(target_os = "macos")]
mod classic_encode_pipeline;
#[cfg(target_os = "macos")]
use self::classic_encode_pipeline::{
    classic_cod_block_style_from_flags, classic_encode_code_blocks_pipeline,
    classic_encode_code_blocks_pipeline_kind, classic_resident_style_flags_from_env,
    classic_tier1_gpu_token_pack_supported, J2kClassicEncodePipelineKind,
};
#[cfg(target_os = "macos")]
mod classic_tier1_stats;
#[cfg(target_os = "macos")]
use self::classic_tier1_stats::{
    accumulate_classic_tier1_scan_estimates, classic_tier1_pass_class_counts,
};
#[cfg(target_os = "macos")]
mod code_block_decoder;
#[cfg(target_os = "macos")]
use self::code_block_decoder::MetalCodeBlockDecoder;
mod direct_cache;
use self::direct_cache::CpuTier1CoefficientCache;
#[cfg(target_os = "macos")]
mod direct_buffers;
#[cfg(target_os = "macos")]
use self::direct_buffers::{
    borrow_mut_slice_buffer, borrow_slice_buffer, copied_recyclable_shared_slice_buffer,
    copied_slice_buffer, owned_slice_buffer, take_classic_coefficients_scratch_buffer,
    take_classic_states_scratch_buffer, wrap_f32_output_buffer, zeroed_recyclable_shared_buffer,
};
#[cfg(target_os = "macos")]
mod direct_commands;
#[cfg(target_os = "macos")]
use self::direct_commands::{
    DecodeHybridSplitCommandBuffers, DirectColorBatchCommandBuffers, DirectIdwtCommandBuffers,
};
#[cfg(target_os = "macos")]
mod direct_cpu;
#[cfg(all(target_os = "macos", test))]
use self::direct_cpu::decode_prepared_classic_sub_band_on_cpu;
#[cfg(target_os = "macos")]
use self::direct_cpu::{
    decode_classic_inputs_on_cpu_with_plan_cache, decode_ht_inputs_on_cpu_with_plan_cache,
    decode_prepared_classic_jobs_on_cpu_with_scratch,
    decode_prepared_classic_jobs_on_cpu_with_scratch_profiled,
    decode_prepared_classic_sub_band_group_on_cpu_profile,
    decode_prepared_classic_sub_band_on_cpu_profile, decode_prepared_ht_jobs_on_cpu_with_workspace,
    decode_prepared_ht_jobs_on_cpu_with_workspace_profiled,
    decode_prepared_ht_sub_band_group_on_cpu_profile, decode_prepared_ht_sub_band_on_cpu_profile,
    ClassicCpuDecodeInput, ClassicCpuDecodeScratch, HtCpuDecodeInput,
};
#[cfg(target_os = "macos")]
mod direct_flattened;
#[cfg(all(target_os = "macos", test))]
use self::direct_flattened::hybrid_cpu_decode_worker_count;
#[cfg(target_os = "macos")]
use self::direct_flattened::{
    build_flattened_cpu_tier1_cache, packed_cpu_decode_coefficients, packed_cpu_decode_output_len,
    FlattenedCpuTier1Cache,
};
mod direct_profile;
#[cfg(target_os = "macos")]
use self::direct_profile::record_completed_decode_split_gpu_stages;
use self::direct_profile::{
    elapsed_us, emit_direct_hybrid_stage_timings, CpuTier1DecodeSubstageCounters,
    DirectHybridStageTimings,
};
mod direct_plan_support;
#[cfg(test)]
use self::direct_plan_support::prepared_direct_color_tier1_input_count;
use self::direct_plan_support::{
    classic_group_shapes_match, classic_sub_band_shapes_match, direct_preflight_invariant,
    ht_group_shapes_match, ht_sub_band_shapes_match, idwt_shapes_match,
    prepared_direct_color_plan_supports_runtime, store_shapes_match,
};
mod direct_scratch;
use self::direct_scratch::{
    recycle_private_buffers, recycle_scratch_buffers, recycle_shared_buffers,
    take_f32_scratch_buffer, take_recyclable_private_buffer, DirectScratchBuffer,
};
#[cfg(target_os = "macos")]
mod direct_status;
#[cfg(target_os = "macos")]
use self::direct_status::{
    decode_classic_status_error, decode_ht_status_error, decode_idwt_status_error,
    decode_mct_status_error, validate_direct_status, DirectStatusCheck,
};
#[cfg(target_os = "macos")]
mod direct_tier1;
#[cfg(target_os = "macos")]
use self::direct_tier1::{
    flattened_hybrid_cpu_tier1_enabled, prepare_direct_tier1_input_buffer,
    record_flattened_hybrid_cpu_decode_batch, record_hybrid_cpu_decode_inputs,
    record_hybrid_cpu_decode_worker_init, record_hybrid_repeated_output_blit,
    record_hybrid_stacked_component_batch, should_flatten_hybrid_cpu_tier1_color_batch,
    DirectTier1Mode, HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK,
};
mod pack_params;
use self::pack_params::{j2k_pack_scale_arrays, j2k_scalar_pack_params, j2k_u32_param};
mod encode_capacity;
use self::encode_capacity::{
    classic_encode_output_capacity, classic_encode_output_capacity_for_mode,
    classic_encode_segment_capacity, classic_packet_output_capacity,
    codestream_progression_order_code, ht_encode_output_capacity,
    ht_packet_output_capacity_for_mode, lossless_codestream_assembly_capacity,
    packet_tree_node_count,
};
pub(crate) use self::encode_capacity::{
    ht_packet_output_capacity_mode_from_env, J2kClassicEncodeOutputCapacityMode,
    J2kHtPacketOutputCapacityMode,
};
mod resident_packet_plan;
use self::resident_packet_plan::{
    build_resident_batch_packet_plan, prepared_lossless_batch_tiles, ResidentBatchPacketPlan,
    ResidentBatchPacketPlanParams,
};
mod resident_types;
pub(crate) use self::resident_types::{
    J2kPendingResidentLosslessCodestream, J2kResidentBatchEncodeItem, J2kResidentEncodeStageStats,
    J2kResidentLosslessCodestream, J2kResidentLosslessCodestreamBatchResult,
};
#[cfg(target_os = "macos")]
mod gpu_timing;
#[cfg(target_os = "macos")]
use self::gpu_timing::{
    completed_command_buffer_gpu_duration, completed_command_buffers_gpu_duration,
    completed_command_buffers_gpu_duration_and_elapsed_window,
};
#[cfg(target_os = "macos")]
mod resident_stage_timing;
#[cfg(target_os = "macos")]
use self::resident_stage_timing::{
    duration_share, finish_resident_encode_split_command_buffer,
    finish_resident_encode_split_command_buffer_timed, new_resident_encode_command_buffer,
    record_completed_resident_encode_gpu_stages, J2kResidentEncodeGpuStage,
    J2kResidentEncodeGpuStageCommandBuffer,
};
#[cfg(target_os = "macos")]
mod shader_source;
#[cfg(target_os = "macos")]
use self::shader_source::SHADER_SOURCE;
#[cfg(test)]
mod test_counters;
#[cfg(test)]
pub(crate) use self::test_counters::{
    classic_gpu_token_pack_dispatches_for_test,
    classic_split_mq_byte_gpu_token_pack_dispatches_for_test,
    direct_tier1_input_buffer_prepares_for_test, flattened_hybrid_cpu_decode_batches_for_test,
    ht_batch_coefficient_copy_blits_for_test, hybrid_cpu_decode_inputs_for_test,
    hybrid_cpu_decode_worker_inits_for_test, hybrid_repeated_output_blits_for_test,
    hybrid_stacked_component_batches_for_test, lossless_deinterleave_rct_fused_dispatches_for_test,
    reset_classic_gpu_token_pack_dispatches_for_test,
    reset_classic_split_mq_byte_gpu_token_pack_dispatches_for_test,
    reset_direct_tier1_input_buffer_prepares_for_test,
    reset_flattened_hybrid_cpu_decode_batches_for_test,
    reset_ht_batch_coefficient_copy_blits_for_test, reset_hybrid_cpu_decode_inputs_for_test,
    reset_hybrid_cpu_decode_worker_inits_for_test, reset_hybrid_repeated_output_blits_for_test,
    reset_hybrid_stacked_component_batches_for_test,
    reset_lossless_deinterleave_rct_fused_dispatches_for_test,
    reset_resident_codestream_command_buffer_waits_for_test,
    reset_resident_gpu_timestamp_queries_for_test,
    resident_codestream_command_buffer_waits_for_test, resident_gpu_timestamp_queries_for_test,
};

#[cfg(target_os = "macos")]
pub(crate) use crate::profile_env::metal_profile_stages_enabled;

#[cfg(all(target_os = "macos", test))]
pub(crate) use crate::buffer_pool::{
    private_buffer_pool_misses_for_test, reset_private_buffer_pool_misses_for_test,
    reset_shared_buffer_pool_misses_for_test, shared_buffer_pool_misses_for_test,
};

#[cfg(all(target_os = "macos", test))]
pub(crate) use crate::profile_env::{
    force_classic_gpu_token_pack_route_for_test, force_metal_profile_stages_for_test,
};

#[cfg(target_os = "macos")]
thread_local! {
    static DEFAULT_METAL_SESSION: RefCell<Option<Result<crate::MetalBackendSession, MetalSupportError>>> = const { RefCell::new(None) };
    static METAL_RUNTIME_OVERRIDE: RefCell<Option<Arc<MetalRuntime>>> = const { RefCell::new(None) };
}

#[cfg(target_os = "macos")]
pub(crate) struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    zero_u32_buffer: ComputePipelineState,
    validate_bytes_equal: ComputePipelineState,
    copy_interleaved_padded: ComputePipelineState,
    lossless_deinterleave_to_planes: ComputePipelineState,
    lossless_deinterleave_rct_rgb8_to_planes: ComputePipelineState,
    lossless_extract_coefficients: ComputePipelineState,
    pack_gray8: ComputePipelineState,
    pack_rgb8: ComputePipelineState,
    pack_mct_rgb8: ComputePipelineState,
    pack_mct_rgb8_batched: ComputePipelineState,
    pack_rgb_opaque_rgba8: ComputePipelineState,
    pack_rgba8: ComputePipelineState,
    pack_gray16: ComputePipelineState,
    pack_rgb16: ComputePipelineState,
    pack_u8_repeated_gray: ComputePipelineState,
    pack_u16_repeated_gray: ComputePipelineState,
    classic_cleanup_plain_batched: ComputePipelineState,
    classic_cleanup_batched: ComputePipelineState,
    classic_cleanup_plain_repeated_batched: ComputePipelineState,
    classic_cleanup_plain_dev_repeated_batched: ComputePipelineState,
    classic_cleanup_repeated_batched: ComputePipelineState,
    classic_store_repeated_batched: ComputePipelineState,
    idwt_interleave: ComputePipelineState,
    idwt_reversible53_horizontal: ComputePipelineState,
    idwt_reversible53_vertical: ComputePipelineState,
    idwt_interleave_batched: ComputePipelineState,
    idwt_reversible53_horizontal_batched: ComputePipelineState,
    idwt_reversible53_vertical_batched: ComputePipelineState,
    idwt_irreversible97_single_decomposition: ComputePipelineState,
    fdwt53_horizontal: ComputePipelineState,
    fdwt53_vertical: ComputePipelineState,
    fdwt53_horizontal_batched: ComputePipelineState,
    fdwt53_vertical_batched: ComputePipelineState,
    fdwt97_lift_horizontal: ComputePipelineState,
    fdwt97_lift_vertical: ComputePipelineState,
    fdwt97_deinterleave_horizontal: ComputePipelineState,
    fdwt97_deinterleave_vertical: ComputePipelineState,
    inverse_mct: ComputePipelineState,
    forward_rct: ComputePipelineState,
    forward_ict: ComputePipelineState,
    quantize_subband: ComputePipelineState,
    store_component: ComputePipelineState,
    store_component_repeated: ComputePipelineState,
    store_component_repeated_gray_u8: ComputePipelineState,
    store_component_repeated_gray_u16: ComputePipelineState,
    store_component_repeated_gray_u8_contiguous: ComputePipelineState,
    store_component_repeated_gray_u16_contiguous: ComputePipelineState,
    store_component_gray_u8: ComputePipelineState,
    store_component_gray_u16: ComputePipelineState,
    ht_cleanup: ComputePipelineState,
    ht_cleanup_batched: ComputePipelineState,
    ht_cleanup_repeated_batched: ComputePipelineState,
    classic_encode_code_block: ComputePipelineState,
    classic_encode_code_blocks: ComputePipelineState,
    classic_encode_code_blocks_32: ComputePipelineState,
    classic_encode_code_blocks_bypass_32: ComputePipelineState,
    classic_encode_code_blocks_bypass_u16_32: ComputePipelineState,
    classic_tier1_density_bypass_u16_32: ComputePipelineState,
    classic_tier1_raw_pack_bypass_u16_32: ComputePipelineState,
    classic_tier1_arithmetic_pack_bypass_u16_32: ComputePipelineState,
    classic_tier1_symbol_plan_bypass_u16_32: ComputePipelineState,
    classic_tier1_pass_plan_bypass_u16_32: ComputePipelineState,
    classic_tier1_token_emit_bypass_u16_32: ComputePipelineState,
    classic_tier1_split_token_emit_bypass_u16_32: ComputePipelineState,
    classic_tier1_split_mq_byte_token_emit_bypass_u16_32: ComputePipelineState,
    classic_tier1_token_pack_bypass_u16_32: ComputePipelineState,
    classic_tier1_split_token_pack_bypass_u16_32: ComputePipelineState,
    classic_encode_code_blocks_style0: ComputePipelineState,
    classic_encode_code_blocks_style0_32: ComputePipelineState,
    ht_encode_code_block: ComputePipelineState,
    ht_encode_code_blocks: ComputePipelineState,
    packet_block_prepare_resident_classic: ComputePipelineState,
    packet_block_prepare_resident_ht: ComputePipelineState,
    packet_encode: ComputePipelineState,
    packet_encode_batched: ComputePipelineState,
    packet_encode_resident_classic_batched: ComputePipelineState,
    packet_payload_copy_batched: ComputePipelineState,
    lossless_codestream_assemble: ComputePipelineState,
    lossless_codestream_assemble_batched: ComputePipelineState,
    ht_vlc_table0: Buffer,
    ht_vlc_table1: Buffer,
    ht_uvlc_table0: Buffer,
    ht_uvlc_table1: Buffer,
    ht_vlc_encode_table0: Buffer,
    ht_vlc_encode_table1: Buffer,
    ht_uvlc_encode_table: Buffer,
    tier1_dummy_buffer: Buffer,
    buffer_pools: MetalBufferPools,
}

#[cfg(target_os = "macos")]
impl MetalRuntime {
    #[cfg(test)]
    fn new() -> Result<Self, MetalSupportError> {
        let device = system_default_device()?;
        Self::new_with_device(&device)
    }

    pub(crate) fn new_with_device(device: &Device) -> Result<Self, MetalSupportError> {
        let loader = MetalPipelineLoader::new(device, SHADER_SOURCE)?;
        let pipeline = |name: &str| loader.pipeline(name);
        let queue = checked_command_queue(device)?;
        Ok(Self {
            device: device.clone(),
            queue,
            zero_u32_buffer: pipeline("j2k_zero_u32_buffer")?,
            validate_bytes_equal: pipeline("j2k_validate_bytes_equal")?,
            copy_interleaved_padded: pipeline("j2k_copy_interleaved_padded")?,
            lossless_deinterleave_to_planes: pipeline("j2k_lossless_deinterleave_to_planes")?,
            lossless_deinterleave_rct_rgb8_to_planes: pipeline(
                "j2k_lossless_deinterleave_rct_rgb8_to_planes",
            )?,
            lossless_extract_coefficients: pipeline("j2k_lossless_extract_coefficients")?,
            pack_gray8: pipeline("j2k_pack_gray8")?,
            pack_rgb8: pipeline("j2k_pack_rgb8")?,
            pack_mct_rgb8: pipeline("j2k_pack_mct_rgb8")?,
            pack_mct_rgb8_batched: pipeline("j2k_pack_mct_rgb8_batched")?,
            pack_rgb_opaque_rgba8: pipeline("j2k_pack_rgb_opaque_rgba8")?,
            pack_rgba8: pipeline("j2k_pack_rgba8")?,
            pack_gray16: pipeline("j2k_pack_gray16")?,
            pack_rgb16: pipeline("j2k_pack_rgb16")?,
            pack_u8_repeated_gray: pipeline("j2k_pack_u8_repeated_gray")?,
            pack_u16_repeated_gray: pipeline("j2k_pack_u16_repeated_gray")?,
            classic_cleanup_plain_batched: pipeline("j2k_decode_classic_cleanup_plain_batched")?,
            classic_cleanup_batched: pipeline("j2k_decode_classic_cleanup_batched")?,
            classic_cleanup_plain_repeated_batched: pipeline(
                "j2k_decode_classic_cleanup_plain_repeated_batched",
            )?,
            classic_cleanup_plain_dev_repeated_batched: pipeline(
                "j2k_decode_classic_cleanup_plain_dev_repeated_batched",
            )?,
            classic_cleanup_repeated_batched: pipeline(
                "j2k_decode_classic_cleanup_repeated_batched",
            )?,
            classic_store_repeated_batched: pipeline("j2k_store_classic_repeated_batched")?,
            idwt_interleave: pipeline("j2k_idwt_interleave")?,
            idwt_reversible53_horizontal: pipeline("j2k_idwt_reversible53_horizontal_pass")?,
            idwt_reversible53_vertical: pipeline("j2k_idwt_reversible53_vertical_pass")?,
            idwt_interleave_batched: pipeline("j2k_idwt_interleave_batched")?,
            idwt_reversible53_horizontal_batched: pipeline(
                "j2k_idwt_reversible53_horizontal_pass_batched",
            )?,
            idwt_reversible53_vertical_batched: pipeline(
                "j2k_idwt_reversible53_vertical_pass_batched",
            )?,
            idwt_irreversible97_single_decomposition: pipeline(
                "j2k_idwt_irreversible97_single_decomposition",
            )?,
            fdwt53_horizontal: pipeline("j2k_forward_dwt53_horizontal")?,
            fdwt53_vertical: pipeline("j2k_forward_dwt53_vertical")?,
            fdwt53_horizontal_batched: pipeline("j2k_forward_dwt53_horizontal_batched")?,
            fdwt53_vertical_batched: pipeline("j2k_forward_dwt53_vertical_batched")?,
            fdwt97_lift_horizontal: pipeline("j2k_forward_dwt97_lift_horizontal")?,
            fdwt97_lift_vertical: pipeline("j2k_forward_dwt97_lift_vertical")?,
            fdwt97_deinterleave_horizontal: pipeline("j2k_forward_dwt97_deinterleave_horizontal")?,
            fdwt97_deinterleave_vertical: pipeline("j2k_forward_dwt97_deinterleave_vertical")?,
            inverse_mct: pipeline("j2k_inverse_mct")?,
            forward_rct: pipeline("j2k_forward_rct")?,
            forward_ict: pipeline("j2k_forward_ict")?,
            quantize_subband: pipeline("j2k_quantize_subband")?,
            store_component: pipeline("j2k_store_component")?,
            store_component_repeated: pipeline("j2k_store_component_repeated")?,
            store_component_repeated_gray_u8: pipeline("j2k_store_component_repeated_gray_u8")?,
            store_component_repeated_gray_u16: pipeline("j2k_store_component_repeated_gray_u16")?,
            store_component_repeated_gray_u8_contiguous: pipeline(
                "j2k_store_component_repeated_gray_u8_contiguous",
            )?,
            store_component_repeated_gray_u16_contiguous: pipeline(
                "j2k_store_component_repeated_gray_u16_contiguous",
            )?,
            store_component_gray_u8: pipeline("j2k_store_component_gray_u8")?,
            store_component_gray_u16: pipeline("j2k_store_component_gray_u16")?,
            ht_cleanup: pipeline("j2k_decode_ht_cleanup")?,
            ht_cleanup_batched: pipeline("j2k_decode_ht_cleanup_batched")?,
            ht_cleanup_repeated_batched: pipeline("j2k_decode_ht_cleanup_repeated_batched")?,
            classic_encode_code_block: pipeline("j2k_encode_classic_code_block")?,
            classic_encode_code_blocks: pipeline("j2k_encode_classic_code_blocks")?,
            classic_encode_code_blocks_32: pipeline("j2k_encode_classic_code_blocks_32")?,
            classic_encode_code_blocks_bypass_32: pipeline(
                "j2k_encode_classic_code_blocks_bypass_32",
            )?,
            classic_encode_code_blocks_bypass_u16_32: pipeline(
                "j2k_encode_classic_code_blocks_bypass_u16_32",
            )?,
            classic_tier1_density_bypass_u16_32: pipeline(
                "j2k_profile_classic_tier1_density_bypass_u16_32",
            )?,
            classic_tier1_raw_pack_bypass_u16_32: pipeline(
                "j2k_profile_classic_tier1_raw_pack_bypass_u16_32",
            )?,
            classic_tier1_arithmetic_pack_bypass_u16_32: pipeline(
                "j2k_profile_classic_tier1_arithmetic_pack_bypass_u16_32",
            )?,
            classic_tier1_symbol_plan_bypass_u16_32: pipeline(
                "j2k_plan_classic_tier1_symbols_bypass_u16_32",
            )?,
            classic_tier1_pass_plan_bypass_u16_32: pipeline(
                "j2k_plan_classic_tier1_passes_bypass_u16_32",
            )?,
            classic_tier1_token_emit_bypass_u16_32: pipeline(
                "j2k_emit_classic_tier1_tokens_bypass_u16_32",
            )?,
            classic_tier1_split_token_emit_bypass_u16_32: pipeline(
                "j2k_emit_classic_tier1_split_tokens_bypass_u16_32",
            )?,
            classic_tier1_split_mq_byte_token_emit_bypass_u16_32: pipeline(
                "j2k_emit_classic_tier1_split_mq_byte_raw_tokens_bypass_u16_32",
            )?,
            classic_tier1_token_pack_bypass_u16_32: pipeline(
                "j2k_pack_classic_tier1_tokens_bypass_u16_32",
            )?,
            classic_tier1_split_token_pack_bypass_u16_32: pipeline(
                "j2k_pack_classic_tier1_split_tokens_bypass_u16_32",
            )?,
            classic_encode_code_blocks_style0: pipeline("j2k_encode_classic_code_blocks_style0")?,
            classic_encode_code_blocks_style0_32: pipeline(
                "j2k_encode_classic_code_blocks_style0_32",
            )?,
            ht_encode_code_block: pipeline("j2k_encode_ht_code_block")?,
            ht_encode_code_blocks: pipeline("j2k_encode_ht_code_blocks")?,
            packet_block_prepare_resident_classic: pipeline(
                "j2k_prepare_packet_blocks_from_classic_status",
            )?,
            packet_block_prepare_resident_ht: pipeline("j2k_prepare_packet_blocks_from_ht_status")?,
            packet_encode: pipeline("j2k_encode_packetization")?,
            packet_encode_batched: pipeline("j2k_encode_packetization_batched")?,
            packet_encode_resident_classic_batched: pipeline(
                "j2k_encode_packetization_resident_classic_batched",
            )?,
            packet_payload_copy_batched: pipeline("j2k_copy_packet_payload_batched")?,
            lossless_codestream_assemble: pipeline("j2k_assemble_lossless_classic_codestream")?,
            lossless_codestream_assemble_batched: pipeline(
                "j2k_assemble_lossless_codestream_batched",
            )?,
            ht_vlc_table0: device.new_buffer_with_data(
                ht_vlc_table0().as_ptr().cast(),
                size_of_val(ht_vlc_table0()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_vlc_table1: device.new_buffer_with_data(
                ht_vlc_table1().as_ptr().cast(),
                size_of_val(ht_vlc_table1()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_uvlc_table0: device.new_buffer_with_data(
                ht_uvlc_table0().as_ptr().cast(),
                size_of_val(ht_uvlc_table0()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_uvlc_table1: device.new_buffer_with_data(
                ht_uvlc_table1().as_ptr().cast(),
                size_of_val(ht_uvlc_table1()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_vlc_encode_table0: device.new_buffer_with_data(
                ht_vlc_encode_table0().as_ptr().cast(),
                size_of_val(ht_vlc_encode_table0()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_vlc_encode_table1: device.new_buffer_with_data(
                ht_vlc_encode_table1().as_ptr().cast(),
                size_of_val(ht_vlc_encode_table1()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_uvlc_encode_table: device.new_buffer_with_data(
                ht_uvlc_encode_table().as_ptr().cast(),
                size_of_val(ht_uvlc_encode_table()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            tier1_dummy_buffer: device.new_buffer(1, MTLResourceOptions::StorageModeShared),
            buffer_pools: MetalBufferPools::new(),
        })
    }

    pub(crate) fn command_queue(&self) -> &metal::CommandQueueRef {
        self.queue.as_ref()
    }

    fn take_private_buffer(&self, bytes: usize) -> Result<Buffer, Error> {
        self.buffer_pools.take_private(&self.device, bytes)
    }

    fn recycle_private_buffer(&self, bytes: usize, buffer: Buffer) -> Result<(), Error> {
        self.buffer_pools.recycle_private(bytes, buffer)
    }

    fn take_shared_buffer(&self, bytes: usize) -> Result<Buffer, Error> {
        self.buffer_pools.take_shared(&self.device, bytes)
    }

    fn recycle_shared_buffer(&self, bytes: usize, buffer: Buffer) -> Result<(), Error> {
        self.buffer_pools.recycle_shared(bytes, buffer)
    }
}

#[cfg(target_os = "macos")]
fn with_runtime<R>(f: impl FnOnce(&MetalRuntime) -> Result<R, Error>) -> Result<R, Error> {
    let override_runtime = METAL_RUNTIME_OVERRIDE.with(|slot| slot.borrow().clone());
    if let Some(runtime) = override_runtime {
        return f(&runtime);
    }

    DEFAULT_METAL_SESSION.with(|session| {
        let mut session = session.borrow_mut();
        if session.is_none() {
            *session = Some(
                j2k_metal_support::system_default_device().map(crate::MetalBackendSession::new),
            );
        }
        let Some(session) = session.as_ref() else {
            return Err(Error::MetalRuntime {
                message: "J2K Metal default session was not initialized".to_string(),
            });
        };
        match session {
            Ok(session) => with_runtime_for_session(session, f),
            Err(error) => Err(runtime_initialization_error(error)),
        }
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn runtime_initialization_error(error: &MetalSupportError) -> Error {
    if error.is_unavailable() {
        Error::MetalUnavailable
    } else {
        Error::MetalRuntime {
            message: error.to_string(),
        }
    }
}

#[cfg(target_os = "macos")]
pub(super) fn commit_and_wait_metal(command_buffer: &CommandBufferRef) -> Result<(), Error> {
    commit_and_wait(command_buffer).map_err(|error| Error::MetalKernel {
        message: error.to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(super) fn wait_for_completion_metal(command_buffer: &CommandBufferRef) -> Result<(), Error> {
    wait_for_completion(command_buffer).map_err(|error| Error::MetalKernel {
        message: error.to_string(),
    })
}

#[cfg(target_os = "macos")]
struct RuntimeOverrideGuard {
    previous: Option<Arc<MetalRuntime>>,
}

#[cfg(target_os = "macos")]
impl Drop for RuntimeOverrideGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        METAL_RUNTIME_OVERRIDE.with(|slot| {
            slot.replace(previous);
        });
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn with_runtime_for_session<R>(
    session: &crate::MetalBackendSession,
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let runtime = session.runtime()?;
    let previous = METAL_RUNTIME_OVERRIDE.with(|slot| slot.replace(Some(runtime.clone())));
    let _guard = RuntimeOverrideGuard { previous };
    f(&runtime)
}

#[cfg(target_os = "macos")]
fn with_runtime_for_device<R>(
    device: &Device,
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let override_runtime = METAL_RUNTIME_OVERRIDE.with(|slot| slot.borrow().clone());
    if let Some(runtime) = override_runtime {
        if runtime.device.as_ptr() == device.as_ptr() {
            return f(&runtime);
        }
    }

    let session = crate::MetalBackendSession::new(device.clone());
    with_runtime_for_session(&session, f)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn with_isolated_runtime_for_device_for_test<R>(
    device: &Device,
    f: impl FnOnce() -> Result<R, Error>,
) -> Result<R, Error> {
    let runtime = Arc::new(
        MetalRuntime::new_with_device(device)
            .map_err(|error| runtime_initialization_error(&error))?,
    );
    let previous = METAL_RUNTIME_OVERRIDE.with(|slot| slot.replace(Some(runtime)));
    let _guard = RuntimeOverrideGuard { previous };
    f()
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(crate) struct PreparedDirectGrayscalePlan {
    dimensions: (u32, u32),
    bit_depth: u8,
    tier1_prepare_mode: DirectTier1Mode,
    steps: Vec<PreparedDirectGrayscaleStep>,
    classic_groups: Vec<PreparedClassicSubBandGroup>,
    ht_groups: Vec<PreparedHtSubBandGroup>,
    cpu_tier1_cache: Arc<CpuTier1CoefficientCache>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(crate) struct PreparedDirectColorPlan {
    dimensions: (u32, u32),
    bit_depths: [u8; 3],
    mct: bool,
    transform: J2kWaveletTransform,
    component_plans: Vec<PreparedDirectGrayscalePlan>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
enum PreparedDirectGrayscaleStep {
    ClassicSubBand(PreparedClassicSubBand),
    HtSubBand(PreparedHtSubBand),
    Idwt(PreparedDirectIdwt),
    Store(J2kDirectStoreStep),
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedDirectIdwt {
    step: J2kDirectIdwtStep,
    output_window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBand {
    band_id: J2kDirectBandId,
    width: u32,
    height: u32,
    zero_fill: bool,
    coded_data: Vec<u8>,
    coded_buffer: Buffer,
    jobs: Vec<J2kClassicCleanupBatchJob>,
    jobs_buffer: Buffer,
    segments: Vec<J2kClassicSegment>,
    segments_buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBandGroup {
    start_step: usize,
    end_step: usize,
    total_coefficients: usize,
    zero_fill: bool,
    coded_data: Vec<u8>,
    coded_buffer: Buffer,
    jobs: Vec<J2kClassicCleanupBatchJob>,
    jobs_buffer: Buffer,
    segments: Vec<J2kClassicSegment>,
    segments_buffer: Buffer,
    members: Vec<PreparedClassicSubBandGroupMember>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBandGroupMember {
    band_id: J2kDirectBandId,
    offset_elements: usize,
    window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBand {
    band_id: J2kDirectBandId,
    width: u32,
    height: u32,
    coded_data: Vec<u8>,
    coded_buffer: Option<Buffer>,
    jobs: Vec<J2kHtCleanupBatchJob>,
    jobs_buffer: Option<Buffer>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct HtCodedArena {
    data: Vec<u8>,
    buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBandGroup {
    start_step: usize,
    end_step: usize,
    total_coefficients: usize,
    coded_arena: HtCodedArena,
    jobs: Vec<J2kHtCleanupBatchJob>,
    jobs_buffer: Buffer,
    members: Vec<PreparedHtSubBandGroupMember>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBandGroupMember {
    band_id: J2kDirectBandId,
    offset_elements: usize,
    window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
struct PlaneStage {
    dims: (u32, u32),
    plane_count: usize,
    color_space: NativeColorSpace,
    has_alpha: bool,
    bit_depths: [u32; 4],
    planes: [Option<Buffer>; 4],
}

#[cfg(target_os = "macos")]
impl PlaneStage {
    fn from_planes(
        device: &Device,
        decoded: &NativeDecodedComponents<'_>,
        roi: Option<Rect>,
    ) -> Result<Self, Error> {
        let full_dims = decoded.dimensions();
        let roi = roi.unwrap_or(Rect {
            x: 0,
            y: 0,
            w: full_dims.0,
            h: full_dims.1,
        });
        let dims = (roi.w, roi.h);
        let plane_count = decoded.planes().len();
        if plane_count == 0 || plane_count > 4 {
            return Err(Error::MetalKernel {
                message: format!("unsupported J2K plane count {plane_count}"),
            });
        }

        let mut bit_depths = [0u32; 4];
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (index, plane) in decoded.planes().iter().enumerate() {
            bit_depths[index] = u32::from(plane.bit_depth());
            let len = dims.0 as usize * dims.1 as usize;
            let buffer = device.new_buffer(
                (len * size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            copy_plane_samples(&buffer, plane.samples(), full_dims.0 as usize, roi);
            planes[index] = Some(buffer);
        }

        Ok(Self {
            dims,
            plane_count,
            color_space: decoded.color_space().clone(),
            has_alpha: decoded.has_alpha(),
            bit_depths,
            planes,
        })
    }

    fn from_captured_planes(
        decoded: &NativeDecodedComponents<'_>,
        captured_planes: Vec<Buffer>,
    ) -> Option<Self> {
        let plane_count = decoded.planes().len();
        let supported_shape = matches!(
            (decoded.color_space(), decoded.has_alpha(), plane_count),
            (NativeColorSpace::Gray, false, 1) | (NativeColorSpace::RGB, false, 3)
        );
        if !supported_shape {
            return None;
        }
        if captured_planes.len() != plane_count || plane_count == 0 || plane_count > 4 {
            return None;
        }

        let mut bit_depths = [0u32; 4];
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (index, (plane, buffer)) in decoded.planes().iter().zip(captured_planes).enumerate() {
            bit_depths[index] = u32::from(plane.bit_depth());
            planes[index] = Some(buffer);
        }

        Some(Self {
            dims: decoded.dimensions(),
            plane_count,
            color_space: decoded.color_space().clone(),
            has_alpha: decoded.has_alpha(),
            bit_depths,
            planes,
        })
    }

    fn finish_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        let command_buffer = runtime.queue.new_command_buffer();
        let surface =
            encode_plane_stage_to_surface_in_command_buffer(runtime, command_buffer, &self, fmt)?;
        commit_and_wait_metal(command_buffer)?;
        Ok(surface)
    }
}

#[cfg(target_os = "macos")]
fn encode_plane_stage_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    stage: &PlaneStage,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = stage.dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * stage.dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (output_channels, opaque_alpha, pipeline) = output_shape_for(
        &stage.color_space,
        stage.has_alpha,
        stage.plane_count,
        fmt,
        runtime,
    )?;
    let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays(stage.bit_depths);

    let params = J2kPackParams {
        width: stage.dims.0,
        height: stage.dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        output_channels,
        opaque_alpha,
        max_values,
        u8_scales,
        u16_scales,
    };

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid plane pack");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(
        0,
        stage.planes[0].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        1,
        stage.planes[1].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        2,
        stage.planes[2].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        3,
        stage.planes[3].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(4, Some(&out_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, pipeline, stage.dims);
    encoder.end_encoding();

    Ok(Surface::from_metal_buffer(out_buffer, stage.dims, fmt))
}

#[cfg(target_os = "macos")]
fn encode_mct_rgb8_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Result<Surface, Error> {
    let pitch_bytes = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (max_values, u8_scales, _) = j2k_pack_scale_arrays([
        u32::from(bit_depths[0]),
        u32::from(bit_depths[1]),
        u32::from(bit_depths[2]),
        0,
    ]);
    let params = J2kMctRgb8PackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        transform: mct_transform_code(transform),
        addends: [
            signed_sample_bias(bit_depths[0]),
            signed_sample_bias(bit_depths[1]),
            signed_sample_bias(bit_depths[2]),
        ],
        max_values: [max_values[0], max_values[1], max_values[2]],
        u8_scales: [u8_scales[0], u8_scales[1], u8_scales[2]],
    };

    let signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid MCT RGB8 pack");
    encoder.set_compute_pipeline_state(&runtime.pack_mct_rgb8);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kMctRgb8PackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, &runtime.pack_mct_rgb8, dims);
    encoder.end_encoding();
    drop(signpost);

    Ok(Surface::from_metal_buffer(
        out_buffer,
        dims,
        PixelFormat::Rgb8,
    ))
}

#[cfg(target_os = "macos")]
fn encode_batched_mct_rgb8_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    count: usize,
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Result<Vec<Surface>, Error> {
    let count_u32 = u32::try_from(count).map_err(|_| Error::MetalKernel {
        message: "J2K MetalDirect color batch count exceeds u32".to_string(),
    })?;
    let pitch_bytes = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect color batch output size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect color batch output size overflow".to_string(),
        })?;
    let out_buffer = runtime
        .device
        .new_buffer(total_bytes as u64, MTLResourceOptions::StorageModeShared);
    let plane_stride = dims
        .0
        .checked_mul(dims.1)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect color batch plane stride overflow".to_string(),
        })?;
    let (max_values, u8_scales, _) = j2k_pack_scale_arrays([
        u32::from(bit_depths[0]),
        u32::from(bit_depths[1]),
        u32::from(bit_depths[2]),
        0,
    ]);
    let params = J2kBatchedMctRgb8PackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        transform: mct_transform_code(transform),
        batch_count: count_u32,
        plane_stride,
        output_stride: u32::try_from(surface_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect color batch surface stride exceeds u32".to_string(),
        })?,
        addends: [
            signed_sample_bias(bit_depths[0]),
            signed_sample_bias(bit_depths[1]),
            signed_sample_bias(bit_depths[2]),
        ],
        max_values: [max_values[0], max_values[1], max_values[2]],
        u8_scales: [u8_scales[0], u8_scales[1], u8_scales[2]],
    };

    let signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid batched MCT RGB8 pack");
    encoder.set_compute_pipeline_state(&runtime.pack_mct_rgb8_batched);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kBatchedMctRgb8PackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        &runtime.pack_mct_rgb8_batched,
        (dims.0, dims.1, count_u32),
    );
    encoder.end_encoding();
    drop(signpost);

    Ok((0..count)
        .map(|index| {
            Surface::from_metal_buffer_with_offset(
                out_buffer.clone(),
                dims,
                PixelFormat::Rgb8,
                index * surface_bytes,
            )
        })
        .collect())
}

#[cfg(target_os = "macos")]
fn encode_repeated_mct_rgb8_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    count: usize,
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Result<Vec<Surface>, Error> {
    let pitch_bytes = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect repeated color batch output size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect repeated color batch output size overflow".to_string(),
        })?;
    let output_len = u64::try_from(total_bytes.max(1)).map_err(|_| Error::MetalKernel {
        message: "J2K MetalDirect repeated output buffer exceeds u64".to_string(),
    })?;
    let out_buffer = runtime
        .device
        .new_buffer(output_len, MTLResourceOptions::StorageModeShared);
    let plane_stride = dims
        .0
        .checked_mul(dims.1)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect repeated color batch plane stride overflow".to_string(),
        })?;
    let (max_values, u8_scales, _) = j2k_pack_scale_arrays([
        u32::from(bit_depths[0]),
        u32::from(bit_depths[1]),
        u32::from(bit_depths[2]),
        0,
    ]);
    let params = J2kBatchedMctRgb8PackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        transform: mct_transform_code(transform),
        batch_count: 1,
        plane_stride,
        output_stride: u32::try_from(surface_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect repeated color batch surface stride exceeds u32".to_string(),
        })?,
        addends: [
            signed_sample_bias(bit_depths[0]),
            signed_sample_bias(bit_depths[1]),
            signed_sample_bias(bit_depths[2]),
        ],
        max_values: [max_values[0], max_values[1], max_values[2]],
        u8_scales: [u8_scales[0], u8_scales[1], u8_scales[2]],
    };

    let signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated MCT RGB8 pack");
    encoder.set_compute_pipeline_state(&runtime.pack_mct_rgb8_batched);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kBatchedMctRgb8PackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, &runtime.pack_mct_rgb8_batched, dims);
    encoder.end_encoding();
    drop(signpost);

    if surface_bytes > 0 && count > 1 {
        let blit = command_buffer.new_blit_command_encoder();
        if metal_profile_stages_enabled() {
            blit.set_label("J2K decode hybrid repeated output blit");
        }
        let mut copied = 1usize;
        while copied < count {
            let copy_count = copied.min(count - copied);
            let dst_offset =
                copied
                    .checked_mul(surface_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K MetalDirect repeated output destination offset overflow"
                            .to_string(),
                    })?;
            let copy_bytes =
                copy_count
                    .checked_mul(surface_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K MetalDirect repeated output copy size overflow".to_string(),
                    })?;
            blit.copy_from_buffer(
                &out_buffer,
                0,
                &out_buffer,
                u64::try_from(dst_offset).map_err(|_| Error::MetalKernel {
                    message: "J2K MetalDirect repeated output destination offset exceeds u64"
                        .to_string(),
                })?,
                u64::try_from(copy_bytes).map_err(|_| Error::MetalKernel {
                    message: "J2K MetalDirect repeated output copy size exceeds u64".to_string(),
                })?,
            );
            record_hybrid_repeated_output_blit();
            copied += copy_count;
        }
        blit.end_encoding();
    }

    Ok((0..count)
        .map(|index| {
            Surface::from_metal_buffer_with_offset(
                out_buffer.clone(),
                dims,
                PixelFormat::Rgb8,
                index * surface_bytes,
            )
        })
        .collect())
}

#[cfg(target_os = "macos")]
fn repeated_shared_direct_color_plan_count(
    plans: &[Arc<PreparedDirectColorPlan>],
) -> Option<usize> {
    let first = plans.first()?;
    (plans.len() > 1 && plans.iter().all(|plan| Arc::ptr_eq(plan, first))).then_some(plans.len())
}

#[cfg(target_os = "macos")]
fn mct_transform_code(transform: J2kWaveletTransform) -> u32 {
    match transform {
        J2kWaveletTransform::Reversible53 => 0,
        J2kWaveletTransform::Irreversible97 => 1,
    }
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band(
    job: &j2k_native::J2kOwnedSubBandPlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedClassicSubBand, Error> {
    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    let mut segments = Vec::new();

    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect segment table exceeds u32".to_string(),
        })?;
        for segment in &block.segments {
            let data_offset = coded_offset
                .checked_add(segment.data_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect segment offset overflow".to_string(),
                })?;
            segments.push(J2kClassicSegment {
                data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            });
        }
        jobs.push(J2kClassicCleanupBatchJob {
            coded_offset,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            segment_offset,
            segment_count: u32::try_from(block.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect segment count exceeds u32".to_string(),
            })?,
            width: block.width,
            height: block.height,
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect output offset overflow".to_string(),
                })?,
            missing_msbs: u32::from(block.missing_bit_planes),
            total_bitplanes: u32::from(block.total_bitplanes),
            roi_shift: u32::from(block.roi_shift),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            sub_band_type: match block.sub_band_type {
                j2k_native::J2kSubBandType::LowLow => 0,
                j2k_native::J2kSubBandType::HighLow => 1,
                j2k_native::J2kSubBandType::LowHigh => 2,
                j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(block.style),
            strict: u32::from(block.strict),
            dequantization_step: block.dequantization_step,
        });
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode);
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode);
        let segments_buffer =
            prepare_direct_tier1_input_buffer(runtime, &segments, tier1_prepare_mode);
        Ok(PreparedClassicSubBand {
            band_id: job.band_id,
            width: job.width,
            height: job.height,
            zero_fill: false,
            coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
            segments,
            segments_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<Vec<PreparedClassicSubBandGroup>, Error> {
    let mut groups = Vec::new();
    let mut step_idx = 0;
    while step_idx < steps.len() {
        let start_step = step_idx;
        let mut sub_bands = Vec::new();
        while let Some(PreparedDirectGrayscaleStep::ClassicSubBand(sub_band)) = steps.get(step_idx)
        {
            sub_bands.push(sub_band);
            step_idx += 1;
        }
        if sub_bands.len() > 1 {
            groups.push(prepare_classic_sub_band_group(
                start_step,
                step_idx,
                &sub_bands,
                tier1_prepare_mode,
            )?);
        }
        if step_idx == start_step {
            step_idx += 1;
        }
    }
    Ok(groups)
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedClassicSubBand],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedClassicSubBandGroup, Error> {
    let mut members = Vec::with_capacity(sub_bands.len());
    let mut jobs = Vec::new();
    let mut segments = Vec::new();
    let mut coded_data = Vec::new();
    let mut output_base = 0usize;

    for sub_band in sub_bands {
        members.push(PreparedClassicSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });

        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped coded payload exceeds u32".to_string(),
        })?;
        let segment_base = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped segment table exceeds u32".to_string(),
        })?;
        let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
        })?;

        for segment in &sub_band.segments {
            let mut grouped_segment = *segment;
            grouped_segment.data_offset =
                coded_base
                    .checked_add(segment.data_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped segment offset overflow"
                            .to_string(),
                    })?;
            segments.push(grouped_segment);
        }

        for job in &sub_band.jobs {
            let mut grouped_job = *job;
            grouped_job.coded_offset =
                coded_base
                    .checked_add(job.coded_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped job coded offset overflow"
                            .to_string(),
                    })?;
            grouped_job.segment_offset =
                segment_base
                    .checked_add(job.segment_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped job segment offset overflow"
                            .to_string(),
                    })?;
            grouped_job.output_offset =
                output_base_u32
                    .checked_add(job.output_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped output offset overflow"
                            .to_string(),
                    })?;
            jobs.push(grouped_job);
        }

        coded_data.extend_from_slice(&sub_band.coded_data);
        let sub_band_len =
            sub_band
                .width
                .checked_mul(sub_band.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect grouped sub-band size overflow".to_string(),
                })? as usize;
        output_base = output_base
            .checked_add(sub_band_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect grouped coefficient arena overflow".to_string(),
            })?;
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode);
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode);
        let segments_buffer =
            prepare_direct_tier1_input_buffer(runtime, &segments, tier1_prepare_mode);
        Ok(PreparedClassicSubBandGroup {
            start_step,
            end_step,
            total_coefficients: output_base,
            zero_fill: sub_bands.iter().any(|sub_band| sub_band.zero_fill),
            coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
            segments,
            segments_buffer,
            members,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band(
    job: &j2k_native::HtOwnedSubBandPlan,
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBand, Error> {
    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        jobs.push(J2kHtCleanupBatchJob {
            coded_offset,
            width: block.width,
            height: block.height,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            missing_msbs: u32::from(block.missing_bit_planes),
            num_bitplanes: u32::from(block.num_bitplanes),
            roi_shift: u32::from(block.roi_shift),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
    }

    Ok(PreparedHtSubBand {
        band_id: job.band_id,
        width: job.width,
        height: job.height,
        coded_data,
        coded_buffer: None,
        jobs,
        jobs_buffer: None,
    })
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<Vec<PreparedHtSubBandGroup>, Error> {
    let mut groups = Vec::new();
    let mut step_idx = 0;
    while step_idx < steps.len() {
        let start_step = step_idx;
        let mut sub_bands = Vec::new();
        while let Some(PreparedDirectGrayscaleStep::HtSubBand(sub_band)) = steps.get(step_idx) {
            sub_bands.push(sub_band);
            step_idx += 1;
        }
        if sub_bands.len() > 1 {
            groups.push(prepare_ht_sub_band_group(
                start_step,
                step_idx,
                &sub_bands,
                tier1_prepare_mode,
            )?);
        }
        if step_idx == start_step {
            step_idx += 1;
        }
    }
    Ok(groups)
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedHtSubBand],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBandGroup, Error> {
    let mut members = Vec::with_capacity(sub_bands.len());
    let mut jobs = Vec::new();
    let mut coded_data = Vec::new();
    let mut output_base = 0usize;

    for sub_band in sub_bands {
        members.push(PreparedHtSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });

        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coded payload exceeds u32".to_string(),
        })?;
        let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
        })?;
        for job in &sub_band.jobs {
            let mut grouped_job = *job;
            grouped_job.coded_offset =
                coded_base
                    .checked_add(job.coded_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped coded offset overflow".to_string(),
                    })?;
            grouped_job.output_offset =
                output_base_u32
                    .checked_add(job.output_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped output offset overflow".to_string(),
                    })?;
            jobs.push(grouped_job);
        }
        coded_data.extend_from_slice(&sub_band.coded_data);
        let sub_band_len =
            sub_band
                .width
                .checked_mul(sub_band.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect grouped sub-band size overflow".to_string(),
                })? as usize;
        output_base = output_base
            .checked_add(sub_band_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect grouped coefficient arena overflow".to_string(),
            })?;
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode);
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode);
        Ok(PreparedHtSubBandGroup {
            start_step,
            end_step,
            total_coefficients: output_base,
            coded_arena: HtCodedArena {
                data: coded_data,
                buffer: coded_buffer,
            },
            jobs,
            jobs_buffer,
            members,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_ungrouped_ht_sub_band_buffers(
    steps: &mut [PreparedDirectGrayscaleStep],
    groups: &[PreparedHtSubBandGroup],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<(), Error> {
    if tier1_prepare_mode != DirectTier1Mode::Metal {
        return Ok(());
    }

    for (step_idx, step) in steps.iter_mut().enumerate() {
        let PreparedDirectGrayscaleStep::HtSubBand(sub_band) = step else {
            continue;
        };
        if groups
            .iter()
            .any(|group| group.start_step <= step_idx && step_idx < group.end_step)
        {
            sub_band.coded_buffer = None;
            sub_band.jobs_buffer = None;
            continue;
        }
        with_runtime(|runtime| {
            sub_band.coded_buffer = Some(prepare_direct_tier1_input_buffer(
                runtime,
                &sub_band.coded_data,
                tier1_prepare_mode,
            ));
            sub_band.jobs_buffer = Some(prepare_direct_tier1_input_buffer(
                runtime,
                &sub_band.jobs,
                tier1_prepare_mode,
            ));
            Ok(())
        })?;
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn prepared_ht_buffer<'a>(buffer: Option<&'a Buffer>, label: &str) -> Result<&'a Buffer, Error> {
    buffer.ok_or_else(|| Error::MetalKernel {
        message: format!("HTJ2K MetalDirect ungrouped sub-band is missing prepared {label} buffer"),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_grayscale_plan(
    plan: &J2kDirectGrayscalePlan,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    prepare_direct_grayscale_plan_with_tier1_mode(plan, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
fn prepare_direct_grayscale_plan_for_cpu_upload(
    plan: &J2kDirectGrayscalePlan,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    prepare_direct_grayscale_plan_with_tier1_mode(plan, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
fn prepare_direct_grayscale_plan_with_tier1_mode(
    plan: &J2kDirectGrayscalePlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    let mut steps = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::ClassicSubBand(
                    prepare_classic_sub_band(sub_band, tier1_prepare_mode)?,
                ));
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::HtSubBand(prepare_ht_sub_band(
                    sub_band,
                    tier1_prepare_mode,
                )?));
            }
            J2kDirectGrayscaleStep::Idwt(idwt) => {
                steps.push(PreparedDirectGrayscaleStep::Idwt(PreparedDirectIdwt {
                    step: *idwt,
                    output_window: BandRequiredRegion::full(idwt.rect.width(), idwt.rect.height()),
                }));
            }
            J2kDirectGrayscaleStep::Store(store) => {
                steps.push(PreparedDirectGrayscaleStep::Store(*store));
            }
        }
    }
    let classic_groups = prepare_classic_sub_band_groups(&steps, tier1_prepare_mode)?;
    let ht_groups = prepare_ht_sub_band_groups(&steps, tier1_prepare_mode)?;
    prepare_ungrouped_ht_sub_band_buffers(&mut steps, &ht_groups, tier1_prepare_mode)?;
    Ok(PreparedDirectGrayscalePlan {
        dimensions: plan.dimensions,
        bit_depth: plan.bit_depth,
        tier1_prepare_mode,
        steps,
        classic_groups,
        ht_groups,
        cpu_tier1_cache: Arc::new(CpuTier1CoefficientCache::default()),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn crop_prepared_direct_grayscale_plan_to_output_region(
    plan: &mut PreparedDirectGrayscalePlan,
    region: Rect,
) -> Result<(), Error> {
    if region.w == 0 || region.h == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect region-scaled grayscale plan has an empty output region"
                .to_string(),
        });
    }
    if region.x == 0
        && region.y == 0
        && region.w == plan.dimensions.0
        && region.h == plan.dimensions.1
    {
        return Ok(());
    }

    plan.clear_cpu_tier1_cache()?;
    let mut store_count = 0;
    for step in &mut plan.steps {
        if let PreparedDirectGrayscaleStep::Store(store) = step {
            crop_direct_store_step_to_output_region(store, region)?;
            store_count += 1;
        }
    }

    if store_count == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect grayscale plan has no store step to crop".to_string(),
        });
    }

    prune_prepared_direct_grayscale_plan_to_store_windows(plan)?;
    plan.dimensions = (region.w, region.h);
    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
struct BandRequiredRegion {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

#[cfg(target_os = "macos")]
impl BandRequiredRegion {
    fn full(width: u32, height: u32) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        }
    }

    fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Option<Self> {
        (x0 < x1 && y0 < y1).then_some(Self { x0, y0, x1, y1 })
    }

    fn width(self) -> u32 {
        self.x1 - self.x0
    }

    fn height(self) -> u32 {
        self.y1 - self.y0
    }

    fn expanded(self, margin: u32, width: u32, height: u32) -> Self {
        Self {
            x0: self.x0.saturating_sub(margin),
            y0: self.y0.saturating_sub(margin),
            x1: self.x1.saturating_add(margin).min(width),
            y1: self.y1.saturating_add(margin).min(height),
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    fn intersects(self, x0: u32, y0: u32, width: u32, height: u32) -> bool {
        let x1 = x0.saturating_add(width);
        let y1 = y0.saturating_add(height);
        self.x0 < x1 && x0 < self.x1 && self.y0 < y1 && y0 < self.y1
    }
}

#[cfg(target_os = "macos")]
fn prune_prepared_direct_grayscale_plan_to_store_windows(
    plan: &mut PreparedDirectGrayscalePlan,
) -> Result<(), Error> {
    let mut required = HashMap::<J2kDirectBandId, BandRequiredRegion>::new();
    for step in &plan.steps {
        if let PreparedDirectGrayscaleStep::Store(store) = step {
            let source_right = store
                .source_x
                .checked_add(store.copy_width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect ROI source width overflows u32".to_string(),
                })?;
            let source_bottom = store
                .source_y
                .checked_add(store.copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect ROI source height overflows u32".to_string(),
                })?;
            if let Some(region) =
                BandRequiredRegion::new(store.source_x, store.source_y, source_right, source_bottom)
            {
                add_required_region(&mut required, store.input_band_id, region);
            }
        }
    }

    let mut idwt_output_windows = HashMap::<J2kDirectBandId, BandRequiredRegion>::new();
    for step in plan.steps.iter().rev() {
        if let PreparedDirectGrayscaleStep::Idwt(idwt) = step {
            let Some(output_region) = required.get(&idwt.step.output_band_id).copied() else {
                continue;
            };
            let expanded = output_region.expanded(
                idwt_required_output_margin(idwt.step.transform),
                idwt.step.rect.width(),
                idwt.step.rect.height(),
            );
            idwt_output_windows.insert(idwt.step.output_band_id, expanded);
            add_idwt_input_required_regions(&mut required, &idwt.step, expanded);
        }
    }

    for step in &mut plan.steps {
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let before = sub_band.jobs.len();
                retain_classic_jobs_for_required_region(
                    &mut sub_band.jobs,
                    required.get(&sub_band.band_id).copied(),
                );
                if sub_band.jobs.len() != before {
                    sub_band.zero_fill = true;
                    if plan.tier1_prepare_mode == DirectTier1Mode::Metal {
                        with_runtime(|runtime| {
                            sub_band.jobs_buffer =
                                borrow_slice_buffer(&runtime.device, &sub_band.jobs);
                            Ok(())
                        })?;
                    }
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let before = sub_band.jobs.len();
                retain_ht_jobs_for_required_region(
                    &mut sub_band.jobs,
                    required.get(&sub_band.band_id).copied(),
                );
                if sub_band.jobs.len() != before {
                    compact_ht_sub_band_coded_data(sub_band, plan.tier1_prepare_mode)?;
                }
            }
            PreparedDirectGrayscaleStep::Idwt(_) | PreparedDirectGrayscaleStep::Store(_) => {}
        }
    }

    apply_prepared_direct_idwt_output_windows(plan, &idwt_output_windows)?;
    plan.classic_groups = prepare_classic_sub_band_groups(&plan.steps, plan.tier1_prepare_mode)?;
    plan.ht_groups = prepare_ht_sub_band_groups(&plan.steps, plan.tier1_prepare_mode)?;
    prepare_ungrouped_ht_sub_band_buffers(
        &mut plan.steps,
        &plan.ht_groups,
        plan.tier1_prepare_mode,
    )?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn apply_prepared_direct_idwt_output_windows(
    plan: &mut PreparedDirectGrayscalePlan,
    windows: &HashMap<J2kDirectBandId, BandRequiredRegion>,
) -> Result<(), Error> {
    for step in &mut plan.steps {
        if let PreparedDirectGrayscaleStep::Idwt(idwt) = step {
            idwt.output_window = windows
                .get(&idwt.step.output_band_id)
                .copied()
                .unwrap_or_else(|| {
                    BandRequiredRegion::full(idwt.step.rect.width(), idwt.step.rect.height())
                });
        }
    }

    for step in &mut plan.steps {
        let PreparedDirectGrayscaleStep::Store(store) = step else {
            continue;
        };
        let Some(window) = windows.get(&store.input_band_id).copied() else {
            continue;
        };

        store.source_x =
            store
                .source_x
                .checked_sub(window.x0)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect cropped IDWT store source x underflow".to_string(),
                })?;
        store.source_y =
            store
                .source_y
                .checked_sub(window.y0)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect cropped IDWT store source y underflow".to_string(),
                })?;
        store.input_rect = j2k_native::J2kRect {
            x0: store.input_rect.x0.saturating_add(window.x0),
            y0: store.input_rect.y0.saturating_add(window.y0),
            x1: store.input_rect.x0.saturating_add(window.x1),
            y1: store.input_rect.y0.saturating_add(window.y1),
        };
    }

    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct PreparedIdwtInputWindows {
    ll: BandRequiredRegion,
    hl: BandRequiredRegion,
    lh: BandRequiredRegion,
    hh: BandRequiredRegion,
}

fn idwt_input_windows_from_slices(
    ll: &DirectBandSlice,
    hl: &DirectBandSlice,
    lh: &DirectBandSlice,
    hh: &DirectBandSlice,
) -> PreparedIdwtInputWindows {
    PreparedIdwtInputWindows {
        ll: BandRequiredRegion::full(ll.window.width(), ll.window.height()),
        hl: BandRequiredRegion::full(hl.window.width(), hl.window.height()),
        lh: BandRequiredRegion::full(lh.window.width(), lh.window.height()),
        hh: BandRequiredRegion::full(hh.window.width(), hh.window.height()),
    }
}

#[cfg(target_os = "macos")]
fn prepared_idwt_params(
    idwt: &PreparedDirectIdwt,
    inputs: PreparedIdwtInputWindows,
) -> J2kIdwtSingleDecompositionParams {
    J2kIdwtSingleDecompositionParams {
        x0: idwt.step.rect.x0,
        y0: idwt.step.rect.y0,
        output_x: idwt.output_window.x0,
        output_y: idwt.output_window.y0,
        width: idwt.output_window.width(),
        height: idwt.output_window.height(),
        ll_x: inputs.ll.x0,
        ll_y: inputs.ll.y0,
        ll_width: inputs.ll.width(),
        ll_height: inputs.ll.height(),
        hl_x: inputs.hl.x0,
        hl_y: inputs.hl.y0,
        hl_width: inputs.hl.width(),
        hl_height: inputs.hl.height(),
        lh_x: inputs.lh.x0,
        lh_y: inputs.lh.y0,
        lh_width: inputs.lh.width(),
        lh_height: inputs.lh.height(),
        hh_x: inputs.hh.x0,
        hh_y: inputs.hh.y0,
        hh_width: inputs.hh.width(),
        hh_height: inputs.hh.height(),
    }
}

#[cfg(target_os = "macos")]
fn repeated_idwt_params(
    idwt: &PreparedDirectIdwt,
    inputs: PreparedIdwtInputWindows,
    strides: PreparedIdwtInputStrides,
    batch_count: usize,
    context: &'static str,
) -> Result<J2kRepeatedIdwtSingleDecompositionParams, Error> {
    Ok(J2kRepeatedIdwtSingleDecompositionParams {
        x0: idwt.step.rect.x0,
        y0: idwt.step.rect.y0,
        output_x: idwt.output_window.x0,
        output_y: idwt.output_window.y0,
        width: idwt.output_window.width(),
        height: idwt.output_window.height(),
        ll_x: inputs.ll.x0,
        ll_y: inputs.ll.y0,
        ll_width: inputs.ll.width(),
        ll_height: inputs.ll.height(),
        hl_x: inputs.hl.x0,
        hl_y: inputs.hl.y0,
        hl_width: inputs.hl.width(),
        hl_height: inputs.hl.height(),
        lh_x: inputs.lh.x0,
        lh_y: inputs.lh.y0,
        lh_width: inputs.lh.width(),
        lh_height: inputs.lh.height(),
        hh_x: inputs.hh.x0,
        hh_y: inputs.hh.y0,
        hh_width: inputs.hh.width(),
        hh_height: inputs.hh.height(),
        ll_instance_stride: strides.ll,
        hl_instance_stride: strides.hl,
        lh_instance_stride: strides.lh,
        hh_instance_stride: strides.hh,
        batch_count: u32::try_from(batch_count).map_err(|_| Error::MetalKernel {
            message: format!("J2K MetalDirect {context} IDWT batch count exceeds u32"),
        })?,
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct PreparedIdwtInputStrides {
    ll: u32,
    hl: u32,
    lh: u32,
    hh: u32,
}

#[cfg(target_os = "macos")]
fn prepared_idwt_output_len(idwt: &PreparedDirectIdwt) -> usize {
    idwt.output_window.width() as usize * idwt.output_window.height() as usize
}

#[cfg(target_os = "macos")]
fn add_required_region(
    required: &mut HashMap<J2kDirectBandId, BandRequiredRegion>,
    band_id: J2kDirectBandId,
    region: BandRequiredRegion,
) {
    required
        .entry(band_id)
        .and_modify(|existing| *existing = existing.union(region))
        .or_insert(region);
}

#[cfg(target_os = "macos")]
fn idwt_required_output_margin(transform: J2kWaveletTransform) -> u32 {
    match transform {
        J2kWaveletTransform::Reversible53 => 16,
        J2kWaveletTransform::Irreversible97 => 40,
    }
}

#[cfg(target_os = "macos")]
fn add_idwt_input_required_regions(
    required: &mut HashMap<J2kDirectBandId, BandRequiredRegion>,
    idwt: &J2kDirectIdwtStep,
    output_region: BandRequiredRegion,
) {
    add_required_region(
        required,
        idwt.ll_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            true,
            true,
            idwt.ll.width(),
            idwt.ll.height(),
        ),
    );
    add_required_region(
        required,
        idwt.hl_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            false,
            true,
            idwt.hl.width(),
            idwt.hl.height(),
        ),
    );
    add_required_region(
        required,
        idwt.lh_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            true,
            false,
            idwt.lh.width(),
            idwt.lh.height(),
        ),
    );
    add_required_region(
        required,
        idwt.hh_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            false,
            false,
            idwt.hh.width(),
            idwt.hh.height(),
        ),
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::fn_params_excessive_bools)]
fn idwt_input_required_region(
    output_region: BandRequiredRegion,
    output_origin_x: u32,
    output_origin_y: u32,
    low_x: bool,
    low_y: bool,
    band_width: u32,
    band_height: u32,
) -> BandRequiredRegion {
    let x0 = j2k_native::idwt_band_index(output_origin_x, output_region.x0, low_x);
    let x1 =
        j2k_native::idwt_band_index(output_origin_x, output_region.x1 - 1, low_x).saturating_add(1);
    let y0 = j2k_native::idwt_band_index(output_origin_y, output_region.y0, low_y);
    let y1 =
        j2k_native::idwt_band_index(output_origin_y, output_region.y1 - 1, low_y).saturating_add(1);
    BandRequiredRegion {
        x0: x0.min(band_width),
        y0: y0.min(band_height),
        x1: x1.min(band_width),
        y1: y1.min(band_height),
    }
}

#[cfg(target_os = "macos")]
fn retain_classic_jobs_for_required_region(
    jobs: &mut Vec<J2kClassicCleanupBatchJob>,
    required: Option<BandRequiredRegion>,
) {
    let Some(required) = required else {
        jobs.clear();
        return;
    };
    jobs.retain(|job| {
        let output_x = job.output_offset % job.output_stride;
        let output_y = job.output_offset / job.output_stride;
        required.intersects(output_x, output_y, job.width, job.height)
    });
}

#[cfg(target_os = "macos")]
fn retain_ht_jobs_for_required_region(
    jobs: &mut Vec<J2kHtCleanupBatchJob>,
    required: Option<BandRequiredRegion>,
) {
    let Some(required) = required else {
        jobs.clear();
        return;
    };
    jobs.retain(|job| {
        let output_x = job.output_offset % job.output_stride;
        let output_y = job.output_offset / job.output_stride;
        required.intersects(output_x, output_y, job.width, job.height)
    });
}

#[cfg(target_os = "macos")]
fn compact_ht_sub_band_coded_data(
    sub_band: &mut PreparedHtSubBand,
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<(), Error> {
    let previous = std::mem::take(&mut sub_band.coded_data);
    let mut compacted = Vec::new();

    for job in &mut sub_band.jobs {
        let start = job.coded_offset as usize;
        let len = job.coded_len as usize;
        let end = start.checked_add(len).ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K MetalDirect cropped coded payload range overflow".to_string(),
        })?;
        if end > previous.len() {
            return Err(Error::MetalKernel {
                message: "HTJ2K MetalDirect cropped coded payload range out of bounds".to_string(),
            });
        }
        job.coded_offset = u32::try_from(compacted.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect cropped coded payload exceeds u32".to_string(),
        })?;
        compacted.extend_from_slice(&previous[start..end]);
    }

    sub_band.coded_data = compacted;
    sub_band.coded_buffer = None;
    sub_band.jobs_buffer = None;
    Ok(())
}

#[cfg(target_os = "macos")]
fn checked_rect_end(origin: u32, length: u32, label: &str) -> Result<u32, Error> {
    origin
        .checked_add(length)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("J2K MetalDirect region-scaled {label} overflows u32"),
        })
}

#[cfg(target_os = "macos")]
fn crop_direct_store_step_to_output_region(
    store: &mut J2kDirectStoreStep,
    region: Rect,
) -> Result<(), Error> {
    let store_bounds = (
        store.output_x,
        store.output_y,
        checked_rect_end(store.output_x, store.copy_width, "store width")?,
        checked_rect_end(store.output_y, store.copy_height, "store height")?,
    );
    let region_bounds = (
        region.x,
        region.y,
        checked_rect_end(region.x, region.w, "ROI width")?,
        checked_rect_end(region.y, region.h, "ROI height")?,
    );
    let intersection = (
        store_bounds.0.max(region_bounds.0),
        store_bounds.1.max(region_bounds.1),
        store_bounds.2.min(region_bounds.2),
        store_bounds.3.min(region_bounds.3),
    );
    if intersection.0 >= intersection.2 || intersection.1 >= intersection.3 {
        return Err(Error::MetalKernel {
            message:
                "J2K MetalDirect region-scaled ROI does not intersect the decoded store window"
                    .to_string(),
        });
    }

    let source_delta = (
        intersection.0 - store_bounds.0,
        intersection.1 - store_bounds.1,
    );
    store.source_x =
        store
            .source_x
            .checked_add(source_delta.0)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect region-scaled source x overflows u32".to_string(),
            })?;
    store.source_y =
        store
            .source_y
            .checked_add(source_delta.1)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect region-scaled source y overflows u32".to_string(),
            })?;
    store.copy_width = intersection.2 - intersection.0;
    store.copy_height = intersection.3 - intersection.1;
    store.output_width = region.w;
    store.output_height = region.h;
    store.output_x = intersection.0 - region_bounds.0;
    store.output_y = intersection.1 - region_bounds.1;
    Ok(())
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_color_plan(
    plan: &J2kDirectColorPlan,
) -> Result<PreparedDirectColorPlan, Error> {
    prepare_direct_color_plan_with_tier1_mode(plan, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_color_plan_for_cpu_upload(
    plan: &J2kDirectColorPlan,
) -> Result<PreparedDirectColorPlan, Error> {
    prepare_direct_color_plan_with_tier1_mode(plan, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
fn prepare_direct_color_plan_with_tier1_mode(
    plan: &J2kDirectColorPlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedDirectColorPlan, Error> {
    let component_plans = plan
        .component_plans
        .iter()
        .map(|component| match tier1_prepare_mode {
            DirectTier1Mode::Metal => prepare_direct_grayscale_plan(component),
            DirectTier1Mode::CpuUpload => prepare_direct_grayscale_plan_for_cpu_upload(component),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color plan expected 3 component plans, got {}",
                component_plans.len()
            ),
        });
    }
    Ok(PreparedDirectColorPlan {
        dimensions: plan.dimensions,
        bit_depths: plan.bit_depths,
        mct: plan.mct,
        transform: plan.transform,
        component_plans,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn crop_prepared_direct_color_plan_to_output_region(
    plan: &mut PreparedDirectColorPlan,
    region: Rect,
) -> Result<(), Error> {
    if region.w == 0 || region.h == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect region-scaled color plan has an empty output region"
                .to_string(),
        });
    }

    for component_plan in &mut plan.component_plans {
        crop_prepared_direct_grayscale_plan_to_output_region(component_plan, region)?;
        if component_plan.dimensions != (region.w, region.h) {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K MetalDirect color component crop produced {:?}, expected {:?}",
                    component_plan.dimensions,
                    (region.w, region.h)
                ),
            });
        }
    }

    plan.dimensions = (region.w, region.h);
    Ok(())
}

#[cfg(target_os = "macos")]
impl PreparedDirectGrayscalePlan {
    fn classic_group_starting_at(&self, step_idx: usize) -> Option<&PreparedClassicSubBandGroup> {
        self.classic_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }

    fn ht_group_starting_at(&self, step_idx: usize) -> Option<&PreparedHtSubBandGroup> {
        self.ht_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }
}

#[cfg(all(test, target_os = "macos"))]
fn prepared_direct_grayscale_plan_compute_encoder_count(
    plan: &PreparedDirectGrayscalePlan,
    _fmt: PixelFormat,
) -> usize {
    usize::from(!plan.steps.is_empty())
}

#[cfg(all(test, target_os = "macos"))]
fn prepared_repeated_direct_ht_cleanup_dispatch_count(plan: &PreparedDirectGrayscalePlan) -> usize {
    let mut dispatches = 0;
    let mut step_idx = 0;
    while step_idx < plan.steps.len() {
        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            dispatches += 1;
            step_idx = group.end_step;
            continue;
        }
        if matches!(
            plan.steps[step_idx],
            PreparedDirectGrayscaleStep::HtSubBand(_)
        ) {
            dispatches += 1;
        }
        step_idx += 1;
    }
    dispatches
}

#[cfg(target_os = "macos")]
fn encode_prepared_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = (|| {
        let mut bands = Vec::<DirectBandSlice>::new();
        let mut final_surface = None;
        let mut step_idx = 0;

        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                let (buffers, status_check) =
                    encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                let (buffers, status_check) =
                    encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            let step = &plan.steps[step_idx];
            match step {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    )?;
                    let (buffers, status_check) =
                        encode_prepared_classic_sub_band_to_buffer_in_encoder(
                            runtime,
                            encoder,
                            sub_band,
                            &output.buffer,
                            scratch_buffers,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    )?;
                    let (buffers, status_check) = encode_prepared_ht_sub_band_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        sub_band,
                        &output.buffer,
                    )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Idwt(idwt) => {
                    let ll =
                        lookup_direct_band_slice_entry(&bands, idwt.step.ll_band_id, idwt.step.ll)?;
                    let hl =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hl_band_id, idwt.step.hl)?;
                    let lh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.lh_band_id, idwt.step.lh)?;
                    let hh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hh_band_id, idwt.step.hh)?;
                    let params = prepared_idwt_params(
                        idwt,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                    );
                    let output = take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt))?;
                    match idwt.step.transform {
                        J2kWaveletTransform::Reversible53 => {
                            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                                runtime,
                                encoder,
                                &ll.buffer,
                                ll.offset_bytes,
                                &hl.buffer,
                                hl.offset_bytes,
                                &lh.buffer,
                                lh.offset_bytes,
                                &hh.buffer,
                                hh.offset_bytes,
                                params,
                                &output.buffer,
                                0,
                            );
                        }
                        J2kWaveletTransform::Irreversible97 => {
                            let status_check =
                                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                                    runtime,
                                    encoder,
                                    &ll.buffer,
                                    ll.offset_bytes,
                                    &hl.buffer,
                                    hl.offset_bytes,
                                    &lh.buffer,
                                    lh.offset_bytes,
                                    &hh.buffer,
                                    hh.offset_bytes,
                                    params,
                                    &output.buffer,
                                    0,
                                );
                            status_checks.push(status_check);
                        }
                    }
                    bands.push(DirectBandSlice {
                        band_id: idwt.step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: idwt.output_window,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Store(store) => {
                    let (input, input_offset) =
                        lookup_direct_band_slice(&bands, store.input_band_id, store.input_rect)?;
                    if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
                        let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
                        final_surface = Some(encode_gray_store_to_surface_in_encoder(
                            runtime,
                            encoder,
                            &input,
                            input_offset,
                            J2kGrayStoreParams {
                                input_width: store.input_rect.width(),
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                max_value: scale.max_value,
                                u8_scale: scale.u8_scale,
                                u16_scale: scale.u16_scale,
                            },
                            plan.dimensions,
                            fmt,
                        )?);
                    } else {
                        let output = take_f32_scratch_buffer(
                            runtime,
                            store.output_width as usize * store.output_height as usize,
                        )?;
                        let params = J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                        };
                        dispatch_store_component_buffer_in_encoder_with_offsets(
                            runtime,
                            encoder,
                            &input,
                            input_offset,
                            &output.buffer,
                            0,
                            params,
                        );
                        retained_buffers.push(output.buffer.clone());
                        final_surface = Some(encode_gray_plane_to_surface_in_encoder(
                            runtime,
                            encoder,
                            &output.buffer,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                        )?);
                        scratch_buffers.push(output);
                    }
                }
            }
            step_idx += 1;
        }

        final_surface.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect prepared grayscale plan did not produce a final stored plane"
                .to_string(),
        })
    })();
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn checked_coefficient_len(width: u32, height: u32, message: &str) -> Result<usize, Error> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: message.to_string(),
        })
}

#[cfg(target_os = "macos")]
fn upload_cpu_decoded_coefficients(
    runtime: &MetalRuntime,
    mut coefficients: Vec<f32>,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
) -> Buffer {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD);
    let buffer = borrow_mut_slice_buffer(&runtime.device, &mut coefficients);
    retained_buffers.push(buffer.clone());
    retained_cpu_coefficients.push(coefficients);
    buffer
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_prepared_direct_component_plane_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    tier1_mode: DirectTier1Mode,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Buffer, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = (|| {
        let mut bands = Vec::<DirectBandSlice>::new();
        let mut final_plane = None;
        let mut step_idx = 0;
        let profile_stages = metal_profile_stages_enabled();

        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                        let (buffers, status_check) =
                            encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                                runtime,
                                encoder,
                                group,
                                &output.buffer,
                                scratch_buffers,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_prepared_classic_sub_band_group_on_cpu_profile(
                            group,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                };
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                        let (buffers, status_check) =
                            encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                                runtime,
                                encoder,
                                group,
                                &output.buffer,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_prepared_ht_sub_band_group_on_cpu_profile(
                            group,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                };
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                step_idx = group.end_step;
                continue;
            }

            match &plan.steps[step_idx] {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let buffer = match tier1_mode {
                        DirectTier1Mode::Metal => {
                            let output = take_f32_scratch_buffer(
                                runtime,
                                sub_band.width as usize * sub_band.height as usize,
                            )?;
                            let (buffers, status_check) =
                                encode_prepared_classic_sub_band_to_buffer_in_encoder(
                                    runtime,
                                    encoder,
                                    sub_band,
                                    &output.buffer,
                                    scratch_buffers,
                                )?;
                            retained_buffers.extend(buffers);
                            status_checks.push(status_check);
                            let buffer = output.buffer.clone();
                            scratch_buffers.push(output);
                            buffer
                        }
                        DirectTier1Mode::CpuUpload => {
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_prepared_classic_sub_band_on_cpu_profile(
                                sub_band,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    };
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer,
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let buffer = match tier1_mode {
                        DirectTier1Mode::Metal => {
                            let output = take_f32_scratch_buffer(
                                runtime,
                                sub_band.width as usize * sub_band.height as usize,
                            )?;
                            let (buffers, status_check) =
                                encode_prepared_ht_sub_band_to_buffer_in_encoder(
                                    runtime,
                                    encoder,
                                    sub_band,
                                    &output.buffer,
                                )?;
                            retained_buffers.extend(buffers);
                            status_checks.push(status_check);
                            let buffer = output.buffer.clone();
                            scratch_buffers.push(output);
                            buffer
                        }
                        DirectTier1Mode::CpuUpload => {
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_prepared_ht_sub_band_on_cpu_profile(
                                sub_band,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    };
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer,
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                PreparedDirectGrayscaleStep::Idwt(idwt) => {
                    let ll =
                        lookup_direct_band_slice_entry(&bands, idwt.step.ll_band_id, idwt.step.ll)?;
                    let hl =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hl_band_id, idwt.step.hl)?;
                    let lh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.lh_band_id, idwt.step.lh)?;
                    let hh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hh_band_id, idwt.step.hh)?;
                    let params = prepared_idwt_params(
                        idwt,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                    );
                    let output = take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt))?;
                    let encode_started = profile_stages.then(Instant::now);
                    match idwt.step.transform {
                        J2kWaveletTransform::Reversible53 => {
                            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                                runtime,
                                encoder,
                                &ll.buffer,
                                ll.offset_bytes,
                                &hl.buffer,
                                hl.offset_bytes,
                                &lh.buffer,
                                lh.offset_bytes,
                                &hh.buffer,
                                hh.offset_bytes,
                                params,
                                &output.buffer,
                                0,
                            );
                        }
                        J2kWaveletTransform::Irreversible97 => {
                            status_checks.push(
                                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                                    runtime,
                                    encoder,
                                    &ll.buffer,
                                    ll.offset_bytes,
                                    &hl.buffer,
                                    hl.offset_bytes,
                                    &lh.buffer,
                                    lh.offset_bytes,
                                    &hh.buffer,
                                    hh.offset_bytes,
                                    params,
                                    &output.buffer,
                                    0,
                                ),
                            );
                        }
                    }
                    if let Some(started) = encode_started {
                        stage_timings.metal_idwt_encode += elapsed_us(started);
                    }
                    bands.push(DirectBandSlice {
                        band_id: idwt.step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: idwt.output_window,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Store(store) => {
                    let (input, input_offset) =
                        lookup_direct_band_slice(&bands, store.input_band_id, store.input_rect)?;
                    let output = take_f32_scratch_buffer(
                        runtime,
                        store.output_width as usize * store.output_height as usize,
                    )?;
                    let encode_started = profile_stages.then(Instant::now);
                    dispatch_store_component_buffer_in_encoder_with_offsets(
                        runtime,
                        encoder,
                        &input,
                        input_offset,
                        &output.buffer,
                        0,
                        J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                        },
                    );
                    if let Some(started) = encode_started {
                        stage_timings.metal_store_encode += elapsed_us(started);
                    }
                    final_plane = Some(output.buffer.clone());
                    scratch_buffers.push(output);
                }
            }
            step_idx += 1;
        }

        final_plane.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect component plan did not produce a stored plane".to_string(),
        })
    })();
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_repeated_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surfaces = encode_repeated_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            count,
            &mut retained_buffers,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surface = encode_prepared_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            &mut retained_buffers,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surface)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_with_device(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_prepared_direct_grayscale_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_batch(
    plans: &[Arc<PreparedDirectGrayscalePlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }

    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut retained_cpu_coefficients = Vec::<Vec<f32>>::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let mut stage_timings = DirectHybridStageTimings::default();
        let mut surfaces = Vec::with_capacity(plans.len());

        let component_plan_refs = plans.iter().map(Arc::as_ref).collect::<Vec<_>>();
        if plans.len() > 1 && supports_stacked_direct_component_plane_batch(&component_plan_refs) {
            let stacked_plane = encode_stacked_direct_component_plane_batch(
                runtime,
                DirectColorBatchCommandBuffers::single(command_buffer),
                &component_plan_refs,
                0,
                None,
                DirectTier1Mode::Metal,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )?;
            let first = plans.first().expect("plans is not empty");
            if stacked_plane.dimensions == first.dimensions && stacked_plane.count == plans.len() {
                surfaces = encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                    runtime,
                    command_buffer,
                    &stacked_plane.buffer,
                    first.dimensions,
                    first.bit_depth,
                    fmt,
                    plans.len(),
                )?;
            }
        }

        for plan in plans {
            if !surfaces.is_empty() {
                break;
            }
            surfaces.push(encode_prepared_direct_grayscale_plan_in_command_buffer(
                runtime,
                command_buffer,
                plan,
                fmt,
                &mut retained_buffers,
                &mut status_checks,
                &mut scratch_buffers,
            )?);
        }

        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        drop(retained_cpu_coefficients);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [Arc::new(plan.clone())];
    let mut surfaces = execute_prepared_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_with_device(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| execute_prepared_direct_color_plan(plan, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [Arc::new(plan.clone())];
    let mut surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect hybrid color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_with_device(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_hybrid_cpu_tier1_direct_color_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
fn execute_direct_color_plan_batch_with_tier1(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, tier1_mode, false)
}

#[cfg(all(target_os = "macos", test))]
fn execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, DirectTier1Mode::CpuUpload, true)
}

#[cfg(target_os = "macos")]
fn execute_direct_color_plan_batch_with_tier1_options(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
    force_flattened_cpu_tier1: bool,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }
    if tier1_mode == DirectTier1Mode::Metal
        && plans
            .iter()
            .any(|plan| !prepared_direct_color_plan_supports_runtime(plan, fmt))
    {
        return Err(Error::MetalKernel {
            message: "unsupported classic kernel input in direct component plan".to_string(),
        });
    }

    with_runtime(|runtime| {
        let mut retained_buffers = Vec::new();
        let mut retained_cpu_coefficients = Vec::<Vec<f32>>::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let mut stage_timings = DirectHybridStageTimings::default();
        let profile_hybrid_stages =
            tier1_mode == DirectTier1Mode::CpuUpload && metal_profile_stages_enabled();

        if fmt == PixelFormat::Rgb8
            && profile_hybrid_stages
            && metal_profile_decode_split_commands_enabled()
        {
            let split_command_buffers = DecodeHybridSplitCommandBuffers::new(runtime);
            if let Some(surfaces) = try_encode_stacked_mct_rgb8_direct_color_batch(
                runtime,
                split_command_buffers.refs(),
                plans,
                tier1_mode,
                force_flattened_cpu_tier1,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )? {
                split_command_buffers.commit_in_order();
                let wait_started = Instant::now();
                let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
                wait_for_completion_metal(&split_command_buffers.mct_pack)?;
                stage_timings.command_wait += elapsed_us(wait_started);
                record_completed_decode_split_gpu_stages(
                    &mut stage_timings,
                    &split_command_buffers,
                );
                for status_check in status_checks {
                    validate_direct_status(status_check)?;
                }
                emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
                drop(retained_buffers);
                drop(retained_cpu_coefficients);
                recycle_scratch_buffers(runtime, scratch_buffers)?;
                return Ok(surfaces);
            }

            drop(split_command_buffers);
            retained_buffers.clear();
            retained_cpu_coefficients.clear();
            status_checks.clear();
            scratch_buffers.clear();
            stage_timings = DirectHybridStageTimings::default();
        }

        let command_buffer = runtime.queue.new_command_buffer();
        if profile_hybrid_stages {
            label_command_buffer(command_buffer, "j2k decode hybrid direct color batch");
        }

        if fmt == PixelFormat::Rgb8 {
            if let Some(surfaces) = try_encode_stacked_mct_rgb8_direct_color_batch(
                runtime,
                DirectColorBatchCommandBuffers::single(command_buffer),
                plans,
                tier1_mode,
                force_flattened_cpu_tier1,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )? {
                command_buffer.commit();
                let wait_started = profile_hybrid_stages.then(Instant::now);
                let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
                wait_for_completion_metal(command_buffer)?;
                if let Some(started) = wait_started {
                    stage_timings.command_wait += elapsed_us(started);
                }
                if profile_hybrid_stages {
                    if let Some(duration) = completed_command_buffer_gpu_duration(command_buffer) {
                        stage_timings.gpu_command += duration.as_micros();
                    }
                }
                for status_check in status_checks {
                    validate_direct_status(status_check)?;
                }
                if tier1_mode == DirectTier1Mode::CpuUpload {
                    emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
                }
                drop(retained_buffers);
                drop(retained_cpu_coefficients);
                recycle_scratch_buffers(runtime, scratch_buffers)?;
                return Ok(surfaces);
            }
        }

        let mut surfaces = Vec::with_capacity(plans.len());

        for plan in plans {
            let surface = encode_prepared_direct_color_plan_in_command_buffer(
                runtime,
                command_buffer,
                plan,
                fmt,
                tier1_mode,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )?;
            surfaces.push(surface);
        }

        command_buffer.commit();
        let wait_started = profile_hybrid_stages.then(Instant::now);
        let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
        wait_for_completion_metal(command_buffer)?;
        if let Some(started) = wait_started {
            stage_timings.command_wait += elapsed_us(started);
        }
        if profile_hybrid_stages {
            if let Some(duration) = completed_command_buffer_gpu_duration(command_buffer) {
                stage_timings.gpu_command += duration.as_micros();
            }
        }
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        if tier1_mode == DirectTier1Mode::CpuUpload {
            emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
        }
        drop(retained_buffers);
        drop(retained_cpu_coefficients);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
fn signed_sample_bias(bit_depth: u8) -> f32 {
    2.0_f32.powi(i32::from(bit_depth) - 1)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_prepared_direct_color_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
    if plan.component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color execution expected 3 component plans, got {}",
                plan.component_plans.len()
            ),
        });
    }

    let mut planes = Vec::with_capacity(3);
    for component_plan in &plan.component_plans {
        planes.push(encode_prepared_direct_component_plane_in_command_buffer(
            runtime,
            command_buffer,
            component_plan,
            tier1_mode,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
            status_checks,
            scratch_buffers,
        )?);
    }

    if plan.mct && fmt == PixelFormat::Rgb8 {
        let encode_started = metal_profile_stages_enabled().then(Instant::now);
        let surface = encode_mct_rgb8_to_surface_in_command_buffer(
            runtime,
            command_buffer,
            [&planes[0], &planes[1], &planes[2]],
            plan.dimensions,
            plan.bit_depths,
            plan.transform,
        )?;
        if let Some(started) = encode_started {
            stage_timings.metal_mct_pack_encode += elapsed_us(started);
        }
        return Ok(surface);
    }

    if plan.mct {
        let len = plan.dimensions.0 as usize * plan.dimensions.1 as usize;
        let encode_started = metal_profile_stages_enabled().then(Instant::now);
        status_checks.push(dispatch_inverse_mct_buffers_in_command_buffer(
            runtime,
            command_buffer,
            [&planes[0], &planes[1], &planes[2]],
            len,
            plan.transform,
            [
                signed_sample_bias(plan.bit_depths[0]),
                signed_sample_bias(plan.bit_depths[1]),
                signed_sample_bias(plan.bit_depths[2]),
            ],
        )?);
        if let Some(started) = encode_started {
            stage_timings.metal_mct_pack_encode += elapsed_us(started);
        }
    }

    let stage = PlaneStage {
        dims: plan.dimensions,
        plane_count: 3,
        color_space: NativeColorSpace::RGB,
        has_alpha: false,
        bit_depths: [
            u32::from(plan.bit_depths[0]),
            u32::from(plan.bit_depths[1]),
            u32::from(plan.bit_depths[2]),
            0,
        ],
        planes: [
            Some(planes[0].clone()),
            Some(planes[1].clone()),
            Some(planes[2].clone()),
            None,
        ],
    };
    let encode_started = metal_profile_stages_enabled().then(Instant::now);
    let surface =
        encode_plane_stage_to_surface_in_command_buffer(runtime, command_buffer, &stage, fmt);
    if let Some(started) = encode_started {
        stage_timings.metal_mct_pack_encode += elapsed_us(started);
    }
    surface
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct DirectBandSlice {
    band_id: J2kDirectBandId,
    buffer: Buffer,
    offset_bytes: usize,
    window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
fn lookup_direct_band_slice_entry(
    bands: &[DirectBandSlice],
    band_id: J2kDirectBandId,
    rect: j2k_native::J2kRect,
) -> Result<DirectBandSlice, Error> {
    bands
        .iter()
        .find(|existing| existing.band_id == band_id)
        .cloned()
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "missing J2K MetalDirect device band {} for rect ({}, {}, {}, {})",
                band_id, rect.x0, rect.y0, rect.x1, rect.y1
            ),
        })
}

#[cfg(target_os = "macos")]
fn lookup_direct_band_slice(
    bands: &[DirectBandSlice],
    band_id: J2kDirectBandId,
    rect: j2k_native::J2kRect,
) -> Result<(Buffer, usize), Error> {
    let entry = lookup_direct_band_slice_entry(bands, band_id, rect)?;
    Ok((entry.buffer, entry.offset_bytes))
}

#[cfg(target_os = "macos")]
fn lookup_repeated_direct_band_layout_entry(
    band_sets: &[Vec<DirectBandSlice>],
    band_id: J2kDirectBandId,
    rect: j2k_native::J2kRect,
) -> Result<(DirectBandSlice, u32), Error> {
    let first_bands = band_sets.first().ok_or_else(|| Error::MetalKernel {
        message: "missing J2K MetalDirect repeated band set".to_string(),
    })?;
    let entry = lookup_direct_band_slice_entry(first_bands, band_id, rect)?;
    let stride_bytes = if let Some(second_bands) = band_sets.get(1) {
        let next = lookup_direct_band_slice_entry(second_bands, band_id, rect)?;
        next.offset_bytes
            .checked_sub(entry.offset_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect repeated band offsets are not monotonic".to_string(),
            })?
    } else {
        entry.window.width() as usize * entry.window.height() as usize * size_of::<f32>()
    };
    if stride_bytes % size_of::<f32>() != 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect repeated band stride is not f32-aligned".to_string(),
        });
    }
    let stride_elements =
        u32::try_from(stride_bytes / size_of::<f32>()).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect repeated band stride exceeds u32".to_string(),
        })?;
    Ok((entry, stride_elements))
}

#[cfg(target_os = "macos")]
struct StackedDirectComponentPlane {
    buffer: Buffer,
    dimensions: (u32, u32),
    count: usize,
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn try_encode_stacked_mct_rgb8_direct_color_batch(
    runtime: &MetalRuntime,
    command_buffers: DirectColorBatchCommandBuffers<'_>,
    plans: &[Arc<PreparedDirectColorPlan>],
    tier1_mode: DirectTier1Mode,
    force_flattened_cpu_tier1: bool,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Option<Vec<Surface>>, Error> {
    let Some(first) = plans.first() else {
        return Ok(Some(Vec::new()));
    };
    let repeated_count = repeated_shared_direct_color_plan_count(plans);
    if plans.len() <= 1
        || !first.mct
        || first.component_plans.len() != 3
        || !plans.iter().all(|plan| {
            plan.mct
                && plan.dimensions == first.dimensions
                && plan.bit_depths == first.bit_depths
                && plan.transform == first.transform
                && plan.component_plans.len() == 3
        })
    {
        return Ok(None);
    }
    let execution_plans = if repeated_count.is_some() {
        &plans[..1]
    } else {
        plans
    };

    let flattened_cpu_tier1_cache = if tier1_mode == DirectTier1Mode::CpuUpload
        && (force_flattened_cpu_tier1
            || flattened_hybrid_cpu_tier1_enabled()
            || should_flatten_hybrid_cpu_tier1_color_batch(execution_plans))
    {
        Some(build_flattened_cpu_tier1_cache(
            runtime,
            execution_plans,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
        )?)
    } else {
        None
    };

    let mut stacked_planes = Vec::with_capacity(3);
    for component_idx in 0..3 {
        let component_plan_refs = execution_plans
            .iter()
            .map(|plan| &plan.component_plans[component_idx])
            .collect::<Vec<_>>();
        if !supports_stacked_direct_component_plane_batch(&component_plan_refs) {
            return Ok(None);
        }
        stacked_planes.push(encode_stacked_direct_component_plane_batch(
            runtime,
            command_buffers,
            &component_plan_refs,
            component_idx,
            flattened_cpu_tier1_cache.as_ref(),
            tier1_mode,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
            status_checks,
            scratch_buffers,
        )?);
    }

    if !stacked_planes
        .iter()
        .all(|plane| plane.dimensions == first.dimensions && plane.count == execution_plans.len())
    {
        return Ok(None);
    }

    let encode_started = metal_profile_stages_enabled().then(Instant::now);
    let mct_plane_buffers = [
        &stacked_planes[0].buffer,
        &stacked_planes[1].buffer,
        &stacked_planes[2].buffer,
    ];
    let surfaces = if let Some(count) = repeated_count {
        encode_repeated_mct_rgb8_to_surfaces_in_command_buffer(
            runtime,
            command_buffers.mct_pack,
            mct_plane_buffers,
            first.dimensions,
            count,
            first.bit_depths,
            first.transform,
        )?
    } else {
        encode_batched_mct_rgb8_to_surfaces_in_command_buffer(
            runtime,
            command_buffers.mct_pack,
            mct_plane_buffers,
            first.dimensions,
            execution_plans.len(),
            first.bit_depths,
            first.transform,
        )?
    };
    if let Some(started) = encode_started {
        stage_timings.metal_mct_pack_encode += elapsed_us(started);
    }
    Ok(Some(surfaces))
}

#[cfg(target_os = "macos")]
fn supports_stacked_direct_component_plane_batch(plans: &[&PreparedDirectGrayscalePlan]) -> bool {
    let Some(first) = plans.first() else {
        return false;
    };
    if plans.iter().any(|plan| {
        plan.dimensions != first.dimensions
            || plan.bit_depth != first.bit_depth
            || plan.steps.len() != first.steps.len()
    }) {
        return false;
    }

    let mut step_idx = 0;
    while step_idx < first.steps.len() {
        if let Some(group) = first.classic_group_starting_at(step_idx) {
            if group.end_step <= step_idx
                || !plans.iter().all(|plan| {
                    plan.classic_group_starting_at(step_idx)
                        .is_some_and(|other| classic_group_shapes_match(group, other))
                })
            {
                return false;
            }
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = first.ht_group_starting_at(step_idx) {
            if group.end_step <= step_idx
                || !plans.iter().all(|plan| {
                    plan.ht_group_starting_at(step_idx)
                        .is_some_and(|other| ht_group_shapes_match(group, other))
                })
            {
                return false;
            }
            step_idx = group.end_step;
            continue;
        }

        match &first.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::ClassicSubBand(other)
                            if classic_sub_band_shapes_match(sub_band, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::HtSubBand(other)
                            if ht_sub_band_shapes_match(sub_band, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::Idwt(other)
                            if idwt_shapes_match(idwt, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::Store(other)
                            if store_shapes_match(store, other)
                    )
                }) {
                    return false;
                }
            }
        }
        step_idx += 1;
    }

    true
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_stacked_direct_component_plane_batch(
    runtime: &MetalRuntime,
    command_buffers: DirectColorBatchCommandBuffers<'_>,
    plans: &[&PreparedDirectGrayscalePlan],
    component_idx: usize,
    flattened_cpu_tier1_cache: Option<&FlattenedCpuTier1Cache>,
    tier1_mode: DirectTier1Mode,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<StackedDirectComponentPlane, Error> {
    let Some(first) = plans.first() else {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect color batch has no component plans".to_string(),
        });
    };

    let count = plans.len();
    let broadcast_tier1_inputs = tier1_mode == DirectTier1Mode::CpuUpload
        && plans.iter().all(|plan| std::ptr::eq(*plan, *first));
    let mut band_sets = vec![Vec::<DirectBandSlice>::new(); count];
    let mut final_plane = None;
    let mut step_idx = 0;
    let profile_stages = tier1_mode == DirectTier1Mode::CpuUpload && metal_profile_stages_enabled();

    while step_idx < first.steps.len() {
        if let Some(group) = first.classic_group_starting_at(step_idx) {
            let groups = plans
                .iter()
                .map(|plan| {
                    plan.classic_group_starting_at(step_idx)
                        .expect("preflight validated classic group")
                })
                .collect::<Vec<_>>();
            let buffer = match tier1_mode {
                DirectTier1Mode::Metal => {
                    let output =
                        take_f32_scratch_buffer(runtime, group.total_coefficients * count)?;
                    let (buffers, status_check) =
                        encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
                            runtime,
                            command_buffers.default,
                            &groups,
                            &output.buffer,
                            scratch_buffers,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    let buffer = output.buffer.clone();
                    scratch_buffers.push(output);
                    buffer
                }
                DirectTier1Mode::CpuUpload => {
                    let input_groups = if broadcast_tier1_inputs {
                        &groups[..1]
                    } else {
                        &groups
                    };
                    if let Some(cache) = flattened_cpu_tier1_cache {
                        cache.buffer_for(
                            component_idx,
                            step_idx,
                            group.total_coefficients,
                            input_groups.len(),
                        )?
                    } else {
                        let inputs = input_groups
                            .iter()
                            .map(|group| ClassicCpuDecodeInput {
                                coded_data: &group.coded_data,
                                segments: &group.segments,
                                jobs: &group.jobs,
                                output_len: group.total_coefficients,
                            })
                            .collect::<Vec<_>>();
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_classic_inputs_on_cpu_with_plan_cache(
                            first,
                            step_idx,
                            &inputs,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                }
            };
            let stride_bytes = group.total_coefficients * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                let source_group = if broadcast_tier1_inputs {
                    groups[0]
                } else {
                    groups[instance_idx]
                };
                let instance_offset = if broadcast_tier1_inputs {
                    0
                } else {
                    instance_idx * stride_bytes
                };
                for member in &source_group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            step_idx = group.end_step;
            continue;
        }

        if let Some(group) = first.ht_group_starting_at(step_idx) {
            let groups = plans
                .iter()
                .map(|plan| {
                    plan.ht_group_starting_at(step_idx)
                        .expect("preflight validated HT group")
                })
                .collect::<Vec<_>>();
            let buffer = match tier1_mode {
                DirectTier1Mode::Metal => {
                    let output =
                        take_f32_scratch_buffer(runtime, group.total_coefficients * count)?;
                    let (buffers, status_check) =
                        encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer(
                            runtime,
                            command_buffers.default,
                            &groups,
                            &output.buffer,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    let buffer = output.buffer.clone();
                    scratch_buffers.push(output);
                    buffer
                }
                DirectTier1Mode::CpuUpload => {
                    let input_groups = if broadcast_tier1_inputs {
                        &groups[..1]
                    } else {
                        &groups
                    };
                    if let Some(cache) = flattened_cpu_tier1_cache {
                        cache.buffer_for(
                            component_idx,
                            step_idx,
                            group.total_coefficients,
                            input_groups.len(),
                        )?
                    } else {
                        let inputs = input_groups
                            .iter()
                            .map(|group| HtCpuDecodeInput {
                                coded_data: &group.coded_arena.data,
                                jobs: &group.jobs,
                                output_len: group.total_coefficients,
                            })
                            .collect::<Vec<_>>();
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_ht_inputs_on_cpu_with_plan_cache(
                            first,
                            step_idx,
                            &inputs,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                }
            };
            let stride_bytes = group.total_coefficients * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                let source_group = if broadcast_tier1_inputs {
                    groups[0]
                } else {
                    groups[instance_idx]
                };
                let instance_offset = if broadcast_tier1_inputs {
                    0
                } else {
                    instance_idx * stride_bytes
                };
                for member in &source_group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            step_idx = group.end_step;
            continue;
        }

        match &first.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let sub_bands = plans
                    .iter()
                    .map(|plan| match &plan.steps[step_idx] {
                        PreparedDirectGrayscaleStep::ClassicSubBand(other) => Ok(other),
                        _ => Err(direct_preflight_invariant(
                            "classic sub-band step mismatch in stacked component batch",
                        )),
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                        let (buffers, status_check) =
                            encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
                                runtime,
                                command_buffers.default,
                                &sub_bands,
                                &output.buffer,
                                scratch_buffers,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let input_sub_bands = if broadcast_tier1_inputs {
                            &sub_bands[..1]
                        } else {
                            &sub_bands
                        };
                        if let Some(cache) = flattened_cpu_tier1_cache {
                            cache.buffer_for(
                                component_idx,
                                step_idx,
                                per_instance_len,
                                input_sub_bands.len(),
                            )?
                        } else {
                            let inputs = input_sub_bands
                                .iter()
                                .map(|sub_band| ClassicCpuDecodeInput {
                                    coded_data: &sub_band.coded_data,
                                    segments: &sub_band.segments,
                                    jobs: &sub_band.jobs,
                                    output_len: per_instance_len,
                                })
                                .collect::<Vec<_>>();
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_classic_inputs_on_cpu_with_plan_cache(
                                first,
                                step_idx,
                                &inputs,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    }
                };
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    let source_sub_band = if broadcast_tier1_inputs {
                        sub_bands[0]
                    } else {
                        sub_bands[instance_idx]
                    };
                    let instance_offset = if broadcast_tier1_inputs {
                        0
                    } else {
                        instance_idx * stride_bytes
                    };
                    bands.push(DirectBandSlice {
                        band_id: source_sub_band.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset,
                        window: BandRequiredRegion::full(
                            source_sub_band.width,
                            source_sub_band.height,
                        ),
                    });
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let sub_bands = plans
                    .iter()
                    .map(|plan| match &plan.steps[step_idx] {
                        PreparedDirectGrayscaleStep::HtSubBand(other) => Ok(other),
                        _ => Err(direct_preflight_invariant(
                            "HT sub-band step mismatch in stacked component batch",
                        )),
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                        let (buffers, status_check) =
                            encode_distinct_ht_sub_bands_to_buffer_in_command_buffer(
                                runtime,
                                command_buffers.default,
                                &sub_bands,
                                &output.buffer,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let input_sub_bands = if broadcast_tier1_inputs {
                            &sub_bands[..1]
                        } else {
                            &sub_bands
                        };
                        if let Some(cache) = flattened_cpu_tier1_cache {
                            cache.buffer_for(
                                component_idx,
                                step_idx,
                                per_instance_len,
                                input_sub_bands.len(),
                            )?
                        } else {
                            let inputs = input_sub_bands
                                .iter()
                                .map(|sub_band| HtCpuDecodeInput {
                                    coded_data: &sub_band.coded_data,
                                    jobs: &sub_band.jobs,
                                    output_len: per_instance_len,
                                })
                                .collect::<Vec<_>>();
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_ht_inputs_on_cpu_with_plan_cache(
                                first,
                                step_idx,
                                &inputs,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    }
                };
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    let source_sub_band = if broadcast_tier1_inputs {
                        sub_bands[0]
                    } else {
                        sub_bands[instance_idx]
                    };
                    let instance_offset = if broadcast_tier1_inputs {
                        0
                    } else {
                        instance_idx * stride_bytes
                    };
                    bands.push(DirectBandSlice {
                        band_id: source_sub_band.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset,
                        window: BandRequiredRegion::full(
                            source_sub_band.width,
                            source_sub_band.height,
                        ),
                    });
                }
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                let per_instance_len = prepared_idwt_output_len(idwt);
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                let encode_started = profile_stages.then(Instant::now);
                match idwt.step.transform {
                    J2kWaveletTransform::Reversible53 => {
                        let (ll, low_low_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.ll_band_id,
                            idwt.step.ll,
                        )?;
                        let (hl, high_low_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.hl_band_id,
                            idwt.step.hl,
                        )?;
                        let (lh, low_high_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.lh_band_id,
                            idwt.step.lh,
                        )?;
                        let (hh, high_high_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.hh_band_id,
                            idwt.step.hh,
                        )?;
                        let params = repeated_idwt_params(
                            idwt,
                            idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                            PreparedIdwtInputStrides {
                                ll: low_low_stride,
                                hl: high_low_stride,
                                lh: low_high_stride,
                                hh: high_high_stride,
                            },
                            count,
                            "color",
                        )?;
                        dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
                            runtime,
                            command_buffers.idwt,
                            &ll.buffer,
                            ll.offset_bytes,
                            &hl.buffer,
                            hl.offset_bytes,
                            &lh.buffer,
                            lh.offset_bytes,
                            &hh.buffer,
                            hh.offset_bytes,
                            params,
                            &output.buffer,
                        );
                    }
                    J2kWaveletTransform::Irreversible97 => {
                        for (instance_idx, bands) in band_sets.iter().enumerate() {
                            let PreparedDirectGrayscaleStep::Idwt(step) =
                                &plans[instance_idx].steps[step_idx]
                            else {
                                return Err(direct_preflight_invariant(
                                    "IDWT step mismatch in stacked component batch",
                                ));
                            };
                            let ll = lookup_direct_band_slice_entry(
                                bands,
                                step.step.ll_band_id,
                                step.step.ll,
                            )?;
                            let hl = lookup_direct_band_slice_entry(
                                bands,
                                step.step.hl_band_id,
                                step.step.hl,
                            )?;
                            let lh = lookup_direct_band_slice_entry(
                                bands,
                                step.step.lh_band_id,
                                step.step.lh,
                            )?;
                            let hh = lookup_direct_band_slice_entry(
                                bands,
                                step.step.hh_band_id,
                                step.step.hh,
                            )?;
                            let params = prepared_idwt_params(
                                step,
                                idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                            );
                            status_checks.push(
                                dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                                    runtime,
                                    command_buffers.idwt.interleave,
                                    &ll.buffer,
                                    ll.offset_bytes,
                                    &hl.buffer,
                                    hl.offset_bytes,
                                    &lh.buffer,
                                    lh.offset_bytes,
                                    &hh.buffer,
                                    hh.offset_bytes,
                                    params,
                                    &output.buffer,
                                    instance_idx * per_instance_len * size_of::<f32>(),
                                ),
                            );
                        }
                    }
                }
                if let Some(started) = encode_started {
                    stage_timings.metal_idwt_encode += elapsed_us(started);
                }
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    let PreparedDirectGrayscaleStep::Idwt(step) =
                        &plans[instance_idx].steps[step_idx]
                    else {
                        return Err(direct_preflight_invariant(
                            "IDWT output step mismatch in stacked component batch",
                        ));
                    };
                    bands.push(DirectBandSlice {
                        band_id: step.step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                        window: step.output_window,
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                let (input, input_instance_stride) = lookup_repeated_direct_band_layout_entry(
                    &band_sets,
                    store.input_band_id,
                    store.input_rect,
                )?;
                let per_instance_len = store.output_width as usize * store.output_height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                let encode_started = profile_stages.then(Instant::now);
                dispatch_store_component_repeated_in_command_buffer(
                    runtime,
                    command_buffers.store,
                    &input.buffer,
                    input.offset_bytes,
                    &output.buffer,
                    J2kRepeatedStoreParams {
                        input_width: store.input_rect.width(),
                        input_height: store.input_rect.height(),
                        input_instance_stride,
                        source_x: store.source_x,
                        source_y: store.source_y,
                        copy_width: store.copy_width,
                        copy_height: store.copy_height,
                        output_width: store.output_width,
                        output_height: store.output_height,
                        output_x: store.output_x,
                        output_y: store.output_y,
                        addend: store.addend,
                        batch_count: u32::try_from(count).map_err(|_| Error::MetalKernel {
                            message: "J2K MetalDirect color store batch count exceeds u32"
                                .to_string(),
                        })?,
                    },
                );
                if let Some(started) = encode_started {
                    stage_timings.metal_store_encode += elapsed_us(started);
                }
                final_plane = Some(output.buffer.clone());
                scratch_buffers.push(output);
            }
        }
        step_idx += 1;
    }

    let buffer = final_plane.ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color component batch did not produce a final plane".to_string(),
    })?;
    record_hybrid_stacked_component_batch(tier1_mode);
    Ok(StackedDirectComponentPlane {
        buffer,
        dimensions: first.dimensions,
        count,
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_repeated_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Vec<Surface>, Error> {
    let mut band_sets = vec![Vec::<DirectBandSlice>::new(); count];
    let mut surfaces = Vec::with_capacity(count);
    let mut stacked_outputs = true;
    let mut step_idx = 0;

    while step_idx < plan.steps.len() {
        if let Some(group) = plan.classic_group_starting_at(step_idx) {
            let per_instance_len = group.total_coefficients;
            let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
            let (buffers, status_check) =
                encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    group,
                    count,
                    &output.buffer,
                    scratch_buffers,
                )?;
            retained_buffers.extend(buffers);
            status_checks.push(status_check);
            let stride_bytes = per_instance_len * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes
                            + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            scratch_buffers.push(output);
            step_idx = group.end_step;
            continue;
        }

        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            let per_instance_len = group.total_coefficients;
            let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
            let (buffers, status_check) =
                encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    group,
                    count,
                    &output.buffer,
                )?;
            retained_buffers.extend(buffers);
            status_checks.push(status_check);
            let stride_bytes = per_instance_len * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes
                            + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            scratch_buffers.push(output);
            step_idx = group.end_step;
            continue;
        }

        let step = &plan.steps[step_idx];
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                let (buffers, status_check) =
                    encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        count,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                let (buffers, status_check) =
                    encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        count,
                        &output.buffer,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => match idwt.step.transform {
                J2kWaveletTransform::Reversible53 if stacked_outputs => {
                    let (ll, low_low_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.ll_band_id,
                        idwt.step.ll,
                    )?;
                    let (hl, high_low_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.hl_band_id,
                        idwt.step.hl,
                    )?;
                    let (lh, low_high_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.lh_band_id,
                        idwt.step.lh,
                    )?;
                    let (hh, high_high_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.hh_band_id,
                        idwt.step.hh,
                    )?;
                    let params = repeated_idwt_params(
                        idwt,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                        PreparedIdwtInputStrides {
                            ll: low_low_stride,
                            hl: high_low_stride,
                            lh: low_high_stride,
                            hh: high_high_stride,
                        },
                        count,
                        "repeated",
                    )?;
                    let per_instance_len = prepared_idwt_output_len(idwt);
                    let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
                        runtime,
                        DirectIdwtCommandBuffers::single(command_buffer),
                        &ll.buffer,
                        ll.offset_bytes,
                        &hl.buffer,
                        hl.offset_bytes,
                        &lh.buffer,
                        lh.offset_bytes,
                        &hh.buffer,
                        hh.offset_bytes,
                        params,
                        &output.buffer,
                    );
                    let stride_bytes = per_instance_len * size_of::<f32>();
                    for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                        bands.push(DirectBandSlice {
                            band_id: idwt.step.output_band_id,
                            buffer: output.buffer.clone(),
                            offset_bytes: instance_idx * stride_bytes,
                            window: idwt.output_window,
                        });
                    }
                    scratch_buffers.push(output);
                }
                _ => {
                    stacked_outputs = false;
                    for bands in &mut band_sets {
                        let ll = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.ll_band_id,
                            idwt.step.ll,
                        )?;
                        let hl = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.hl_band_id,
                            idwt.step.hl,
                        )?;
                        let lh = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.lh_band_id,
                            idwt.step.lh,
                        )?;
                        let hh = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.hh_band_id,
                            idwt.step.hh,
                        )?;
                        let params = prepared_idwt_params(
                            idwt,
                            idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                        );
                        let output =
                            take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt))?;
                        match idwt.step.transform {
                                J2kWaveletTransform::Reversible53 => {
                                    dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
                                        runtime,
                                        command_buffer,
                                        &ll.buffer,
                                        ll.offset_bytes,
                                        &hl.buffer,
                                        hl.offset_bytes,
                                        &lh.buffer,
                                        lh.offset_bytes,
                                        &hh.buffer,
                                        hh.offset_bytes,
                                        params,
                                        &output.buffer,
                                        0,
                                    );
                                }
                                J2kWaveletTransform::Irreversible97 => status_checks.push(
                                    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                                        runtime,
                                        command_buffer,
                                        &ll.buffer,
                                        ll.offset_bytes,
                                        &hl.buffer,
                                        hl.offset_bytes,
                                        &lh.buffer,
                                        lh.offset_bytes,
                                        &hh.buffer,
                                        hh.offset_bytes,
                                        params,
                                        &output.buffer,
                                        0,
                                    ),
                                ),
                            }
                        bands.push(DirectBandSlice {
                            band_id: idwt.step.output_band_id,
                            buffer: output.buffer.clone(),
                            offset_bytes: 0,
                            window: idwt.output_window,
                        });
                        scratch_buffers.push(output);
                    }
                }
            },
            PreparedDirectGrayscaleStep::Store(store) => {
                if stacked_outputs {
                    let (input, _) = lookup_direct_band_slice(
                        &band_sets[0],
                        store.input_band_id,
                        store.input_rect,
                    )?;
                    let batch_count = u32::try_from(count).map_err(|_| Error::MetalKernel {
                        message: "J2K MetalDirect repeated store batch count exceeds u32"
                            .to_string(),
                    })?;
                    if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
                        let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
                        surfaces.extend(encode_repeated_gray_store_to_surfaces_in_command_buffer(
                            runtime,
                            command_buffer,
                            &input,
                            J2kRepeatedGrayStoreParams {
                                input_width: store.input_rect.width(),
                                input_height: store.input_rect.height(),
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_height: store.output_height,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                batch_count,
                                max_value: scale.max_value,
                                u8_scale: scale.u8_scale,
                                u16_scale: scale.u16_scale,
                            },
                            plan.dimensions,
                            fmt,
                            count,
                        )?);
                    } else {
                        let per_instance_len =
                            store.output_width as usize * store.output_height as usize;
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                        dispatch_store_component_repeated_in_command_buffer(
                            runtime,
                            command_buffer,
                            &input,
                            0,
                            &output.buffer,
                            J2kRepeatedStoreParams {
                                input_width: store.input_rect.width(),
                                input_height: store.input_rect.height(),
                                input_instance_stride: store
                                    .input_rect
                                    .width()
                                    .checked_mul(store.input_rect.height())
                                    .ok_or_else(|| Error::MetalKernel {
                                        message: "J2K MetalDirect repeated store input stride overflows u32"
                                            .to_string(),
                                    })?,
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_height: store.output_height,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                batch_count,
                            },
                        );
                        retained_buffers.push(output.buffer.clone());
                        surfaces.extend(encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                            runtime,
                            command_buffer,
                            &output.buffer,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                            count,
                        )?);
                        scratch_buffers.push(output);
                    }
                } else {
                    for bands in &band_sets {
                        let (input, input_offset) =
                            lookup_direct_band_slice(bands, store.input_band_id, store.input_rect)?;
                        let output = take_f32_scratch_buffer(
                            runtime,
                            store.output_width as usize * store.output_height as usize,
                        )?;
                        let params = J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                        };
                        dispatch_store_component_buffer_in_command_buffer_with_offsets(
                            runtime,
                            command_buffer,
                            &input,
                            input_offset,
                            &output.buffer,
                            0,
                            params,
                        );
                        retained_buffers.push(output.buffer.clone());
                        surfaces.push(encode_gray_plane_to_surface_in_command_buffer_with_offset(
                            runtime,
                            command_buffer,
                            &output.buffer,
                            0,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                        )?);
                        scratch_buffers.push(output);
                    }
                }
            }
        }
        step_idx += 1;
    }

    if surfaces.len() != count {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect repeated grayscale plan produced {} surfaces for count {}",
                surfaces.len(),
                count
            ),
        });
    }

    Ok(surfaces)
}

#[cfg(target_os = "macos")]
fn copy_plane_samples(buffer: &Buffer, samples: &[f32], image_width: usize, roi: Rect) {
    let row_width = roi.w as usize;
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), row_width * roi.h as usize)
    };

    for row in 0..roi.h as usize {
        let src_start = (roi.y as usize + row) * image_width + roi.x as usize;
        let src_end = src_start + row_width;
        let dst_start = row * row_width;
        dst[dst_start..dst_start + row_width].copy_from_slice(&samples[src_start..src_end]);
    }
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    encode_gray_plane_to_surface_in_encoder_with_offset(
        runtime, encoder, plane, 0, dims, bit_depth, fmt,
    )
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_command_buffer_with_offset(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    plane_offset_bytes: usize,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = encode_gray_plane_to_surface_in_encoder_with_offset(
        runtime,
        encoder,
        plane,
        plane_offset_bytes,
        dims,
        bit_depth,
        fmt,
    );
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_encoder_with_offset(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    plane: &Buffer,
    plane_offset_bytes: usize,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (output_channels, opaque_alpha, pipeline) =
        output_shape_for(&NativeColorSpace::Gray, false, 1, fmt, runtime)?;
    let mut bit_depths = [0u32; 4];
    bit_depths[0] = u32::from(bit_depth);
    let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays(bit_depths);
    let params = J2kPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        output_channels,
        opaque_alpha,
        max_values,
        u8_scales,
        u16_scales,
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), plane_offset_bytes as u64);
    encoder.set_buffer(1, None, 0);
    encoder.set_buffer(2, None, 0);
    encoder.set_buffer(3, None, 0);
    encoder.set_buffer(4, Some(&out_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, pipeline, dims);

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
fn encode_repeated_gray_plane_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    let count_u32 = u32::try_from(count).map_err(|_| Error::MetalKernel {
        message: "J2K Metal repeated grayscale surface count exceeds u32".to_string(),
    })?;
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal repeated grayscale surface size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal repeated grayscale output size overflow".to_string(),
        })?;
    let out_buffer = runtime
        .device
        .new_buffer(total_bytes as u64, MTLResourceOptions::StorageModeShared);
    let scale = j2k_scalar_pack_params(u32::from(bit_depth));
    let params = J2kRepeatedGrayPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        batch_count: count_u32,
        max_value: scale.max_value,
        u8_scale: scale.u8_scale,
        u16_scale: scale.u16_scale,
    };
    let pipeline = match fmt {
        PixelFormat::Gray8 => &runtime.pack_u8_repeated_gray,
        PixelFormat::Gray16 => &runtime.pack_u16_repeated_gray,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("J2K Metal repeated grayscale pack does not support {fmt:?}"),
            })
        }
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), 0);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedGrayPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(encoder, pipeline, (dims.0, dims.1, count_u32));
    encoder.end_encoding();

    let mut surfaces = Vec::with_capacity(count);
    for instance_idx in 0..count {
        surfaces.push(Surface::from_metal_buffer_with_offset(
            out_buffer.clone(),
            dims,
            fmt,
            instance_idx * surface_bytes,
        ));
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
fn j2k_pack_kernel_name_for(
    color_space: &NativeColorSpace,
    has_alpha: bool,
    plane_count: usize,
    fmt: PixelFormat,
) -> Option<&'static str> {
    match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8) => Some("j2k_pack_gray8"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => Some("j2k_pack_rgb8"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => Some("j2k_pack_rgb_opaque_rgba8"),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => Some("j2k_pack_rgba8"),
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray16) => Some("j2k_pack_gray16"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb16) => Some("j2k_pack_rgb16"),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn j2k_pack_pipeline_for<'a>(
    runtime: &'a MetalRuntime,
    kernel_name: &str,
) -> Result<&'a ComputePipelineState, Error> {
    let pipeline = match kernel_name {
        "j2k_pack_gray8" => &runtime.pack_gray8,
        "j2k_pack_rgb8" => &runtime.pack_rgb8,
        "j2k_pack_rgb_opaque_rgba8" => &runtime.pack_rgb_opaque_rgba8,
        "j2k_pack_rgba8" => &runtime.pack_rgba8,
        "j2k_pack_gray16" => &runtime.pack_gray16,
        "j2k_pack_rgb16" => &runtime.pack_rgb16,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("unsupported validated J2K Metal pack kernel `{kernel_name}`"),
            });
        }
    };
    Ok(pipeline)
}

#[cfg(target_os = "macos")]
fn output_shape_for<'a>(
    color_space: &NativeColorSpace,
    has_alpha: bool,
    plane_count: usize,
    fmt: PixelFormat,
    runtime: &'a MetalRuntime,
) -> Result<(u32, u32, &'a ComputePipelineState), Error> {
    let Some(kernel_name) = j2k_pack_kernel_name_for(color_space, has_alpha, plane_count, fmt)
    else {
        return Err(Error::MetalKernel {
            message: format!(
                "unsupported J2K Metal mapping for {color_space:?}, alpha={has_alpha}, planes={plane_count}, fmt={fmt:?}"
            ),
        });
    };
    let (output_channels, opaque_alpha) = match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8 | PixelFormat::Gray16) => (1, 0),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8 | PixelFormat::Rgb16)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => (3, 0),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => (4, 1),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => (4, 0),
        _ => {
            return Err(Error::MetalKernel {
                message: format!(
                    "unsupported validated J2K Metal pack shape for {color_space:?}, alpha={has_alpha}, planes={plane_count}, fmt={fmt:?}"
                ),
            });
        }
    };
    Ok((
        output_channels,
        opaque_alpha,
        j2k_pack_pipeline_for(runtime, kernel_name)?,
    ))
}

#[cfg(target_os = "macos")]
fn required_classic_output_len(job: J2kCodeBlockDecodeJob<'_>) -> Result<usize, Error> {
    if job.height == 0 {
        return Ok(0);
    }

    job.output_stride
        .checked_mul(job.height as usize - 1)
        .and_then(|prefix| prefix.checked_add(job.width as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal output size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn classic_style_flags(style: j2k_native::J2kCodeBlockStyle) -> u32 {
    let mut flags = 0u32;
    if style.reset_context_probabilities {
        flags |= J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES;
    }
    if style.termination_on_each_pass {
        flags |= J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS;
    }
    if style.vertically_causal_context {
        flags |= J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT;
    }
    if style.segmentation_symbols {
        flags |= J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS;
    }
    if style.selective_arithmetic_coding_bypass {
        flags |= J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS;
    }
    flags
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_forward_dwt53(
    samples: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
) -> Result<J2kForwardDwt53Output, Error> {
    if width == 0 || height == 0 {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions must be non-zero".to_string(),
        });
    }
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions overflow".to_string(),
        })?;
    if samples.len() != expected_len {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT sample length mismatch".to_string(),
        });
    }

    with_runtime(|runtime| {
        let bytes = size_of_val(samples);
        let buffer_a = copied_slice_buffer(&runtime.device, samples);
        let buffer_b = runtime
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        let command_buffer = runtime.queue.new_command_buffer();

        let mut current_width = width;
        let mut current_height = height;
        let mut shapes = Vec::new();
        let mut levels_run = 0u8;
        let mut active_is_a = true;

        while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
            let low_width = current_width.div_ceil(2);
            let low_height = current_height.div_ceil(2);
            let params = J2kForwardDwt53Params {
                full_width: width,
                current_width,
                current_height,
                low_width,
                low_height,
            };

            if current_height >= 2 {
                let (input, output) =
                    active_forward_dwt53_buffers(&buffer_a, &buffer_b, active_is_a);
                dispatch_forward_dwt53_pass(
                    &runtime.fdwt53_vertical,
                    command_buffer,
                    input,
                    output,
                    params,
                    "J2K forward DWT 5/3 vertical",
                );
                active_is_a = !active_is_a;
            }
            if current_width >= 2 {
                let (input, output) =
                    active_forward_dwt53_buffers(&buffer_a, &buffer_b, active_is_a);
                dispatch_forward_dwt53_pass(
                    &runtime.fdwt53_horizontal,
                    command_buffer,
                    input,
                    output,
                    params,
                    "J2K forward DWT 5/3 horizontal",
                );
                active_is_a = !active_is_a;
            }

            shapes.push(J2kForwardDwt53Level {
                hl: Vec::new(),
                lh: Vec::new(),
                hh: Vec::new(),
                width: current_width,
                height: current_height,
                low_width,
                low_height,
                high_width: current_width / 2,
                high_height: current_height / 2,
            });
            current_width = low_width;
            current_height = low_height;
            levels_run = levels_run.saturating_add(1);
        }

        commit_and_wait_metal(command_buffer)?;

        let active_buffer = if active_is_a { &buffer_a } else { &buffer_b };
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let transformed = unsafe {
            core::slice::from_raw_parts(active_buffer.contents().cast::<f32>(), samples.len())
        };
        let output = extract_forward_dwt53_output(
            transformed,
            width,
            current_width,
            current_height,
            shapes,
        )?;
        Ok(output)
    })
}

#[cfg(target_os = "macos")]
const FDWT97_ALPHA: f32 = -1.586_134_3;
#[cfg(target_os = "macos")]
const FDWT97_BETA: f32 = -0.052_980_117;
#[cfg(target_os = "macos")]
const FDWT97_GAMMA: f32 = 0.882_911_1;
#[cfg(target_os = "macos")]
const FDWT97_DELTA: f32 = 0.443_506_87;
#[cfg(target_os = "macos")]
const FDWT97_HIGH_PASS: u32 = 1;
#[cfg(target_os = "macos")]
const FDWT97_LOW_PASS: u32 = 0;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kForwardDwt97Params {
    full_width: u32,
    current_width: u32,
    current_height: u32,
    low_width: u32,
    low_height: u32,
    parity: u32,
    coefficient: f32,
    _reserved: u32,
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_forward_dwt97(
    samples: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
) -> Result<J2kForwardDwt97Output, Error> {
    if width == 0 || height == 0 {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions must be non-zero".to_string(),
        });
    }
    let width_usize = usize::try_from(width).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward DWT width does not fit usize".to_string(),
    })?;
    let height_usize = usize::try_from(height).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward DWT height does not fit usize".to_string(),
    })?;
    let expected_len = width_usize
        .checked_mul(height_usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions overflow".to_string(),
        })?;
    if samples.len() != expected_len {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT sample length mismatch".to_string(),
        });
    }
    let bytes = size_of_val(samples);
    let bytes_u64 = u64::try_from(bytes).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward DWT buffer size exceeds u64".to_string(),
    })?;

    with_runtime(|runtime| {
        let buffer_a = copied_slice_buffer(&runtime.device, samples);
        let buffer_b = runtime
            .device
            .new_buffer(bytes_u64, MTLResourceOptions::StorageModeShared);
        let command_buffer = runtime.queue.new_command_buffer();

        let mut current_width = width;
        let mut current_height = height;
        let mut shapes = Vec::new();
        let mut levels_run = 0u8;
        let mut active_is_a = true;

        while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
            let low_width = current_width.div_ceil(2);
            let low_height = current_height.div_ceil(2);
            let base_params = J2kForwardDwt97Params {
                full_width: width,
                current_width,
                current_height,
                low_width,
                low_height,
                parity: FDWT97_HIGH_PASS,
                coefficient: 0.0,
                _reserved: 0,
            };

            if current_height >= 2 {
                dispatch_forward_dwt97_lift_steps(
                    &runtime.fdwt97_lift_vertical,
                    command_buffer,
                    &buffer_a,
                    &buffer_b,
                    active_is_a,
                    base_params,
                    "J2K forward DWT 9/7 vertical",
                );
                let (input, output) =
                    active_forward_dwt53_buffers(&buffer_a, &buffer_b, active_is_a);
                dispatch_forward_dwt97_pass(
                    &runtime.fdwt97_deinterleave_vertical,
                    command_buffer,
                    input,
                    output,
                    base_params,
                    "J2K forward DWT 9/7 vertical deinterleave",
                );
                active_is_a = !active_is_a;
            }
            if current_width >= 2 {
                dispatch_forward_dwt97_lift_steps(
                    &runtime.fdwt97_lift_horizontal,
                    command_buffer,
                    &buffer_a,
                    &buffer_b,
                    active_is_a,
                    base_params,
                    "J2K forward DWT 9/7 horizontal",
                );
                let (input, output) =
                    active_forward_dwt53_buffers(&buffer_a, &buffer_b, active_is_a);
                dispatch_forward_dwt97_pass(
                    &runtime.fdwt97_deinterleave_horizontal,
                    command_buffer,
                    input,
                    output,
                    base_params,
                    "J2K forward DWT 9/7 horizontal deinterleave",
                );
                active_is_a = !active_is_a;
            }

            shapes.push(J2kForwardDwt97Level {
                hl: Vec::new(),
                lh: Vec::new(),
                hh: Vec::new(),
                width: current_width,
                height: current_height,
                low_width,
                low_height,
                high_width: current_width / 2,
                high_height: current_height / 2,
            });
            current_width = low_width;
            current_height = low_height;
            levels_run = levels_run.saturating_add(1);
        }

        commit_and_wait_metal(command_buffer)?;

        let active_buffer = if active_is_a { &buffer_a } else { &buffer_b };
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let transformed = unsafe {
            core::slice::from_raw_parts(active_buffer.contents().cast::<f32>(), samples.len())
        };
        extract_forward_dwt97_output(transformed, width, current_width, current_height, shapes)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_deinterleave_to_f32(
    job: J2kDeinterleaveToF32Job<'_>,
) -> Result<Option<Vec<Vec<f32>>>, Error> {
    validate_encode_deinterleave_to_f32_job(job)?;
    let pixel_count = u32::try_from(job.num_pixels).map_err(|_| Error::MetalKernel {
        message: "J2K Metal encode deinterleave pixel count exceeds u32".to_string(),
    })?;
    let bytes_per_sample = encode_deinterleave_bytes_per_sample(job.bit_depth);
    let sample_count = job
        .num_pixels
        .checked_mul(usize::from(job.num_components))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal encode deinterleave sample count overflow".to_string(),
        })?;
    let expected_len = job
        .num_pixels
        .checked_mul(usize::from(job.num_components))
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal encode deinterleave input length overflow".to_string(),
        })?;
    if job.pixels.len() != expected_len {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K Metal encode deinterleave input length mismatch: expected {expected_len} bytes, got {}",
                job.pixels.len()
            ),
        });
    }
    let src_stride = u32::try_from(expected_len).map_err(|_| Error::MetalKernel {
        message: "J2K Metal encode deinterleave row stride exceeds u32".to_string(),
    })?;
    let plane_bytes = job
        .num_pixels
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal encode deinterleave output length overflow".to_string(),
        })?;

    with_runtime(|runtime| {
        let input_buffer = copied_slice_buffer(&runtime.device, job.pixels);
        let plane_buffers = (0..4)
            .map(|_| {
                runtime
                    .device
                    .new_buffer(plane_bytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .collect::<Vec<_>>();
        let params = J2kLosslessDeinterleaveParams {
            src_width: pixel_count,
            src_height: 1,
            src_stride,
            dst_width: pixel_count,
            dst_height: 1,
            components: u32::from(job.num_components),
            bytes_per_sample: bytes_per_sample as u32,
            bit_depth: u32::from(job.bit_depth),
            sample_offset: encode_deinterleave_sample_offset(job.bit_depth, job.signed),
            signed_samples: u32::from(job.signed),
        };

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "j2k encode-stage deinterleave");
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "J2K encode-stage deinterleave");
        encoder.set_compute_pipeline_state(&runtime.lossless_deinterleave_to_planes);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&plane_buffers[0]), 0);
        encoder.set_buffer(2, Some(&plane_buffers[1]), 0);
        encoder.set_buffer(3, Some(&plane_buffers[2]), 0);
        encoder.set_bytes(
            4,
            size_of::<J2kLosslessDeinterleaveParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(5, Some(&plane_buffers[3]), 0);
        dispatch_2d_pipeline(
            encoder,
            &runtime.lossless_deinterleave_to_planes,
            (pixel_count, 1),
        );
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        let planes = plane_buffers
            .iter()
            .take(usize::from(job.num_components))
            .map(|buffer| {
                // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
                let samples = unsafe {
                    core::slice::from_raw_parts(buffer.contents().cast::<f32>(), job.num_pixels)
                };
                samples.to_vec()
            })
            .collect();
        debug_assert_eq!(
            sample_count.checked_mul(bytes_per_sample),
            Some(expected_len)
        );
        Ok(Some(planes))
    })
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt97_lift_steps(
    pipeline: &ComputePipelineState,
    command_buffer: &CommandBufferRef,
    buffer_a: &Buffer,
    buffer_b: &Buffer,
    active_is_a: bool,
    base_params: J2kForwardDwt97Params,
    label_prefix: &str,
) {
    let active_buffer = if active_is_a { buffer_a } else { buffer_b };
    for (parity, coefficient) in [
        (FDWT97_HIGH_PASS, FDWT97_ALPHA),
        (FDWT97_LOW_PASS, FDWT97_BETA),
        (FDWT97_HIGH_PASS, FDWT97_GAMMA),
        (FDWT97_LOW_PASS, FDWT97_DELTA),
    ] {
        let params = J2kForwardDwt97Params {
            parity,
            coefficient,
            ..base_params
        };
        dispatch_forward_dwt97_pass(
            pipeline,
            command_buffer,
            active_buffer,
            active_buffer,
            params,
            label_prefix,
        );
    }
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt97_pass(
    pipeline: &ComputePipelineState,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: J2kForwardDwt97Params,
    label: &str,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, label);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kForwardDwt97Params>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        pipeline,
        (params.current_width, params.current_height),
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn validate_encode_deinterleave_to_f32_job(job: J2kDeinterleaveToF32Job<'_>) -> Result<(), Error> {
    if job.num_pixels == 0 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode deinterleave requires at least one pixel",
        });
    }
    if !(1..=4).contains(&job.num_components) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode deinterleave supports 1-4 component samples",
        });
    }
    if job.bit_depth == 0 || job.bit_depth > 16 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode deinterleave supports 1-16 bits per sample",
        });
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_deinterleave_bytes_per_sample(bit_depth: u8) -> usize {
    if bit_depth <= 8 {
        1
    } else {
        2
    }
}

#[cfg(target_os = "macos")]
fn encode_deinterleave_sample_offset(bit_depth: u8, signed: bool) -> u32 {
    if signed {
        0
    } else {
        1u32 << (u32::from(bit_depth) - 1)
    }
}

#[cfg(target_os = "macos")]
fn active_forward_dwt53_buffers<'a>(
    buffer_a: &'a Buffer,
    buffer_b: &'a Buffer,
    active_is_a: bool,
) -> (&'a Buffer, &'a Buffer) {
    if active_is_a {
        (buffer_a, buffer_b)
    } else {
        (buffer_b, buffer_a)
    }
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt53_pass(
    pipeline: &ComputePipelineState,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: J2kForwardDwt53Params,
    label: &str,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, label);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kForwardDwt53Params>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        pipeline,
        (params.current_width, params.current_height),
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt53_batched_pass(
    pipeline: &ComputePipelineState,
    command_buffer: &CommandBufferRef,
    inputs: &[Buffer],
    outputs: &[Buffer],
    params: J2kForwardDwt53BatchedParams,
    label: &str,
) {
    debug_assert!(!inputs.is_empty());
    debug_assert!(!outputs.is_empty());
    debug_assert!(params.component_count >= 1 && params.component_count <= 3);
    let first_input_buffer = &inputs[0];
    let second_input_buffer = inputs.get(1).unwrap_or(first_input_buffer);
    let third_input_buffer = inputs.get(2).unwrap_or(first_input_buffer);
    let first_output_buffer = &outputs[0];
    let second_output_buffer = outputs.get(1).unwrap_or(first_output_buffer);
    let third_output_buffer = outputs.get(2).unwrap_or(first_output_buffer);

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, label);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(first_input_buffer), 0);
    encoder.set_buffer(1, Some(second_input_buffer), 0);
    encoder.set_buffer(2, Some(third_input_buffer), 0);
    encoder.set_buffer(3, Some(first_output_buffer), 0);
    encoder.set_buffer(4, Some(second_output_buffer), 0);
    encoder.set_buffer(5, Some(third_output_buffer), 0);
    encoder.set_bytes(
        6,
        size_of::<J2kForwardDwt53BatchedParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        pipeline,
        (
            params.current_width,
            params.current_height,
            params.component_count,
        ),
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn extract_forward_dwt53_output(
    transformed: &[f32],
    full_width: u32,
    ll_width: u32,
    ll_height: u32,
    mut shapes: Vec<J2kForwardDwt53Level>,
) -> Result<J2kForwardDwt53Output, Error> {
    let full_width_usize = full_width as usize;
    let mut ll = Vec::with_capacity((ll_width as usize) * (ll_height as usize));
    for y in 0..ll_height as usize {
        let row_start = y
            .checked_mul(full_width_usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT LL row offset overflow".to_string(),
            })?;
        ll.extend_from_slice(&transformed[row_start..row_start + ll_width as usize]);
    }

    for shape in &mut shapes {
        shape.hl = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            0,
            shape.high_width,
            shape.low_height,
        )?;
        shape.lh = extract_subband(
            transformed,
            full_width_usize,
            0,
            shape.low_height,
            shape.low_width,
            shape.high_height,
        )?;
        shape.hh = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            shape.low_height,
            shape.high_width,
            shape.high_height,
        )?;
    }
    shapes.reverse();

    Ok(J2kForwardDwt53Output {
        ll,
        ll_width,
        ll_height,
        levels: shapes,
    })
}

#[cfg(target_os = "macos")]
fn extract_forward_dwt97_output(
    transformed: &[f32],
    full_width: u32,
    ll_width: u32,
    ll_height: u32,
    mut shapes: Vec<J2kForwardDwt97Level>,
) -> Result<J2kForwardDwt97Output, Error> {
    let full_width_usize = usize::try_from(full_width).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward DWT full width does not fit usize".to_string(),
    })?;
    let ll_width_usize = usize::try_from(ll_width).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward DWT LL width does not fit usize".to_string(),
    })?;
    let ll_height_usize = usize::try_from(ll_height).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward DWT LL height does not fit usize".to_string(),
    })?;
    let ll_len = ll_width_usize
        .checked_mul(ll_height_usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal forward DWT LL dimensions overflow".to_string(),
        })?;
    let mut ll = Vec::with_capacity(ll_len);
    for y in 0..ll_height_usize {
        let row_start = y
            .checked_mul(full_width_usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT LL row offset overflow".to_string(),
            })?;
        let row_end = row_start
            .checked_add(ll_width_usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT LL row end overflow".to_string(),
            })?;
        let row = transformed
            .get(row_start..row_end)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT LL row out of bounds".to_string(),
            })?;
        ll.extend_from_slice(row);
    }

    for shape in &mut shapes {
        shape.hl = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            0,
            shape.high_width,
            shape.low_height,
        )?;
        shape.lh = extract_subband(
            transformed,
            full_width_usize,
            0,
            shape.low_height,
            shape.low_width,
            shape.high_height,
        )?;
        shape.hh = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            shape.low_height,
            shape.high_width,
            shape.high_height,
        )?;
    }
    shapes.reverse();

    Ok(J2kForwardDwt97Output {
        ll,
        ll_width,
        ll_height,
        levels: shapes,
    })
}

#[cfg(target_os = "macos")]
fn extract_subband(
    transformed: &[f32],
    full_width: usize,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, Error> {
    let mut out = Vec::with_capacity((width as usize) * (height as usize));
    for y in 0..height as usize {
        let row_start = (y0 as usize)
            .checked_add(y)
            .and_then(|row| row.checked_mul(full_width))
            .and_then(|row| row.checked_add(x0 as usize))
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT subband offset overflow".to_string(),
            })?;
        out.extend_from_slice(&transformed[row_start..row_start + width as usize]);
    }
    Ok(out)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct J2kLosslessDeviceCodeBlock {
    pub(crate) coefficient_offset: u32,
    pub(crate) component: u32,
    pub(crate) subband_x: u32,
    pub(crate) subband_y: u32,
    pub(crate) block_x: u32,
    pub(crate) block_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) sub_band_type: j2k_native::J2kSubBandType,
    pub(crate) total_bitplanes: u8,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct J2kLosslessDevicePrepareJob<'a> {
    pub(crate) input: &'a Buffer,
    pub(crate) input_byte_offset: usize,
    pub(crate) input_width: u32,
    pub(crate) input_height: u32,
    pub(crate) input_pitch_bytes: usize,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) components: u8,
    pub(crate) bytes_per_sample: u8,
    pub(crate) bit_depth: u8,
    pub(crate) num_decomposition_levels: u8,
    pub(crate) coefficient_count: usize,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kLosslessDeviceBatchPrepareItem<'a> {
    pub(crate) tile_index: usize,
    pub(crate) job: J2kLosslessDevicePrepareJob<'a>,
    pub(crate) code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kPreparedLosslessDeviceCodeBlocks {
    coefficient_buffer: Buffer,
    coefficient_byte_offset: usize,
    coefficient_byte_len: usize,
    coefficient_buffer_is_batch_shared: bool,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    recyclable_private_buffers: Vec<(usize, Buffer)>,
    _prepare_command_buffer: CommandBuffer,
    _prepare_deinterleave_rct_command_buffer: Option<CommandBuffer>,
    _prepare_dwt53_command_buffer: Option<CommandBuffer>,
    _prepare_dwt53_vertical_command_buffers: Vec<CommandBuffer>,
    _prepare_dwt53_horizontal_command_buffers: Vec<CommandBuffer>,
    _prepare_coefficient_extract_command_buffer: Option<CommandBuffer>,
    _deinterleave_status_buffer: Buffer,
    _plane_buffers: Vec<Buffer>,
    _scratch_buffers: Vec<Buffer>,
    _coefficient_job_buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct J2kResidentPacketizationSubband {
    pub(crate) code_block_start: u32,
    pub(crate) code_block_count: u32,
    pub(crate) num_cbs_x: u32,
    pub(crate) num_cbs_y: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub(crate) struct J2kResidentPacketizationResolution {
    pub(crate) subbands: Vec<J2kResidentPacketizationSubband>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(crate) struct J2kResidentPacketizationEncodeJob<'a> {
    pub(crate) resolution_count: u32,
    pub(crate) num_layers: u8,
    pub(crate) num_components: u8,
    pub(crate) code_block_count: u32,
    pub(crate) packet_descriptors: &'a [J2kPacketizationPacketDescriptor],
    pub(crate) resolutions: &'a [J2kResidentPacketizationResolution],
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum J2kLosslessCodestreamBlockCodingMode {
    Classic,
    HighThroughput,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct J2kLosslessCodestreamAssemblyJob {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) num_components: u8,
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
    pub(crate) num_decomposition_levels: u8,
    pub(crate) use_mct: bool,
    pub(crate) guard_bits: u8,
    pub(crate) code_block_width_exp: u8,
    pub(crate) code_block_height_exp: u8,
    pub(crate) progression_order: EncodeProgressionOrder,
    pub(crate) write_tlm: bool,
    pub(crate) block_coding_mode: J2kLosslessCodestreamBlockCodingMode,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kResidentLosslessTier1CodeBlocks {
    output_buffer: Buffer,
    status_buffer: Buffer,
    job_buffer: Buffer,
    batch_jobs: Vec<J2kClassicEncodeBatchJob>,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    output_capacity_total: usize,
    _segment_buffer: Buffer,
    tier1_command_buffer: CommandBuffer,
    _coefficient_buffer: Buffer,
    prepare_command_buffer: CommandBuffer,
    _deinterleave_status_buffer: Buffer,
    _plane_buffers: Vec<Buffer>,
    _scratch_buffers: Vec<Buffer>,
    _coefficient_job_buffer: Buffer,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kResidentLosslessHtCodeBlocks {
    output_buffer: Buffer,
    status_buffer: Buffer,
    job_buffer: Buffer,
    batch_jobs: Vec<J2kHtEncodeBatchJob>,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    output_capacity_total: usize,
    tier1_command_buffer: CommandBuffer,
    _coefficient_buffer: Buffer,
    prepare_command_buffer: CommandBuffer,
    _deinterleave_status_buffer: Buffer,
    _plane_buffers: Vec<Buffer>,
    _scratch_buffers: Vec<Buffer>,
    _coefficient_job_buffer: Buffer,
}

/// Family descriptor that lets the resident Tier-1 codestream drivers run
/// generically over the classic and HT code-block tables. Associated consts
/// keep diagnostics byte-identical to the pre-convergence per-family bodies.
#[cfg(target_os = "macos")]
pub(crate) trait ResidentLosslessTier1Metal {
    /// Prefix for Tier-2 resident packet diagnostics ("" for classic).
    const TIER2_PREFIX: &'static str;
    /// Family name used in codestream assembly diagnostics.
    const FAMILY_NAME: &'static str;
    /// Value stored in `J2kResidentPacketBlock::block_coding_mode`.
    const BLOCK_CODING_MODE: u32;
    const CODESTREAM_STATUS_STAGE: &'static str;
    const CODESTREAM_LENGTH_ERROR: &'static str;
    const CODESTREAM_CAPACITY_ERROR: &'static str;

    fn batch_job_count(&self) -> usize;
    fn code_block_count(&self) -> usize;
    fn output_capacity_total(&self) -> usize;
    fn output_buffer(&self) -> &Buffer;
    fn status_buffer(&self) -> &Buffer;
    fn job_buffer(&self) -> &Buffer;
    fn prepare_command_buffer(&self) -> &CommandBuffer;
    fn tier1_command_buffer(&self) -> &CommandBuffer;
    fn packet_block_prepare_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
}

#[cfg(target_os = "macos")]
impl ResidentLosslessTier1Metal for J2kResidentLosslessTier1CodeBlocks {
    const TIER2_PREFIX: &'static str = "";
    const FAMILY_NAME: &'static str = "J2K";
    const BLOCK_CODING_MODE: u32 = 0;
    const CODESTREAM_STATUS_STAGE: &'static str = "J2K codestream assembly";
    const CODESTREAM_LENGTH_ERROR: &'static str =
        "J2K Metal codestream output length exceeds usize";
    const CODESTREAM_CAPACITY_ERROR: &'static str =
        "J2K Metal codestream output length exceeds buffer";

    fn batch_job_count(&self) -> usize {
        self.batch_jobs.len()
    }
    fn code_block_count(&self) -> usize {
        self.code_blocks.len()
    }
    fn output_capacity_total(&self) -> usize {
        self.output_capacity_total
    }
    fn output_buffer(&self) -> &Buffer {
        &self.output_buffer
    }
    fn status_buffer(&self) -> &Buffer {
        &self.status_buffer
    }
    fn job_buffer(&self) -> &Buffer {
        &self.job_buffer
    }
    fn prepare_command_buffer(&self) -> &CommandBuffer {
        &self.prepare_command_buffer
    }
    fn tier1_command_buffer(&self) -> &CommandBuffer {
        &self.tier1_command_buffer
    }
    fn packet_block_prepare_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.packet_block_prepare_resident_classic
    }
}

#[cfg(target_os = "macos")]
impl ResidentLosslessTier1Metal for J2kResidentLosslessHtCodeBlocks {
    const TIER2_PREFIX: &'static str = "HTJ2K ";
    const FAMILY_NAME: &'static str = "HTJ2K";
    const BLOCK_CODING_MODE: u32 = 1;
    const CODESTREAM_STATUS_STAGE: &'static str = "HTJ2K codestream assembly";
    const CODESTREAM_LENGTH_ERROR: &'static str =
        "HTJ2K Metal codestream output length exceeds usize";
    const CODESTREAM_CAPACITY_ERROR: &'static str =
        "HTJ2K Metal codestream output length exceeds buffer";

    fn batch_job_count(&self) -> usize {
        self.batch_jobs.len()
    }
    fn code_block_count(&self) -> usize {
        self.code_blocks.len()
    }
    fn output_capacity_total(&self) -> usize {
        self.output_capacity_total
    }
    fn output_buffer(&self) -> &Buffer {
        &self.output_buffer
    }
    fn status_buffer(&self) -> &Buffer {
        &self.status_buffer
    }
    fn job_buffer(&self) -> &Buffer {
        &self.job_buffer
    }
    fn prepare_command_buffer(&self) -> &CommandBuffer {
        &self.prepare_command_buffer
    }
    fn tier1_command_buffer(&self) -> &CommandBuffer {
        &self.tier1_command_buffer
    }
    fn packet_block_prepare_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.packet_block_prepare_resident_ht
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
enum J2kResidentTier1StatusKind {
    Classic,
    HighThroughput,
}

#[cfg(target_os = "macos")]
struct J2kResidentTier1StatusReadback {
    buffer: Buffer,
    kind: J2kResidentTier1StatusKind,
    classic_style_flags: u32,
    classic_jobs: Option<Vec<J2kClassicEncodeBatchJob>>,
    count: usize,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1DensityReadback {
    buffer: Buffer,
    count: usize,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1SymbolPlanReadback {
    buffer: Buffer,
    count: usize,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1PassPlanReadback {
    buffer: Buffer,
    count: usize,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1TokenEmitReadback {
    counter_buffer: Buffer,
    token_buffer: Option<Buffer>,
    segment_buffer: Option<Buffer>,
    token_stride_bytes: usize,
    token_segment_stride: usize,
    count: usize,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1GpuTokenBuffers {
    counter_buffer: Buffer,
    token_buffer: Buffer,
    segment_buffer: Buffer,
    job_count: u32,
    token_stride_bytes: u32,
    token_segment_stride: u32,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1SplitTokenBuffers {
    counter_buffer: Buffer,
    mq_token_buffer: Buffer,
    raw_token_buffer: Buffer,
    segment_buffer: Buffer,
    job_count: u32,
    mq_token_stride_bytes: u32,
    raw_token_stride_bytes: u32,
    token_segment_stride: u32,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kPendingResidentLosslessCodestreamBatch {
    runtime: Arc<MetalRuntime>,
    buffer: Buffer,
    byte_offsets: Vec<usize>,
    capacities: Vec<usize>,
    status_buffer: Buffer,
    packet_status_buffer: Buffer,
    tier1_status_readback: Option<J2kResidentTier1StatusReadback>,
    classic_tier1_density_readback: Option<J2kResidentClassicTier1DensityReadback>,
    classic_tier1_symbol_plan_readback: Option<J2kResidentClassicTier1SymbolPlanReadback>,
    classic_tier1_pass_plan_readback: Option<J2kResidentClassicTier1PassPlanReadback>,
    classic_tier1_token_emit_readback: Option<J2kResidentClassicTier1TokenEmitReadback>,
    classic_tier1_split_token_emit_readback: Option<J2kResidentClassicTier1SplitTokenBuffers>,
    classic_gpu_token_pack_used: bool,
    command_buffer: CommandBuffer,
    retained_command_buffers: Vec<CommandBuffer>,
    _retained_buffers: Vec<Buffer>,
    recyclable_private_buffers: Vec<(usize, Buffer)>,
    recyclable_shared_buffers: Vec<(usize, Buffer)>,
    gpu_stage_command_buffers: Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    stage_stats: J2kResidentEncodeStageStats,
    codestream_payload_copy_dispatched: bool,
    status_stage: &'static str,
    length_error: &'static str,
    capacity_error: &'static str,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct J2kBatchedPacketPayloadCopyDispatch<'a> {
    payload_buffer: &'a Buffer,
    packet_output_buffer: &'a Buffer,
    packet_job_buffer: &'a Buffer,
    packet_status_buffer: &'a Buffer,
    packet_payload_copy_job_buffer: &'a Buffer,
    tile_count: u64,
    max_payload_copy_jobs_per_tile: u64,
    label: &'a str,
    signpost_name: HybridSignpostName,
}

#[cfg(target_os = "macos")]
pub(crate) fn wait_resident_lossless_codestream(
    pending: J2kPendingResidentLosslessCodestream,
) -> Result<J2kResidentLosslessCodestream, Error> {
    wait_resident_codestream_command_buffer(&pending.command_buffer)?;
    let gpu_duration = completed_command_buffers_gpu_duration(
        &pending.retained_command_buffers,
        &pending.command_buffer,
    );
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let status = unsafe {
        pending
            .status_buffer
            .contents()
            .cast::<J2kCodestreamAssemblyStatus>()
            .read()
    };
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            pending.status_stage,
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: pending.length_error.to_string(),
    })?;
    if data_len > pending.capacity {
        return Err(Error::MetalKernel {
            message: pending.capacity_error.to_string(),
        });
    }
    Ok(J2kResidentLosslessCodestream {
        buffer: pending.buffer,
        byte_offset: 0,
        byte_len: data_len,
        capacity: pending.capacity,
        gpu_duration,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn wait_resident_lossless_codestream_batch(
    pending: J2kPendingResidentLosslessCodestreamBatch,
) -> Result<J2kResidentLosslessCodestreamBatchResult, Error> {
    wait_resident_codestream_command_buffer(&pending.command_buffer)?;
    finish_completed_resident_lossless_codestream_batch(pending)
}

#[cfg(target_os = "macos")]
pub(crate) fn wait_resident_lossless_codestream_batches(
    pending_batches: Vec<J2kPendingResidentLosslessCodestreamBatch>,
) -> Result<Vec<J2kResidentLosslessCodestreamBatchResult>, Error> {
    if let Some(last) = pending_batches.last() {
        // These command buffers are submitted on the same Metal queue before
        // harvest, so completing the final one implies earlier chunks are done.
        wait_resident_codestream_command_buffer(&last.command_buffer)?;
    }
    pending_batches
        .into_iter()
        .map(finish_completed_resident_lossless_codestream_batch)
        .collect()
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn schedule_resident_tier1_status_readback(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    status_buffer: &Buffer,
    kind: J2kResidentTier1StatusKind,
    classic_style_flags: u32,
    classic_jobs: Option<&[J2kClassicEncodeBatchJob]>,
    count: usize,
    status_size: usize,
    profile_stages: bool,
) -> Result<Option<J2kResidentTier1StatusReadback>, Error> {
    if !profile_stages || count == 0 {
        return Ok(None);
    }
    let byte_len = count
        .checked_mul(status_size)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident Tier-1 status readback size overflow".to_string(),
        })?;
    let readback = runtime.device.new_buffer(
        byte_len.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_buffer(status_buffer, 0, &readback, 0, byte_len as u64);
    blit.end_encoding();
    Ok(Some(J2kResidentTier1StatusReadback {
        buffer: readback,
        kind,
        classic_style_flags,
        classic_jobs: classic_jobs.map(<[J2kClassicEncodeBatchJob]>::to_vec),
        count,
    }))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_density_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<Option<J2kResidentClassicTier1DensityReadback>, Error> {
    if !metal_profile_classic_tier1_density_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 density profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = runtime.device.new_buffer(
        (tier1_jobs.len().max(1) * size_of::<J2kClassicTier1DensityCounters>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 density job count exceeds u32".to_string(),
    })?;
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 density profile");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_density_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_density_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(J2kResidentClassicTier1DensityReadback {
        buffer: counter_buffer,
        count: tier1_jobs.len(),
    }))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_raw_pack_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
    tier1_output_capacity_total: usize,
) -> Result<Option<Buffer>, Error> {
    if !metal_profile_classic_tier1_raw_pack_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 raw-pack profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let raw_output_buffer = runtime.device.new_buffer(
        tier1_output_capacity_total.max(1) as u64,
        MTLResourceOptions::StorageModePrivate,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 raw-pack job count exceeds u32".to_string(),
    })?;
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 raw-pack profile");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_raw_pack_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&raw_output_buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_raw_pack_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(raw_output_buffer))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_arithmetic_pack_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
    tier1_output_capacity_total: usize,
) -> Result<Option<Buffer>, Error> {
    if !metal_profile_classic_tier1_arithmetic_pack_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 arithmetic-pack profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let arithmetic_output_buffer = runtime.device.new_buffer(
        tier1_output_capacity_total.max(1) as u64,
        MTLResourceOptions::StorageModePrivate,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 arithmetic-pack job count exceeds u32".to_string(),
    })?;
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 arithmetic-pack profile");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_arithmetic_pack_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&arithmetic_output_buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_arithmetic_pack_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(arithmetic_output_buffer))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_symbol_plan_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<Option<J2kResidentClassicTier1SymbolPlanReadback>, Error> {
    if !metal_profile_classic_tier1_symbol_plan_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 symbol-plan profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = runtime.device.new_buffer(
        (tier1_jobs.len().max(1) * size_of::<J2kClassicTier1SymbolPlanCounters>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 symbol-plan job count exceeds u32".to_string(),
    })?;
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 symbol plan");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_symbol_plan_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_symbol_plan_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(J2kResidentClassicTier1SymbolPlanReadback {
        buffer: counter_buffer,
        count: tier1_jobs.len(),
    }))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_pass_plan_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<Option<J2kResidentClassicTier1PassPlanReadback>, Error> {
    if !metal_profile_classic_tier1_pass_plan_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 pass-plan profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = runtime.device.new_buffer(
        (tier1_jobs.len().max(1) * size_of::<J2kClassicTier1PassPlanCounters>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 pass-plan job count exceeds u32".to_string(),
    })?;
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 pass plan");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_pass_plan_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_pass_plan_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(J2kResidentClassicTier1PassPlanReadback {
        buffer: counter_buffer,
        count: tier1_jobs.len(),
    }))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_token_emit_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<Option<J2kResidentClassicTier1TokenEmitReadback>, Error> {
    if !metal_profile_classic_tier1_token_emit_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token-emitter profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = runtime.device.new_buffer(
        (tier1_jobs.len().max(1) * size_of::<J2kClassicTier1SymbolPlanCounters>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token buffer size overflow".to_string(),
        })?;
    let token_buffer = runtime.device.new_buffer(
        token_buffer_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let segment_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY)
        .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token segment buffer size overflow".to_string(),
        })?;
    let segment_buffer = runtime.device.new_buffer(
        segment_buffer_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 token-emitter job count exceeds u32".to_string(),
    })?;
    let token_stride_bytes =
        u32::try_from(CLASSIC_TIER1_TOKEN_ARENA_BYTES).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token arena stride exceeds u32".to_string(),
        })?;
    let token_segment_stride =
        u32::try_from(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token segment stride exceeds u32".to_string(),
        })?;

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 token emit");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_token_emit_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_buffer(3, Some(&token_buffer), 0);
    encoder.set_buffer(4, Some(&segment_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        (&raw const token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        (&raw const token_segment_stride).cast(),
    );
    encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_token_emit_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(J2kResidentClassicTier1TokenEmitReadback {
        counter_buffer,
        token_buffer: Some(token_buffer),
        segment_buffer: Some(segment_buffer),
        token_stride_bytes: CLASSIC_TIER1_TOKEN_ARENA_BYTES,
        token_segment_stride: CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY,
        count: tier1_jobs.len(),
    }))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_split_token_emit_for_cpu_pack(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<J2kResidentClassicTier1SplitTokenBuffers, Error> {
    if !classic_tier1_gpu_token_pack_supported(tier1_jobs) {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic split-token route currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = runtime.device.new_buffer(
        (tier1_jobs.len().max(1) * size_of::<J2kClassicTier1SymbolPlanCounters>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mq_token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split-token MQ buffer size overflow".to_string(),
        })?;
    let mq_token_buffer = runtime.device.new_buffer(
        mq_token_buffer_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let raw_token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split-token raw buffer size overflow".to_string(),
        })?;
    let raw_token_buffer = runtime.device.new_buffer(
        raw_token_buffer_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let segment_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY)
        .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split-token segment buffer size overflow".to_string(),
        })?;
    let segment_buffer = runtime.device.new_buffer(
        segment_buffer_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic split-token job count exceeds u32".to_string(),
    })?;
    let mq_token_stride_bytes =
        u32::try_from(CLASSIC_TIER1_TOKEN_ARENA_BYTES).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split-token MQ arena stride exceeds u32".to_string(),
        })?;
    let raw_token_stride_bytes =
        u32::try_from(CLASSIC_TIER1_TOKEN_ARENA_BYTES).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split-token raw arena stride exceeds u32".to_string(),
        })?;
    let token_segment_stride =
        u32::try_from(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split-token segment stride exceeds u32".to_string(),
        })?;

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 split token emit");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_split_token_emit_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_buffer(3, Some(&mq_token_buffer), 0);
    encoder.set_buffer(4, Some(&raw_token_buffer), 0);
    encoder.set_buffer(5, Some(&segment_buffer), 0);
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        (&raw const mq_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        7,
        size_of::<u32>() as u64,
        (&raw const raw_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        8,
        size_of::<u32>() as u64,
        (&raw const token_segment_stride).cast(),
    );
    encoder.set_bytes(9, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_split_token_emit_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(J2kResidentClassicTier1SplitTokenBuffers {
        counter_buffer,
        mq_token_buffer,
        raw_token_buffer,
        segment_buffer,
        job_count,
        mq_token_stride_bytes,
        raw_token_stride_bytes,
        token_segment_stride,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_split_token_emit_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<Option<J2kResidentClassicTier1SplitTokenBuffers>, Error> {
    if !metal_profile_classic_tier1_split_token_emit_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    dispatch_classic_tier1_split_token_emit_for_cpu_pack(
        runtime,
        command_buffer,
        coefficient_buffer,
        tier1_job_buffer,
        tier1_jobs,
    )
    .map(Some)
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_split_token_emit_for_gpu_pack(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
    recyclable_private_buffers: &mut Vec<(usize, Buffer)>,
    use_mq_byte_emit: bool,
) -> Result<J2kResidentClassicTier1SplitTokenBuffers, Error> {
    if !classic_tier1_gpu_token_pack_supported(tier1_jobs) {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic split GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }
    #[cfg(test)]
    if use_mq_byte_emit {
        test_counters::record_classic_split_mq_byte_gpu_token_pack_dispatch();
    }

    let counter_buffer = take_recyclable_private_buffer(
        runtime,
        tier1_jobs
            .len()
            .max(1)
            .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal classic split GPU token counter buffer size overflow"
                    .to_string(),
            })?,
        recyclable_private_buffers,
    )?;
    let mq_token_arena_bytes = if use_mq_byte_emit {
        CLASSIC_TIER1_MQ_BYTE_TOKEN_ARENA_BYTES
    } else {
        CLASSIC_TIER1_TOKEN_ARENA_BYTES
    };
    let mq_token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(mq_token_arena_bytes)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split GPU token MQ buffer size overflow".to_string(),
        })?;
    let mq_token_buffer =
        take_recyclable_private_buffer(runtime, mq_token_buffer_len, recyclable_private_buffers)?;
    let raw_token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split GPU token raw buffer size overflow".to_string(),
        })?;
    let raw_token_buffer =
        take_recyclable_private_buffer(runtime, raw_token_buffer_len, recyclable_private_buffers)?;
    let segment_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY)
        .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split GPU token segment buffer size overflow".to_string(),
        })?;
    let segment_buffer =
        take_recyclable_private_buffer(runtime, segment_buffer_len, recyclable_private_buffers)?;
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic split GPU token job count exceeds u32".to_string(),
    })?;
    let mq_token_stride_bytes =
        u32::try_from(mq_token_arena_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split GPU token MQ arena stride exceeds u32".to_string(),
        })?;
    let raw_token_stride_bytes =
        u32::try_from(CLASSIC_TIER1_TOKEN_ARENA_BYTES).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split GPU token raw arena stride exceeds u32".to_string(),
        })?;
    let token_segment_stride =
        u32::try_from(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split GPU token segment stride exceeds u32".to_string(),
        })?;

    let emit_pipeline = if use_mq_byte_emit {
        &runtime.classic_tier1_split_mq_byte_token_emit_bypass_u16_32
    } else {
        &runtime.classic_tier1_split_token_emit_bypass_u16_32
    };

    let encoder = command_buffer.new_compute_command_encoder();
    if use_mq_byte_emit {
        label_compute_encoder(encoder, "J2K classic Tier-1 split MQ-byte token emit");
    } else {
        label_compute_encoder(encoder, "J2K classic Tier-1 split token emit");
    }
    encoder.set_compute_pipeline_state(emit_pipeline);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_buffer(3, Some(&mq_token_buffer), 0);
    encoder.set_buffer(4, Some(&raw_token_buffer), 0);
    encoder.set_buffer(5, Some(&segment_buffer), 0);
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        (&raw const mq_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        7,
        size_of::<u32>() as u64,
        (&raw const raw_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        8,
        size_of::<u32>() as u64,
        (&raw const token_segment_stride).cast(),
    );
    encoder.set_bytes(9, size_of::<u32>() as u64, (&raw const job_count).cast());
    dispatch_1d_pipeline(encoder, emit_pipeline, u64::from(job_count));
    encoder.end_encoding();

    Ok(J2kResidentClassicTier1SplitTokenBuffers {
        counter_buffer,
        mq_token_buffer,
        raw_token_buffer,
        segment_buffer,
        job_count,
        mq_token_stride_bytes,
        raw_token_stride_bytes,
        token_segment_stride,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_token_emit_for_gpu_pack(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
    recyclable_private_buffers: &mut Vec<(usize, Buffer)>,
) -> Result<J2kResidentClassicTier1GpuTokenBuffers, Error> {
    if !classic_tier1_gpu_token_pack_supported(tier1_jobs) {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = take_recyclable_private_buffer(
        runtime,
        tier1_jobs
            .len()
            .max(1)
            .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token counter buffer size overflow".to_string(),
            })?,
        recyclable_private_buffers,
    )?;
    let token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token buffer size overflow".to_string(),
        })?;
    let token_buffer =
        take_recyclable_private_buffer(runtime, token_buffer_len, recyclable_private_buffers)?;
    let segment_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY)
        .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token segment buffer size overflow".to_string(),
        })?;
    let segment_buffer =
        take_recyclable_private_buffer(runtime, segment_buffer_len, recyclable_private_buffers)?;
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 token-emitter job count exceeds u32".to_string(),
    })?;
    let token_stride_bytes =
        u32::try_from(CLASSIC_TIER1_TOKEN_ARENA_BYTES).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token arena stride exceeds u32".to_string(),
        })?;
    let token_segment_stride =
        u32::try_from(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token segment stride exceeds u32".to_string(),
        })?;

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 token emit");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_token_emit_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_buffer(3, Some(&token_buffer), 0);
    encoder.set_buffer(4, Some(&segment_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        (&raw const token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        (&raw const token_segment_stride).cast(),
    );
    encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_token_emit_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(J2kResidentClassicTier1GpuTokenBuffers {
        counter_buffer,
        token_buffer,
        segment_buffer,
        job_count,
        token_stride_bytes,
        token_segment_stride,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_token_pack_from_gpu_tokens(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    tier1_job_buffer: &Buffer,
    token_buffers: &J2kResidentClassicTier1GpuTokenBuffers,
    tier1_output_buffer: &Buffer,
    tier1_status_buffer: &Buffer,
    tier1_segment_buffer: &Buffer,
) {
    #[cfg(test)]
    test_counters::record_classic_gpu_token_pack_dispatch();

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 token pack");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_token_pack_bypass_u16_32);
    encoder.set_buffer(0, Some(tier1_job_buffer), 0);
    encoder.set_buffer(1, Some(&token_buffers.counter_buffer), 0);
    encoder.set_buffer(2, Some(&token_buffers.token_buffer), 0);
    encoder.set_buffer(3, Some(&token_buffers.segment_buffer), 0);
    encoder.set_buffer(4, Some(tier1_output_buffer), 0);
    encoder.set_buffer(5, Some(tier1_status_buffer), 0);
    encoder.set_buffer(6, Some(tier1_segment_buffer), 0);
    encoder.set_bytes(
        7,
        size_of::<u32>() as u64,
        (&raw const token_buffers.token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        8,
        size_of::<u32>() as u64,
        (&raw const token_buffers.token_segment_stride).cast(),
    );
    encoder.set_bytes(
        9,
        size_of::<u32>() as u64,
        (&raw const token_buffers.job_count).cast(),
    );
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(token_buffers.job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_token_pack_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_split_token_pack_from_gpu_tokens(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    tier1_job_buffer: &Buffer,
    token_buffers: &J2kResidentClassicTier1SplitTokenBuffers,
    tier1_output_buffer: &Buffer,
    tier1_status_buffer: &Buffer,
    tier1_segment_buffer: &Buffer,
) {
    #[cfg(test)]
    test_counters::record_classic_gpu_token_pack_dispatch();

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 split token pack");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_split_token_pack_bypass_u16_32);
    encoder.set_buffer(0, Some(tier1_job_buffer), 0);
    encoder.set_buffer(1, Some(&token_buffers.counter_buffer), 0);
    encoder.set_buffer(2, Some(&token_buffers.mq_token_buffer), 0);
    encoder.set_buffer(3, Some(&token_buffers.raw_token_buffer), 0);
    encoder.set_buffer(4, Some(&token_buffers.segment_buffer), 0);
    encoder.set_buffer(5, Some(tier1_output_buffer), 0);
    encoder.set_buffer(6, Some(tier1_status_buffer), 0);
    encoder.set_buffer(7, Some(tier1_segment_buffer), 0);
    encoder.set_bytes(
        8,
        size_of::<u32>() as u64,
        (&raw const token_buffers.mq_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        9,
        size_of::<u32>() as u64,
        (&raw const token_buffers.raw_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        10,
        size_of::<u32>() as u64,
        (&raw const token_buffers.token_segment_stride).cast(),
    );
    encoder.set_bytes(
        11,
        size_of::<u32>() as u64,
        (&raw const token_buffers.job_count).cast(),
    );
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(token_buffers.job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_split_token_pack_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn schedule_classic_tier1_gpu_token_pack_readback(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    token_buffers: &J2kResidentClassicTier1GpuTokenBuffers,
    profile_stages: bool,
) -> Result<Option<J2kResidentClassicTier1TokenEmitReadback>, Error> {
    if !profile_stages || token_buffers.job_count == 0 {
        return Ok(None);
    }

    let count = usize::try_from(token_buffers.job_count).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic GPU token-pack readback job count exceeds usize".to_string(),
    })?;
    let token_stride_bytes =
        usize::try_from(token_buffers.token_stride_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic GPU token-pack token stride exceeds usize".to_string(),
        })?;
    let token_segment_stride =
        usize::try_from(token_buffers.token_segment_stride).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic GPU token-pack segment stride exceeds usize".to_string(),
        })?;
    let counter_byte_len = count
        .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic GPU token-pack counter readback size overflow".to_string(),
        })?;
    let counter_readback = runtime.device.new_buffer(
        counter_byte_len.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let copy_token_payloads = metal_profile_classic_tier1_token_pack_enabled();
    let (token_readback, token_byte_len) = if copy_token_payloads {
        let byte_len = count
            .checked_mul(token_stride_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal classic GPU token-pack token readback size overflow"
                    .to_string(),
            })?;
        (
            Some(runtime.device.new_buffer(
                byte_len.max(1) as u64,
                MTLResourceOptions::StorageModeShared,
            )),
            byte_len,
        )
    } else {
        (None, 0)
    };
    let (segment_readback, segment_byte_len) = if copy_token_payloads {
        let byte_len = count
            .checked_mul(token_segment_stride)
            .and_then(|segment_count| {
                segment_count.checked_mul(size_of::<J2kClassicTier1TokenSegment>())
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal classic GPU token-pack segment readback size overflow"
                    .to_string(),
            })?;
        (
            Some(runtime.device.new_buffer(
                byte_len.max(1) as u64,
                MTLResourceOptions::StorageModeShared,
            )),
            byte_len,
        )
    } else {
        (None, 0)
    };

    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_buffer(
        &token_buffers.counter_buffer,
        0,
        &counter_readback,
        0,
        counter_byte_len as u64,
    );
    if let Some(token_readback) = token_readback.as_ref() {
        blit.copy_from_buffer(
            &token_buffers.token_buffer,
            0,
            token_readback,
            0,
            token_byte_len as u64,
        );
    }
    if let Some(segment_readback) = segment_readback.as_ref() {
        blit.copy_from_buffer(
            &token_buffers.segment_buffer,
            0,
            segment_readback,
            0,
            segment_byte_len as u64,
        );
    }
    blit.end_encoding();

    Ok(Some(J2kResidentClassicTier1TokenEmitReadback {
        counter_buffer: counter_readback,
        token_buffer: token_readback,
        segment_buffer: segment_readback,
        token_stride_bytes,
        token_segment_stride,
        count,
    }))
}

#[cfg(target_os = "macos")]
fn record_classic_tier1_density_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1DensityReadback,
) -> Result<(), Error> {
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .buffer
                .contents()
                .cast::<J2kClassicTier1DensityCounters>(),
            readback.count,
        )
    };
    for counter in counters {
        stage_stats.tier1_sigprop_active_candidate_count_total = stage_stats
            .tier1_sigprop_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 sigprop candidate count exceeds usize"
                            .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_sigprop_new_significant_count_total = stage_stats
            .tier1_sigprop_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 sigprop significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_magref_active_candidate_count_total = stage_stats
            .tier1_magref_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.magref_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 magref candidate count exceeds usize"
                            .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_arithmetic_sigprop_active_candidate_count_total = stage_stats
            .tier1_arithmetic_sigprop_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.arithmetic_sigprop_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 arithmetic sigprop candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_arithmetic_sigprop_new_significant_count_total = stage_stats
            .tier1_arithmetic_sigprop_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.arithmetic_sigprop_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 arithmetic sigprop significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_raw_sigprop_active_candidate_count_total = stage_stats
            .tier1_raw_sigprop_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.raw_sigprop_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 raw sigprop candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_raw_sigprop_new_significant_count_total = stage_stats
            .tier1_raw_sigprop_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.raw_sigprop_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 raw sigprop significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_arithmetic_magref_active_candidate_count_total = stage_stats
            .tier1_arithmetic_magref_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.arithmetic_magref_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 arithmetic magref candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_raw_magref_active_candidate_count_total = stage_stats
            .tier1_raw_magref_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.raw_magref_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 raw magref candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_cleanup_active_candidate_count_total = stage_stats
            .tier1_cleanup_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 cleanup candidate count exceeds usize"
                            .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_cleanup_new_significant_count_total = stage_stats
            .tier1_cleanup_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 cleanup significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_cleanup_rlc_stripe_count_total = stage_stats
            .tier1_cleanup_rlc_stripe_count_total
            .saturating_add(usize::try_from(counter.cleanup_rlc_stripes).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 cleanup RLC stripe count exceeds usize"
                        .to_string(),
                }
            })?);
        stage_stats.tier1_cleanup_rlc_zero_stripe_count_total = stage_stats
            .tier1_cleanup_rlc_zero_stripe_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_rlc_zero_stripes).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 cleanup zero-RLC stripe count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn record_classic_tier1_symbol_plan_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1SymbolPlanReadback,
) -> Result<(), Error> {
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            readback.count,
        )
    };
    for counter in counters {
        if counter.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1 symbol plan",
                counter.code,
                counter.detail,
            ));
        }
        stage_stats.tier1_symbol_plan_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_mq_symbol_count_total
            .saturating_add(usize::try_from(counter.mq_symbol_count).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 symbol-plan MQ count exceeds usize"
                        .to_string(),
                }
            })?);
        stage_stats.tier1_symbol_plan_raw_bit_count_total = stage_stats
            .tier1_symbol_plan_raw_bit_count_total
            .saturating_add(usize::try_from(counter.raw_bit_count).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 symbol-plan raw bit count exceeds usize"
                        .to_string(),
                }
            })?);
        let mq_symbol_count =
            usize::try_from(counter.mq_symbol_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan MQ count exceeds usize".to_string(),
            })?;
        let raw_bit_count =
            usize::try_from(counter.raw_bit_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan raw bit count exceeds usize"
                    .to_string(),
            })?;
        stage_stats.max_tier1_symbol_plan_mq_symbols_per_block = stage_stats
            .max_tier1_symbol_plan_mq_symbols_per_block
            .max(mq_symbol_count);
        stage_stats.max_tier1_symbol_plan_raw_bits_per_block = stage_stats
            .max_tier1_symbol_plan_raw_bits_per_block
            .max(raw_bit_count);
        let mq_packed_bytes = mq_symbol_count
            .saturating_mul(6)
            .saturating_add(7)
            .checked_div(8)
            .unwrap_or(usize::MAX);
        let raw_packed_bytes = raw_bit_count
            .saturating_add(7)
            .checked_div(8)
            .unwrap_or(usize::MAX);
        let packed_token_bytes = mq_packed_bytes.saturating_add(raw_packed_bytes);
        stage_stats.tier1_symbol_plan_packed_token_bytes_total = stage_stats
            .tier1_symbol_plan_packed_token_bytes_total
            .saturating_add(packed_token_bytes);
        stage_stats.max_tier1_symbol_plan_packed_token_bytes_per_block = stage_stats
            .max_tier1_symbol_plan_packed_token_bytes_per_block
            .max(packed_token_bytes);
        stage_stats.tier1_symbol_plan_cleanup_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_cleanup_mq_symbol_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_mq_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan cleanup MQ count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_sigprop_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_sigprop_mq_symbol_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_mq_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan sigprop MQ count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_magref_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_magref_mq_symbol_count_total
            .saturating_add(
                usize::try_from(counter.magref_mq_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan magref MQ count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_raw_sigprop_bit_count_total = stage_stats
            .tier1_symbol_plan_raw_sigprop_bit_count_total
            .saturating_add(usize::try_from(counter.raw_sigprop_bit_count).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 symbol-plan raw sigprop bit count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.tier1_symbol_plan_raw_magref_bit_count_total = stage_stats
            .tier1_symbol_plan_raw_magref_bit_count_total
            .saturating_add(usize::try_from(counter.raw_magref_bit_count).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 symbol-plan raw magref bit count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.tier1_symbol_plan_cleanup_sign_symbol_count_total = stage_stats
            .tier1_symbol_plan_cleanup_sign_symbol_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_sign_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan cleanup sign count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_sigprop_sign_symbol_count_total = stage_stats
            .tier1_symbol_plan_sigprop_sign_symbol_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_sign_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan sigprop sign count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_mq_symbol_hash_xor ^= usize::try_from(counter.mq_symbol_hash)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan MQ hash exceeds usize".to_string(),
            })?;
        stage_stats.tier1_symbol_plan_raw_bit_hash_xor ^= usize::try_from(counter.raw_bit_hash)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan raw hash exceeds usize".to_string(),
            })?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn record_classic_tier1_pass_plan_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1PassPlanReadback,
) -> Result<(), Error> {
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .buffer
                .contents()
                .cast::<J2kClassicTier1PassPlanCounters>(),
            readback.count,
        )
    };
    for counter in counters {
        if counter.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1 pass plan",
                counter.code,
                counter.detail,
            ));
        }
        let mq_symbol_count =
            usize::try_from(counter.mq_symbol_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 pass-plan MQ count exceeds usize".to_string(),
            })?;
        let raw_bit_count =
            usize::try_from(counter.raw_bit_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 pass-plan raw bit count exceeds usize"
                    .to_string(),
            })?;
        stage_stats.tier1_pass_plan_mq_symbol_count_total = stage_stats
            .tier1_pass_plan_mq_symbol_count_total
            .saturating_add(mq_symbol_count);
        stage_stats.tier1_pass_plan_raw_bit_count_total = stage_stats
            .tier1_pass_plan_raw_bit_count_total
            .saturating_add(raw_bit_count);
        stage_stats.tier1_pass_plan_nonempty_mq_pass_count_total = stage_stats
            .tier1_pass_plan_nonempty_mq_pass_count_total
            .saturating_add(usize::try_from(counter.nonempty_mq_passes).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 pass-plan nonempty MQ pass count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.tier1_pass_plan_nonempty_raw_pass_count_total = stage_stats
            .tier1_pass_plan_nonempty_raw_pass_count_total
            .saturating_add(usize::try_from(counter.nonempty_raw_passes).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 pass-plan nonempty raw pass count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.max_tier1_pass_plan_mq_symbols_per_pass =
            stage_stats.max_tier1_pass_plan_mq_symbols_per_pass.max(
                usize::try_from(counter.max_mq_symbols_per_pass).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 pass-plan max MQ pass count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.max_tier1_pass_plan_raw_bits_per_pass =
            stage_stats.max_tier1_pass_plan_raw_bits_per_pass.max(
                usize::try_from(counter.max_raw_bits_per_pass).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 pass-plan max raw pass count exceeds usize"
                        .to_string(),
                })?,
            );

        let pass_mq_total = counter.mq_symbols_by_pass.iter().try_fold(
            0usize,
            |acc, &value| -> Result<usize, Error> {
                Ok(acc.saturating_add(
                    usize::try_from(value).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 pass-plan MQ pass count exceeds usize"
                            .to_string(),
                    })?,
                ))
            },
        )?;
        let pass_raw_total = counter.raw_bits_by_pass.iter().try_fold(
            0usize,
            |acc, &value| -> Result<usize, Error> {
                Ok(acc.saturating_add(usize::try_from(value).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 pass-plan raw pass count exceeds usize"
                            .to_string(),
                    }
                })?))
            },
        )?;
        if pass_mq_total != mq_symbol_count || pass_raw_total != raw_bit_count {
            return Err(Error::MetalKernel {
                message: "J2K Metal classic Tier-1 pass-plan per-pass totals are inconsistent"
                    .to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn compare_classic_tier1_symbol_plan_and_pass_plan_counters(
    symbol_plan: &J2kResidentClassicTier1SymbolPlanReadback,
    pass_plan: &J2kResidentClassicTier1PassPlanReadback,
) -> Result<(), Error> {
    if symbol_plan.count != pass_plan.count {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 pass-plan comparison count mismatch".to_string(),
        });
    }
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let symbol_plan_counters = unsafe {
        core::slice::from_raw_parts(
            symbol_plan
                .buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            symbol_plan.count,
        )
    };
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let pass_plan_counters = unsafe {
        core::slice::from_raw_parts(
            pass_plan
                .buffer
                .contents()
                .cast::<J2kClassicTier1PassPlanCounters>(),
            pass_plan.count,
        )
    };
    for (idx, (plan, pass)) in symbol_plan_counters
        .iter()
        .zip(pass_plan_counters)
        .enumerate()
    {
        let plan_values = [
            plan.code,
            plan.detail,
            plan.coding_passes,
            plan.missing_bit_planes,
            plan.segment_count,
            plan.mq_symbol_count,
            plan.raw_bit_count,
        ];
        let pass_values = [
            pass.code,
            pass.detail,
            pass.coding_passes,
            pass.missing_bit_planes,
            pass.segment_count,
            pass.mq_symbol_count,
            pass.raw_bit_count,
        ];
        if plan_values != pass_values {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal classic Tier-1 pass-plan diverged from symbol plan at block {idx}"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn record_classic_tier1_token_emit_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1TokenEmitReadback,
) -> Result<(), Error> {
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .counter_buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            readback.count,
        )
    };
    for counter in counters {
        if counter.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1 token emit",
                counter.code,
                counter.detail,
            ));
        }
        let mq_symbol_count =
            usize::try_from(counter.mq_symbol_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token-emitter MQ count exceeds usize"
                    .to_string(),
            })?;
        let raw_bit_count =
            usize::try_from(counter.raw_bit_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token-emitter raw bit count exceeds usize"
                    .to_string(),
            })?;
        let segment_count =
            usize::try_from(counter.segment_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token-emitter segment count exceeds usize"
                    .to_string(),
            })?;
        let token_bytes = mq_symbol_count
            .saturating_mul(6)
            .saturating_add(raw_bit_count)
            .saturating_add(7)
            .checked_div(8)
            .unwrap_or(usize::MAX);
        stage_stats.tier1_token_emit_mq_symbol_count_total = stage_stats
            .tier1_token_emit_mq_symbol_count_total
            .saturating_add(mq_symbol_count);
        stage_stats.tier1_token_emit_raw_bit_count_total = stage_stats
            .tier1_token_emit_raw_bit_count_total
            .saturating_add(raw_bit_count);
        stage_stats.tier1_token_emit_token_bytes_total = stage_stats
            .tier1_token_emit_token_bytes_total
            .saturating_add(token_bytes);
        stage_stats.max_tier1_token_emit_token_bytes_per_block = stage_stats
            .max_tier1_token_emit_token_bytes_per_block
            .max(token_bytes);
        stage_stats.tier1_token_emit_segment_count_total = stage_stats
            .tier1_token_emit_segment_count_total
            .saturating_add(segment_count);
        stage_stats.max_tier1_token_emit_segments_per_block = stage_stats
            .max_tier1_token_emit_segments_per_block
            .max(segment_count);
        stage_stats.tier1_token_emit_mq_symbol_hash_xor ^= usize::try_from(counter.mq_symbol_hash)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token-emitter MQ hash exceeds usize".to_string(),
            })?;
        stage_stats.tier1_token_emit_raw_bit_hash_xor ^= usize::try_from(counter.raw_bit_hash)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token-emitter raw hash exceeds usize"
                    .to_string(),
            })?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn compare_classic_tier1_symbol_plan_and_token_emit_counters(
    symbol_plan: &J2kResidentClassicTier1SymbolPlanReadback,
    token_emit: &J2kResidentClassicTier1TokenEmitReadback,
) -> Result<(), Error> {
    if symbol_plan.count != token_emit.count {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token-emitter comparison count mismatch".to_string(),
        });
    }
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let symbol_plan_counters = unsafe {
        core::slice::from_raw_parts(
            symbol_plan
                .buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            symbol_plan.count,
        )
    };
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let token_emit_counters = unsafe {
        core::slice::from_raw_parts(
            token_emit
                .counter_buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            token_emit.count,
        )
    };
    for (idx, (plan, emit)) in symbol_plan_counters
        .iter()
        .zip(token_emit_counters)
        .enumerate()
    {
        let plan_values = [
            plan.code,
            plan.detail,
            plan.coding_passes,
            plan.missing_bit_planes,
            plan.segment_count,
            plan.mq_symbol_count,
            plan.raw_bit_count,
            plan.cleanup_mq_symbol_count,
            plan.sigprop_mq_symbol_count,
            plan.magref_mq_symbol_count,
            plan.raw_sigprop_bit_count,
            plan.raw_magref_bit_count,
            plan.cleanup_sign_symbol_count,
            plan.sigprop_sign_symbol_count,
            plan.mq_symbol_hash,
            plan.raw_bit_hash,
        ];
        let emit_values = [
            emit.code,
            emit.detail,
            emit.coding_passes,
            emit.missing_bit_planes,
            emit.segment_count,
            emit.mq_symbol_count,
            emit.raw_bit_count,
            emit.cleanup_mq_symbol_count,
            emit.sigprop_mq_symbol_count,
            emit.magref_mq_symbol_count,
            emit.raw_sigprop_bit_count,
            emit.raw_magref_bit_count,
            emit.cleanup_sign_symbol_count,
            emit.sigprop_sign_symbol_count,
            emit.mq_symbol_hash,
            emit.raw_bit_hash,
        ];
        if plan_values != emit_values {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal classic Tier-1 token-emitter diverged from symbol plan at block {idx}"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_classic_tier1_split_token_emit_counters(
    readback: &J2kResidentClassicTier1SplitTokenBuffers,
) -> Result<(), Error> {
    if readback.mq_token_stride_bytes == 0
        || readback.raw_token_stride_bytes == 0
        || readback.token_segment_stride == 0
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 split-token readback has empty stride".to_string(),
        });
    }
    let count = usize::try_from(readback.job_count).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 split-token counter count exceeds usize".to_string(),
    })?;
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .counter_buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            count,
        )
    };
    for counter in counters {
        if counter.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1 split-token emit",
                counter.code,
                counter.detail,
            ));
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn compare_classic_tier1_symbol_plan_and_split_token_emit_counters(
    symbol_plan: &J2kResidentClassicTier1SymbolPlanReadback,
    split_emit: &J2kResidentClassicTier1SplitTokenBuffers,
) -> Result<(), Error> {
    let split_count = usize::try_from(split_emit.job_count).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 split-token comparison count exceeds usize".to_string(),
    })?;
    if symbol_plan.count != split_count {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 split-token comparison count mismatch".to_string(),
        });
    }
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let symbol_plan_counters = unsafe {
        core::slice::from_raw_parts(
            symbol_plan
                .buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            symbol_plan.count,
        )
    };
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let split_emit_counters = unsafe {
        core::slice::from_raw_parts(
            split_emit
                .counter_buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            split_count,
        )
    };
    for (idx, (plan, emit)) in symbol_plan_counters
        .iter()
        .zip(split_emit_counters)
        .enumerate()
    {
        let plan_values = [
            plan.code,
            plan.detail,
            plan.coding_passes,
            plan.missing_bit_planes,
            plan.segment_count,
            plan.mq_symbol_count,
            plan.raw_bit_count,
            plan.cleanup_mq_symbol_count,
            plan.sigprop_mq_symbol_count,
            plan.magref_mq_symbol_count,
            plan.raw_sigprop_bit_count,
            plan.raw_magref_bit_count,
            plan.cleanup_sign_symbol_count,
            plan.sigprop_sign_symbol_count,
            plan.mq_symbol_hash,
            plan.raw_bit_hash,
        ];
        let emit_values = [
            emit.code,
            emit.detail,
            emit.coding_passes,
            emit.missing_bit_planes,
            emit.segment_count,
            emit.mq_symbol_count,
            emit.raw_bit_count,
            emit.cleanup_mq_symbol_count,
            emit.sigprop_mq_symbol_count,
            emit.magref_mq_symbol_count,
            emit.raw_sigprop_bit_count,
            emit.raw_magref_bit_count,
            emit.cleanup_sign_symbol_count,
            emit.sigprop_sign_symbol_count,
            emit.mq_symbol_hash,
            emit.raw_bit_hash,
        ];
        if plan_values != emit_values {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal classic Tier-1 split-token emitter diverged from symbol plan at block {idx}"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn profile_classic_tier1_token_pack(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1TokenEmitReadback,
) -> Result<(), Error> {
    if !metal_profile_classic_tier1_token_pack_enabled() {
        return Ok(());
    }
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .counter_buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            readback.count,
        )
    };
    let token_buffer = readback
        .token_buffer
        .as_ref()
        .ok_or_else(|| Error::MetalKernel {
            message:
                "J2K Metal classic Tier-1 token-pack profiling requires token payload readback"
                    .to_string(),
        })?;
    let segment_buffer = readback
        .segment_buffer
        .as_ref()
        .ok_or_else(|| Error::MetalKernel {
            message:
                "J2K Metal classic Tier-1 token-pack profiling requires token segment readback"
                    .to_string(),
        })?;
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let token_bytes = unsafe {
        core::slice::from_raw_parts(
            token_buffer.contents().cast::<u8>(),
            readback.count.saturating_mul(readback.token_stride_bytes),
        )
    };
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let token_segments = unsafe {
        core::slice::from_raw_parts(
            segment_buffer
                .contents()
                .cast::<J2kClassicTier1TokenSegment>(),
            readback.count.saturating_mul(readback.token_segment_stride),
        )
    };
    let token_stride_bytes = readback.token_stride_bytes;
    let token_segment_stride = readback.token_segment_stride;

    let started = Instant::now();
    let packed_lengths = (0..readback.count)
        .into_par_iter()
        .map(|block_idx| -> Result<usize, String> {
            let counter = &counters[block_idx];
            if counter.code != J2K_ENCODE_STATUS_OK {
                return Err(format!(
                "classic Tier-1 token pack input failed at block {block_idx}: code={} detail={}",
                counter.code, counter.detail
            ));
            }
            let segment_count = usize::try_from(counter.segment_count)
                .map_err(|_| "J2K Metal classic Tier-1 token-pack segment count exceeds usize")?;
            if segment_count > CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY {
                return Err(
                    "J2K Metal classic Tier-1 token-pack segment count exceeds capacity"
                        .to_string(),
                );
            }
            let token_start = block_idx
                .checked_mul(token_stride_bytes)
                .ok_or("J2K Metal classic Tier-1 token-pack byte offset overflow")?;
            let segment_start = block_idx
                .checked_mul(token_segment_stride)
                .ok_or("J2K Metal classic Tier-1 token-pack segment offset overflow")?;
            let mut native_segments = Vec::with_capacity(segment_count);
            for segment in &token_segments[segment_start..segment_start + segment_count] {
                let start_coding_pass = u8::try_from(segment.pass_range & 0xFFFF)
                    .map_err(|_| "J2K Metal classic Tier-1 token-pack start pass exceeds u8")?;
                let end_coding_pass = u8::try_from(segment.pass_range >> 16)
                    .map_err(|_| "J2K Metal classic Tier-1 token-pack end pass exceeds u8")?;
                native_segments.push(J2kTier1TokenSegment {
                    token_bit_offset: segment.token_bit_offset,
                    token_bit_count: segment.token_bit_count,
                    start_coding_pass,
                    end_coding_pass,
                    use_arithmetic: (segment.flags & 1) != 0,
                });
            }
            let packed = pack_j2k_code_block_scalar_from_tier1_tokens(
                &token_bytes[token_start..token_start + token_stride_bytes],
                &native_segments,
                u8::try_from(counter.coding_passes).map_err(|_| {
                    "J2K Metal classic Tier-1 token-pack coding-pass count exceeds u8"
                })?,
                u8::try_from(counter.missing_bit_planes).map_err(|_| {
                    "J2K Metal classic Tier-1 token-pack missing bitplanes exceed u8"
                })?,
            )
            .map_err(|message| format!("J2K Metal classic Tier-1 token-pack failed: {message}"))?;
            Ok(packed.data.len())
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|message| Error::MetalKernel { message })?;
    for output_len in packed_lengths {
        stage_stats.tier1_token_pack_output_bytes_total = stage_stats
            .tier1_token_pack_output_bytes_total
            .saturating_add(output_len);
        stage_stats.max_tier1_token_pack_output_bytes_per_block = stage_stats
            .max_tier1_token_pack_output_bytes_per_block
            .max(output_len);
    }
    stage_stats.classic_tier1_token_pack_duration = started.elapsed();
    Ok(())
}

#[cfg(target_os = "macos")]
fn record_resident_tier1_output_usage(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentTier1StatusReadback,
    classic_gpu_token_pack_used: bool,
) -> Result<(), Error> {
    match readback.kind {
        J2kResidentTier1StatusKind::Classic => {
            let classic_jobs =
                readback
                    .classic_jobs
                    .as_ref()
                    .ok_or_else(|| Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 profile readback is missing job metadata"
                                .to_string(),
                    })?;
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            let statuses = unsafe {
                core::slice::from_raw_parts(
                    readback.buffer.contents().cast::<J2kClassicEncodeStatus>(),
                    readback.count,
                )
            };
            if classic_jobs.len() != statuses.len() {
                return Err(Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 profile readback job/status count mismatch"
                        .to_string(),
                });
            }
            for (status, job) in statuses.iter().zip(classic_jobs) {
                if status.code != J2K_ENCODE_STATUS_OK {
                    return Err(encode_status_error(
                        "classic Tier-1",
                        status.code,
                        status.detail,
                    ));
                }
                let data_len =
                    usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 output length exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_output_used_bytes_total = stage_stats
                    .tier1_output_used_bytes_total
                    .saturating_add(data_len);
                stage_stats.max_tier1_output_used_bytes =
                    stage_stats.max_tier1_output_used_bytes.max(data_len);
                if classic_gpu_token_pack_used {
                    stage_stats.tier1_token_pack_output_bytes_total = stage_stats
                        .tier1_token_pack_output_bytes_total
                        .saturating_add(data_len);
                    stage_stats.max_tier1_token_pack_output_bytes_per_block = stage_stats
                        .max_tier1_token_pack_output_bytes_per_block
                        .max(data_len);
                }
                let coding_passes =
                    usize::try_from(status.number_of_coding_passes).map_err(|_| {
                        Error::MetalKernel {
                            message: "J2K Metal classic Tier-1 coding-pass count exceeds usize"
                                .to_string(),
                        }
                    })?;
                stage_stats.tier1_coding_pass_count_total = stage_stats
                    .tier1_coding_pass_count_total
                    .saturating_add(coding_passes);
                stage_stats.max_tier1_coding_passes_per_block = stage_stats
                    .max_tier1_coding_passes_per_block
                    .max(coding_passes);
                let pass_counts =
                    classic_tier1_pass_class_counts(coding_passes, readback.classic_style_flags);
                let coeff_count = usize::try_from(job.width)
                    .and_then(|width| {
                        usize::try_from(job.height).map(|height| width.saturating_mul(height))
                    })
                    .map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 code-block dimensions exceed usize"
                            .to_string(),
                    })?;
                accumulate_classic_tier1_scan_estimates(stage_stats, pass_counts, coeff_count);
                stage_stats.tier1_arithmetic_pass_count_total = stage_stats
                    .tier1_arithmetic_pass_count_total
                    .saturating_add(pass_counts.arithmetic);
                stage_stats.tier1_raw_pass_count_total = stage_stats
                    .tier1_raw_pass_count_total
                    .saturating_add(pass_counts.raw);
                stage_stats.tier1_cleanup_pass_count_total = stage_stats
                    .tier1_cleanup_pass_count_total
                    .saturating_add(pass_counts.cleanup);
                stage_stats.tier1_sigprop_pass_count_total = stage_stats
                    .tier1_sigprop_pass_count_total
                    .saturating_add(pass_counts.sigprop);
                stage_stats.tier1_magref_pass_count_total = stage_stats
                    .tier1_magref_pass_count_total
                    .saturating_add(pass_counts.magref);
                stage_stats.tier1_arithmetic_cleanup_pass_count_total = stage_stats
                    .tier1_arithmetic_cleanup_pass_count_total
                    .saturating_add(pass_counts.arithmetic_cleanup);
                stage_stats.tier1_arithmetic_sigprop_pass_count_total = stage_stats
                    .tier1_arithmetic_sigprop_pass_count_total
                    .saturating_add(pass_counts.arithmetic_sigprop);
                stage_stats.tier1_arithmetic_magref_pass_count_total = stage_stats
                    .tier1_arithmetic_magref_pass_count_total
                    .saturating_add(pass_counts.arithmetic_magref);
                stage_stats.tier1_raw_sigprop_pass_count_total = stage_stats
                    .tier1_raw_sigprop_pass_count_total
                    .saturating_add(pass_counts.raw_sigprop);
                stage_stats.tier1_raw_magref_pass_count_total = stage_stats
                    .tier1_raw_magref_pass_count_total
                    .saturating_add(pass_counts.raw_magref);
                if coding_passes == 0 {
                    stage_stats.tier1_zero_block_count_total =
                        stage_stats.tier1_zero_block_count_total.saturating_add(1);
                } else {
                    stage_stats.tier1_nonzero_block_count_total = stage_stats
                        .tier1_nonzero_block_count_total
                        .saturating_add(1);
                }
                let missing_bitplanes =
                    usize::try_from(status.missing_bit_planes).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 missing-bitplane count exceeds usize"
                            .to_string(),
                    })?;
                stage_stats.tier1_missing_bitplane_count_total = stage_stats
                    .tier1_missing_bitplane_count_total
                    .saturating_add(missing_bitplanes);
                stage_stats.max_tier1_missing_bitplanes_per_block = stage_stats
                    .max_tier1_missing_bitplanes_per_block
                    .max(missing_bitplanes);
                let segment_count =
                    usize::try_from(status.segment_count).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 segment count exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_segment_count_total = stage_stats
                    .tier1_segment_count_total
                    .saturating_add(segment_count);
                stage_stats.max_tier1_segments_per_block =
                    stage_stats.max_tier1_segments_per_block.max(segment_count);
            }
        }
        J2kResidentTier1StatusKind::HighThroughput => {
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            let statuses = unsafe {
                core::slice::from_raw_parts(
                    readback.buffer.contents().cast::<J2kHtEncodeStatus>(),
                    readback.count,
                )
            };
            for status in statuses {
                if status.code != J2K_ENCODE_STATUS_OK {
                    return Err(encode_status_error(
                        "HTJ2K Tier-1",
                        status.code,
                        status.detail,
                    ));
                }
                let data_len =
                    usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal Tier-1 output length exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_output_used_bytes_total = stage_stats
                    .tier1_output_used_bytes_total
                    .saturating_add(data_len);
                stage_stats.max_tier1_output_used_bytes =
                    stage_stats.max_tier1_output_used_bytes.max(data_len);
                let coding_passes =
                    usize::try_from(status.num_coding_passes).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal Tier-1 coding-pass count exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_coding_pass_count_total = stage_stats
                    .tier1_coding_pass_count_total
                    .saturating_add(coding_passes);
                stage_stats.max_tier1_coding_passes_per_block = stage_stats
                    .max_tier1_coding_passes_per_block
                    .max(coding_passes);
                if coding_passes == 0 {
                    stage_stats.tier1_zero_block_count_total =
                        stage_stats.tier1_zero_block_count_total.saturating_add(1);
                } else {
                    stage_stats.tier1_nonzero_block_count_total = stage_stats
                        .tier1_nonzero_block_count_total
                        .saturating_add(1);
                }
                let missing_bitplanes =
                    usize::try_from(status.num_zero_bitplanes).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal Tier-1 missing-bitplane count exceeds usize"
                            .to_string(),
                    })?;
                stage_stats.tier1_missing_bitplane_count_total = stage_stats
                    .tier1_missing_bitplane_count_total
                    .saturating_add(missing_bitplanes);
                stage_stats.max_tier1_missing_bitplanes_per_block = stage_stats
                    .max_tier1_missing_bitplanes_per_block
                    .max(missing_bitplanes);
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn wait_resident_codestream_command_buffer(command_buffer: &CommandBufferRef) -> Result<(), Error> {
    #[cfg(test)]
    test_counters::record_resident_codestream_command_buffer_wait();
    let _signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT);
    wait_for_completion_metal(command_buffer)
}

#[cfg(target_os = "macos")]
fn finish_completed_resident_lossless_codestream_batch(
    pending: J2kPendingResidentLosslessCodestreamBatch,
) -> Result<J2kResidentLosslessCodestreamBatchResult, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST);
    let profile_stages = metal_profile_stages_enabled();
    let result_harvest_started = profile_stages.then(Instant::now);
    let gpu_timings = completed_command_buffers_gpu_duration_and_elapsed_window(
        &pending.retained_command_buffers,
        &pending.command_buffer,
    );
    let gpu_duration = gpu_timings.map(|timings| timings.0);
    let gpu_elapsed_wall_duration = gpu_timings.map(|timings| timings.1);
    let mut stage_stats = pending.stage_stats;
    if let Some(duration) = gpu_elapsed_wall_duration {
        stage_stats.gpu_elapsed_wall_duration = duration;
    }
    if profile_stages {
        record_completed_resident_encode_gpu_stages(
            &mut stage_stats,
            &pending.gpu_stage_command_buffers,
        );
    }
    if let Some(readback) = pending.tier1_status_readback.as_ref() {
        record_resident_tier1_output_usage(
            &mut stage_stats,
            readback,
            pending.classic_gpu_token_pack_used,
        )?;
    }
    if let Some(readback) = pending.classic_tier1_density_readback.as_ref() {
        record_classic_tier1_density_counters(&mut stage_stats, readback)?;
    }
    if let Some(readback) = pending.classic_tier1_symbol_plan_readback.as_ref() {
        record_classic_tier1_symbol_plan_counters(&mut stage_stats, readback)?;
    }
    if let (Some(symbol_plan), Some(pass_plan)) = (
        pending.classic_tier1_symbol_plan_readback.as_ref(),
        pending.classic_tier1_pass_plan_readback.as_ref(),
    ) {
        compare_classic_tier1_symbol_plan_and_pass_plan_counters(symbol_plan, pass_plan)?;
    }
    if let Some(readback) = pending.classic_tier1_pass_plan_readback.as_ref() {
        record_classic_tier1_pass_plan_counters(&mut stage_stats, readback)?;
    }
    if let (Some(symbol_plan), Some(token_emit)) = (
        pending.classic_tier1_symbol_plan_readback.as_ref(),
        pending.classic_tier1_token_emit_readback.as_ref(),
    ) {
        compare_classic_tier1_symbol_plan_and_token_emit_counters(symbol_plan, token_emit)?;
    }
    if let Some(readback) = pending.classic_tier1_token_emit_readback.as_ref() {
        record_classic_tier1_token_emit_counters(&mut stage_stats, readback)?;
        profile_classic_tier1_token_pack(&mut stage_stats, readback)?;
    }
    if let Some(readback) = pending.classic_tier1_split_token_emit_readback.as_ref() {
        validate_classic_tier1_split_token_emit_counters(readback)?;
    }
    if let (Some(symbol_plan), Some(split_emit)) = (
        pending.classic_tier1_symbol_plan_readback.as_ref(),
        pending.classic_tier1_split_token_emit_readback.as_ref(),
    ) {
        compare_classic_tier1_symbol_plan_and_split_token_emit_counters(symbol_plan, split_emit)?;
    }
    let runtime = pending.runtime.clone();
    let recyclable_private_buffers = pending.recyclable_private_buffers;
    let private_recycle_started = profile_stages.then(Instant::now);
    recycle_private_buffers(&runtime, recyclable_private_buffers)?;
    if let Some(started) = private_recycle_started {
        stage_stats.result_private_recycle_duration = started.elapsed();
    }
    let gpu_duration_share =
        gpu_duration.map(|duration| duration_share(duration, pending.capacities.len()));
    let status_copy_started = profile_stages.then(Instant::now);
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let statuses = unsafe {
        core::slice::from_raw_parts(
            pending
                .status_buffer
                .contents()
                .cast::<J2kCodestreamAssemblyStatus>(),
            pending.capacities.len(),
        )
    }
    .to_vec();
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let packet_statuses = unsafe {
        core::slice::from_raw_parts(
            pending
                .packet_status_buffer
                .contents()
                .cast::<J2kPacketEncodeStatus>(),
            pending.capacities.len(),
        )
    }
    .to_vec();
    if let Some(started) = status_copy_started {
        stage_stats.result_status_copy_duration = started.elapsed();
    }
    let recyclable_shared_buffers = pending.recyclable_shared_buffers;
    let shared_recycle_started = profile_stages.then(Instant::now);
    recycle_shared_buffers(&runtime, recyclable_shared_buffers)?;
    if let Some(started) = shared_recycle_started {
        stage_stats.result_shared_recycle_duration = started.elapsed();
    }
    let codestream_collect_started = profile_stages.then(Instant::now);
    let mut codestreams = Vec::with_capacity(pending.capacities.len());
    for (index, status) in statuses.into_iter().enumerate() {
        let packet_status = packet_statuses
            .get(index)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal packetization status missing for resident batch tile"
                    .to_string(),
            })?;
        if packet_status.code != J2K_ENCODE_STATUS_OK {
            return Err(packet_encode_status_error(*packet_status));
        }
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                pending.status_stage,
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: pending.length_error.to_string(),
        })?;
        let capacity = pending.capacities[index];
        if data_len > capacity {
            return Err(Error::MetalKernel {
                message: pending.capacity_error.to_string(),
            });
        }
        let packet_output_used =
            usize::try_from(packet_status.data_len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packet output length exceeds usize".to_string(),
            })?;
        let packet_payload_copy_jobs =
            usize::try_from(packet_status.detail).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packet payload-copy count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_bytes =
            usize::try_from(packet_status.payload_copy_bytes).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packet payload-copy byte count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_small_jobs = usize::try_from(packet_status.payload_copy_small_jobs)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal small packet payload-copy count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_medium_jobs =
            usize::try_from(packet_status.payload_copy_medium_jobs).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal medium packet payload-copy count exceeds usize".to_string(),
                }
            })?;
        let packet_payload_copy_large_jobs = usize::try_from(packet_status.payload_copy_large_jobs)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal large packet payload-copy count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_active_stripes =
            packet_payload_copy_jobs.saturating_mul(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB as usize);
        stage_stats.packet_output_used_bytes_total = stage_stats
            .packet_output_used_bytes_total
            .saturating_add(packet_output_used);
        stage_stats.max_packet_output_used_bytes = stage_stats
            .max_packet_output_used_bytes
            .max(packet_output_used);
        stage_stats.packet_payload_copy_job_count_total = stage_stats
            .packet_payload_copy_job_count_total
            .saturating_add(packet_payload_copy_jobs);
        stage_stats.max_packet_payload_copy_jobs_used_per_tile = stage_stats
            .max_packet_payload_copy_jobs_used_per_tile
            .max(packet_payload_copy_jobs);
        stage_stats.packet_payload_copy_bytes_total = stage_stats
            .packet_payload_copy_bytes_total
            .saturating_add(packet_payload_copy_bytes);
        stage_stats.max_packet_payload_copy_bytes_per_tile = stage_stats
            .max_packet_payload_copy_bytes_per_tile
            .max(packet_payload_copy_bytes);
        stage_stats.packet_payload_copy_small_job_count_total = stage_stats
            .packet_payload_copy_small_job_count_total
            .saturating_add(packet_payload_copy_small_jobs);
        stage_stats.packet_payload_copy_medium_job_count_total = stage_stats
            .packet_payload_copy_medium_job_count_total
            .saturating_add(packet_payload_copy_medium_jobs);
        stage_stats.packet_payload_copy_large_job_count_total = stage_stats
            .packet_payload_copy_large_job_count_total
            .saturating_add(packet_payload_copy_large_jobs);
        stage_stats.packet_payload_copy_active_stripe_count_total = stage_stats
            .packet_payload_copy_active_stripe_count_total
            .saturating_add(packet_payload_copy_active_stripes);
        if pending.codestream_payload_copy_dispatched {
            stage_stats.codestream_payload_copy_bytes_total = stage_stats
                .codestream_payload_copy_bytes_total
                .saturating_add(packet_output_used);
        }
        codestreams.push(J2kResidentLosslessCodestream {
            buffer: pending.buffer.clone(),
            byte_offset: pending.byte_offsets[index],
            byte_len: data_len,
            capacity,
            gpu_duration: gpu_duration_share,
        });
    }
    if let Some(started) = codestream_collect_started {
        stage_stats.result_codestream_collect_duration = started.elapsed();
    }
    if let Some(started) = result_harvest_started {
        stage_stats.result_harvest_duration = started.elapsed();
    }
    Ok(J2kResidentLosslessCodestreamBatchResult {
        codestreams,
        stage_stats,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_batched_packet_payload_copy(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    dispatch: J2kBatchedPacketPayloadCopyDispatch<'_>,
) -> bool {
    if dispatch.tile_count == 0 || dispatch.max_payload_copy_jobs_per_tile == 0 {
        return false;
    }

    let signpost = hybrid_stage_signpost(dispatch.signpost_name);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, dispatch.label);
    encoder.set_compute_pipeline_state(&runtime.packet_payload_copy_batched);
    encoder.set_buffer(0, Some(dispatch.payload_buffer), 0);
    encoder.set_buffer(1, Some(dispatch.packet_output_buffer), 0);
    encoder.set_buffer(2, Some(dispatch.packet_job_buffer), 0);
    encoder.set_buffer(3, Some(dispatch.packet_status_buffer), 0);
    encoder.set_buffer(4, Some(dispatch.packet_payload_copy_job_buffer), 0);
    let params = J2kPacketPayloadCopyParams {
        bytes_per_thread: PACKET_PAYLOAD_COPY_BYTES_PER_STRIPE,
        stripes_per_job: PACKET_PAYLOAD_COPY_STRIPES_PER_JOB,
    };
    encoder.set_bytes(
        5,
        size_of::<J2kPacketPayloadCopyParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.dispatch_threads(
        MTLSize {
            width: dispatch.max_payload_copy_jobs_per_tile,
            height: dispatch.tile_count,
            depth: u64::from(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB),
        },
        MTLSize {
            width: runtime
                .packet_payload_copy_batched
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    drop(signpost);
    true
}

#[cfg(target_os = "macos")]
fn dispatch_lossless_deinterleave(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: J2kLosslessDevicePrepareJob<'_>,
    plane0: &Buffer,
    plane1: &Buffer,
    plane2: &Buffer,
) -> Result<(), Error> {
    let input_byte_offset =
        u64::try_from(job.input_byte_offset).map_err(|_| Error::MetalKernel {
            message: "J2K Metal resident encode input offset exceeds u64".to_string(),
        })?;
    let src_stride = u32::try_from(job.input_pitch_bytes).map_err(|_| Error::MetalKernel {
        message: "J2K Metal resident encode input pitch exceeds u32".to_string(),
    })?;
    let sample_offset = if job.bit_depth == 0 || job.bit_depth > 16 {
        return Err(Error::MetalKernel {
            message: "J2K Metal resident encode bit depth must be 1-16".to_string(),
        });
    } else {
        1u32 << (u32::from(job.bit_depth) - 1)
    };
    let params = J2kLosslessDeinterleaveParams {
        src_width: job.input_width,
        src_height: job.input_height,
        src_stride,
        dst_width: job.output_width,
        dst_height: job.output_height,
        components: u32::from(job.components),
        bytes_per_sample: u32::from(job.bytes_per_sample),
        bit_depth: u32::from(job.bit_depth),
        sample_offset,
        signed_samples: 0,
    };
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep deinterleave");
    encoder.set_compute_pipeline_state(&runtime.lossless_deinterleave_to_planes);
    encoder.set_buffer(0, Some(job.input), input_byte_offset);
    encoder.set_buffer(1, Some(plane0), 0);
    encoder.set_buffer(2, Some(plane1), 0);
    encoder.set_buffer(3, Some(plane2), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kLosslessDeinterleaveParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(5, Some(plane2), 0);
    dispatch_2d_pipeline(
        encoder,
        &runtime.lossless_deinterleave_to_planes,
        (job.output_width, job.output_height),
    );
    encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_lossless_deinterleave_rct_rgb8(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: J2kLosslessDevicePrepareJob<'_>,
    plane0: &Buffer,
    plane1: &Buffer,
    plane2: &Buffer,
    status_buffer: &Buffer,
) -> Result<(), Error> {
    let input_byte_offset =
        u64::try_from(job.input_byte_offset).map_err(|_| Error::MetalKernel {
            message: "J2K Metal resident encode input offset exceeds u64".to_string(),
        })?;
    let src_stride = u32::try_from(job.input_pitch_bytes).map_err(|_| Error::MetalKernel {
        message: "J2K Metal resident encode input pitch exceeds u32".to_string(),
    })?;
    let sample_offset = if job.bit_depth == 0 || job.bit_depth > 16 {
        return Err(Error::MetalKernel {
            message: "J2K Metal resident encode bit depth must be 1-16".to_string(),
        });
    } else {
        1u32 << (u32::from(job.bit_depth) - 1)
    };
    let params = J2kLosslessDeinterleaveParams {
        src_width: job.input_width,
        src_height: job.input_height,
        src_stride,
        dst_width: job.output_width,
        dst_height: job.output_height,
        components: u32::from(job.components),
        bytes_per_sample: u32::from(job.bytes_per_sample),
        bit_depth: u32::from(job.bit_depth),
        sample_offset,
        signed_samples: 0,
    };
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep deinterleave + RCT");
    encoder.set_compute_pipeline_state(&runtime.lossless_deinterleave_rct_rgb8_to_planes);
    encoder.set_buffer(0, Some(job.input), input_byte_offset);
    encoder.set_buffer(1, Some(plane0), 0);
    encoder.set_buffer(2, Some(plane1), 0);
    encoder.set_buffer(3, Some(plane2), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kLosslessDeinterleaveParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(5, Some(status_buffer), 0);
    dispatch_2d_pipeline(
        encoder,
        &runtime.lossless_deinterleave_rct_rgb8_to_planes,
        (job.output_width, job.output_height),
    );
    encoder.end_encoding();
    #[cfg(test)]
    test_counters::record_lossless_deinterleave_rct_fused_dispatch();
    Ok(())
}

#[cfg(target_os = "macos")]
fn lossless_deinterleave_rct_rgb8_supported(job: J2kLosslessDevicePrepareJob<'_>) -> bool {
    job.components == 3 && job.bytes_per_sample == 1
}

#[cfg(target_os = "macos")]
fn dispatch_forward_rct_on_buffers(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane0: &Buffer,
    plane1: &Buffer,
    plane2: &Buffer,
    len: usize,
    status_buffer: &Buffer,
) -> Result<(), Error> {
    if len == 0 {
        return Ok(());
    }
    let params = J2kForwardRctParams {
        _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
            message: "J2K Metal resident encode RCT length exceeds u32".to_string(),
        })?,
        _reserved0: 0,
        _reserved1: 0,
        _reserved2: 0,
    };
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep RCT");
    encoder.set_compute_pipeline_state(&runtime.forward_rct);
    encoder.set_buffer(0, Some(plane0), 0);
    encoder.set_buffer(1, Some(plane1), 0);
    encoder.set_buffer(2, Some(plane2), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kForwardRctParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(4, Some(status_buffer), 0);
    let width = runtime
        .forward_rct
        .thread_execution_width()
        .max(1)
        .min(len as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt53_on_buffers(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    scratch: &Buffer,
    width: u32,
    height: u32,
    num_levels: u8,
) -> Buffer {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53Params {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
        };

        if current_height >= 2 {
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_vertical,
                command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_horizontal,
                command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    if active_is_input {
        input.to_owned()
    } else {
        scratch.to_owned()
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_forward_dwt53_components_on_buffers(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane_buffers: &[Buffer],
    scratch_buffers: &[Buffer],
    width: u32,
    height: u32,
    num_levels: u8,
    component_count: usize,
) -> Vec<Buffer> {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;
    let component_count_u32 = component_count as u32;

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53BatchedParams {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
            component_count: component_count_u32,
        };

        if current_height >= 2 {
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_vertical_batched,
                command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_horizontal_batched,
                command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    let active_buffers = if active_is_input {
        plane_buffers
    } else {
        scratch_buffers
    };
    active_buffers[..component_count].to_vec()
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt53_on_buffers_split_profile(
    runtime: &MetalRuntime,
    input: &Buffer,
    scratch: &Buffer,
    width: u32,
    height: u32,
    num_levels: u8,
) -> (Buffer, Vec<CommandBuffer>, Vec<CommandBuffer>) {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;
    let mut vertical_command_buffers = Vec::new();
    let mut horizontal_command_buffers = Vec::new();

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53Params {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
        };

        if current_height >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "j2k coefficient prep DWT 5/3 vertical",
            );
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_vertical,
                &command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            command_buffer.commit();
            vertical_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "j2k coefficient prep DWT 5/3 horizontal",
            );
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_horizontal,
                &command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            command_buffer.commit();
            horizontal_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    let active = if active_is_input {
        input.to_owned()
    } else {
        scratch.to_owned()
    };
    (active, vertical_command_buffers, horizontal_command_buffers)
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt53_components_split_profile(
    runtime: &MetalRuntime,
    plane_buffers: &[Buffer],
    scratch_buffers: &[Buffer],
    width: u32,
    height: u32,
    num_levels: u8,
    component_count: usize,
) -> (Vec<Buffer>, Vec<CommandBuffer>, Vec<CommandBuffer>) {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;
    let mut vertical_command_buffers = Vec::new();
    let mut horizontal_command_buffers = Vec::new();
    let component_count_u32 = component_count as u32;

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53BatchedParams {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
            component_count: component_count_u32,
        };

        if current_height >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "j2k coefficient prep DWT 5/3 vertical",
            );
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_vertical_batched,
                &command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            command_buffer.commit();
            vertical_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "j2k coefficient prep DWT 5/3 horizontal",
            );
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_horizontal_batched,
                &command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            command_buffer.commit();
            horizontal_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    let active_buffers = if active_is_input {
        plane_buffers
    } else {
        scratch_buffers
    };
    (
        active_buffers[..component_count].to_vec(),
        vertical_command_buffers,
        horizontal_command_buffers,
    )
}

#[cfg(target_os = "macos")]
fn dispatch_lossless_extract_coefficients(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: &[Buffer],
    coefficient_buffer: &Buffer,
    coefficient_jobs: &[J2kLosslessCoefficientJob],
    output_width: u32,
) -> Result<Buffer, Error> {
    let coefficient_job_buffer = copied_slice_buffer(&runtime.device, coefficient_jobs);
    let job_count = u32::try_from(coefficient_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal resident encode coefficient job count exceeds u32".to_string(),
    })?;
    let max_block_width = coefficient_jobs
        .iter()
        .map(|job| job.block_width)
        .max()
        .unwrap_or(1);
    let max_block_height = coefficient_jobs
        .iter()
        .map(|job| job.block_height)
        .max()
        .unwrap_or(1);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep extract");
    encoder.set_compute_pipeline_state(&runtime.lossless_extract_coefficients);
    encoder.set_buffer(0, planes.first().map(|buffer| &**buffer), 0);
    encoder.set_buffer(
        1,
        planes
            .get(1)
            .or_else(|| planes.first())
            .map(|buffer| &**buffer),
        0,
    );
    encoder.set_buffer(
        2,
        planes
            .get(2)
            .or_else(|| planes.first())
            .map(|buffer| &**buffer),
        0,
    );
    encoder.set_buffer(3, Some(coefficient_buffer), 0);
    encoder.set_buffer(4, Some(&coefficient_job_buffer), 0);
    encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
    dispatch_3d_pipeline(
        encoder,
        &runtime.lossless_extract_coefficients,
        (max_block_width, max_block_height, job_count),
    );
    encoder.end_encoding();
    let _ = output_width;
    Ok(coefficient_job_buffer)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct J2kLosslessPrepareSizes {
    plane_len: usize,
    plane_bytes: usize,
    coefficient_bytes: usize,
}

#[cfg(target_os = "macos")]
fn lossless_prepare_sizes(
    job: J2kLosslessDevicePrepareJob<'_>,
) -> Result<J2kLosslessPrepareSizes, Error> {
    if job.components != 1 && job.components != 3 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode supports grayscale or RGB input",
        });
    }
    if job.bytes_per_sample != 1 && job.bytes_per_sample != 2 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode supports 8-bit or 16-bit samples",
        });
    }
    let plane_len = (job.output_width as usize)
        .checked_mul(job.output_height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident encode plane size overflow".to_string(),
        })?;
    let plane_bytes =
        plane_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal resident encode plane byte size overflow".to_string(),
            })?;
    let coefficient_bytes = job
        .coefficient_count
        .max(1)
        .checked_mul(size_of::<i32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident encode coefficient size overflow".to_string(),
        })?;
    Ok(J2kLosslessPrepareSizes {
        plane_len,
        plane_bytes,
        coefficient_bytes,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_lossless_device_code_blocks(
    session: &crate::MetalBackendSession,
    job: J2kLosslessDevicePrepareJob<'_>,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
) -> Result<J2kPreparedLosslessDeviceCodeBlocks, Error> {
    let sizes = lossless_prepare_sizes(job)?;

    with_runtime_for_session(session, |runtime| {
        let mut plane_buffers = Vec::with_capacity(3);
        let mut scratch_buffers = Vec::with_capacity(usize::from(job.components));
        for _ in 0..3 {
            plane_buffers.push(runtime.device.new_buffer(
                sizes.plane_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            ));
        }
        for _ in 0..job.components {
            scratch_buffers.push(runtime.device.new_buffer(
                sizes.plane_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            ));
        }
        let coefficient_buffer = runtime.device.new_buffer(
            sizes.coefficient_bytes as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let deinterleave_status = J2kMctStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const deinterleave_status).cast(),
            size_of::<J2kMctStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = runtime.queue.new_command_buffer();

        if lossless_deinterleave_rct_rgb8_supported(job) {
            dispatch_lossless_deinterleave_rct_rgb8(
                runtime,
                command_buffer,
                job,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
                &status_buffer,
            )?;
        } else {
            dispatch_lossless_deinterleave(
                runtime,
                command_buffer,
                job,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
            )?;
        }
        if job.components == 3 && !lossless_deinterleave_rct_rgb8_supported(job) {
            dispatch_forward_rct_on_buffers(
                runtime,
                command_buffer,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
                sizes.plane_len,
                &status_buffer,
            )?;
        }

        let mut active_planes = Vec::with_capacity(usize::from(job.components));
        for component in 0..usize::from(job.components) {
            if job.num_decomposition_levels == 0 {
                active_planes.push(plane_buffers[component].clone());
            } else {
                active_planes.push(dispatch_forward_dwt53_on_buffers(
                    runtime,
                    command_buffer,
                    &plane_buffers[component],
                    &scratch_buffers[component],
                    job.output_width,
                    job.output_height,
                    job.num_decomposition_levels,
                ));
            }
        }
        while active_planes.len() < 3 {
            active_planes.push(active_planes[0].clone());
        }

        let coefficient_jobs = code_blocks
            .iter()
            .map(|block| J2kLosslessCoefficientJob {
                coefficient_offset: block.coefficient_offset,
                component: block.component,
                subband_x: block.subband_x,
                subband_y: block.subband_y,
                block_x: block.block_x,
                block_y: block.block_y,
                block_width: block.width,
                block_height: block.height,
                full_width: job.output_width,
            })
            .collect::<Vec<_>>();
        let coefficient_job_buffer = dispatch_lossless_extract_coefficients(
            runtime,
            command_buffer,
            &active_planes,
            &coefficient_buffer,
            &coefficient_jobs,
            job.output_width,
        )?;

        command_buffer.commit();
        Ok(J2kPreparedLosslessDeviceCodeBlocks {
            coefficient_buffer,
            coefficient_byte_offset: 0,
            coefficient_byte_len: sizes.coefficient_bytes,
            coefficient_buffer_is_batch_shared: false,
            code_blocks,
            recyclable_private_buffers: Vec::new(),
            _prepare_command_buffer: command_buffer.to_owned(),
            _prepare_deinterleave_rct_command_buffer: None,
            _prepare_dwt53_command_buffer: None,
            _prepare_dwt53_vertical_command_buffers: Vec::new(),
            _prepare_dwt53_horizontal_command_buffers: Vec::new(),
            _prepare_coefficient_extract_command_buffer: None,
            _deinterleave_status_buffer: status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_lossless_device_code_blocks_batch(
    session: &crate::MetalBackendSession,
    items: Vec<J2kLosslessDeviceBatchPrepareItem<'_>>,
) -> Result<Vec<J2kPreparedLosslessDeviceCodeBlocks>, Error> {
    if items.is_empty() {
        return Ok(Vec::new());
    }

    let mut sizes = Vec::with_capacity(items.len());
    let mut coefficient_byte_offsets = Vec::with_capacity(items.len());
    let mut total_coefficient_bytes = 0usize;
    for item in &items {
        let item_sizes = lossless_prepare_sizes(item.job).map_err(|err| Error::MetalKernel {
            message: format!(
                "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                item.tile_index
            ),
        })?;
        coefficient_byte_offsets.push(total_coefficient_bytes);
        total_coefficient_bytes = total_coefficient_bytes
            .checked_add(item_sizes.coefficient_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal resident batch coefficient size overflow".to_string(),
            })?;
        sizes.push(item_sizes);
    }

    with_runtime_for_session(session, |runtime| {
        let mut shared_recyclable_private_buffers = Vec::new();
        let coefficient_buffer = take_recyclable_private_buffer(
            runtime,
            total_coefficient_bytes.max(1),
            &mut shared_recyclable_private_buffers,
        )?;
        let split_prepare_command_buffers = metal_profile_coefficient_prep_split_commands_enabled();
        let shared_command_buffer = if split_prepare_command_buffers {
            None
        } else {
            Some(runtime.queue.new_command_buffer().to_owned())
        };
        let mut prepared = Vec::with_capacity(items.len());

        for ((item, item_sizes), coefficient_byte_offset) in
            items.into_iter().zip(sizes).zip(coefficient_byte_offsets)
        {
            let job = item.job;
            let mut recyclable_private_buffers = Vec::new();
            if !shared_recyclable_private_buffers.is_empty() {
                recyclable_private_buffers.append(&mut shared_recyclable_private_buffers);
            }
            let mut plane_buffers = Vec::with_capacity(3);
            let mut scratch_buffers = Vec::with_capacity(usize::from(job.components));
            for _ in 0..3 {
                plane_buffers.push(take_recyclable_private_buffer(
                    runtime,
                    item_sizes.plane_bytes,
                    &mut recyclable_private_buffers,
                )?);
            }
            for _ in 0..job.components {
                scratch_buffers.push(take_recyclable_private_buffer(
                    runtime,
                    item_sizes.plane_bytes,
                    &mut recyclable_private_buffers,
                )?);
            }

            let deinterleave_status = J2kMctStatus::default();
            let status_buffer = runtime.device.new_buffer_with_data(
                (&raw const deinterleave_status).cast(),
                size_of::<J2kMctStatus>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let mut prepare_deinterleave_rct_command_buffer = None;
            let prepare_dwt53_command_buffer = None;
            let mut prepare_dwt53_vertical_command_buffers = Vec::new();
            let mut prepare_dwt53_horizontal_command_buffers = Vec::new();
            let mut prepare_coefficient_extract_command_buffer = None;
            let deinterleave_command_buffer = if split_prepare_command_buffers {
                new_resident_encode_command_buffer(runtime, "j2k coefficient prep deinterleave rct")
            } else {
                shared_command_buffer
                    .as_ref()
                    .expect("shared coefficient prep command buffer exists")
                    .clone()
            };
            if lossless_deinterleave_rct_rgb8_supported(job) {
                dispatch_lossless_deinterleave_rct_rgb8(
                    runtime,
                    &deinterleave_command_buffer,
                    job,
                    &plane_buffers[0],
                    &plane_buffers[1],
                    &plane_buffers[2],
                    &status_buffer,
                )
            } else {
                dispatch_lossless_deinterleave(
                    runtime,
                    &deinterleave_command_buffer,
                    job,
                    &plane_buffers[0],
                    &plane_buffers[1],
                    &plane_buffers[2],
                )
            }
            .map_err(|err| Error::MetalKernel {
                message: format!(
                    "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                    item.tile_index
                ),
            })?;
            if job.components == 3 && !lossless_deinterleave_rct_rgb8_supported(job) {
                dispatch_forward_rct_on_buffers(
                    runtime,
                    &deinterleave_command_buffer,
                    &plane_buffers[0],
                    &plane_buffers[1],
                    &plane_buffers[2],
                    item_sizes.plane_len,
                    &status_buffer,
                )
                .map_err(|err| Error::MetalKernel {
                    message: format!(
                        "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                        item.tile_index
                    ),
                })?;
            }
            if split_prepare_command_buffers {
                deinterleave_command_buffer.commit();
                prepare_deinterleave_rct_command_buffer = Some(deinterleave_command_buffer);
            }

            let mut active_planes = Vec::with_capacity(usize::from(job.components));
            if job.num_decomposition_levels == 0 {
                active_planes.extend(
                    plane_buffers
                        .iter()
                        .take(usize::from(job.components))
                        .cloned(),
                );
            } else if split_prepare_command_buffers {
                let component_count = usize::from(job.components);
                if component_count > 1 {
                    let (
                        mut component_active_planes,
                        mut vertical_command_buffers,
                        mut horizontal_command_buffers,
                    ) = dispatch_forward_dwt53_components_split_profile(
                        runtime,
                        &plane_buffers,
                        &scratch_buffers,
                        job.output_width,
                        job.output_height,
                        job.num_decomposition_levels,
                        component_count,
                    );
                    active_planes.append(&mut component_active_planes);
                    prepare_dwt53_vertical_command_buffers.append(&mut vertical_command_buffers);
                    prepare_dwt53_horizontal_command_buffers
                        .append(&mut horizontal_command_buffers);
                } else {
                    for component in 0..component_count {
                        let (
                            active_plane,
                            mut vertical_command_buffers,
                            mut horizontal_command_buffers,
                        ) = dispatch_forward_dwt53_on_buffers_split_profile(
                            runtime,
                            &plane_buffers[component],
                            &scratch_buffers[component],
                            job.output_width,
                            job.output_height,
                            job.num_decomposition_levels,
                        );
                        active_planes.push(active_plane);
                        prepare_dwt53_vertical_command_buffers
                            .append(&mut vertical_command_buffers);
                        prepare_dwt53_horizontal_command_buffers
                            .append(&mut horizontal_command_buffers);
                    }
                }
            } else {
                let dwt_command_buffer_ref = shared_command_buffer
                    .as_ref()
                    .expect("shared coefficient prep command buffer exists");
                let component_count = usize::from(job.components);
                if component_count > 1 {
                    active_planes = dispatch_forward_dwt53_components_on_buffers(
                        runtime,
                        dwt_command_buffer_ref,
                        &plane_buffers,
                        &scratch_buffers,
                        job.output_width,
                        job.output_height,
                        job.num_decomposition_levels,
                        component_count,
                    );
                } else {
                    for component in 0..component_count {
                        active_planes.push(dispatch_forward_dwt53_on_buffers(
                            runtime,
                            dwt_command_buffer_ref,
                            &plane_buffers[component],
                            &scratch_buffers[component],
                            job.output_width,
                            job.output_height,
                            job.num_decomposition_levels,
                        ));
                    }
                }
            }
            while active_planes.len() < 3 {
                active_planes.push(active_planes[0].clone());
            }

            let coefficient_word_offset = coefficient_byte_offset
                .checked_div(size_of::<i32>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal resident batch coefficient offset division failed"
                        .to_string(),
                })?;
            let coefficient_word_offset_u32 =
                u32::try_from(coefficient_word_offset).map_err(|_| Error::MetalKernel {
                    message: format!(
                        "J2K Metal resident batch coefficient offset exceeds u32 at tile {}",
                        item.tile_index
                    ),
                })?;
            let coefficient_jobs = item
                .code_blocks
                .iter()
                .map(|block| {
                    let coefficient_offset = block
                        .coefficient_offset
                        .checked_add(coefficient_word_offset_u32)
                        .ok_or_else(|| Error::MetalKernel {
                            message: format!(
                                "J2K Metal resident batch coefficient offset overflow at tile {}",
                                item.tile_index
                            ),
                        })?;
                    Ok(J2kLosslessCoefficientJob {
                        coefficient_offset,
                        component: block.component,
                        subband_x: block.subband_x,
                        subband_y: block.subband_y,
                        block_x: block.block_x,
                        block_y: block.block_y,
                        block_width: block.width,
                        block_height: block.height,
                        full_width: job.output_width,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let extract_command_buffer = if split_prepare_command_buffers {
                new_resident_encode_command_buffer(runtime, "j2k coefficient prep extract")
            } else {
                shared_command_buffer
                    .as_ref()
                    .expect("shared coefficient prep command buffer exists")
                    .clone()
            };
            let coefficient_job_buffer = dispatch_lossless_extract_coefficients(
                runtime,
                &extract_command_buffer,
                &active_planes,
                &coefficient_buffer,
                &coefficient_jobs,
                job.output_width,
            )
            .map_err(|err| Error::MetalKernel {
                message: format!(
                    "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                    item.tile_index
                ),
            })?;
            let prepare_command_buffer = extract_command_buffer.clone();
            if split_prepare_command_buffers {
                extract_command_buffer.commit();
                prepare_coefficient_extract_command_buffer = Some(extract_command_buffer);
            }

            prepared.push(J2kPreparedLosslessDeviceCodeBlocks {
                coefficient_buffer: coefficient_buffer.clone(),
                coefficient_byte_offset,
                coefficient_byte_len: item_sizes.coefficient_bytes,
                coefficient_buffer_is_batch_shared: true,
                code_blocks: item.code_blocks,
                recyclable_private_buffers,
                _prepare_command_buffer: prepare_command_buffer,
                _prepare_deinterleave_rct_command_buffer: prepare_deinterleave_rct_command_buffer,
                _prepare_dwt53_command_buffer: prepare_dwt53_command_buffer,
                _prepare_dwt53_vertical_command_buffers: prepare_dwt53_vertical_command_buffers,
                _prepare_dwt53_horizontal_command_buffers: prepare_dwt53_horizontal_command_buffers,
                _prepare_coefficient_extract_command_buffer:
                    prepare_coefficient_extract_command_buffer,
                _deinterleave_status_buffer: status_buffer,
                _plane_buffers: plane_buffers,
                _scratch_buffers: scratch_buffers,
                _coefficient_job_buffer: coefficient_job_buffer,
            });
        }

        if let Some(command_buffer) = shared_command_buffer {
            command_buffer.commit();
        }
        Ok(prepared)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_forward_rct(
    plane0: &mut [f32],
    plane1: &mut [f32],
    plane2: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::MetalKernel {
                message: "J2K Metal forward RCT plane lengths must match".to_string(),
            });
        }

        let params = J2kForwardRctParams {
            _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal forward RCT plane length exceeds u32".to_string(),
            })?,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let plane0_buffer = borrow_mut_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = borrow_mut_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = borrow_mut_slice_buffer(&runtime.device, plane2);
        let status = J2kMctStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kMctStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.forward_rct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kForwardRctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .forward_rct
            .thread_execution_width()
            .max(1)
            .min(len as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let status = unsafe { status_buffer.contents().cast::<J2kMctStatus>().read() };
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_forward_ict(
    plane0: &mut [f32],
    plane1: &mut [f32],
    plane2: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::UnsupportedMetalRequest {
                reason: "J2K Metal forward ICT plane lengths must match",
            });
        }

        let params = J2kForwardIctParams {
            _len: u32::try_from(len).map_err(|_| Error::UnsupportedMetalRequest {
                reason: "J2K Metal forward ICT plane length exceeds u32",
            })?,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let plane0_buffer = borrow_mut_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = borrow_mut_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = borrow_mut_slice_buffer(&runtime.device, plane2);
        let status = J2kMctStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kMctStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.forward_ict);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kForwardIctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .forward_ict
            .thread_execution_width()
            .max(1)
            .min(len as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let status = unsafe { status_buffer.contents().cast::<J2kMctStatus>().read() };
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        Ok(())
    })
}

#[cfg(target_os = "macos")]
fn validate_encode_quantize_subband_job(job: J2kQuantizeSubbandJob<'_>) -> Result<(), Error> {
    if job.step_exponent > 31 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode quantize_subband supports step exponents <= 31",
        });
    }
    if job.step_mantissa > 2047 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode quantize_subband supports step mantissas <= 2047",
        });
    }
    if job.range_bits == 0 || job.range_bits > 31 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode quantize_subband supports range bits 1-31",
        });
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_quantize_subband(job: J2kQuantizeSubbandJob<'_>) -> Result<Vec<i32>, Error> {
    validate_encode_quantize_subband_job(job)?;
    let len = job.coefficients.len();
    if len == 0 {
        return Ok(Vec::new());
    }
    let len_u32 = u32::try_from(len).map_err(|_| Error::UnsupportedMetalRequest {
        reason: "J2K Metal encode quantize_subband coefficient count exceeds u32",
    })?;
    let output_bytes = len
        .checked_mul(size_of::<i32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal encode quantize_subband output length overflow".to_string(),
        })?;

    with_runtime(|runtime| {
        let input_buffer = copied_slice_buffer(&runtime.device, job.coefficients);
        let output_buffer = runtime
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let params = J2kQuantizeSubbandParams {
            _len: len_u32,
            _step_exponent: u32::from(job.step_exponent),
            _step_mantissa: u32::from(job.step_mantissa),
            _range_bits: u32::from(job.range_bits),
            _reversible: u32::from(job.reversible),
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "j2k encode-stage quantize_subband");
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "J2K encode-stage quantize_subband");
        encoder.set_compute_pipeline_state(&runtime.quantize_subband);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kQuantizeSubbandParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_1d_pipeline(encoder, &runtime.quantize_subband, u64::from(len_u32));
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let coefficients =
            unsafe { core::slice::from_raw_parts(output_buffer.contents().cast::<i32>(), len) };
        Ok(coefficients.to_vec())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_inverse_mct(job: J2kInverseMctJob<'_>) -> Result<Vec<Buffer>, Error> {
    let J2kInverseMctJob {
        transform,
        plane0,
        plane1,
        plane2,
        addend0,
        addend1,
        addend2,
    } = job;
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(Vec::new());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::MetalKernel {
                message: "J2K Metal inverse MCT plane lengths must match".to_string(),
            });
        }

        let transform = match transform {
            J2kWaveletTransform::Reversible53 => 0,
            J2kWaveletTransform::Irreversible97 => 1,
        };
        let params = J2kInverseMctParams {
            _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal inverse MCT plane length exceeds u32".to_string(),
            })?,
            _transform: transform,
            _addend0: addend0,
            _addend1: addend1,
            _addend2: addend2,
        };
        let plane0_buffer = copied_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = copied_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = copied_slice_buffer(&runtime.device, plane2);
        let status = J2kMctStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kMctStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.inverse_mct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kInverseMctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .inverse_mct
            .thread_execution_width()
            .max(1)
            .min(len as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let status = unsafe { status_buffer.contents().cast::<J2kMctStatus>().read() };
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        let plane0_host =
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            unsafe { core::slice::from_raw_parts(plane0_buffer.contents().cast::<f32>(), len) };
        let plane1_host =
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            unsafe { core::slice::from_raw_parts(plane1_buffer.contents().cast::<f32>(), len) };
        let plane2_host =
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            unsafe { core::slice::from_raw_parts(plane2_buffer.contents().cast::<f32>(), len) };
        for (dst, sample) in plane0.iter_mut().zip(plane0_host.iter().copied()) {
            *dst = sample - addend0;
        }
        for (dst, sample) in plane1.iter_mut().zip(plane1_host.iter().copied()) {
            *dst = sample - addend1;
        }
        for (dst, sample) in plane2.iter_mut().zip(plane2_host.iter().copied()) {
            *dst = sample - addend2;
        }
        Ok(vec![plane0_buffer, plane1_buffer, plane2_buffer])
    })
}

#[cfg(target_os = "macos")]
fn dispatch_inverse_mct_buffers_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    len: usize,
    transform: J2kWaveletTransform,
    addends: [f32; 3],
) -> Result<DirectStatusCheck, Error> {
    if len == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect color MCT cannot run on an empty plane".to_string(),
        });
    }

    let transform = match transform {
        J2kWaveletTransform::Reversible53 => 0,
        J2kWaveletTransform::Irreversible97 => 1,
    };
    let params = J2kInverseMctParams {
        _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect color MCT plane length exceeds u32".to_string(),
        })?,
        _transform: transform,
        _addend0: addends[0],
        _addend1: addends[1],
        _addend2: addends[2],
    };
    let status = J2kMctStatus::default();
    let status_buffer = runtime.device.new_buffer_with_data(
        (&raw const status).cast(),
        size_of::<J2kMctStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.inverse_mct);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kInverseMctParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(4, Some(&status_buffer), 0);
    let width = runtime
        .inverse_mct
        .thread_execution_width()
        .max(1)
        .min(len as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Mct(status_buffer))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_store_component_and_capture(
    job: J2kStoreComponentJob<'_>,
) -> Result<Buffer, Error> {
    let J2kStoreComponentJob {
        input,
        input_width,
        source_x,
        source_y,
        copy_width,
        copy_height,
        output,
        output_width,
        output_x,
        output_y,
        addend,
    } = job;
    with_runtime(|runtime| {
        if copy_width == 0 || copy_height == 0 {
            return Ok(wrap_f32_output_buffer(&runtime.device, output));
        }

        let required_input_height =
            source_y
                .checked_add(copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal store source height overflow".to_string(),
                })?;
        let required_output_height =
            output_y
                .checked_add(copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal store destination height overflow".to_string(),
                })?;
        if source_x
            .checked_add(copy_width)
            .is_none_or(|end| end > input_width)
            || output_x
                .checked_add(copy_width)
                .is_none_or(|end| end > output_width)
        {
            return Err(Error::MetalKernel {
                message: "J2K Metal store copy rectangle exceeds row bounds".to_string(),
            });
        }
        if input.len()
            < input_width as usize
                * usize::try_from(required_input_height).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal store source height exceeds usize".to_string(),
                })?
            || output.len()
                < output_width as usize
                    * usize::try_from(required_output_height).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal store destination height exceeds usize".to_string(),
                    })?
        {
            return Err(Error::MetalKernel {
                message: "J2K Metal store buffers are smaller than required".to_string(),
            });
        }

        let params = J2kStoreParams {
            input_width,
            source_x,
            source_y,
            copy_width,
            copy_height,
            output_width,
            output_x,
            output_y,
            addend,
        };
        let input_buffer = borrow_slice_buffer(&runtime.device, input);
        let output_buffer = wrap_f32_output_buffer(&runtime.device, output);
        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.store_component);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kStoreParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(encoder, &runtime.store_component, (copy_width, copy_height));
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;
        Ok(output_buffer)
    })
}

#[cfg(target_os = "macos")]
fn dispatch_store_component_buffer_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    input_offset_bytes: usize,
    output: &Buffer,
    output_offset_bytes: usize,
    params: J2kStoreParams,
) {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid component store");
    dispatch_store_component_buffer_in_encoder_with_offsets(
        runtime,
        encoder,
        input,
        input_offset_bytes,
        output,
        output_offset_bytes,
        params,
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn dispatch_store_component_buffer_in_encoder_with_offsets(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    input: &Buffer,
    input_offset_bytes: usize,
    output: &Buffer,
    output_offset_bytes: usize,
    params: J2kStoreParams,
) {
    encoder.set_compute_pipeline_state(&runtime.store_component);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    encoder.set_buffer(1, Some(output), output_offset_bytes as u64);
    encoder.set_bytes(
        2,
        size_of::<J2kStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        &runtime.store_component,
        (params.copy_width, params.copy_height),
    );
}

fn dispatch_store_component_repeated_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    input_offset_bytes: usize,
    output: &Buffer,
    params: J2kRepeatedStoreParams,
) {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated component store");
    encoder.set_compute_pipeline_state(&runtime.store_component_repeated);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        &runtime.store_component_repeated,
        (params.copy_width, params.copy_height, params.batch_count),
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn repeated_gray_store_is_contiguous_full_surface(params: J2kRepeatedGrayStoreParams) -> bool {
    params.source_x == 0
        && params.source_y == 0
        && params.output_x == 0
        && params.output_y == 0
        && params.copy_width == params.input_width
        && params.copy_height == params.input_height
        && params.copy_width == params.output_width
        && params.copy_height == params.output_height
}

#[cfg(target_os = "macos")]
fn encode_repeated_gray_store_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    params: J2kRepeatedGrayStoreParams,
    dims: (u32, u32),
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    let bytes_per_pixel = fmt.bytes_per_pixel();
    let pitch_bytes = dims.0 as usize * bytes_per_pixel;
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal repeated grayscale fused store size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal repeated grayscale fused store total size overflow".to_string(),
        })?;
    let out_buffer = runtime
        .device
        .new_buffer(total_bytes as u64, MTLResourceOptions::StorageModeShared);
    let contiguous_full_surface = repeated_gray_store_is_contiguous_full_surface(params);
    let pipeline = match (fmt, contiguous_full_surface) {
        (PixelFormat::Gray8, true) => &runtime.store_component_repeated_gray_u8_contiguous,
        (PixelFormat::Gray8, false) => &runtime.store_component_repeated_gray_u8,
        (PixelFormat::Gray16, true) => &runtime.store_component_repeated_gray_u16_contiguous,
        (PixelFormat::Gray16, false) => &runtime.store_component_repeated_gray_u16,
        _ => {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal repeated grayscale fused store does not support {fmt:?}"
                ),
            })
        }
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedGrayStoreParams>() as u64,
        (&raw const params).cast(),
    );
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    if contiguous_full_surface {
        let total_samples = u64::from(params.input_width)
            * u64::from(params.input_height)
            * u64::from(params.batch_count);
        encoder.dispatch_threads(
            MTLSize {
                width: total_samples,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: max_threads,
                height: 1,
                depth: 1,
            },
        );
    } else {
        dispatch_3d_pipeline(
            encoder,
            pipeline,
            (params.copy_width, params.copy_height, params.batch_count),
        );
    }
    encoder.end_encoding();

    let mut surfaces = Vec::with_capacity(count);
    for instance_idx in 0..count {
        surfaces.push(Surface::from_metal_buffer_with_offset(
            out_buffer.clone(),
            dims,
            fmt,
            instance_idx * surface_bytes,
        ));
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
fn encode_gray_store_to_surface_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    input: &Buffer,
    input_offset_bytes: usize,
    params: J2kGrayStoreParams,
    dims: (u32, u32),
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pipeline = match fmt {
        PixelFormat::Gray8 => &runtime.store_component_gray_u8,
        PixelFormat::Gray16 => &runtime.store_component_gray_u16,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("J2K Metal grayscale fused store does not support {fmt:?}"),
            })
        }
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kGrayStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, pipeline, (params.copy_width, params.copy_height));

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_reversible53_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let required_len = job.rect.width() as usize * job.rect.height() as usize;
        if output.len() < required_len {
            return Err(Error::MetalKernel {
                message: "J2K Metal IDWT output slice is too small".to_string(),
            });
        }

        let params = J2kIdwtSingleDecompositionParams {
            x0: job.rect.x0,
            y0: job.rect.y0,
            output_x: 0,
            output_y: 0,
            width: job.rect.width(),
            height: job.rect.height(),
            ll_x: 0,
            ll_y: 0,
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_x: 0,
            hl_y: 0,
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_x: 0,
            lh_y: 0,
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_x: 0,
            hh_y: 0,
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let ll = borrow_slice_buffer(&runtime.device, job.ll.coefficients);
        let hl = borrow_slice_buffer(&runtime.device, job.hl.coefficients);
        let lh = borrow_slice_buffer(&runtime.device, job.lh.coefficients);
        let hh = borrow_slice_buffer(&runtime.device, job.hh.coefficients);
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let command_buffer = runtime.queue.new_command_buffer();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
        encoder.set_buffer(0, Some(&ll), 0);
        encoder.set_buffer(1, Some(&hl), 0);
        encoder.set_buffer(2, Some(&lh), 0);
        encoder.set_buffer(3, Some(&hh), 0);
        encoder.set_buffer(4, Some(&decoded), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(
            encoder,
            &runtime.idwt_interleave,
            (params.width, params.height),
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes(
            1,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let horizontal_width = runtime
            .idwt_reversible53_horizontal
            .thread_execution_width()
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.height),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: horizontal_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes(
            1,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let vertical_width = runtime
            .idwt_reversible53_vertical
            .thread_execution_width()
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: vertical_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid reversible53 IDWT");
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
        runtime,
        encoder,
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
        params,
        decoded,
        decoded_offset,
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) {
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        &runtime.idwt_interleave,
        (params.width, params.height),
    );

    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let horizontal_width = runtime
        .idwt_reversible53_horizontal
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.height),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: horizontal_width,
            height: 1,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let vertical_width = runtime
        .idwt_reversible53_vertical
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: vertical_width,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffers: DirectIdwtCommandBuffers<'_>,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kRepeatedIdwtSingleDecompositionParams,
    decoded: &Buffer,
) {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = command_buffers.interleave.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated IDWT interleave");
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave_batched);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        &runtime.idwt_interleave_batched,
        (params.width, params.height, params.batch_count),
    );
    encoder.end_encoding();

    let encoder = command_buffers.horizontal.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated IDWT horizontal");
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let horizontal_width = runtime
        .idwt_reversible53_horizontal_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.height),
            height: u64::from(params.batch_count),
            depth: 1,
        },
        MTLSize {
            width: horizontal_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffers.vertical.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated IDWT vertical");
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let vertical_width = runtime
        .idwt_reversible53_vertical_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: u64::from(params.batch_count),
            depth: 1,
        },
        MTLSize {
            width: vertical_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_irreversible97_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let required_len = job.rect.width() as usize * job.rect.height() as usize;
        if output.len() < required_len {
            return Err(Error::MetalKernel {
                message: "J2K Metal IDWT output slice is too small".to_string(),
            });
        }

        let params = J2kIdwtSingleDecompositionParams {
            x0: job.rect.x0,
            y0: job.rect.y0,
            output_x: 0,
            output_y: 0,
            width: job.rect.width(),
            height: job.rect.height(),
            ll_x: 0,
            ll_y: 0,
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_x: 0,
            hl_y: 0,
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_x: 0,
            lh_y: 0,
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_x: 0,
            hh_y: 0,
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let ll = borrow_slice_buffer(&runtime.device, job.ll.coefficients);
        let hl = borrow_slice_buffer(&runtime.device, job.hl.coefficients);
        let lh = borrow_slice_buffer(&runtime.device, job.lh.coefficients);
        let hh = borrow_slice_buffer(&runtime.device, job.hh.coefficients);
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kIdwtStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
        encoder.set_buffer(0, Some(&ll), 0);
        encoder.set_buffer(1, Some(&hl), 0);
        encoder.set_buffer(2, Some(&lh), 0);
        encoder.set_buffer(3, Some(&hh), 0);
        encoder.set_buffer(4, Some(&decoded), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(6, Some(&status_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let status = unsafe { status_buffer.contents().cast::<J2kIdwtStatus>().read() };
        if status.code != J2K_IDWT_STATUS_OK {
            return Err(decode_idwt_status_error(status));
        }
        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) -> DirectStatusCheck {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kIdwtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid irreversible97 IDWT");
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
        runtime,
        encoder,
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
        params,
        decoded,
        decoded_offset,
        &status_buffer,
    );
    encoder.end_encoding();

    DirectStatusCheck::Idwt(status_buffer)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) -> DirectStatusCheck {
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kIdwtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
        runtime,
        encoder,
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
        params,
        decoded,
        decoded_offset,
        &status_buffer,
    );

    DirectStatusCheck::Idwt(status_buffer)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
    status_buffer: &Buffer,
) {
    encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(6, Some(status_buffer), 0);
    dispatch_single_thread(encoder);
}

#[cfg(target_os = "macos")]
fn classic_batch_uses_plain_fast_path(
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    jobs.iter().all(|job| {
        if job.style_flags != 0
            || job.width > J2K_CLASSIC_MAX_WIDTH
            || job.height > J2K_CLASSIC_MAX_HEIGHT
        {
            return false;
        }
        let start = job.segment_offset as usize;
        let Some(end) = start.checked_add(job.segment_count as usize) else {
            return false;
        };
        segments.get(start..end).is_some_and(|job_segments| {
            job_segments
                .iter()
                .all(|segment| segment.use_arithmetic != 0)
        })
    })
}

#[cfg(target_os = "macos")]
fn classic_repeated_uses_plain_fast_path(
    count: usize,
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    let _ = (count, jobs, segments);
    // Batch-16 WSI benches are faster with device-state cleanup plus the separate parallel store.
    false
}

#[cfg(target_os = "macos")]
fn classic_batch_is_plain_arithmetic(
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    jobs.iter().all(|job| {
        job.style_flags == 0
            && segments[job.segment_offset as usize
                ..job.segment_offset as usize + job.segment_count as usize]
                .iter()
                .all(|segment| segment.use_arithmetic != 0)
    })
}

#[cfg(target_os = "macos")]
fn dispatch_classic_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, jobs);
    let segments_buffer = borrow_slice_buffer(&runtime.device, segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(jobs, segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_batched
    } else {
        &runtime.classic_cleanup_batched
    };
    let status_buffer = runtime.device.new_buffer(
        (jobs.len().max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&segments_buffer), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(&coefficients_scratch.buffer), 0);
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: jobs.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(jobs.len() as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: jobs.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.end_encoding();
    commit_and_wait_metal(command_buffer)?;

    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let statuses = unsafe {
        core::slice::from_raw_parts(
            status_buffer.contents().cast::<J2kClassicStatus>(),
            jobs.len(),
        )
    };
    let status = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_CLASSIC_STATUS_OK);
    runtime.recycle_private_buffer(coefficients_scratch.bytes, coefficients_scratch.buffer)?;
    if let Some(status) = status {
        return Err(decode_classic_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) -> (DirectStatusCheck, Option<Buffer>) {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_classic_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        use_plain_fast_path,
        segments,
        decoded,
        coefficients_scratch,
        &status_buffer,
    );
    encoder.end_encoding();

    (
        DirectStatusCheck::Classic {
            buffer: status_buffer,
            len: job_count,
        },
        None,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_batched_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) -> (DirectStatusCheck, Option<Buffer>) {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    dispatch_classic_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        use_plain_fast_path,
        segments,
        decoded,
        coefficients_scratch,
        &status_buffer,
    );

    (
        DirectStatusCheck::Classic {
            buffer: status_buffer,
            len: job_count,
        },
        None,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_batched_in_encoder_with_status(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
    status_buffer: &Buffer,
) {
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_batched
    } else {
        &runtime.classic_cleanup_batched
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: job_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(job_count as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: job_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) -> Result<DirectStatusCheck, Error> {
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_repeated_batched
    } else {
        &runtime.classic_cleanup_repeated_batched
    };
    let status_buffer = runtime.device.new_buffer(
        (total_job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let repeated = J2kClassicRepeatedBatchParams {
        job_count: j2k_u32_param(job_count, "classic repeated base job count exceeds u32")?,
        output_plane_len: j2k_u32_param(
            output_plane_len,
            "classic repeated output plane len exceeds u32",
        )?,
        batch_count: j2k_u32_param(
            total_job_count / job_count.max(1),
            "classic repeated batch count exceeds u32",
        )?,
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    encoder.set_bytes(
        6,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: job_count as u64,
                height: u64::from(repeated.batch_count),
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(job_count as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: job_count as u64,
                height: u64::from(repeated.batch_count),
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.end_encoding();

    Ok(DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: total_job_count,
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
    states_scratch: &Buffer,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer = runtime.device.new_buffer(
        (total_job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let repeated = J2kClassicRepeatedBatchParams {
        job_count: j2k_u32_param(job_count, "classic repeated base job count exceeds u32")?,
        output_plane_len: j2k_u32_param(
            output_plane_len,
            "classic repeated output plane len exceeds u32",
        )?,
        batch_count: j2k_u32_param(
            total_job_count / job_count.max(1),
            "classic repeated batch count exceeds u32",
        )?,
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.classic_cleanup_plain_dev_repeated_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    encoder.set_buffer(6, Some(states_scratch), 0);
    encoder.set_bytes(
        7,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    let width = runtime
        .classic_cleanup_plain_dev_repeated_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: total_job_count,
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_store_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    jobs: &Buffer,
    job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) -> Result<(), Error> {
    let repeated = J2kClassicRepeatedBatchParams {
        job_count: j2k_u32_param(job_count, "classic repeated base job count exceeds u32")?,
        output_plane_len: j2k_u32_param(
            output_plane_len,
            "classic repeated output plane len exceeds u32",
        )?,
        batch_count: j2k_u32_param(
            total_job_count / job_count.max(1),
            "classic repeated batch count exceeds u32",
        )?,
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.classic_store_repeated_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_buffer(1, Some(jobs), 0);
    encoder.set_buffer(2, Some(coefficients_scratch), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    sub_bands: &[&PreparedClassicSubBand],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = sub_bands.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.width as usize * first.height as usize;
    encode_distinct_classic_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        sub_bands.iter().map(|sub_band| DistinctClassicBatch {
            coded_data: &sub_band.coded_data,
            jobs: &sub_band.jobs,
            segments: &sub_band.segments,
            output_base: sub_bands
                .iter()
                .position(|candidate| core::ptr::eq(*candidate, *sub_band))
                .expect("sub-band exists")
                * per_instance_len,
        }),
        output,
        scratch_buffers,
    )
}

#[cfg(target_os = "macos")]
fn encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    groups: &[&PreparedClassicSubBandGroup],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = groups.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.total_coefficients;
    encode_distinct_classic_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctClassicBatch {
                coded_data: &group.coded_data,
                jobs: &group.jobs,
                segments: &group.segments,
                output_base: index * per_instance_len,
            }),
        output,
        scratch_buffers,
    )
}

#[cfg(target_os = "macos")]
struct DistinctClassicBatch<'a> {
    coded_data: &'a [u8],
    jobs: &'a [J2kClassicCleanupBatchJob],
    segments: &'a [J2kClassicSegment],
    output_base: usize,
}

#[cfg(target_os = "macos")]
fn encode_distinct_classic_batches_to_buffer_in_command_buffer<'a>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batches: impl IntoIterator<Item = DistinctClassicBatch<'a>>,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let mut coded_data = Vec::new();
    let mut jobs = Vec::new();
    let mut segments = Vec::new();

    for batch in batches {
        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color coded payload exceeds u32".to_string(),
        })?;
        let segment_base = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color segment table exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(batch.coded_data);
        for segment in batch.segments {
            let mut adjusted = *segment;
            adjusted.data_offset =
                adjusted
                    .data_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect distinct color segment offset overflow"
                            .to_string(),
                    })?;
            segments.push(adjusted);
        }
        let output_base = u32::try_from(batch.output_base).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color output offset exceeds u32".to_string(),
        })?;
        for job in batch.jobs {
            let mut adjusted = *job;
            adjusted.coded_offset =
                adjusted
                    .coded_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect distinct color job coded offset overflow"
                            .to_string(),
                    })?;
            adjusted.segment_offset = adjusted
                .segment_offset
                .checked_add(segment_base)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect distinct color job segment offset overflow"
                        .to_string(),
                })?;
            adjusted.output_offset =
                adjusted
                    .output_offset
                    .checked_add(output_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message:
                            "classic J2K MetalDirect distinct color job output offset overflow"
                                .to_string(),
                    })?;
            jobs.push(adjusted);
        }
    }

    if jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = owned_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = owned_slice_buffer(&runtime.device, &jobs);
    let segments_buffer = owned_slice_buffer(&runtime.device, &segments);
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&jobs, &segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        jobs.len(),
        use_plain_fast_path,
        &segments_buffer,
        output,
        &coefficients_scratch.buffer,
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn encode_distinct_ht_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    sub_bands: &[&PreparedHtSubBand],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = sub_bands.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.width as usize * first.height as usize;
    encode_distinct_ht_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        sub_bands
            .iter()
            .enumerate()
            .map(|(index, sub_band)| DistinctHtBatch {
                coded_data: &sub_band.coded_data,
                jobs: &sub_band.jobs,
                output_base: index * per_instance_len,
            }),
        output,
    )
}

#[cfg(target_os = "macos")]
fn encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    groups: &[&PreparedHtSubBandGroup],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = groups.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.total_coefficients;
    encode_distinct_ht_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctHtBatch {
                coded_data: &group.coded_arena.data,
                jobs: &group.jobs,
                output_base: index * per_instance_len,
            }),
        output,
    )
}

#[cfg(target_os = "macos")]
struct DistinctHtBatch<'a> {
    coded_data: &'a [u8],
    jobs: &'a [J2kHtCleanupBatchJob],
    output_base: usize,
}

#[cfg(target_os = "macos")]
fn encode_distinct_ht_batches_to_buffer_in_command_buffer<'a>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batches: impl IntoIterator<Item = DistinctHtBatch<'a>>,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let mut coded_data = Vec::new();
    let mut jobs = Vec::new();

    for batch in batches {
        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect distinct grayscale coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(batch.coded_data);
        let output_base = u32::try_from(batch.output_base).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect distinct grayscale output offset exceeds u32".to_string(),
        })?;
        for job in batch.jobs {
            let mut adjusted = *job;
            adjusted.coded_offset =
                adjusted
                    .coded_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect distinct grayscale job coded offset overflow"
                            .to_string(),
                    })?;
            adjusted.output_offset =
                adjusted
                    .output_offset
                    .checked_add(output_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect distinct grayscale job output offset overflow"
                            .to_string(),
                    })?;
            jobs.push(adjusted);
        }
    }

    if jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = owned_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = owned_slice_buffer(&runtime.device, &jobs);
    let status_check = dispatch_ht_cleanup_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        jobs.len(),
        output,
        ht_batch_output_word_count(&jobs)?,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedClassicSubBand,
    count: usize,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    if job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = job
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect repeated job count overflow".to_string(),
        })?;
    let coded_buffer = job.coded_buffer.clone();
    let jobs_buffer = job.jobs_buffer.clone();
    let segments_buffer = job.segments_buffer.clone();
    let use_plain_fast_path =
        classic_repeated_uses_plain_fast_path(count, &job.jobs, &job.segments)
            && runtime
                .classic_cleanup_plain_repeated_batched
                .max_total_threads_per_threadgroup()
                >= 32;
    let use_plain_dev_path = !use_plain_fast_path
        && count <= 16
        && classic_batch_is_plain_arithmetic(&job.jobs, &job.segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, total_jobs)?;
    let states_scratch = if use_plain_dev_path {
        Some(take_classic_states_scratch_buffer(runtime, total_jobs)?)
    } else {
        None
    };
    let status_check = if use_plain_fast_path {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            true,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )?
    } else if let Some(states_scratch) = states_scratch.as_ref() {
        dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
            &states_scratch.buffer,
        )?
    } else {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            use_plain_fast_path,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )?
    };
    if !use_plain_fast_path {
        dispatch_classic_store_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            output,
            &coefficients_scratch.buffer,
        )?;
    }
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        scratch_buffers.push(states_scratch);
    }
    let retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedClassicSubBandGroup,
    count: usize,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || group.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = group
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect repeated grouped job count overflow".to_string(),
        })?;
    let coded_buffer = group.coded_buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let segments_buffer = group.segments_buffer.clone();
    let use_plain_fast_path =
        classic_repeated_uses_plain_fast_path(count, &group.jobs, &group.segments)
            && runtime
                .classic_cleanup_plain_repeated_batched
                .max_total_threads_per_threadgroup()
                >= 32;
    let use_plain_dev_path = !use_plain_fast_path
        && count <= 16
        && classic_batch_is_plain_arithmetic(&group.jobs, &group.segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, total_jobs)?;
    let states_scratch = if use_plain_dev_path {
        Some(take_classic_states_scratch_buffer(runtime, total_jobs)?)
    } else {
        None
    };
    let status_check = if use_plain_fast_path {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            true,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )?
    } else if let Some(states_scratch) = states_scratch.as_ref() {
        dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
            &states_scratch.buffer,
        )?
    } else {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            use_plain_fast_path,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )?
    };
    if !use_plain_fast_path {
        dispatch_classic_store_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            output,
            &coefficients_scratch.buffer,
        )?;
    }
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        scratch_buffers.push(states_scratch);
    }
    let retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn encode_prepared_classic_sub_band_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    job: &PreparedClassicSubBand,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if job.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(
            runtime,
            encoder,
            output,
            job.width as usize * job.height as usize,
        )?;
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = job.coded_buffer.clone();
    let jobs_buffer = job.jobs_buffer.clone();
    let segments_buffer = job.segments_buffer.clone();
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&job.jobs, &job.segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, job.jobs.len())?;
    if job.zero_fill {
        dispatch_zero_u32_buffer_in_encoder(
            runtime,
            encoder,
            output,
            job.width as usize * job.height as usize,
        )?;
    }
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        job.jobs.len(),
        use_plain_fast_path,
        &segments_buffer,
        output,
        &coefficients_scratch.buffer,
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    group: &PreparedClassicSubBandGroup,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if group.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, group.total_coefficients)?;
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = group.coded_buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let segments_buffer = group.segments_buffer.clone();
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&group.jobs, &group.segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, group.jobs.len())?;
    if group.zero_fill {
        dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, group.total_coefficients)?;
    }
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        group.jobs.len(),
        use_plain_fast_path,
        &segments_buffer,
        output,
        &coefficients_scratch.buffer,
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn required_ht_output_len(job: HtCodeBlockDecodeJob<'_>) -> Result<usize, Error> {
    if job.height == 0 {
        return Ok(0);
    }

    job.output_stride
        .checked_mul(job.height as usize - 1)
        .and_then(|prefix| prefix.checked_add(job.width as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal output size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedHtSubBand,
    count: usize,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = job
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated job count overflow".to_string(),
        })?;
    let coded_buffer = prepared_ht_buffer(job.coded_buffer.as_ref(), "coded")?.clone();
    let jobs_buffer = prepared_ht_buffer(job.jobs_buffer.as_ref(), "jobs")?.clone();
    let status_check = dispatch_ht_cleanup_repeated_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        job.jobs.len(),
        total_jobs,
        job.width as usize * job.height as usize,
        output,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedHtSubBandGroup,
    count: usize,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || group.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = group
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated grouped job count overflow".to_string(),
        })?;
    let coded_buffer = group.coded_arena.buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let status_check = dispatch_ht_cleanup_repeated_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        group.jobs.len(),
        total_jobs,
        group.total_coefficients,
        output,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn encode_prepared_ht_sub_band_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    job: &PreparedHtSubBand,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if job.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(
            runtime,
            encoder,
            output,
            job.width as usize * job.height as usize,
        )?;
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = prepared_ht_buffer(job.coded_buffer.as_ref(), "coded")?.clone();
    let jobs_buffer = prepared_ht_buffer(job.jobs_buffer.as_ref(), "jobs")?.clone();
    let status_check = dispatch_ht_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        job.jobs.len(),
        output,
        job.width as usize * job.height as usize,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    group: &PreparedHtSubBandGroup,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if group.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, group.total_coefficients)?;
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = group.coded_arena.buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let status_check = dispatch_ht_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        group.jobs.len(),
        output,
        group.total_coefficients,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn ht_output_word_count(
    output_offset: u32,
    output_stride: u32,
    width: u32,
    height: u32,
) -> Result<usize, Error> {
    let end = if width == 0 || height == 0 {
        u64::from(output_offset)
    } else {
        u64::from(output_offset)
            .checked_add(u64::from(height - 1) * u64::from(output_stride))
            .and_then(|offset| offset.checked_add(u64::from(width)))
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal output span overflow".to_string(),
            })?
    };
    usize::try_from(end).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal output span exceeds usize".to_string(),
    })
}

#[cfg(target_os = "macos")]
fn ht_batch_output_word_count(jobs: &[J2kHtCleanupBatchJob]) -> Result<usize, Error> {
    let mut word_count = 0usize;
    for job in jobs {
        let job_word_count =
            ht_output_word_count(job.output_offset, job.output_stride, job.width, job.height)?;
        word_count = word_count.max(job_word_count);
    }
    Ok(word_count)
}

#[cfg(target_os = "macos")]
fn dispatch_zero_u32_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    buffer: &Buffer,
    word_count: usize,
) -> Result<(), Error> {
    let word_count = u32::try_from(word_count).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal zero-fill word count exceeds u32".to_string(),
    })?;
    if word_count == 0 {
        return Ok(());
    }

    encoder.set_compute_pipeline_state(&runtime.zero_u32_buffer);
    encoder.set_buffer(0, Some(buffer), 0);
    encoder.set_bytes(1, size_of::<u32>() as u64, (&raw const word_count).cast());
    dispatch_1d_pipeline(encoder, &runtime.zero_u32_buffer, u64::from(word_count));
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_status_error(stage: &str, code: u32, detail: u32) -> Error {
    let kind = match code {
        J2K_ENCODE_STATUS_FAIL => "failure",
        J2K_ENCODE_STATUS_UNSUPPORTED => "unsupported input",
        _ => "unexpected status",
    };
    Error::MetalKernel {
        message: format!("{stage} Metal encode kernel {kind} (detail={detail})"),
    }
}

#[cfg(target_os = "macos")]
fn packet_encode_status_error(status: J2kPacketEncodeStatus) -> Error {
    if status.code == J2K_ENCODE_STATUS_FAIL && status.detail == 7 {
        return Error::MetalKernel {
            message: format!(
                "packetization Metal encode kernel failure (detail=7, tier1_detail={})",
                status.data_len
            ),
        };
    }
    encode_status_error("packetization", status.code, status.detail)
}

fn classic_encode_sub_band_code(sub_band_type: j2k_native::J2kSubBandType) -> u32 {
    match sub_band_type {
        j2k_native::J2kSubBandType::LowLow => 0,
        j2k_native::J2kSubBandType::HighLow => 1,
        j2k_native::J2kSubBandType::LowHigh => 2,
        j2k_native::J2kSubBandType::HighHigh => 3,
    }
}

#[cfg(target_os = "macos")]
fn read_classic_encoded_code_block(
    status: J2kClassicEncodeStatus,
    output: &Buffer,
    output_offset: usize,
    output_capacity: usize,
    segment_buffer: &Buffer,
    segment_offset: usize,
    segment_capacity: usize,
) -> Result<EncodedJ2kCodeBlock, Error> {
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "classic Tier-1",
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: "classic J2K Metal encode length exceeds usize".to_string(),
    })?;
    let payload_skip = usize::try_from(status.reserved0).map_err(|_| Error::MetalKernel {
        message: "classic J2K Metal encode payload skip exceeds usize".to_string(),
    })?;
    let number_of_coding_passes =
        u8::try_from(status.number_of_coding_passes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode pass count exceeds u8".to_string(),
        })?;
    let missing_bit_planes =
        u8::try_from(status.missing_bit_planes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode missing bitplanes exceeds u8".to_string(),
        })?;
    let segment_count = usize::try_from(status.segment_count).map_err(|_| Error::MetalKernel {
        message: "classic J2K Metal encode segment count exceeds usize".to_string(),
    })?;
    if segment_count > segment_capacity {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal encode segment count exceeds buffer".to_string(),
        });
    }
    let raw_segments = if segment_count == 0 {
        &[][..]
    } else {
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        unsafe {
            core::slice::from_raw_parts(
                segment_buffer
                    .contents()
                    .cast::<J2kClassicSegment>()
                    .add(segment_offset),
                segment_count,
            )
        }
    };
    let data = if data_len == 0 {
        Vec::new()
    } else {
        let payload_span =
            data_len
                .checked_add(payload_skip)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode payload span overflow".to_string(),
                })?;
        if payload_span > output_capacity {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode length exceeds output buffer".to_string(),
            });
        }
        let payload_offset =
            output_offset
                .checked_add(payload_skip)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode payload offset overflow".to_string(),
                })?;
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        unsafe {
            core::slice::from_raw_parts(
                output.contents().cast::<u8>().add(payload_offset),
                data_len,
            )
        }
        .to_vec()
    };
    let segments = raw_segments
        .iter()
        .map(|segment| {
            Ok(J2kCodeBlockSegment {
                data_offset: segment.data_offset,
                data_length: segment.data_length,
                start_coding_pass: u8::try_from(segment.start_coding_pass).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment start pass exceeds u8"
                            .to_string(),
                    }
                })?,
                end_coding_pass: u8::try_from(segment.end_coding_pass).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment end pass exceeds u8".to_string(),
                    }
                })?,
                use_arithmetic: segment.use_arithmetic != 0,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(EncodedJ2kCodeBlock {
        data,
        segments,
        number_of_coding_passes,
        missing_bit_planes,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_classic_tier1_code_blocks(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient count overflow".to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient slice is too small".to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient table exceeds u32".to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment table exceeds u32".to_string(),
                })?;
            let style_flags = classic_style_flags(job.style);
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, job.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode output capacity exceeds u32".to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode segment buffer overflow".to_string(),
                })?;
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            (jobs.len() * size_of::<J2kClassicEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let classic_encode_pipeline = classic_encode_code_blocks_pipeline(runtime, &batch_jobs);
        encoder.set_compute_pipeline_state(classic_encode_pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
        dispatch_1d_pipeline(encoder, classic_encode_pipeline, u64::from(job_count));
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let statuses = unsafe {
            core::slice::from_raw_parts(
                status_buffer.contents().cast::<J2kClassicEncodeStatus>(),
                jobs.len(),
            )
        };
        let mut results = Vec::with_capacity(jobs.len());
        for (idx, status) in statuses.iter().copied().enumerate() {
            let batch_job = batch_jobs[idx];
            results.push(read_classic_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output capacity exceeds usize".to_string(),
                })?,
                &segment_buffer,
                usize::try_from(batch_job.segment_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.segment_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
                })?,
            )?);
        }

        Ok(results)
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_gpu_token_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal token-pack coefficient count overflow".to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal token-pack coefficient slice is too small"
                        .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment table exceeds u32".to_string(),
                })?;
            let style_flags = classic_style_flags(job.style);
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, job.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal token-pack output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal token-pack segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment buffer overflow".to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message:
                    "classic J2K Metal token-pack parity helper supports only bypass_u16_32 jobs"
                        .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            (jobs.len() * size_of::<J2kClassicEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal token-pack job count exceeds u32".to_string(),
        })?;
        let command_buffer = runtime.queue.new_command_buffer();
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let token_buffers = dispatch_classic_tier1_token_emit_for_gpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
            &mut recyclable_private_buffers,
        )?;
        debug_assert_eq!(token_buffers.job_count, job_count);
        dispatch_classic_tier1_token_pack_from_gpu_tokens(
            runtime,
            command_buffer,
            &job_buffer,
            &token_buffers,
            &output,
            &status_buffer,
            &segment_buffer,
        );
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let statuses = unsafe {
            core::slice::from_raw_parts(
                status_buffer.contents().cast::<J2kClassicEncodeStatus>(),
                jobs.len(),
            )
        };
        let mut results = Vec::with_capacity(jobs.len());
        for (idx, status) in statuses.iter().copied().enumerate() {
            let batch_job = batch_jobs[idx];
            results.push(read_classic_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output capacity exceeds usize"
                        .to_string(),
                })?,
                &segment_buffer,
                usize::try_from(batch_job.segment_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment offset exceeds usize"
                        .to_string(),
                })?,
                usize::try_from(batch_job.segment_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment capacity exceeds usize"
                        .to_string(),
                })?,
            )?);
        }

        Ok(results)
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test_with_emit_route(
        jobs, false,
    )
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_split_mq_byte_raw_tokens_gpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test_with_emit_route(
        jobs, true,
    )
}

#[cfg(all(test, target_os = "macos"))]
fn encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test_with_emit_route(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    use_mq_byte_emit: bool,
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack coefficient count overflow"
                        .to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message:
                        "classic J2K Metal split GPU token-pack coefficient slice is too small"
                            .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output table exceeds u32"
                        .to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack segment table exceeds u32"
                        .to_string(),
                })?;
            let style_flags = classic_style_flags(job.style);
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, job.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "classic J2K Metal split GPU token-pack output capacity exceeds u32"
                                .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "classic J2K Metal split GPU token-pack segment capacity exceeds u32"
                                .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output buffer overflow"
                        .to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack segment buffer overflow"
                        .to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message:
                    "classic J2K Metal split GPU token-pack helper supports only bypass_u16_32 jobs"
                        .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            (jobs.len() * size_of::<J2kClassicEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = runtime.queue.new_command_buffer();
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let split_buffers = dispatch_classic_tier1_split_token_emit_for_gpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
            &mut recyclable_private_buffers,
            use_mq_byte_emit,
        )?;
        dispatch_classic_tier1_split_token_pack_from_gpu_tokens(
            runtime,
            command_buffer,
            &job_buffer,
            &split_buffers,
            &output,
            &status_buffer,
            &segment_buffer,
        );
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let statuses = unsafe {
            core::slice::from_raw_parts(
                status_buffer.contents().cast::<J2kClassicEncodeStatus>(),
                jobs.len(),
            )
        };
        let mut results = Vec::with_capacity(jobs.len());
        for (idx, status) in statuses.iter().copied().enumerate() {
            let batch_job = batch_jobs[idx];
            results.push(read_classic_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output offset exceeds usize"
                        .to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output capacity exceeds usize"
                        .to_string(),
                })?,
                &segment_buffer,
                usize::try_from(batch_job.segment_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack segment offset exceeds usize"
                        .to_string(),
                })?,
                usize::try_from(batch_job.segment_capacity).map_err(|_| Error::MetalKernel {
                    message:
                        "classic J2K Metal split GPU token-pack segment capacity exceeds usize"
                            .to_string(),
                })?,
            )?);
        }

        Ok(results)
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_ordered_tokens_cpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token coefficient count overflow"
                        .to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal ordered-token coefficient slice is too small"
                        .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment table exceeds u32"
                        .to_string(),
                })?;
            let style_flags = classic_style_flags(job.style);
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, job.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal ordered-token output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal ordered-token segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment buffer overflow".to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal ordered-token helper supports only bypass_u16_32 jobs"
                    .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let command_buffer = runtime.queue.new_command_buffer();
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let token_buffers = dispatch_classic_tier1_token_emit_for_gpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
            &mut recyclable_private_buffers,
        )?;
        let job_count =
            usize::try_from(token_buffers.job_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal ordered-token job count exceeds usize".to_string(),
            })?;
        let token_stride_bytes =
            usize::try_from(token_buffers.token_stride_bytes).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal ordered-token byte stride exceeds usize".to_string(),
            })?;
        let token_segment_stride =
            usize::try_from(token_buffers.token_segment_stride).map_err(|_| {
                Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment stride exceeds usize"
                        .to_string(),
                }
            })?;
        let counter_byte_len = job_count
            .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K Metal ordered-token counter readback overflow".to_string(),
            })?;
        let token_byte_len =
            job_count
                .checked_mul(token_stride_bytes)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token byte readback overflow".to_string(),
                })?;
        let token_segment_byte_len = job_count
            .checked_mul(token_segment_stride)
            .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K Metal ordered-token segment readback overflow".to_string(),
            })?;
        let counter_readback = runtime.device.new_buffer(
            counter_byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let token_readback = runtime.device.new_buffer(
            token_byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let token_segment_readback = runtime.device.new_buffer(
            token_segment_byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &token_buffers.counter_buffer,
            0,
            &counter_readback,
            0,
            counter_byte_len as u64,
        );
        blit.copy_from_buffer(
            &token_buffers.token_buffer,
            0,
            &token_readback,
            0,
            token_byte_len as u64,
        );
        blit.copy_from_buffer(
            &token_buffers.segment_buffer,
            0,
            &token_segment_readback,
            0,
            token_segment_byte_len as u64,
        );
        blit.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let counters = unsafe {
            core::slice::from_raw_parts(
                counter_readback
                    .contents()
                    .cast::<J2kClassicTier1SymbolPlanCounters>(),
                job_count,
            )
        };
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let token_bytes = unsafe {
            core::slice::from_raw_parts(token_readback.contents().cast::<u8>(), token_byte_len)
        };
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let token_segments = unsafe {
            core::slice::from_raw_parts(
                token_segment_readback
                    .contents()
                    .cast::<J2kClassicTier1TokenSegment>(),
                job_count.saturating_mul(token_segment_stride),
            )
        };

        let mut results = Vec::with_capacity(job_count);
        for (block_idx, counter) in counters.iter().enumerate() {
            if counter.code != J2K_ENCODE_STATUS_OK {
                return Err(encode_status_error(
                    "classic Tier-1 ordered-token emit",
                    counter.code,
                    counter.detail,
                ));
            }
            let segment_count =
                usize::try_from(counter.segment_count).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment count exceeds usize"
                        .to_string(),
                })?;
            if segment_count > token_segment_stride {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment count exceeds capacity"
                        .to_string(),
                });
            }
            let token_start =
                block_idx
                    .checked_mul(token_stride_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token byte offset overflow".to_string(),
                    })?;
            let segment_start =
                block_idx
                    .checked_mul(token_segment_stride)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token segment offset overflow"
                            .to_string(),
                    })?;
            let mut native_segments = Vec::with_capacity(segment_count);
            for segment in &token_segments[segment_start..segment_start + segment_count] {
                let start_coding_pass =
                    u8::try_from(segment.pass_range & 0xFFFF).map_err(|_| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token start pass exceeds u8"
                            .to_string(),
                    })?;
                let end_coding_pass =
                    u8::try_from(segment.pass_range >> 16).map_err(|_| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token end pass exceeds u8".to_string(),
                    })?;
                native_segments.push(J2kTier1TokenSegment {
                    token_bit_offset: segment.token_bit_offset,
                    token_bit_count: segment.token_bit_count,
                    start_coding_pass,
                    end_coding_pass,
                    use_arithmetic: (segment.flags & 1) != 0,
                });
            }
            let packed = pack_j2k_code_block_scalar_from_tier1_tokens(
                &token_bytes[token_start..token_start + token_stride_bytes],
                &native_segments,
                u8::try_from(counter.coding_passes).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token coding-pass count exceeds u8"
                        .to_string(),
                })?,
                u8::try_from(counter.missing_bit_planes).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token missing bitplanes exceed u8"
                        .to_string(),
                })?,
            )
            .map_err(|message| Error::MetalKernel {
                message: format!("classic J2K Metal ordered-token CPU pack failed: {message}"),
            })?;
            results.push(packed);
        }

        Ok(results)
    })
}

#[cfg(all(test, target_os = "macos"))]
#[derive(Default)]
struct ClassicTier1MsbBitWriter {
    bytes: Vec<u8>,
    current_byte: u8,
    bits_in_current: u8,
    bit_count: usize,
}

#[cfg(all(test, target_os = "macos"))]
impl ClassicTier1MsbBitWriter {
    fn write_bit(&mut self, bit: u8) {
        self.current_byte = (self.current_byte << 1) | (bit & 1);
        self.bits_in_current += 1;
        self.bit_count += 1;
        if self.bits_in_current == 8 {
            self.bytes.push(self.current_byte);
            self.current_byte = 0;
            self.bits_in_current = 0;
        }
    }

    fn bit_count_u32(&self) -> Result<u32, Error> {
        u32::try_from(self.bit_count).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal split-token combined bit offset exceeds u32".to_string(),
        })
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits_in_current != 0 {
            self.bytes
                .push(self.current_byte << (8 - self.bits_in_current));
        }
        self.bytes
    }
}

#[cfg(all(test, target_os = "macos"))]
fn classic_tier1_split_token_bit(source: &[u8], bit_offset: usize) -> Result<u8, Error> {
    if bit_offset >= source.len().saturating_mul(8) {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal split-token bit offset exceeds stream".to_string(),
        });
    }
    let byte = source[bit_offset / 8];
    let shift = 7 - (bit_offset % 8);
    Ok((byte >> shift) & 1)
}

#[cfg(all(test, target_os = "macos"))]
fn classic_tier1_append_split_token_bits(
    writer: &mut ClassicTier1MsbBitWriter,
    source: &[u8],
    bit_offset: usize,
    bit_count: usize,
) -> Result<(), Error> {
    let end = bit_offset
        .checked_add(bit_count)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal split-token bit range overflow".to_string(),
        })?;
    if end > source.len().saturating_mul(8) {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal split-token bit range exceeds stream".to_string(),
        });
    }
    for bit_idx in 0..bit_count {
        writer.write_bit(classic_tier1_split_token_bit(source, bit_offset + bit_idx)?);
    }
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
fn pack_classic_split_mq_raw_tokens_for_test(
    mq_token_bytes: &[u8],
    raw_token_bytes: &[u8],
    split_segments: &[J2kClassicTier1TokenSegment],
    counter: J2kClassicTier1SymbolPlanCounters,
) -> Result<EncodedJ2kCodeBlock, Error> {
    if counter.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "classic Tier-1 split-token emit",
            counter.code,
            counter.detail,
        ));
    }

    let mut combined = ClassicTier1MsbBitWriter::default();
    let mut native_segments = Vec::with_capacity(split_segments.len());
    for segment in split_segments {
        if (segment.flags & !1) != 0 {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal split-token segment has unsupported flags".to_string(),
            });
        }
        let use_arithmetic = (segment.flags & 1) != 0;
        let source = if use_arithmetic {
            mq_token_bytes
        } else {
            raw_token_bytes
        };
        let source_bit_offset =
            usize::try_from(segment.token_bit_offset).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token bit offset exceeds usize".to_string(),
            })?;
        let source_bit_count =
            usize::try_from(segment.token_bit_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token bit count exceeds usize".to_string(),
            })?;
        let combined_bit_offset = combined.bit_count_u32()?;
        classic_tier1_append_split_token_bits(
            &mut combined,
            source,
            source_bit_offset,
            source_bit_count,
        )?;
        let start_coding_pass =
            u8::try_from(segment.pass_range & 0xFFFF).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token start pass exceeds u8".to_string(),
            })?;
        let end_coding_pass =
            u8::try_from(segment.pass_range >> 16).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token end pass exceeds u8".to_string(),
            })?;
        native_segments.push(J2kTier1TokenSegment {
            token_bit_offset: combined_bit_offset,
            token_bit_count: segment.token_bit_count,
            start_coding_pass,
            end_coding_pass,
            use_arithmetic,
        });
    }

    pack_j2k_code_block_scalar_from_tier1_tokens(
        &combined.finish(),
        &native_segments,
        u8::try_from(counter.coding_passes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal split-token coding-pass count exceeds u8".to_string(),
        })?,
        u8::try_from(counter.missing_bit_planes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal split-token missing bitplanes exceed u8".to_string(),
        })?,
    )
    .map_err(|message| Error::MetalKernel {
        message: format!("classic J2K Metal split-token CPU pack failed: {message}"),
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_cpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split-token coefficient count overflow".to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal split-token coefficient slice is too small"
                        .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token segment table exceeds u32".to_string(),
                })?;
            let style_flags = classic_style_flags(job.style);
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, job.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal split-token output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal split-token segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split-token output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split-token segment buffer overflow".to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal split-token helper supports only bypass_u16_32 jobs"
                    .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let command_buffer = runtime.queue.new_command_buffer();
        let split_buffers = dispatch_classic_tier1_split_token_emit_for_cpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
        )?;
        commit_and_wait_metal(command_buffer)?;

        let job_count =
            usize::try_from(split_buffers.job_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token job count exceeds usize".to_string(),
            })?;
        let mq_token_stride_bytes =
            usize::try_from(split_buffers.mq_token_stride_bytes).map_err(|_| {
                Error::MetalKernel {
                    message: "classic J2K Metal split-token MQ byte stride exceeds usize"
                        .to_string(),
                }
            })?;
        let raw_token_stride_bytes = usize::try_from(split_buffers.raw_token_stride_bytes)
            .map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token raw byte stride exceeds usize".to_string(),
            })?;
        let token_segment_stride =
            usize::try_from(split_buffers.token_segment_stride).map_err(|_| {
                Error::MetalKernel {
                    message: "classic J2K Metal split-token segment stride exceeds usize"
                        .to_string(),
                }
            })?;
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let counters = unsafe {
            core::slice::from_raw_parts(
                split_buffers
                    .counter_buffer
                    .contents()
                    .cast::<J2kClassicTier1SymbolPlanCounters>(),
                job_count,
            )
        };
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let mq_token_bytes = unsafe {
            core::slice::from_raw_parts(
                split_buffers.mq_token_buffer.contents().cast::<u8>(),
                job_count.saturating_mul(mq_token_stride_bytes),
            )
        };
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let raw_token_bytes = unsafe {
            core::slice::from_raw_parts(
                split_buffers.raw_token_buffer.contents().cast::<u8>(),
                job_count.saturating_mul(raw_token_stride_bytes),
            )
        };
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let token_segments = unsafe {
            core::slice::from_raw_parts(
                split_buffers
                    .segment_buffer
                    .contents()
                    .cast::<J2kClassicTier1TokenSegment>(),
                job_count.saturating_mul(token_segment_stride),
            )
        };

        let mut results = Vec::with_capacity(job_count);
        for (block_idx, counter) in counters.iter().copied().enumerate() {
            let segment_count =
                usize::try_from(counter.segment_count).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token segment count exceeds usize"
                        .to_string(),
                })?;
            if segment_count > token_segment_stride {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal split-token segment count exceeds capacity"
                        .to_string(),
                });
            }
            let mq_token_start = block_idx
                .checked_mul(mq_token_stride_bytes)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split-token MQ byte offset overflow".to_string(),
                })?;
            let raw_token_start =
                block_idx
                    .checked_mul(raw_token_stride_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal split-token raw byte offset overflow"
                            .to_string(),
                    })?;
            let segment_start =
                block_idx
                    .checked_mul(token_segment_stride)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal split-token segment offset overflow"
                            .to_string(),
                    })?;
            results.push(pack_classic_split_mq_raw_tokens_for_test(
                &mq_token_bytes[mq_token_start..mq_token_start + mq_token_stride_bytes],
                &raw_token_bytes[raw_token_start..raw_token_start + raw_token_stride_bytes],
                &token_segments[segment_start..segment_start + segment_count],
                counter,
            )?);
        }

        Ok(results)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_classic_tier1_prepared_device_code_blocks_resident(
    session: &crate::MetalBackendSession,
    prepared: J2kPreparedLosslessDeviceCodeBlocks,
) -> Result<J2kResidentLosslessTier1CodeBlocks, Error> {
    let J2kPreparedLosslessDeviceCodeBlocks {
        coefficient_buffer,
        coefficient_byte_offset: _,
        coefficient_byte_len: _,
        coefficient_buffer_is_batch_shared: _,
        code_blocks,
        recyclable_private_buffers: _,
        _prepare_command_buffer: prepare_command_buffer,
        _prepare_deinterleave_rct_command_buffer: _,
        _prepare_dwt53_command_buffer: _,
        _prepare_dwt53_vertical_command_buffers: _,
        _prepare_dwt53_horizontal_command_buffers: _,
        _prepare_coefficient_extract_command_buffer: _,
        _deinterleave_status_buffer: deinterleave_status_buffer,
        _plane_buffers: plane_buffers,
        _scratch_buffers: scratch_buffers,
        _coefficient_job_buffer: coefficient_job_buffer,
    } = prepared;
    with_runtime_for_session(session, |runtime| {
        if code_blocks.is_empty() {
            let output = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let status_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let segment_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let job_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModeShared);
            let command_buffer = runtime.queue.new_command_buffer();
            command_buffer.commit();
            return Ok(J2kResidentLosslessTier1CodeBlocks {
                output_buffer: output,
                status_buffer,
                job_buffer,
                batch_jobs: Vec::new(),
                code_blocks,
                output_capacity_total: 0,
                _segment_buffer: segment_buffer,
                tier1_command_buffer: command_buffer.to_owned(),
                _coefficient_buffer: coefficient_buffer,
                prepare_command_buffer,
                _deinterleave_status_buffer: deinterleave_status_buffer,
                _plane_buffers: plane_buffers,
                _scratch_buffers: scratch_buffers,
                _coefficient_job_buffer: coefficient_job_buffer,
            });
        }
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(code_blocks.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for block in &code_blocks {
            let output_capacity =
                classic_encode_output_capacity(block.width, block.height, block.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal resident encode output table exceeds u32"
                        .to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal resident encode segment table exceeds u32"
                        .to_string(),
                })?;
            let style_flags = 0;
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, block.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset: block.coefficient_offset,
                output_offset,
                segment_offset,
                width: block.width,
                height: block.height,
                sub_band_type: classic_encode_sub_band_code(block.sub_band_type),
                total_bitplanes: u32::from(block.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal resident encode output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal resident encode segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal resident encode output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal resident encode segment buffer overflow"
                        .to_string(),
                })?;
        }

        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer = runtime.device.new_buffer(
            (batch_jobs.len() * size_of::<J2kClassicEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal resident encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let classic_encode_pipeline = classic_encode_code_blocks_pipeline(runtime, &batch_jobs);
        encoder.set_compute_pipeline_state(classic_encode_pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
        dispatch_1d_pipeline(encoder, classic_encode_pipeline, u64::from(job_count));
        encoder.end_encoding();
        command_buffer.commit();

        Ok(J2kResidentLosslessTier1CodeBlocks {
            output_buffer: output,
            status_buffer,
            job_buffer,
            batch_jobs,
            code_blocks,
            output_capacity_total,
            _segment_buffer: segment_buffer,
            tier1_command_buffer: command_buffer.to_owned(),
            _coefficient_buffer: coefficient_buffer,
            prepare_command_buffer,
            _deinterleave_status_buffer: deinterleave_status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_ht_prepared_device_code_blocks_resident(
    session: &crate::MetalBackendSession,
    prepared: J2kPreparedLosslessDeviceCodeBlocks,
) -> Result<J2kResidentLosslessHtCodeBlocks, Error> {
    let J2kPreparedLosslessDeviceCodeBlocks {
        coefficient_buffer,
        coefficient_byte_offset: _,
        coefficient_byte_len: _,
        coefficient_buffer_is_batch_shared: _,
        code_blocks,
        recyclable_private_buffers: _,
        _prepare_command_buffer: prepare_command_buffer,
        _prepare_deinterleave_rct_command_buffer: _,
        _prepare_dwt53_command_buffer: _,
        _prepare_dwt53_vertical_command_buffers: _,
        _prepare_dwt53_horizontal_command_buffers: _,
        _prepare_coefficient_extract_command_buffer: _,
        _deinterleave_status_buffer: deinterleave_status_buffer,
        _plane_buffers: plane_buffers,
        _scratch_buffers: scratch_buffers,
        _coefficient_job_buffer: coefficient_job_buffer,
    } = prepared;
    with_runtime_for_session(session, |runtime| {
        if code_blocks.is_empty() {
            let output = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let status_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let job_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModeShared);
            let command_buffer = runtime.queue.new_command_buffer();
            command_buffer.commit();
            return Ok(J2kResidentLosslessHtCodeBlocks {
                output_buffer: output,
                status_buffer,
                job_buffer,
                batch_jobs: Vec::new(),
                code_blocks,
                output_capacity_total: 0,
                tier1_command_buffer: command_buffer.to_owned(),
                _coefficient_buffer: coefficient_buffer,
                prepare_command_buffer,
                _deinterleave_status_buffer: deinterleave_status_buffer,
                _plane_buffers: plane_buffers,
                _scratch_buffers: scratch_buffers,
                _coefficient_job_buffer: coefficient_job_buffer,
            });
        }

        let mut batch_jobs = Vec::<J2kHtEncodeBatchJob>::with_capacity(code_blocks.len());
        let mut output_capacity_total = 0usize;

        for block in &code_blocks {
            let output_capacity = ht_encode_output_capacity(block.width, block.height)?;
            let output_capacity_u32 =
                u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal resident encode output capacity exceeds u32".to_string(),
                })?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal resident encode output table exceeds u32".to_string(),
                })?;
            batch_jobs.push(J2kHtEncodeBatchJob {
                coefficient_offset: block.coefficient_offset,
                output_offset,
                width: block.width,
                height: block.height,
                total_bitplanes: u32::from(block.total_bitplanes),
                output_capacity: output_capacity_u32,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal resident encode output buffer overflow".to_string(),
                })?;
        }

        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer = runtime.device.new_buffer(
            (batch_jobs.len() * size_of::<J2kHtEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal resident encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "j2k htj2k resident tier1");
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "HTJ2K Tier-1 encode");
        let pipeline = &runtime.ht_encode_code_blocks;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
        encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
        encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
        encoder.set_buffer(6, Some(&status_buffer), 0);
        encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const job_count).cast());
        dispatch_1d_pipeline(encoder, pipeline, u64::from(job_count));
        encoder.end_encoding();
        command_buffer.commit();

        Ok(J2kResidentLosslessHtCodeBlocks {
            output_buffer: output,
            status_buffer,
            job_buffer,
            batch_jobs,
            code_blocks,
            output_capacity_total,
            tier1_command_buffer: command_buffer.to_owned(),
            _coefficient_buffer: coefficient_buffer,
            prepare_command_buffer,
            _deinterleave_status_buffer: deinterleave_status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_classic_tier1_code_block(
    job: J2kTier1CodeBlockEncodeJob<'_>,
) -> Result<EncodedJ2kCodeBlock, Error> {
    with_runtime(|runtime| {
        let expected_coefficients = usize::try_from(job.width)
            .ok()
            .and_then(|w| {
                usize::try_from(job.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K Metal encode coefficient count overflow".to_string(),
            })?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode coefficient slice is too small".to_string(),
            });
        }

        let output_capacity =
            classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let style_flags = classic_style_flags(job.style);
        let segment_capacity = classic_encode_segment_capacity(style_flags, job.total_bitplanes);
        let params = J2kClassicEncodeParams {
            width: job.width,
            height: job.height,
            sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
            total_bitplanes: u32::from(job.total_bitplanes),
            style_flags,
            output_capacity: output_capacity_u32,
            segment_capacity: u32::try_from(segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds u32".to_string(),
            })?,
        };
        let coefficients =
            borrow_slice_buffer(&runtime.device, &job.coefficients[..expected_coefficients]);
        let output = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kClassicEncodeStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let segment_buffer = runtime.device.new_buffer(
            (usize::try_from(params.segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
            })? * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.classic_encode_code_block);
        encoder.set_buffer(0, Some(&coefficients), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kClassicEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let status = unsafe {
            status_buffer
                .contents()
                .cast::<J2kClassicEncodeStatus>()
                .read()
        };
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1",
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode length exceeds usize".to_string(),
        })?;
        let payload_skip = usize::try_from(status.reserved0).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode payload skip exceeds usize".to_string(),
        })?;
        let payload_span =
            data_len
                .checked_add(payload_skip)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode payload span overflow".to_string(),
                })?;
        if payload_span > output_capacity {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode length exceeds output buffer".to_string(),
            });
        }
        let payload_offset = payload_skip;
        let data = if data_len == 0 {
            Vec::new()
        } else {
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            unsafe {
                core::slice::from_raw_parts(
                    output.contents().cast::<u8>().add(payload_offset),
                    data_len,
                )
            }
            .to_vec()
        };
        let number_of_coding_passes =
            u8::try_from(status.number_of_coding_passes).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode pass count exceeds u8".to_string(),
            })?;
        let missing_bit_planes =
            u8::try_from(status.missing_bit_planes).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode missing bitplanes exceeds u8".to_string(),
            })?;
        let segment_count =
            usize::try_from(status.segment_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment count exceeds usize".to_string(),
            })?;
        let segment_capacity =
            usize::try_from(params.segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
            })?;
        if segment_count > segment_capacity {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode segment count exceeds buffer".to_string(),
            });
        }
        let raw_segments = if segment_count == 0 {
            &[][..]
        } else {
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            unsafe {
                core::slice::from_raw_parts(
                    segment_buffer.contents().cast::<J2kClassicSegment>(),
                    segment_count,
                )
            }
        };
        let segments = raw_segments
            .iter()
            .map(|segment| {
                Ok(J2kCodeBlockSegment {
                    data_offset: segment.data_offset,
                    data_length: segment.data_length,
                    start_coding_pass: u8::try_from(segment.start_coding_pass).map_err(|_| {
                        Error::MetalKernel {
                            message: "classic J2K Metal encode segment start pass exceeds u8"
                                .to_string(),
                        }
                    })?,
                    end_coding_pass: u8::try_from(segment.end_coding_pass).map_err(|_| {
                        Error::MetalKernel {
                            message: "classic J2K Metal encode segment end pass exceeds u8"
                                .to_string(),
                        }
                    })?,
                    use_arithmetic: segment.use_arithmetic != 0,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(EncodedJ2kCodeBlock {
            data,
            segments,
            number_of_coding_passes,
            missing_bit_planes,
        })
    })
}

#[cfg(target_os = "macos")]
fn read_ht_encoded_code_block(
    status: J2kHtEncodeStatus,
    output: &Buffer,
    output_offset: usize,
    output_capacity: usize,
) -> Result<EncodedHtJ2kCodeBlock, Error> {
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "HTJ2K cleanup",
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal encode length exceeds usize".to_string(),
    })?;
    if data_len > output_capacity {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal encode length exceeds output buffer".to_string(),
        });
    }
    let data = if data_len == 0 {
        Vec::new()
    } else {
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        unsafe {
            core::slice::from_raw_parts(output.contents().cast::<u8>().add(output_offset), data_len)
        }
        .to_vec()
    };
    Ok(EncodedHtJ2kCodeBlock {
        data,
        cleanup_length: status.data_len,
        refinement_length: 0,
        num_coding_passes: u8::try_from(status.num_coding_passes).map_err(|_| {
            Error::MetalKernel {
                message: "HTJ2K Metal encode pass count exceeds u8".to_string(),
            }
        })?,
        num_zero_bitplanes: u8::try_from(status.num_zero_bitplanes).map_err(|_| {
            Error::MetalKernel {
                message: "HTJ2K Metal encode zero bitplanes exceeds u8".to_string(),
            }
        })?,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn read_resident_ht_tier1_code_blocks_for_cpu_packetization(
    session: &crate::MetalBackendSession,
    tier1: &J2kResidentLosslessHtCodeBlocks,
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    with_runtime_for_session(session, |runtime| {
        if tier1.batch_jobs.is_empty() {
            return Ok(Vec::new());
        }
        let output_bytes = tier1.output_capacity_total.max(1);
        let status_bytes = tier1
            .batch_jobs
            .len()
            .checked_mul(size_of::<J2kHtEncodeStatus>())
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal resident status readback size overflow".to_string(),
            })?;
        let output = runtime
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let status_buffer = runtime
            .device
            .new_buffer(status_bytes as u64, MTLResourceOptions::StorageModeShared);

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "j2k htj2k resident tier1 cpu readback");
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &tier1.output_buffer,
            0,
            &output,
            0,
            tier1.output_capacity_total as u64,
        );
        blit.copy_from_buffer(
            &tier1.status_buffer,
            0,
            &status_buffer,
            0,
            status_bytes as u64,
        );
        blit.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let statuses = unsafe {
            core::slice::from_raw_parts(
                status_buffer.contents().cast::<J2kHtEncodeStatus>(),
                tier1.batch_jobs.len(),
            )
        };
        tier1
            .batch_jobs
            .iter()
            .zip(statuses.iter().copied())
            .map(|(batch_job, status)| {
                read_ht_encoded_code_block(
                    status,
                    &output,
                    usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal resident output offset exceeds usize".to_string(),
                    })?,
                    usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal resident output capacity exceeds usize".to_string(),
                    })?,
                )
            })
            .collect()
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_ht_cleanup_code_blocks(
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    with_runtime(|runtime| encode_ht_cleanup_code_blocks_with_runtime(runtime, jobs))
}

#[cfg(target_os = "macos")]
fn encode_ht_cleanup_code_blocks_with_runtime(
    runtime: &MetalRuntime,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    encode_ht_cleanup_code_blocks_with_runtime_and_statuses(runtime, jobs).map(|blocks| {
        blocks
            .into_iter()
            .map(|(encoded, _status)| encoded)
            .collect()
    })
}

#[cfg(target_os = "macos")]
fn encode_ht_cleanup_code_blocks_with_runtime_and_statuses(
    runtime: &MetalRuntime,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<(EncodedHtJ2kCodeBlock, J2kHtEncodeStatus)>, Error> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    if jobs.iter().any(|job| job.target_coding_passes != 1) {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal cleanup encode supports one coding pass".to_string(),
        });
    }

    let mut coefficients = Vec::<i32>::new();
    let mut batch_jobs = Vec::<J2kHtEncodeBatchJob>::with_capacity(jobs.len());
    let mut output_capacity_total = 0usize;

    for job in jobs {
        let output_capacity = ht_encode_output_capacity(job.width, job.height)?;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let expected_coefficients = usize::try_from(job.width)
            .ok()
            .and_then(|w| {
                usize::try_from(job.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient count overflow".to_string(),
            })?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient slice is too small".to_string(),
            });
        }
        let coefficient_offset =
            u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient table exceeds u32".to_string(),
            })?;
        coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
        let output_offset =
            u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output table exceeds u32".to_string(),
            })?;
        batch_jobs.push(J2kHtEncodeBatchJob {
            coefficient_offset,
            output_offset,
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_capacity: output_capacity_u32,
        });
        output_capacity_total = output_capacity_total
            .checked_add(output_capacity)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode output buffer overflow".to_string(),
            })?;
    }

    let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
    let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
    let output = runtime.device.new_buffer(
        output_capacity_total.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let status_buffer = runtime.device.new_buffer(
        (jobs.len() * size_of::<J2kHtEncodeStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal encode job count exceeds u32".to_string(),
    })?;

    let command_buffer = runtime.queue.new_command_buffer();
    label_command_buffer(command_buffer, "j2k htj2k tier1 batch");
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "HTJ2K Tier-1 encode");
    let pipeline = &runtime.ht_encode_code_blocks;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&coefficient_buffer), 0);
    encoder.set_buffer(1, Some(&output), 0);
    encoder.set_buffer(2, Some(&job_buffer), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
    encoder.set_buffer(6, Some(&status_buffer), 0);
    encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const job_count).cast());
    dispatch_1d_pipeline(encoder, pipeline, u64::from(job_count));
    encoder.end_encoding();
    commit_and_wait_metal(command_buffer)?;

    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let statuses = unsafe {
        core::slice::from_raw_parts(
            status_buffer.contents().cast::<J2kHtEncodeStatus>(),
            jobs.len(),
        )
    };
    let mut results = Vec::with_capacity(jobs.len());
    for (idx, status) in statuses.iter().copied().enumerate() {
        let batch_job = batch_jobs[idx];
        let encoded_block = read_ht_encoded_code_block(
            status,
            &output,
            usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output offset exceeds usize".to_string(),
            })?,
            usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds usize".to_string(),
            })?,
        )?;
        results.push((encoded_block, status));
    }

    Ok(results)
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_ht_cleanup_code_block(
    job: J2kHtCodeBlockEncodeJob<'_>,
) -> Result<EncodedHtJ2kCodeBlock, Error> {
    with_runtime(|runtime| {
        if job.target_coding_passes != 1 {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal cleanup encode supports one coding pass".to_string(),
            });
        }
        let expected_coefficients = usize::try_from(job.width)
            .ok()
            .and_then(|w| {
                usize::try_from(job.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient count overflow".to_string(),
            })?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient slice is too small".to_string(),
            });
        }
        let output_capacity = ht_encode_output_capacity(job.width, job.height)?;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let params = J2kHtEncodeParams {
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_capacity: output_capacity_u32,
        };
        let coefficients =
            borrow_slice_buffer(&runtime.device, &job.coefficients[..expected_coefficients]);
        let output = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kHtEncodeStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.ht_encode_code_block);
        encoder.set_buffer(0, Some(&coefficients), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kHtEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
        encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
        encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
        encoder.set_buffer(6, Some(&status_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let status = unsafe { status_buffer.contents().cast::<J2kHtEncodeStatus>().read() };
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "HTJ2K cleanup",
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal encode length exceeds usize".to_string(),
        })?;
        if data_len > output_capacity {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode length exceeds output buffer".to_string(),
            });
        }
        let data = if data_len == 0 {
            Vec::new()
        } else {
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            unsafe { core::slice::from_raw_parts(output.contents().cast::<u8>(), data_len) }
                .to_vec()
        };
        Ok(EncodedHtJ2kCodeBlock {
            data,
            cleanup_length: status.data_len,
            refinement_length: 0,
            num_coding_passes: u8::try_from(status.num_coding_passes).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Metal encode pass count exceeds u8".to_string(),
                }
            })?,
            num_zero_bitplanes: u8::try_from(status.num_zero_bitplanes).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Metal encode zero bitplanes exceeds u8".to_string(),
                }
            })?,
        })
    })
}

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
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kPacketEncodeStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

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

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let status = unsafe {
            status_buffer
                .contents()
                .cast::<J2kPacketEncodeStatus>()
                .read()
        };
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
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            unsafe { core::slice::from_raw_parts(output_buffer.contents().cast::<u8>(), data_len) }
                .to_vec()
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
            num_components: u32::from(job.num_components),
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
            num_components: u32::from(codestream_job.num_components),
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
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kPacketEncodeStatus>() as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let codestream_status_buffer = runtime.device.new_buffer(
            size_of::<J2kCodestreamAssemblyStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

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
            runtime,
            &command_buffer,
            &tier1_status_buffer,
            J2kResidentTier1StatusKind::HighThroughput,
            0,
            None,
            tier1_jobs.len(),
            size_of::<J2kHtEncodeStatus>(),
            profile_stages,
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
                    let next_label = if profile_classic_tier1_density {
                        "j2k classic resident Tier-1 density profile"
                    } else if profile_classic_tier1_raw_pack {
                        "j2k classic resident Tier-1 raw-pack profile"
                    } else if profile_classic_tier1_arithmetic_pack {
                        "j2k classic resident Tier-1 arithmetic-pack profile"
                    } else if profile_classic_tier1_symbol_plan {
                        "j2k classic resident Tier-1 symbol plan"
                    } else if profile_classic_tier1_token_emit {
                        "j2k classic resident Tier-1 token emit"
                    } else if profile_classic_tier1_split_token_emit {
                        "j2k classic resident Tier-1 split token emit"
                    } else {
                        "j2k classic resident packetization"
                    };
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
                    let next_label = if profile_classic_tier1_density {
                        "j2k classic resident Tier-1 density profile"
                    } else if profile_classic_tier1_raw_pack {
                        "j2k classic resident Tier-1 raw-pack profile"
                    } else if profile_classic_tier1_arithmetic_pack {
                        "j2k classic resident Tier-1 arithmetic-pack profile"
                    } else if profile_classic_tier1_symbol_plan {
                        "j2k classic resident Tier-1 symbol plan"
                    } else if profile_classic_tier1_token_emit {
                        "j2k classic resident Tier-1 token emit"
                    } else if profile_classic_tier1_split_token_emit {
                        "j2k classic resident Tier-1 split token emit"
                    } else {
                        "j2k classic resident packetization"
                    };
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
                    let next_label = if profile_classic_tier1_density {
                        "j2k classic resident Tier-1 density profile"
                    } else if profile_classic_tier1_raw_pack {
                        "j2k classic resident Tier-1 raw-pack profile"
                    } else if profile_classic_tier1_arithmetic_pack {
                        "j2k classic resident Tier-1 arithmetic-pack profile"
                    } else if profile_classic_tier1_symbol_plan {
                        "j2k classic resident Tier-1 symbol plan"
                    } else if profile_classic_tier1_token_emit {
                        "j2k classic resident Tier-1 token emit"
                    } else if profile_classic_tier1_split_token_emit {
                        "j2k classic resident Tier-1 split token emit"
                    } else {
                        "j2k classic resident packetization"
                    };
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
                let next_label = if profile_classic_tier1_raw_pack {
                    "j2k classic resident Tier-1 raw-pack profile"
                } else if profile_classic_tier1_arithmetic_pack {
                    "j2k classic resident Tier-1 arithmetic-pack profile"
                } else if profile_classic_tier1_symbol_plan {
                    "j2k classic resident Tier-1 symbol plan"
                } else if profile_classic_tier1_token_emit {
                    "j2k classic resident Tier-1 token emit"
                } else if profile_classic_tier1_split_token_emit {
                    "j2k classic resident Tier-1 split token emit"
                } else {
                    "j2k classic resident packetization"
                };
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
                let next_label = if profile_classic_tier1_arithmetic_pack {
                    "j2k classic resident Tier-1 arithmetic-pack profile"
                } else if profile_classic_tier1_symbol_plan {
                    "j2k classic resident Tier-1 symbol plan"
                } else if profile_classic_tier1_token_emit {
                    "j2k classic resident Tier-1 token emit"
                } else if profile_classic_tier1_split_token_emit {
                    "j2k classic resident Tier-1 split token emit"
                } else {
                    "j2k classic resident packetization"
                };
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
                let next_label = if profile_classic_tier1_symbol_plan {
                    "j2k classic resident Tier-1 symbol plan"
                } else if profile_classic_tier1_token_emit {
                    "j2k classic resident Tier-1 token emit"
                } else if profile_classic_tier1_split_token_emit {
                    "j2k classic resident Tier-1 split token emit"
                } else {
                    "j2k classic resident packetization"
                };
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
                let next_label = if profile_classic_tier1_pass_plan {
                    "j2k classic resident Tier-1 pass plan"
                } else if profile_classic_tier1_token_emit {
                    "j2k classic resident Tier-1 token emit"
                } else if profile_classic_tier1_split_token_emit {
                    "j2k classic resident Tier-1 split token emit"
                } else {
                    "j2k classic resident packetization"
                };
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
                let next_label = if profile_classic_tier1_token_emit {
                    "j2k classic resident Tier-1 token emit"
                } else if profile_classic_tier1_split_token_emit {
                    "j2k classic resident Tier-1 split token emit"
                } else {
                    "j2k classic resident packetization"
                };
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
                let next_label = if profile_classic_tier1_split_token_emit {
                    "j2k classic resident Tier-1 split token emit"
                } else {
                    "j2k classic resident packetization"
                };
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
        let codestream_status_buffer = runtime.device.new_buffer(
            (assembly_jobs.len() * size_of::<J2kCodestreamAssemblyStatus>()) as u64,
            MTLResourceOptions::StorageModeShared,
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
        let tier1_status_readback = schedule_resident_tier1_status_readback(
            runtime,
            &command_buffer,
            &tier1_status_buffer,
            J2kResidentTier1StatusKind::Classic,
            classic_resident_style_flags,
            Some(&tier1_jobs),
            tier1_jobs.len(),
            size_of::<J2kClassicEncodeStatus>(),
            profile_stages,
        )?;
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

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    params: J2kHtCleanupParams,
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kHtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(
        runtime,
        encoder,
        decoded,
        ht_output_word_count(
            params.output_offset,
            params.output_stride,
            params.width,
            params.height,
        )?,
    )?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kHtCleanupParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    dispatch_single_thread(encoder);
    encoder.end_encoding();
    commit_and_wait_metal(command_buffer)?;

    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let status = unsafe { status_buffer.contents().cast::<J2kHtStatus>().read() };
    if status.code != J2K_HT_STATUS_OK {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, jobs);
    let status_buffer = runtime.device.new_buffer(
        (jobs.len().max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(
        runtime,
        encoder,
        decoded,
        ht_batch_output_word_count(jobs)?,
    )?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_batched);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    let width = runtime
        .ht_cleanup_batched
        .thread_execution_width()
        .max(1)
        .min(jobs.len() as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: jobs.len() as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    commit_and_wait_metal(command_buffer)?;

    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let statuses = unsafe {
        core::slice::from_raw_parts(status_buffer.contents().cast::<J2kHtStatus>(), jobs.len())
    };
    if let Some(status) = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_HT_STATUS_OK)
    {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    decoded_word_count: usize,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, decoded_word_count)?;
    dispatch_ht_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        decoded,
        &status_buffer,
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: job_count,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    decoded_word_count: usize,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, decoded_word_count)?;
    dispatch_ht_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        decoded,
        &status_buffer,
    );

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: job_count,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched_in_encoder_with_status(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    status_buffer: &Buffer,
) {
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(status_buffer), 0);
    let width = runtime
        .ht_cleanup_batched
        .thread_execution_width()
        .max(1)
        .min(job_count as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: job_count as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_ht_cleanup_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    base_job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    decoded: &Buffer,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer = runtime.device.new_buffer(
        (total_job_count.max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let batch_count =
        total_job_count
            .checked_div(base_job_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect repeated base job count is zero".to_string(),
            })?;
    let decoded_word_count =
        output_plane_len
            .checked_mul(batch_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect repeated output span overflow".to_string(),
            })?;
    let repeated = J2kHtRepeatedBatchParams {
        job_count: u32::try_from(base_job_count).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated base job count exceeds u32".to_string(),
        })?,
        output_plane_len: u32::try_from(output_plane_len).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated output plane length exceeds u32".to_string(),
        })?,
        batch_count: u32::try_from(batch_count).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated batch count exceeds u32".to_string(),
        })?,
    };

    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, decoded_word_count)?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_repeated_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kHtRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(5, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(7, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(8, Some(&status_buffer), 0);
    let width = runtime
        .ht_cleanup_repeated_batched
        .thread_execution_width()
        .max(1)
        .min(base_job_count as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: base_job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: total_job_count,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_classic_cleanup_code_block(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = required_classic_output_len(job)?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal output slice is too small".to_string(),
        });
    }

    if job.width == 0 || job.height == 0 {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        let batch_job = J2kClassicCleanupBatchJob {
            coded_offset: 0,
            coded_len: u32::try_from(job.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal coded payload exceeds u32".to_string(),
            })?,
            segment_offset: 0,
            segment_count: u32::try_from(job.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal segment count exceeds u32".to_string(),
            })?,
            width: job.width,
            height: job.height,
            output_stride: u32::try_from(job.output_stride).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal output stride exceeds u32".to_string(),
            })?,
            output_offset: 0,
            missing_msbs: u32::from(job.missing_bit_planes),
            total_bitplanes: u32::from(job.total_bitplanes),
            roi_shift: u32::from(job.roi_shift),
            number_of_coding_passes: u32::from(job.number_of_coding_passes),
            sub_band_type: match job.sub_band_type {
                j2k_native::J2kSubBandType::LowLow => 0,
                j2k_native::J2kSubBandType::HighLow => 1,
                j2k_native::J2kSubBandType::LowHigh => 2,
                j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(job.style),
            strict: u32::from(job.strict),
            dequantization_step: job.dequantization_step,
        };
        let segments: Vec<_> = job
            .segments
            .iter()
            .map(|segment| J2kClassicSegment {
                data_offset: segment.data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            })
            .collect();
        dispatch_classic_cleanup_batched(runtime, job.data, &[batch_job], &segments, &decoded)?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_classic_cleanup_sub_band(
    job: J2kSubBandDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal sub-band size overflow".to_string(),
        })?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal sub-band output slice is too small".to_string(),
        });
    }
    if job.jobs.is_empty() {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let mut jobs = Vec::with_capacity(job.jobs.len());
        let mut coded_data = Vec::new();
        let mut segments = Vec::new();

        for block in job.jobs {
            let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal batched coded payload exceeds u32".to_string(),
            })?;
            coded_data.extend_from_slice(block.code_block.data);
            let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal segment table exceeds u32".to_string(),
            })?;
            let end_x = block
                .output_x
                .checked_add(block.code_block.width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal batched block width overflow".to_string(),
                })?;
            let end_y = block
                .output_y
                .checked_add(block.code_block.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal batched block height overflow".to_string(),
                })?;
            if end_x > job.width || end_y > job.height {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal batched block lies outside sub-band bounds"
                        .to_string(),
                });
            }
            for segment in block.code_block.segments {
                let data_offset =
                    coded_offset
                        .checked_add(segment.data_offset)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "classic J2K Metal segment offset overflow".to_string(),
                        })?;
                segments.push(J2kClassicSegment {
                    data_offset,
                    data_length: segment.data_length,
                    start_coding_pass: u32::from(segment.start_coding_pass),
                    end_coding_pass: u32::from(segment.end_coding_pass),
                    use_arithmetic: u32::from(segment.use_arithmetic),
                });
            }
            jobs.push(J2kClassicCleanupBatchJob {
                coded_offset,
                coded_len: u32::try_from(block.code_block.data.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal coded payload exceeds u32".to_string(),
                    }
                })?,
                segment_offset,
                segment_count: u32::try_from(block.code_block.segments.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal segment count exceeds u32".to_string(),
                    }
                })?,
                width: block.code_block.width,
                height: block.code_block.height,
                output_stride: job.width,
                output_offset: block
                    .output_y
                    .checked_mul(job.width)
                    .and_then(|row| row.checked_add(block.output_x))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal output offset overflow".to_string(),
                    })?,
                missing_msbs: u32::from(block.code_block.missing_bit_planes),
                total_bitplanes: u32::from(block.code_block.total_bitplanes),
                roi_shift: u32::from(block.code_block.roi_shift),
                number_of_coding_passes: u32::from(block.code_block.number_of_coding_passes),
                sub_band_type: match block.code_block.sub_band_type {
                    j2k_native::J2kSubBandType::LowLow => 0,
                    j2k_native::J2kSubBandType::HighLow => 1,
                    j2k_native::J2kSubBandType::LowHigh => 2,
                    j2k_native::J2kSubBandType::HighHigh => 3,
                },
                style_flags: classic_style_flags(block.code_block.style),
                strict: u32::from(block.code_block.strict),
                dequantization_step: block.code_block.dequantization_step,
            });
        }

        dispatch_classic_cleanup_batched(runtime, &coded_data, &jobs, &segments, &decoded)?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_ht_cleanup_code_block(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = required_ht_output_len(job)?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal output slice is too small".to_string(),
        });
    }

    if job.width == 0 || job.height == 0 {
        return Ok(());
    }

    with_runtime(|runtime| {
        let params = J2kHtCleanupParams {
            width: job.width,
            height: job.height,
            coded_len: u32::try_from(job.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            missing_msbs: u32::from(job.missing_bit_planes),
            num_bitplanes: u32::from(job.num_bitplanes),
            number_of_coding_passes: u32::from(job.number_of_coding_passes),
            output_stride: u32::try_from(job.output_stride).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal output stride exceeds u32".to_string(),
            })?,
            output_offset: 0,
            dequantization_step: job.dequantization_step,
            stripe_causal: u32::from(job.stripe_causal),
        };
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        dispatch_ht_cleanup(runtime, job.data, params, &decoded)?;

        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_ht_cleanup_sub_band(
    job: HtSubBandDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal sub-band size overflow".to_string(),
        })?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal sub-band output slice is too small".to_string(),
        });
    }

    if job.jobs.is_empty() {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let mut jobs = Vec::with_capacity(job.jobs.len());
        let mut coded_data = Vec::new();

        for block in job.jobs {
            let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal batched coded payload exceeds u32".to_string(),
            })?;
            coded_data.extend_from_slice(block.code_block.data);

            jobs.push(J2kHtCleanupBatchJob {
                coded_offset,
                width: block.code_block.width,
                height: block.code_block.height,
                coded_len: u32::try_from(block.code_block.data.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal coded payload exceeds u32".to_string(),
                    }
                })?,
                cleanup_length: block.code_block.cleanup_length,
                refinement_length: block.code_block.refinement_length,
                missing_msbs: u32::from(block.code_block.missing_bit_planes),
                num_bitplanes: u32::from(block.code_block.num_bitplanes),
                roi_shift: u32::from(block.code_block.roi_shift),
                number_of_coding_passes: u32::from(block.code_block.number_of_coding_passes),
                output_stride: job.width,
                output_offset: block
                    .output_y
                    .checked_mul(job.width)
                    .and_then(|row| row.checked_add(block.output_x))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal output offset overflow".to_string(),
                    })?,
                dequantization_step: block.code_block.dequantization_step,
                stripe_causal: u32::from(block.code_block.stripe_causal),
            });

            let end_x = block
                .output_x
                .checked_add(block.code_block.width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batched block width overflow".to_string(),
                })?;
            let end_y = block
                .output_y
                .checked_add(block.code_block.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batched block height overflow".to_string(),
                })?;
            if end_x > job.width || end_y > job.height {
                return Err(Error::MetalKernel {
                    message: "HTJ2K Metal batched block lies outside sub-band bounds".to_string(),
                });
            }
        }

        dispatch_ht_cleanup_batched(runtime, &coded_data, &jobs, &decoded)?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
mod surface_decode;
#[cfg(all(test, target_os = "macos"))]
mod tests;
#[cfg(target_os = "macos")]
pub(crate) use self::surface_decode::{
    decode_image_region_to_surface, decode_image_region_to_surface_with_device,
    decode_image_to_surface, decode_image_to_surface_with_device, decode_region_scaled_to_surface,
    decode_region_scaled_to_surface_with_device, decode_scaled_to_surface,
    decode_scaled_to_surface_with_device,
};
