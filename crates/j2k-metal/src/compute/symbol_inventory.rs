// SPDX-License-Identifier: MIT OR Apache-2.0

//! Explicit symbol wiring for the `compute` facade.
//!
//! Keeping this inventory separate prevents import bookkeeping from obscuring
//! the facade's runtime helpers while preserving its established root paths.

macro_rules! wire_compute_symbols {
    () => {
        pub(crate) use crate::compute::abi::J2K_HT_STATUS_OK;
        #[cfg(target_os = "macos")]
        pub(crate) use crate::compute::abi::{
            J2kBatchedCodestreamAssemblyJob, J2kBatchedMctRgb8PackParams,
            J2kBatchedPacketEncodeJob, J2kClassicCleanupBatchJob, J2kClassicEncodeBatchJob,
            J2kClassicEncodeParams, J2kClassicEncodeStatus, J2kClassicRepeatedBatchParams,
            J2kClassicSegment, J2kClassicStatus, J2kClassicTier1DensityCounters,
            J2kClassicTier1PassPlanCounters, J2kClassicTier1SymbolPlanCounters,
            J2kClassicTier1TokenSegment, J2kCodestreamAssemblyStatus, J2kCopyInterleavedParams,
            J2kForwardDwt53BatchedParams, J2kForwardDwt53Params, J2kForwardIctParams,
            J2kForwardRctParams, J2kGrayStoreParams, J2kHtCleanupBatchJob, J2kHtCleanupParams,
            J2kHtEncodeBatchJob, J2kHtEncodeParams, J2kHtEncodeStatus, J2kHtRepeatedBatchParams,
            J2kHtStatus, J2kIdwtSingleDecompositionParams, J2kIdwtStatus, J2kInverseMctParams,
            J2kLosslessCodestreamAssemblyParams, J2kLosslessCoefficientJob,
            J2kLosslessDeinterleaveParams, J2kMctRgb8PackParams, J2kMctStatus, J2kPackParams,
            J2kPacketBlock, J2kPacketDescriptor, J2kPacketEncodeParams, J2kPacketEncodeStatus,
            J2kPacketPayloadCopyJob, J2kPacketPayloadCopyParams, J2kPacketResolution,
            J2kPacketStateBlock, J2kPacketSubband, J2kQuantizeSubbandParams,
            J2kRepeatedGrayPackParams, J2kRepeatedGrayStoreParams,
            J2kRepeatedIdwtSingleDecompositionParams, J2kRepeatedStoreParams,
            J2kResidentPacketBlock, J2kResidentPacketBlockParams, J2kStoreParams,
            J2kValidateBytesParams, J2kValidateBytesStatus,
            CLASSIC_TIER1_MQ_BYTE_TOKEN_ARENA_BYTES, CLASSIC_TIER1_TOKEN_ARENA_BYTES,
            CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY, HT_PACKET_CAPACITY_ENV,
            J2K_CLASSIC_ENCODE_32_MAX_HEIGHT, J2K_CLASSIC_ENCODE_32_MAX_WIDTH,
            J2K_CLASSIC_MAX_COEFF_COUNT, J2K_CLASSIC_MAX_HEIGHT, J2K_CLASSIC_MAX_WIDTH,
            J2K_CLASSIC_STATUS_FAIL, J2K_CLASSIC_STATUS_OK, J2K_CLASSIC_STATUS_UNSUPPORTED,
            J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES,
            J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS,
            J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
            J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS,
            J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT, J2K_ENCODE_STATUS_FAIL,
            J2K_ENCODE_STATUS_OK, J2K_ENCODE_STATUS_UNSUPPORTED,
            J2K_HT_ENCODE_BASE_OUTPUT_SIZE, J2K_HT_ENCODE_MAX_SAMPLES, J2K_HT_ENCODE_MEL_SIZE,
            J2K_HT_ENCODE_MS_BYTES_PER_SAMPLE_FLOOR, J2K_HT_ENCODE_MS_SIZE,
            J2K_HT_ENCODE_VLC_SIZE, J2K_HT_STATUS_FAIL, J2K_HT_STATUS_UNSUPPORTED,
            J2K_IDWT_STATUS_FAIL, J2K_IDWT_STATUS_OK, J2K_MCT_STATUS_FAIL, J2K_MCT_STATUS_OK,
            PACKET_PAYLOAD_COPY_BYTES_PER_STRIPE, PACKET_PAYLOAD_COPY_STRIPES_PER_JOB,
        };

        #[cfg(target_os = "macos")]
        pub(crate) use crate::compute::direct_prepare::prepare_direct_grayscale_plan;
        #[cfg(all(target_os = "macos", test))]
        use crate::compute::direct_prepare::prepare_sub_band_groups;
        #[cfg(target_os = "macos")]
        use crate::compute::direct_prepare::{
            prepare_classic_sub_band_groups, prepare_ht_sub_band_groups,
            prepare_ungrouped_ht_sub_band_buffers, prepared_ht_buffer,
        };

        #[cfg(target_os = "macos")]
        pub(crate) use crate::compute::direct_roi::crop_prepared_direct_grayscale_plan_to_output_region;
        #[cfg(all(target_os = "macos", test))]
        use crate::compute::direct_roi::retain_ht_jobs_for_required_region;
        #[cfg(target_os = "macos")]
        use crate::compute::direct_roi::{
            idwt_input_windows_from_slices, prepared_idwt_output_len, prepared_idwt_params,
            repeated_idwt_params, BandRequiredRegion, PreparedIdwtInputStrides,
        };

        #[cfg(all(target_os = "macos", test))]
        use crate::compute::direct_grayscale_execute::execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test;
        #[cfg(target_os = "macos")]
        use crate::compute::direct_grayscale_execute::{
            checked_coefficient_len, encode_prepared_direct_component_plane_in_command_buffer,
            upload_cpu_decoded_coefficients, DirectComponentPlaneRequest,
        };
        #[cfg(target_os = "macos")]
        pub(crate) use crate::compute::direct_grayscale_execute::{
            execute_hybrid_cpu_tier1_direct_color_plan,
            execute_hybrid_cpu_tier1_direct_color_plan_batch,
            execute_hybrid_cpu_tier1_direct_color_plan_with_device,
            execute_prepared_direct_color_plan, execute_prepared_direct_color_plan_batch,
            execute_prepared_direct_color_plan_with_device, execute_prepared_direct_grayscale_plan,
            execute_prepared_direct_grayscale_plan_batch,
            execute_prepared_direct_grayscale_plan_with_device,
            execute_repeated_prepared_direct_grayscale_plan,
        };

        #[cfg(target_os = "macos")]
        use crate::compute::forward_transform::{
            active_forward_dwt53_buffers, dispatch_forward_dwt53_batched_pass,
            dispatch_forward_dwt53_pass,
        };
        #[cfg(target_os = "macos")]
        pub(crate) use crate::compute::forward_transform::{
            encode_deinterleave_to_f32, encode_forward_dwt53, encode_forward_dwt97,
        };

        #[cfg(all(target_os = "macos", test))]
        use crate::compute::resident_tier1::dispatch_classic_tier1_split_token_emit_for_cpu_pack;
        #[cfg(target_os = "macos")]
        use crate::compute::resident_tier1::{
            dispatch_classic_tier1_arithmetic_pack_profile,
            dispatch_classic_tier1_density_profile, dispatch_classic_tier1_pass_plan_profile,
            dispatch_classic_tier1_raw_pack_profile,
            dispatch_classic_tier1_split_token_emit_for_gpu_pack,
            dispatch_classic_tier1_split_token_emit_profile,
            dispatch_classic_tier1_split_token_pack_from_gpu_tokens,
            dispatch_classic_tier1_symbol_plan_profile,
            dispatch_classic_tier1_token_emit_for_gpu_pack,
            dispatch_classic_tier1_token_emit_profile,
            dispatch_classic_tier1_token_pack_from_gpu_tokens,
            schedule_classic_tier1_gpu_token_pack_readback, schedule_resident_tier1_status_readback,
            J2kBatchedPacketPayloadCopyDispatch, ResidentTier1StatusReadbackRequest,
        };
        #[cfg(target_os = "macos")]
        pub(crate) use crate::compute::resident_tier1::{
            wait_resident_lossless_codestream, wait_resident_lossless_codestream_batch,
            J2kLosslessCodestreamAssemblyJob, J2kLosslessCodestreamBlockCodingMode,
            J2kLosslessDeviceBatchPrepareItem,
            J2kLosslessDeviceCodeBlock, J2kLosslessDevicePrepareJob,
            J2kPendingResidentLosslessCodestreamBatch, J2kPreparedLosslessDeviceCodeBlocks,
            J2kResidentLosslessHtCodeBlocks, J2kResidentLosslessTier1CodeBlocks,
            J2kResidentPacketizationEncodeJob, J2kResidentPacketizationResolution,
            J2kResidentPacketizationSubband, ResidentLosslessTier1Metal,
        };

        #[cfg(target_os = "macos")]
        use crate::compute::lossless_prepare::dispatch_batched_packet_payload_copy;
        #[cfg(target_os = "macos")]
        pub(crate) use crate::compute::lossless_prepare::{
            encode_forward_ict, encode_forward_rct, encode_quantize_subband,
            prepare_lossless_device_code_blocks, prepare_lossless_device_code_blocks_batch,
        };

        #[cfg(all(target_os = "macos", test))]
        use crate::compute::decode_dispatch::{
            classic_batch_uses_plain_fast_path, classic_repeated_uses_plain_fast_path,
            repeated_gray_store_is_contiguous_full_surface,
        };
        #[cfg(target_os = "macos")]
        pub(crate) use crate::compute::decode_dispatch::{
            decode_inverse_mct, decode_irreversible97_single_decomposition_idwt,
            decode_reversible53_single_decomposition_idwt, decode_store_component_and_capture,
        };
        #[cfg(target_os = "macos")]
        use crate::compute::decode_dispatch::{
            dispatch_classic_cleanup_batched, dispatch_inverse_mct_buffers_in_command_buffer,
            dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets,
            dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
            dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets,
            dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets,
            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets,
            dispatch_store_component_buffer_in_command_buffer_with_offsets,
            dispatch_store_component_buffer_in_encoder_with_offsets,
            dispatch_store_component_repeated_in_command_buffer, dispatch_zero_u32_buffer_in_encoder,
            encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer,
            encode_distinct_classic_sub_bands_to_buffer_in_command_buffer,
            encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer,
            encode_distinct_ht_sub_bands_to_buffer_in_command_buffer,
            encode_gray_store_to_surface_in_encoder,
            encode_prepared_classic_sub_band_group_to_buffer_in_encoder,
            encode_prepared_classic_sub_band_to_buffer_in_encoder,
            encode_prepared_ht_sub_band_group_to_buffer_in_encoder,
            encode_prepared_ht_sub_band_to_buffer_in_encoder,
            encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer,
            encode_repeated_classic_sub_band_to_buffer_in_command_buffer,
            encode_repeated_gray_store_to_surfaces_in_command_buffer,
            encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer,
            encode_repeated_ht_sub_band_to_buffer_in_command_buffer, ht_batch_output_word_count,
            ht_output_word_count, required_ht_output_len, IdwtSubBandBuffers, RepeatedIdwtDispatch,
            SingleIdwtDispatch,
        };

        #[cfg(target_os = "macos")]
        use crate::compute::tier1_encode::{
            classic_encode_sub_band_code, encode_status_error, packet_encode_status_error,
        };
        #[cfg(target_os = "macos")]
        pub(crate) use crate::compute::tier1_encode::{
            encode_classic_tier1_code_block, encode_classic_tier1_code_blocks,
            encode_classic_tier1_prepared_device_code_blocks_resident, encode_ht_cleanup_code_block,
            encode_ht_cleanup_code_blocks, encode_ht_prepared_device_code_blocks_resident,
            read_resident_ht_tier1_code_blocks_for_cpu_packetization,
        };
        #[cfg(all(target_os = "macos", test))]
        pub(crate) use crate::compute::tier1_encode::{
            encode_classic_tier1_code_blocks_via_gpu_token_pack_for_test,
            encode_classic_tier1_code_blocks_via_ordered_tokens_cpu_pack_for_test,
            encode_classic_tier1_code_blocks_via_split_mq_byte_raw_tokens_gpu_pack_for_test,
            encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_cpu_pack_for_test,
            encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test,
        };

        #[cfg(target_os = "macos")]
        use crate::compute::resident_codestream::{
            dispatch_ht_cleanup, dispatch_ht_cleanup_batched,
            dispatch_ht_cleanup_batched_in_command_buffer, dispatch_ht_cleanup_batched_in_encoder,
            dispatch_ht_cleanup_repeated_batched_in_command_buffer, HtRepeatedCleanupDispatch,
        };
        #[cfg(target_os = "macos")]
        pub(crate) use crate::compute::resident_codestream::{
            encode_lossless_codestream_buffer_from_resident_tier1, encode_tier2_packetization,
            submit_lossless_codestream_buffers_from_prepared_classic_batch,
            submit_lossless_codestream_buffers_from_prepared_ht_batch,
        };

        #[cfg(target_os = "macos")]
        pub(crate) use crate::compute::decode_cleanup::{
            decode_classic_cleanup_code_block, decode_classic_cleanup_sub_band,
            decode_ht_cleanup_code_block, decode_ht_cleanup_sub_band,
        };
    };
}

pub(super) use wire_compute_symbols;
