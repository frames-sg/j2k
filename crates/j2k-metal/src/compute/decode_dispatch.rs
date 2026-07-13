// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_buffer_read, checked_buffer_slice, checked_metal_surface_len, commit_and_wait_metal,
    copied_slice_buffer, decode_classic_status_error, decode_idwt_status_error,
    decode_mct_status_error, dispatch_1d_pipeline, dispatch_2d_pipeline, dispatch_3d_pipeline,
    dispatch_ht_cleanup_batched_in_command_buffer, dispatch_ht_cleanup_batched_in_encoder,
    dispatch_ht_cleanup_repeated_batched_in_command_buffer, hybrid_stage_signpost, j2k_u32_param,
    label_compute_encoder, new_command_buffer, new_compute_command_encoder, new_shared_buffer,
    prepared_ht_buffer, size_of, take_classic_coefficients_scratch_buffer,
    take_classic_states_scratch_buffer, with_runtime, zeroed_shared_buffer, Buffer,
    CommandBufferRef, ComputeCommandEncoderRef, DirectIdwtCommandBuffers, DirectScratchBuffer,
    DirectStatusCheck, Error, HtCodeBlockDecodeJob, HtRepeatedCleanupDispatch,
    J2kClassicCleanupBatchJob, J2kClassicRepeatedBatchParams, J2kClassicSegment, J2kClassicStatus,
    J2kGrayStoreParams, J2kHtCleanupBatchJob, J2kIdwt97StepParams,
    J2kIdwtSingleDecompositionParams, J2kIdwtStatus, J2kInverseMctJob, J2kInverseMctParams,
    J2kMctStatus, J2kRepeatedGrayStoreParams, J2kRepeatedIdwtSingleDecompositionParams,
    J2kRepeatedStoreParams, J2kSingleDecompositionIdwtJob, J2kStoreComponentJob, J2kStoreParams,
    J2kWaveletTransform, MTLSize, MetalRuntime, PixelFormat, PreparedClassicSubBand,
    PreparedClassicSubBandGroup, PreparedHtSubBand, PreparedHtSubBandGroup, Surface,
    J2K_CLASSIC_MAX_HEIGHT, J2K_CLASSIC_MAX_WIDTH, J2K_CLASSIC_STATUS_OK, J2K_IDWT_STATUS_OK,
    J2K_MCT_STATUS_OK, SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
    SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE, SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE,
};

mod classic_cleanup;
mod classic_subband;
mod ht_distinct;
mod ht_subband;
mod idwt;
mod mct;
mod store;

pub(in crate::compute) use self::classic_cleanup::{
    classic_batch_is_plain_arithmetic, classic_batch_uses_plain_fast_path,
    classic_repeated_uses_plain_fast_path, dispatch_classic_cleanup_batched,
    dispatch_classic_cleanup_batched_in_encoder,
    dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer,
    dispatch_classic_cleanup_repeated_batched_in_command_buffer,
    dispatch_classic_store_repeated_batched_in_command_buffer,
    encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer,
    encode_distinct_classic_sub_bands_to_buffer_in_command_buffer, ClassicCleanupBatchDispatch,
    ClassicPlainDevRepeatedCleanupDispatch, ClassicRepeatedCleanupDispatch,
    ClassicRepeatedStoreDispatch,
};
pub(in crate::compute) use self::classic_subband::{
    encode_prepared_classic_sub_band_group_to_buffer_in_encoder,
    encode_prepared_classic_sub_band_to_buffer_in_encoder,
    encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer,
    encode_repeated_classic_sub_band_to_buffer_in_command_buffer,
};
pub(in crate::compute) use self::ht_distinct::{
    encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer,
    encode_distinct_ht_sub_bands_to_buffer_in_command_buffer,
};
pub(in crate::compute) use self::ht_subband::{
    dispatch_zero_u32_buffer_in_encoder, encode_prepared_ht_sub_band_group_to_buffer_in_encoder,
    encode_prepared_ht_sub_band_to_buffer_in_encoder,
    encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer,
    encode_repeated_ht_sub_band_to_buffer_in_command_buffer, ht_batch_output_word_count,
    ht_output_word_count, required_ht_output_len,
};
#[cfg(test)]
pub(crate) use self::idwt::decode_irreversible97_staged_single_decomposition_idwt;
pub(crate) use self::idwt::{
    decode_irreversible97_single_decomposition_idwt, decode_reversible53_single_decomposition_idwt,
};
pub(in crate::compute) use self::idwt::{
    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets, IdwtSubBandBuffers,
    RepeatedIdwtDispatch, SingleIdwtDispatch,
};
pub(crate) use self::mct::decode_inverse_mct;
pub(in crate::compute) use self::mct::dispatch_inverse_mct_buffers_in_command_buffer;
pub(crate) use self::store::decode_store_component_and_capture;
#[cfg(test)]
pub(in crate::compute) use self::store::repeated_gray_store_is_contiguous_full_surface;
pub(in crate::compute) use self::store::{
    dispatch_store_component_buffer_in_command_buffer_with_offsets,
    dispatch_store_component_buffer_in_encoder_with_offsets,
    dispatch_store_component_repeated_in_command_buffer, encode_gray_store_to_surface_in_encoder,
    encode_repeated_gray_store_to_surfaces_in_command_buffer,
};
