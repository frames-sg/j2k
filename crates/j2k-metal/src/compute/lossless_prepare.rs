// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use j2k_metal_support::{dispatch_1d_pipeline, dispatch_2d_pipeline, dispatch_3d_pipeline};

use crate::profile_env::{
    hybrid_stage_signpost, label_command_buffer, label_compute_encoder,
    metal_profile_coefficient_prep_split_commands_enabled,
};

use super::abi::{
    J2kForwardDwt53BatchedParams, J2kForwardDwt53Params, J2kForwardIctParams, J2kForwardRctParams,
    J2kLosslessCoefficientJob, J2kLosslessDeinterleaveParams, J2kMctStatus,
    J2kPacketPayloadCopyParams, J2kQuantizeSubbandParams, J2K_MCT_STATUS_OK,
    PACKET_PAYLOAD_COPY_BYTES_PER_STRIPE, PACKET_PAYLOAD_COPY_STRIPES_PER_JOB,
};
use super::forward_transform::{
    active_forward_dwt53_buffers, dispatch_forward_dwt53_batched_pass, dispatch_forward_dwt53_pass,
};
use super::resident_tier1::J2kBatchedPacketPayloadCopyDispatch;
#[cfg(test)]
use super::test_counters;
use super::{
    checked_buffer_read, checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer,
    decode_mct_status_error, new_command_buffer, new_compute_command_encoder, new_private_buffer,
    new_resident_encode_command_buffer, new_shared_buffer, take_recyclable_private_buffer,
    with_runtime, with_runtime_for_session, zeroed_shared_buffer, Buffer, CommandBuffer,
    CommandBufferRef, Error, J2kLosslessDeviceBatchPrepareItem, J2kLosslessDeviceCodeBlock,
    J2kLosslessDevicePrepareJob, J2kPreparedLosslessDeviceCodeBlocks, J2kQuantizeSubbandJob,
    MTLSize, MetalRuntime,
};

mod batch;
mod batch_item;
mod commands;
mod forward_encode;
mod single;
mod sizes;

pub(crate) use self::batch::prepare_lossless_device_code_blocks_batch;
use self::batch_item::{prepare_lossless_batch_item, BatchPrepareItemRequest};
pub(in crate::compute) use self::commands::{
    dispatch_batched_packet_payload_copy, dispatch_forward_dwt53_components_on_buffers,
    dispatch_forward_dwt53_components_split_profile, dispatch_forward_dwt53_on_buffers,
    dispatch_forward_dwt53_on_buffers_split_profile, dispatch_forward_rct_on_buffers,
    dispatch_lossless_deinterleave, dispatch_lossless_deinterleave_rct_rgb8,
    dispatch_lossless_extract_coefficients, lossless_deinterleave_rct_rgb8_supported,
    ForwardDwt53ComponentsDispatch, ForwardDwt53SplitProfile,
};
pub(crate) use self::forward_encode::{
    encode_forward_ict, encode_forward_rct, encode_quantize_subband,
};
pub(crate) use self::single::prepare_lossless_device_code_blocks;
pub(in crate::compute) use self::sizes::{lossless_prepare_sizes, J2kLosslessPrepareSizes};
