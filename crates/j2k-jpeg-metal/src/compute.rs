// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use j2k_core::BackendRequest;
use j2k_core::{BufferError, PixelFormat, Rect};
use j2k_jpeg::{
    adapter::{
        JpegEntropyCheckpointV1, JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1,
        JpegHuffmanTable,
    },
    ColorSpace as JpegColorSpace, Decoder as CpuDecoder,
};
#[cfg(all(target_os = "macos", test))]
use j2k_metal_support::system_default_device;
#[cfg(target_os = "macos")]
use j2k_metal_support::{
    checked_blit_command_encoder, checked_command_buffer, checked_command_queue,
    checked_compute_command_encoder, commit_and_wait, wait_for_completion, MetalPipelineLoader,
    MetalSupportError,
};
#[cfg(all(target_os = "macos", test))]
use metal::foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use metal::{
    BlitCommandEncoder, Buffer, CommandBuffer, CommandBufferRef, CommandQueue, CommandQueueRef,
    ComputeCommandEncoder, ComputePipelineState, Device, MTLPixelFormat, MTLResourceOptions,
};
#[cfg(target_os = "macos")]
use std::{
    cell::RefCell,
    sync::{Arc, Mutex, MutexGuard},
};

#[cfg(target_os = "macos")]
pub(crate) use crate::abi::{
    JpegBaselineEncodeHuffmanTable, JpegBaselineEncodeParams, JpegBaselineEncodeStatus,
    JpegBaselineEntropyEncodeBatchJob, JpegBaselineEntropyEncodeJob, JpegDecodeStatus,
    JpegEntropyCheckpointHost, JpegFast420BatchParams, JpegFast420Params, JpegFast420ScaledParams,
    JpegFast420TextureBatchParams, JpegFast420WindowedPackParams, JpegFast444Params,
    JpegFast444ScaledParams, JpegFast444TextureBatchParams, JpegFastRegionScaledBatchParams,
    JpegPackParams, JpegRgb8ToRgbaTextureParams, JpegTexturePackBatchParams,
    JpegWindowedPackBatchParams, JpegWindowedTexturePackBatchParams, PreparedHuffmanHost,
    FAST420_TEXTURE_BOUNDARY_META_WORDS, FAST420_TEXTURE_BOUNDARY_SAMPLE_BYTES,
    FAST420_TEXTURE_VERTICAL_META_WORDS, FAST420_TEXTURE_VERTICAL_SAMPLE_BYTES,
    FAST422_TEXTURE_BOUNDARY_META_WORDS, FAST422_TEXTURE_BOUNDARY_SAMPLE_BYTES, MODE_GRAY,
    MODE_RGB, MODE_YCBCR, OUT_GRAY, OUT_RGB, OUT_RGBA,
};

#[cfg(target_os = "macos")]
use crate::buffers::{
    new_decode_plane_buffer, new_private_buffer, new_shared_buffer_with_data, MetalBatchScratch,
};
#[cfg(target_os = "macos")]
use crate::error::{metal_kernel_support_error, metal_runtime_support_error};
use crate::{batch, Error, JpegFastPackets, Surface};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn new_command_buffer(
    queue: &CommandQueueRef,
) -> Result<CommandBuffer, Error> {
    checked_command_buffer(queue).map_err(|source| {
        metal_kernel_support_error("JPEG Metal command buffer creation failed", source)
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn new_compute_command_encoder(
    command_buffer: &CommandBufferRef,
) -> Result<ComputeCommandEncoder, Error> {
    checked_compute_command_encoder(command_buffer).map_err(|source| {
        metal_kernel_support_error("JPEG Metal compute encoder creation failed", source)
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn new_blit_command_encoder(
    command_buffer: &CommandBufferRef,
) -> Result<BlitCommandEncoder, Error> {
    checked_blit_command_encoder(command_buffer).map_err(|source| {
        metal_kernel_support_error("JPEG Metal blit encoder creation failed", source)
    })
}

#[cfg(target_os = "macos")]
mod batch_entry;
#[cfg(target_os = "macos")]
mod batch_full;
mod batch_plan;
#[cfg(target_os = "macos")]
mod batch_region;
#[cfg(target_os = "macos")]
mod batch_support;
#[cfg(target_os = "macos")]
mod encode;
mod fast_packets;
#[cfg(target_os = "macos")]
mod kernel_helpers;
#[cfg(target_os = "macos")]
mod pack_dispatch;
#[cfg(target_os = "macos")]
mod region_scaled_plan;
#[cfg(target_os = "macos")]
mod single_decode;
#[cfg(target_os = "macos")]
mod status;
mod viewport_cache;
#[cfg(target_os = "macos")]
mod viewport_compose;
#[cfg(all(target_os = "macos", test))]
pub(crate) use self::batch_entry::decode_full_batch_to_surfaces;
#[cfg(target_os = "macos")]
pub(crate) use self::batch_entry::{
    decode_full_batch_to_surfaces_with_session, decode_full_batch_to_surfaces_with_session_state,
    decode_full_rgb8_batch_into_output_with_session,
    decode_full_rgb8_batch_into_textures_with_session,
    decode_region_scaled_rgb8_batch_into_output_with_session,
    decode_region_scaled_rgb8_batch_into_textures_with_session,
};
#[cfg(all(target_os = "macos", test))]
use self::batch_full::try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output;
#[cfg(target_os = "macos")]
use self::batch_full::{
    try_decode_fast444_full_rgb_batch_to_surfaces,
    try_decode_fast444_full_rgb_batch_to_surfaces_into_output,
    try_decode_fast444_full_rgba_batch_to_textures,
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces,
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output,
    try_decode_fast_subsampled_full_rgba_batch_to_textures,
};
use self::batch_plan::{
    batched_fast_packets, core_rect_to_jpeg, BatchDeviceBufferCache, BatchedDecodeItem,
    BatchedFastPacket,
};
#[cfg(target_os = "macos")]
use self::batch_region::{
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast420_region_scaled_rgba_batch_to_textures,
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast422_region_scaled_rgba_batch_to_textures,
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast444_region_scaled_rgba_batch_to_textures,
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output,
    try_decode_repeated_region_scaled_batch_to_surfaces,
};
#[cfg(target_os = "macos")]
use self::batch_support::{
    batch_entropy_buffers, batch_entropy_host_data, fast420_batch_timing_enabled,
    fast_batch_decode_mode, region_scaled_batch_error_results, surface_batch_error_results,
    surface_batch_success_results, texture_batch_error_results, BatchEntropyBufferKeys,
    BatchEntropyBufferPlan, BatchEntropyBuffers, BatchEntropyHostData, BatchEntropyLabels,
    FastBatchDecodeMode, FastBatchTiming,
};
#[cfg(all(test, target_os = "macos"))]
use self::batch_support::{fast420_batch_timing_value_enabled, fast420_batch_timing_value_mode};
#[cfg(target_os = "macos")]
pub(crate) use self::encode::{
    encode_jpeg_baseline_entropy_batch_with_session, encode_jpeg_baseline_entropy_with_session,
};
use self::fast_packets::{
    checked_entropy_segment_count, entropy_checkpoints_buffer, entropy_decode_thread_count,
    fast444_params, fast444_region_params, fast444_scaled_params, fast444_scaled_region_params,
    fast_subsampled_full_mcu_scaled_window, fast_subsampled_full_mcu_window,
    fast_subsampled_params, fast_subsampled_region_params, fast_subsampled_scaled_params,
    fast_subsampled_scaled_region_params, fast_subsampled_windowed_pack_params_for_dims,
    mcu_range_for_rect, restart_offsets_buffer, restart_work_for_mcu_range, FastRegionScaledMetal,
    FastScratchKeys, FastSubsampledMetal, FastSubsampledPacket, FastTextureRepairCtx,
};
#[cfg(all(test, target_os = "macos"))]
use self::kernel_helpers::choose_1d_threadgroup_width;
#[cfg(target_os = "macos")]
use self::kernel_helpers::{
    bind_fast_decode_entropy_inputs, bind_three_plane_pack, dispatch_1d_pipeline,
    dispatch_2d_pipeline, dispatch_3d_pipeline, fast_packet_huffman_tables, packed_pair_extent,
    pixel_format_to_out_format, plane_mode_to_u32, FastDecodeEntropyInputs,
};
#[cfg(target_os = "macos")]
use self::pack_dispatch::{
    batch_output_buffer_or_new, checked_u32, copy_grouped_surfaces_to_output,
    copy_rgb8_surfaces_to_rgba_textures, dispatch_rgba_texture_pack,
    dispatch_windowed_rgba_texture_pack, encode_fast444_batch_item,
    encode_fast444_region_batch_item, encode_fast444_scaled_batch_item,
    encode_fast444_scaled_region_batch_item, encode_fast_subsampled_op_batch_item,
    encode_fast_subsampled_region_batch_item, encode_fast_subsampled_scaled_batch_item,
    texture_batch_success_results, validate_rgba_texture_batch_output,
    Fast444ScaledRegionBatchItemRequest, FastSubsampledOpBatchItemRequest,
};
#[cfg(all(target_os = "macos", test))]
use self::pack_dispatch::{encode_split_coeff_idct_passes, SplitCoeffIdctPasses};
#[cfg(target_os = "macos")]
use self::region_scaled_plan::{
    fast444_packets_share_region_scaled_batch_shape, fast444_region_scaled_batch_groups,
    fast_subsampled_full_rgb_batch_groups, fast_subsampled_packets_share_full_rgb_batch_shape,
    fast_subsampled_region_scaled_batch_groups, fast_subsampled_region_scaled_batch_plan,
    windowed_texture_pack_params, RegionScaledBatchPlan,
};
#[cfg(target_os = "macos")]
pub(crate) use self::single_decode::{
    decode_private_rgb8_tile_with_session, decode_region_scaled_to_surface,
    decode_region_to_surface, decode_scaled_to_surface, decode_to_surface,
    decode_to_surface_with_session,
};
#[cfg(all(target_os = "macos", test))]
use self::single_decode::{
    try_decode_fast420_region_to_surface, try_decode_fast420_scaled_region_to_surface,
    try_decode_fast420_scaled_to_surface, try_decode_fast422_region_to_surface,
    try_decode_fast422_scaled_to_surface, try_decode_fast422_to_surface,
    try_decode_fast444_region_to_surface, try_decode_fast444_scaled_region_to_surface,
    try_decode_fast444_scaled_to_surface, try_decode_fast444_to_surface,
};
#[cfg(target_os = "macos")]
use self::single_decode::{
    try_decode_fast420_scaled_region_to_surface_with_status,
    try_decode_fast422_scaled_region_to_surface,
    try_decode_fast444_scaled_region_to_surface_with_mode_and_status,
};
#[cfg(target_os = "macos")]
use self::status::{
    decode_status_buffer, fast422_status_error, fast_decode_status_error, first_decode_error_status,
};
use self::viewport_cache::{
    cached_plane_stage, CachedViewportPlanes, PlaneMode, PlaneStage, ViewportPlaneCacheGate,
    ViewportPlaneCacheLease,
};
#[cfg(target_os = "macos")]
pub(crate) use self::viewport_compose::compose_rgb_viewport_from_regions;
#[cfg(all(target_os = "macos", test))]
pub(crate) use self::viewport_compose::{
    compose_rgb_viewport_from_regions_into_output_with_session,
    compose_rgb_viewport_from_regions_into_textures_with_session,
};

#[cfg(all(target_os = "macos", test))]
pub(crate) use crate::buffers::{
    jpeg_private_buffer_allocations_for_test, jpeg_shared_buffer_allocations_for_test,
    reset_jpeg_private_buffer_allocations_for_test, reset_jpeg_shared_buffer_allocations_for_test,
};

#[cfg(target_os = "macos")]
const SHADER_SOURCE: &str = concat!(
    include_str!("shaders_shared.metal"),
    include_str!("shaders_encode.metal"),
    include_str!("shaders_decode_helpers.metal"),
    include_str!("shaders_pack_444.metal"),
    include_str!("shaders_decode_fast420.metal"),
    include_str!("shaders_decode_fast422_regions.metal"),
    include_str!("shaders_decode_fast444.metal"),
    include_str!("shaders_pack_subsampled.metal"),
);

#[cfg(target_os = "macos")]
const REGION_SCALED_BATCH_CHUNK: usize = 8;

#[cfg(target_os = "macos")]
struct FastRgbDecodeBuffer {
    buffer: Buffer,
    dimensions: (u32, u32),
    status_buffer: Buffer,
    command_buffer: CommandBuffer,
}

#[cfg(target_os = "macos")]
fn private_jpeg_tile_from_fast_rgb_buffer(
    decoded: FastRgbDecodeBuffer,
) -> Result<crate::ResidentPrivateJpegTile, Error> {
    crate::ResidentPrivateJpegTile::new(
        decoded.buffer,
        0,
        decoded.dimensions,
        PixelFormat::Rgb8,
        decoded.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel(),
        decoded.status_buffer,
        decoded.command_buffer,
    )
}

#[cfg(target_os = "macos")]
thread_local! {
    static DEFAULT_METAL_SESSION: RefCell<Option<Result<crate::MetalBackendSession, MetalSupportError>>> = const { RefCell::new(None) };
}

#[cfg(target_os = "macos")]
pub(crate) struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    pack_pipeline: ComputePipelineState,
    jpeg_baseline_encode_pipeline: ComputePipelineState,
    jpeg_baseline_encode_batch_pipeline: ComputePipelineState,
    pack_420_pipeline: ComputePipelineState,
    pack_420_rgb_pipeline: ComputePipelineState,
    pack_420_rgba_pipeline: ComputePipelineState,
    pack_420_rgb_batch_pipeline: ComputePipelineState,
    pack_420_rgba_texture_pipeline: ComputePipelineState,
    pack_420_windowed_rgb_batch_pipeline: ComputePipelineState,
    pack_420_windowed_rgba_texture_pipeline: ComputePipelineState,
    pack_422_rgb_pipeline: ComputePipelineState,
    pack_422_rgba_pipeline: ComputePipelineState,
    pack_422_rgb_batch_pipeline: ComputePipelineState,
    pack_422_rgba_texture_pipeline: ComputePipelineState,
    pack_422_windowed_rgb_batch_pipeline: ComputePipelineState,
    pack_422_windowed_rgba_texture_pipeline: ComputePipelineState,
    pack_444_rgb_batch_pipeline: ComputePipelineState,
    pack_444_rgba_texture_pipeline: ComputePipelineState,
    pack_422_windowed_pipeline: ComputePipelineState,
    pack_422_windowed_rgb_pipeline: ComputePipelineState,
    pack_422_windowed_rgba_pipeline: ComputePipelineState,
    pack_420_windowed_pipeline: ComputePipelineState,
    pack_420_windowed_rgb_pipeline: ComputePipelineState,
    pack_420_windowed_rgba_pipeline: ComputePipelineState,
    fast420_decode_pipeline: ComputePipelineState,
    fast420_batch_decode_pipeline: ComputePipelineState,
    #[cfg(test)]
    fast420_batch_coeffs_decode_pipeline: ComputePipelineState,
    #[cfg(test)]
    fast420_batch_idct_deposit_pipeline: ComputePipelineState,
    fast420_scaled_region_batch_decode_pipeline: ComputePipelineState,
    fast420_rgba_texture_batch_decode_pipeline: ComputePipelineState,
    fast420_rgba_texture_boundary_pipeline: ComputePipelineState,
    fast420_rgba_texture_vertical_boundary_pipeline: ComputePipelineState,
    fast420_rgba_texture_corner_pipeline: ComputePipelineState,
    fast422_decode_pipeline: ComputePipelineState,
    fast422_batch_decode_pipeline: ComputePipelineState,
    fast422_scaled_region_batch_decode_pipeline: ComputePipelineState,
    fast422_rgba_texture_batch_decode_pipeline: ComputePipelineState,
    fast422_rgba_texture_boundary_pipeline: ComputePipelineState,
    fast422_region_decode_pipeline: ComputePipelineState,
    fast422_scaled_decode_pipeline: ComputePipelineState,
    fast422_scaled_region_decode_pipeline: ComputePipelineState,
    fast420_region_decode_pipeline: ComputePipelineState,
    fast420_scaled_decode_pipeline: ComputePipelineState,
    fast420_scaled_region_decode_pipeline: ComputePipelineState,
    fast444_decode_pipeline: ComputePipelineState,
    fast444_region_decode_pipeline: ComputePipelineState,
    fast444_scaled_decode_pipeline: ComputePipelineState,
    fast444_scaled_region_decode_pipeline: ComputePipelineState,
    fast444_scaled_region_batch_decode_pipeline: ComputePipelineState,
    fast444_rgba_texture_batch_decode_pipeline: ComputePipelineState,
    rgb8_to_rgba_texture_pipeline: ComputePipelineState,
    batch_scratch: Mutex<MetalBatchScratch>,
    viewport_plane_cache: Mutex<Option<CachedViewportPlanes>>,
    viewport_plane_cache_gate: Arc<ViewportPlaneCacheGate>,
}

#[cfg(target_os = "macos")]
impl MetalRuntime {
    #[cfg(test)]
    fn new() -> Result<Self, MetalSupportError> {
        let device = system_default_device()?;
        Self::new_with_device(device)
    }

    pub(crate) fn new_with_device(device: Device) -> Result<Self, MetalSupportError> {
        let loader = MetalPipelineLoader::new(&device, SHADER_SOURCE)?;
        let pipeline = |name: &str| loader.pipeline(name);
        let queue = checked_command_queue(&device)?;
        Ok(Self {
            device,
            queue,
            pack_pipeline: pipeline("jpeg_pack")?,
            jpeg_baseline_encode_pipeline: pipeline("jpeg_encode_baseline_entropy")?,
            jpeg_baseline_encode_batch_pipeline: pipeline("jpeg_encode_baseline_entropy_batch")?,
            pack_420_pipeline: pipeline("jpeg_pack_420")?,
            pack_420_rgb_pipeline: pipeline("jpeg_pack_420_rgb")?,
            pack_420_rgba_pipeline: pipeline("jpeg_pack_420_rgba")?,
            pack_420_rgb_batch_pipeline: pipeline("jpeg_pack_420_rgb_batch")?,
            pack_420_rgba_texture_pipeline: pipeline("jpeg_pack_420_rgba_texture")?,
            pack_420_windowed_rgb_batch_pipeline: pipeline("jpeg_pack_420_windowed_rgb_batch")?,
            pack_420_windowed_rgba_texture_pipeline: pipeline(
                "jpeg_pack_420_windowed_rgba_texture",
            )?,
            pack_422_rgb_pipeline: pipeline("jpeg_pack_422_rgb")?,
            pack_422_rgba_pipeline: pipeline("jpeg_pack_422_rgba")?,
            pack_422_rgb_batch_pipeline: pipeline("jpeg_pack_422_rgb_batch")?,
            pack_422_rgba_texture_pipeline: pipeline("jpeg_pack_422_rgba_texture")?,
            pack_422_windowed_rgb_batch_pipeline: pipeline("jpeg_pack_422_windowed_rgb_batch")?,
            pack_422_windowed_rgba_texture_pipeline: pipeline(
                "jpeg_pack_422_windowed_rgba_texture",
            )?,
            pack_444_rgb_batch_pipeline: pipeline("jpeg_pack_444_rgb_batch")?,
            pack_444_rgba_texture_pipeline: pipeline("jpeg_pack_444_rgba_texture")?,
            pack_422_windowed_pipeline: pipeline("jpeg_pack_422_windowed")?,
            pack_422_windowed_rgb_pipeline: pipeline("jpeg_pack_422_windowed_rgb")?,
            pack_422_windowed_rgba_pipeline: pipeline("jpeg_pack_422_windowed_rgba")?,
            pack_420_windowed_pipeline: pipeline("jpeg_pack_420_windowed")?,
            pack_420_windowed_rgb_pipeline: pipeline("jpeg_pack_420_windowed_rgb")?,
            pack_420_windowed_rgba_pipeline: pipeline("jpeg_pack_420_windowed_rgba")?,
            fast420_decode_pipeline: pipeline("jpeg_decode_fast420")?,
            fast420_batch_decode_pipeline: pipeline("jpeg_decode_fast420_batch")?,
            #[cfg(test)]
            fast420_batch_coeffs_decode_pipeline: pipeline("jpeg_decode_fast420_batch_coeffs")?,
            #[cfg(test)]
            fast420_batch_idct_deposit_pipeline: pipeline("jpeg_idct_deposit_fast420_batch")?,
            fast420_scaled_region_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast420_scaled_region_batch",
            )?,
            fast420_rgba_texture_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast420_rgba_texture_batch",
            )?,
            fast420_rgba_texture_boundary_pipeline: pipeline(
                "jpeg_resolve_fast420_rgba_texture_boundaries",
            )?,
            fast420_rgba_texture_vertical_boundary_pipeline: pipeline(
                "jpeg_resolve_fast420_rgba_texture_vertical_boundaries",
            )?,
            fast420_rgba_texture_corner_pipeline: pipeline(
                "jpeg_resolve_fast420_rgba_texture_corners",
            )?,
            fast422_decode_pipeline: pipeline("jpeg_decode_fast422")?,
            fast422_batch_decode_pipeline: pipeline("jpeg_decode_fast422_batch")?,
            fast422_scaled_region_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast422_scaled_region_batch",
            )?,
            fast422_rgba_texture_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast422_rgba_texture_batch",
            )?,
            fast422_rgba_texture_boundary_pipeline: pipeline(
                "jpeg_resolve_fast422_rgba_texture_boundaries",
            )?,
            fast422_region_decode_pipeline: pipeline("jpeg_decode_fast422_region")?,
            fast422_scaled_decode_pipeline: pipeline("jpeg_decode_fast422_scaled")?,
            fast422_scaled_region_decode_pipeline: pipeline("jpeg_decode_fast422_scaled_region")?,
            fast420_region_decode_pipeline: pipeline("jpeg_decode_fast420_region")?,
            fast420_scaled_decode_pipeline: pipeline("jpeg_decode_fast420_scaled")?,
            fast420_scaled_region_decode_pipeline: pipeline("jpeg_decode_fast420_scaled_region")?,
            fast444_decode_pipeline: pipeline("jpeg_decode_fast444")?,
            fast444_region_decode_pipeline: pipeline("jpeg_decode_fast444_region")?,
            fast444_scaled_decode_pipeline: pipeline("jpeg_decode_fast444_scaled")?,
            fast444_scaled_region_decode_pipeline: pipeline("jpeg_decode_fast444_scaled_region")?,
            fast444_scaled_region_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast444_scaled_region_batch",
            )?,
            fast444_rgba_texture_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast444_rgba_texture_batch",
            )?,
            rgb8_to_rgba_texture_pipeline: pipeline("jpeg_copy_rgb8_to_rgba_texture")?,
            batch_scratch: Mutex::new(MetalBatchScratch::default()),
            viewport_plane_cache: Mutex::new(None),
            viewport_plane_cache_gate: ViewportPlaneCacheGate::new(),
        })
    }

    fn batch_scratch(&self) -> Result<MutexGuard<'_, MetalBatchScratch>, Error> {
        self.batch_scratch
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "JPEG Metal batch scratch",
            })
    }

    fn viewport_plane_cache(&self) -> Result<MutexGuard<'_, Option<CachedViewportPlanes>>, Error> {
        self.viewport_plane_cache
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "JPEG Metal viewport plane cache",
            })
    }

    fn viewport_plane_cache_lease(&self) -> Result<ViewportPlaneCacheLease, Error> {
        self.viewport_plane_cache_gate.acquire()
    }

    #[cfg(test)]
    fn viewport_plane_cache_id_for_test(&self) -> Result<Option<usize>, Error> {
        Ok(self
            .viewport_plane_cache()?
            .as_ref()
            .map(|cached| cached.plane0.as_ptr() as usize))
    }
}

#[cfg(target_os = "macos")]
fn pack_420_pipeline_for_format(runtime: &MetalRuntime, fmt: PixelFormat) -> &ComputePipelineState {
    match fmt {
        PixelFormat::Rgb8 => &runtime.pack_420_rgb_pipeline,
        PixelFormat::Rgba8 => &runtime.pack_420_rgba_pipeline,
        _ => &runtime.pack_420_pipeline,
    }
}

#[cfg(target_os = "macos")]
fn pack_420_windowed_pipeline_for_format(
    runtime: &MetalRuntime,
    fmt: PixelFormat,
) -> &ComputePipelineState {
    match fmt {
        PixelFormat::Rgb8 => &runtime.pack_420_windowed_rgb_pipeline,
        PixelFormat::Rgba8 => &runtime.pack_420_windowed_rgba_pipeline,
        _ => &runtime.pack_420_windowed_pipeline,
    }
}

#[cfg(target_os = "macos")]
fn pack_422_pipeline_for_format(
    runtime: &MetalRuntime,
    fmt: PixelFormat,
) -> Option<&ComputePipelineState> {
    match fmt {
        PixelFormat::Rgb8 => Some(&runtime.pack_422_rgb_pipeline),
        PixelFormat::Rgba8 => Some(&runtime.pack_422_rgba_pipeline),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn pack_422_windowed_pipeline_for_format(
    runtime: &MetalRuntime,
    fmt: PixelFormat,
) -> &ComputePipelineState {
    match fmt {
        PixelFormat::Rgb8 => &runtime.pack_422_windowed_rgb_pipeline,
        PixelFormat::Rgba8 => &runtime.pack_422_windowed_rgba_pipeline,
        _ => &runtime.pack_422_windowed_pipeline,
    }
}

#[cfg(target_os = "macos")]
fn with_runtime<R>(f: impl FnOnce(&MetalRuntime) -> Result<R, Error>) -> Result<R, Error> {
    DEFAULT_METAL_SESSION.with(|session| {
        let mut session = session.borrow_mut();
        if session.is_none() {
            *session = Some(
                j2k_metal_support::system_default_device().map(crate::MetalBackendSession::new),
            );
        }
        let Some(session) = session.as_ref() else {
            return Err(Error::MetalRuntime {
                message: "JPEG Metal default session was not initialized".to_string(),
            });
        };
        match session {
            Ok(session) => with_runtime_for_session(session, f),
            Err(error) => Err(runtime_initialization_error(error)),
        }
    })
}

#[cfg(target_os = "macos")]
fn with_runtime_for_session<R>(
    session: &crate::MetalBackendSession,
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    match session.runtime_result() {
        Ok(runtime) => f(runtime),
        Err(error) => Err(runtime_initialization_error(error)),
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn runtime_initialization_error(error: &MetalSupportError) -> Error {
    metal_runtime_support_error(error)
}

#[cfg(target_os = "macos")]
pub(super) fn commit_and_wait_jpeg(command_buffer: &CommandBufferRef) -> Result<(), Error> {
    commit_and_wait(command_buffer)
        .map_err(|error| metal_kernel_support_error(error.to_string(), error))
}

#[cfg(target_os = "macos")]
fn wait_for_completion_jpeg(command_buffer: &CommandBufferRef) -> Result<(), Error> {
    wait_for_completion(command_buffer)
        .map_err(|error| metal_kernel_support_error(error.to_string(), error))
}

#[cfg(all(test, target_os = "macos"))]
mod tests;
