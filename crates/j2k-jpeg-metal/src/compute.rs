// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names)]

#[cfg(all(target_os = "macos", test))]
use j2k_metal_support::system_default_device;
#[cfg(all(target_os = "macos", test))]
use metal::foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};

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
#[cfg(target_os = "macos")]
use j2k_metal_support::{
    checked_command_queue, commit_and_wait, wait_for_completion, MetalPipelineLoader,
    MetalSupportError,
};
#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device,
    MTLPixelFormat, MTLResourceOptions, MTLSize,
};

#[cfg(target_os = "macos")]
pub(crate) use crate::abi::*;

#[cfg(target_os = "macos")]
use crate::buffers::{
    checked_buffer_read, checked_buffer_slice, new_decode_plane_buffer, new_private_buffer,
    new_shared_buffer_with_data, MetalBatchScratch,
};
use crate::{batch, Error, JpegFastPackets, Surface};

mod batch_plan;
#[cfg(target_os = "macos")]
mod batch_support;
#[cfg(target_os = "macos")]
mod kernel_helpers;
#[cfg(target_os = "macos")]
mod region_scaled_plan;
#[cfg(target_os = "macos")]
mod status;
mod viewport_cache;
#[cfg(target_os = "macos")]
mod viewport_compose;
use self::batch_plan::{
    batched_fast_packets, core_rect_to_jpeg, BatchDeviceBufferCache, BatchedDecodeItem,
    BatchedFastPacket,
};
#[cfg(target_os = "macos")]
use self::batch_support::{
    batch_entropy_buffers, batch_entropy_host_data, fast420_batch_timing_enabled,
    fast_batch_decode_mode, region_scaled_batch_error_results, surface_batch_error_results,
    surface_batch_success_results, texture_batch_error_results, BatchEntropyBufferKeys,
    BatchEntropyBuffers, BatchEntropyHostData, BatchEntropyLabels, FastBatchDecodeMode,
    FastBatchTiming,
};
#[cfg(all(test, target_os = "macos"))]
use self::batch_support::{fast420_batch_timing_value_enabled, fast420_batch_timing_value_mode};
#[cfg(all(test, target_os = "macos"))]
use self::kernel_helpers::choose_1d_threadgroup_width;
#[cfg(target_os = "macos")]
use self::kernel_helpers::{
    bind_fast_decode_entropy_inputs, bind_three_plane_pack, dispatch_1d_pipeline,
    dispatch_2d_pipeline, dispatch_3d_pipeline, fast_packet_huffman_tables, packed_pair_extent,
    pixel_format_to_out_format, plane_mode_to_u32, FastDecodeEntropyInputs,
};
#[cfg(target_os = "macos")]
use self::region_scaled_plan::{
    fast444_packets_share_region_scaled_batch_shape, fast444_region_scaled_batch_groups,
    fast_subsampled_full_rgb_batch_groups, fast_subsampled_packets_share_full_rgb_batch_shape,
    fast_subsampled_region_scaled_batch_groups, fast_subsampled_region_scaled_batch_plan,
    windowed_texture_pack_params, RegionScaledBatchPlan,
};
#[cfg(target_os = "macos")]
use self::status::{
    decode_error_from_cpu, decode_status_buffer, fast422_status_error, first_decode_error_status,
    jpeg_baseline_encode_status_error,
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
) -> crate::ResidentPrivateJpegTile {
    crate::ResidentPrivateJpegTile {
        buffer: decoded.buffer,
        byte_offset: 0,
        dimensions: decoded.dimensions,
        pixel_format: PixelFormat::Rgb8,
        pitch_bytes: decoded.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel(),
        status_buffer: decoded.status_buffer,
        command_buffer: decoded.command_buffer,
    }
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
    if error.is_unavailable() {
        Error::MetalUnavailable
    } else {
        Error::MetalRuntime {
            message: error.to_string(),
        }
    }
}

#[cfg(target_os = "macos")]
pub(super) fn commit_and_wait_jpeg(command_buffer: &CommandBufferRef) -> Result<(), Error> {
    commit_and_wait(command_buffer).map_err(|error| Error::MetalKernel {
        message: error.to_string(),
    })
}

#[cfg(target_os = "macos")]
fn wait_for_completion_jpeg(command_buffer: &CommandBufferRef) -> Result<(), Error> {
    wait_for_completion(command_buffer).map_err(|error| Error::MetalKernel {
        message: error.to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_jpeg_baseline_entropy_with_session(
    session: &crate::MetalBackendSession,
    job: &JpegBaselineEntropyEncodeJob<'_>,
) -> Result<Vec<u8>, Error> {
    with_runtime_for_session(session, |runtime| {
        let entropy_buffer = runtime.device.new_buffer(
            job.entropy_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status = JpegBaselineEncodeStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<JpegBaselineEncodeStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.jpeg_baseline_encode_pipeline);
        encoder.set_buffer(0, Some(job.input), job.input_offset as u64);
        encoder.set_buffer(1, Some(&entropy_buffer), 0);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<JpegBaselineEncodeParams>() as u64,
            (&raw const job.params).cast(),
        );
        encoder.set_bytes(
            4,
            size_of_val(&job.q_luma) as u64,
            job.q_luma.as_ptr().cast(),
        );
        encoder.set_bytes(
            5,
            size_of_val(&job.q_chroma) as u64,
            job.q_chroma.as_ptr().cast(),
        );
        encoder.set_bytes(
            6,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_luma).cast(),
        );
        encoder.set_bytes(
            7,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_luma).cast(),
        );
        encoder.set_bytes(
            8,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_chroma).cast(),
        );
        encoder.set_bytes(
            9,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_chroma).cast(),
        );
        encoder.dispatch_threads(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait_jpeg(command_buffer)?;

        let status = checked_buffer_read::<JpegBaselineEncodeStatus>(
            &status_buffer,
            "baseline encode status",
        )?;
        if status.code != JPEG_BASELINE_ENCODE_STATUS_OK {
            return Err(jpeg_baseline_encode_status_error(status));
        }
        let entropy_len = usize::try_from(status.entropy_len).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal encode entropy length exceeds usize".to_string(),
        })?;
        if entropy_len > job.entropy_capacity {
            return Err(Error::MetalKernel {
                message: "JPEG Baseline Metal encode reported length exceeds output capacity"
                    .to_string(),
            });
        }
        let entropy =
            checked_buffer_slice::<u8>(&entropy_buffer, entropy_len, "baseline encode entropy")?;
        Ok(entropy)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_jpeg_baseline_entropy_batch_with_session(
    session: &crate::MetalBackendSession,
    job: &JpegBaselineEntropyEncodeBatchJob<'_>,
) -> Result<Vec<Vec<u8>>, Error> {
    if job.params.is_empty() {
        return Ok(Vec::new());
    }
    with_runtime_for_session(session, |runtime| {
        let entropy_buffer = runtime.device.new_buffer(
            job.entropy_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let statuses = vec![JpegBaselineEncodeStatus::default(); job.params.len()];
        let status_buffer = runtime.device.new_buffer_with_data(
            statuses.as_ptr().cast(),
            size_of::<JpegBaselineEncodeStatus>() as u64 * statuses.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params_buffer = runtime.device.new_buffer_with_data(
            job.params.as_ptr().cast(),
            size_of::<JpegBaselineEncodeParams>() as u64 * job.params.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tile_count = u32::try_from(job.params.len()).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal batch tile count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.jpeg_baseline_encode_batch_pipeline);
        encoder.set_buffer(0, Some(job.input), 0);
        encoder.set_buffer(1, Some(&entropy_buffer), 0);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_buffer(3, Some(&params_buffer), 0);
        encoder.set_bytes(
            4,
            size_of_val(&job.q_luma) as u64,
            job.q_luma.as_ptr().cast(),
        );
        encoder.set_bytes(
            5,
            size_of_val(&job.q_chroma) as u64,
            job.q_chroma.as_ptr().cast(),
        );
        encoder.set_bytes(
            6,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_luma).cast(),
        );
        encoder.set_bytes(
            7,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_luma).cast(),
        );
        encoder.set_bytes(
            8,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_chroma).cast(),
        );
        encoder.set_bytes(
            9,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_chroma).cast(),
        );
        encoder.set_bytes(10, size_of::<u32>() as u64, (&raw const tile_count).cast());
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(tile_count),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait_jpeg(command_buffer)?;

        let status_slice = checked_buffer_slice::<JpegBaselineEncodeStatus>(
            &status_buffer,
            job.params.len(),
            "baseline batch encode statuses",
        )?;
        let entropy_bytes = checked_buffer_slice::<u8>(
            &entropy_buffer,
            job.entropy_capacity,
            "baseline batch encode entropy",
        )?;
        let mut out = Vec::with_capacity(job.params.len());
        for (status, params) in status_slice.iter().copied().zip(job.params.iter()) {
            if status.code != JPEG_BASELINE_ENCODE_STATUS_OK {
                return Err(jpeg_baseline_encode_status_error(status));
            }
            let entropy_len =
                usize::try_from(status.entropy_len).map_err(|_| Error::MetalKernel {
                    message: "JPEG Baseline Metal encode entropy length exceeds usize".to_string(),
                })?;
            let offset =
                usize::try_from(params.entropy_offset_bytes).map_err(|_| Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy offset exceeds usize".to_string(),
                })?;
            let capacity =
                usize::try_from(params.entropy_capacity).map_err(|_| Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy capacity exceeds usize".to_string(),
                })?;
            if entropy_len > capacity {
                return Err(Error::MetalKernel {
                    message:
                        "JPEG Baseline Metal encode reported length exceeds tile output capacity"
                            .to_string(),
                });
            }
            let end = offset
                .checked_add(entropy_len)
                .ok_or_else(|| Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy range overflow".to_string(),
                })?;
            if end > entropy_bytes.len() {
                return Err(Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy range exceeds buffer".to_string(),
                });
            }
            out.push(entropy_bytes[offset..end].to_vec());
        }
        Ok(out)
    })
}

include!("compute/fast_packets_impl.rs");

include!("compute/pack_dispatch_impl.rs");

include!("compute/batch_decode_full.rs");

include!("compute/batch_decode_region.rs");

include!("compute/batch_decode_entry.rs");

include!("compute/single_decode_impl.rs");

#[cfg(all(test, target_os = "macos"))]
mod tests;
