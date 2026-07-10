// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use super::super::{
    checked_u32, dispatch_1d_pipeline, fast420_batch_timing_enabled, pack_420_pipeline_for_format,
    pack_420_windowed_pipeline_for_format, pack_422_pipeline_for_format,
    pack_422_windowed_pipeline_for_format, BatchedFastPacket, ComputePipelineState, Error,
    JpegFast420PacketV1, JpegFast420TextureBatchParams, JpegFast422PacketV1, JpegFast444PacketV1,
    MetalRuntime, PixelFormat, PlaneMode, FAST420_TEXTURE_BOUNDARY_META_WORDS,
    FAST420_TEXTURE_BOUNDARY_SAMPLE_BYTES, FAST420_TEXTURE_VERTICAL_META_WORDS,
    FAST420_TEXTURE_VERTICAL_SAMPLE_BYTES, FAST422_TEXTURE_BOUNDARY_META_WORDS,
    FAST422_TEXTURE_BOUNDARY_SAMPLE_BYTES,
};
use super::descriptors::{
    FastRegionScaledMetal, FastScratchKeys, FastSubsampledMetal, FastTextureRepairCtx,
    FastVerticalRepairSpec,
};

#[cfg(target_os = "macos")]
impl FastSubsampledMetal for JpegFast420PacketV1 {
    const REGION_SCALED_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast420_region_scaled_y",
        cb: "fast420_region_scaled_cb",
        cr: "fast420_region_scaled_cr",
        entropy: "fast420_region_scaled_entropy",
        entropy_offsets: "fast420_region_scaled_entropy_offsets",
        entropy_lens: "fast420_region_scaled_entropy_lens",
        entropy_checkpoints: "fast420_region_scaled_entropy_checkpoints",
        status: "fast420_region_scaled_status",
    };
    const REGION_SCALED_TEXTURE_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast420_region_scaled_texture_y",
        cb: "fast420_region_scaled_texture_cb",
        cr: "fast420_region_scaled_texture_cr",
        entropy: "fast420_region_scaled_texture_entropy",
        entropy_offsets: "fast420_region_scaled_texture_entropy_offsets",
        entropy_lens: "fast420_region_scaled_texture_entropy_lens",
        entropy_checkpoints: "fast420_region_scaled_texture_entropy_checkpoints",
        status: "fast420_region_scaled_texture_status",
    };

    const FULL_BATCH_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast420_full_y",
        cb: "fast420_full_cb",
        cr: "fast420_full_cr",
        entropy: "fast420_full_entropy",
        entropy_offsets: "fast420_full_entropy_offsets",
        entropy_lens: "fast420_full_entropy_lens",
        entropy_checkpoints: "fast420_full_entropy_checkpoints",
        status: "fast420_full_status",
    };
    const TEXTURE_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast420_texture_y",
        cb: "fast420_texture_cb",
        cr: "fast420_texture_cr",
        entropy: "fast420_texture_entropy",
        entropy_offsets: "fast420_texture_entropy_offsets",
        entropy_lens: "fast420_texture_entropy_lens",
        entropy_checkpoints: "fast420_texture_entropy_checkpoints",
        status: "fast420_texture_status",
    };
    const FULL_RGB_BATCH_TIMING_TAG: &'static str = "metal_fast420_batch";
    const TEXTURE_BOUNDARY_META_WORDS: usize = FAST420_TEXTURE_BOUNDARY_META_WORDS;
    const TEXTURE_BOUNDARY_SAMPLE_BYTES: usize = FAST420_TEXTURE_BOUNDARY_SAMPLE_BYTES;
    const TEXTURE_BOUNDARY_META_KEY: &'static str = "fast420_texture_boundary_meta";
    const TEXTURE_BOUNDARY_SAMPLES_KEY: &'static str = "fast420_texture_boundary_samples";
    const TEXTURE_VERTICAL_REPAIR: Option<FastVerticalRepairSpec> = Some(FastVerticalRepairSpec {
        meta_words: FAST420_TEXTURE_VERTICAL_META_WORDS,
        sample_bytes: FAST420_TEXTURE_VERTICAL_SAMPLE_BYTES,
        meta_key: "fast420_texture_vertical_meta",
        samples_key: "fast420_texture_vertical_samples",
    });
    const USE_FAST444_TEXTURE_PARAMS: bool = false;

    fn from_batched<'a>(packet: &BatchedFastPacket<'a>) -> Option<&'a Self> {
        match packet {
            BatchedFastPacket::Fast420(packet) => Some(packet),
            _ => None,
        }
    }
    fn to_batched(&self) -> BatchedFastPacket<'_> {
        BatchedFastPacket::Fast420(self)
    }
    fn decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_decode_pipeline
    }
    fn region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_region_decode_pipeline
    }
    fn scaled_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_scaled_decode_pipeline
    }
    fn scaled_region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_scaled_region_decode_pipeline
    }
    fn pack_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Option<&ComputePipelineState> {
        Some(pack_420_pipeline_for_format(runtime, fmt))
    }
    fn pack_windowed_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> &ComputePipelineState {
        pack_420_windowed_pipeline_for_format(runtime, fmt)
    }
    fn scaled_region_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_scaled_region_batch_decode_pipeline
    }
    fn pack_windowed_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_420_windowed_rgba_texture_pipeline
    }
    fn full_rgb_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_batch_decode_pipeline
    }
    fn pack_full_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_420_rgb_batch_pipeline
    }
    fn pack_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_420_rgba_texture_pipeline
    }
    fn rgba_texture_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_rgba_texture_batch_decode_pipeline
    }
    fn rgba_texture_boundary_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_rgba_texture_boundary_pipeline
    }
    fn full_rgb_batch_timing_enabled() -> bool {
        fast420_batch_timing_enabled()
    }
    fn texture_mcu_dispatch_threads(total_mcus: usize) -> Result<Option<u32>, Error> {
        checked_u32(total_mcus, "fast420 texture batch MCU count").map(Some)
    }
    fn texture_repair_record_count(
        tile_count: usize,
        total_mcus: usize,
        _total_decode_threads: u32,
    ) -> Result<usize, Error> {
        tile_count
            .checked_mul(total_mcus)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal fast420 texture repair record count overflowed".to_string(),
            })
    }
    fn horizontal_repair_threads(
        first: &Self,
        _segment_count_u32: u32,
        mcu_threads: Option<u32>,
    ) -> Option<u32> {
        mcu_threads.filter(|_| first.mcus_per_row > 1)
    }
    fn encode_extra_texture_repair_passes(
        runtime: &MetalRuntime,
        ctx: &FastTextureRepairCtx<'_>,
    ) -> Result<(), Error> {
        let (vertical_meta_buffer, vertical_samples_buffer) =
            ctx.vertical_buffers.ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal fast420 texture vertical repair scratch was missing"
                    .to_string(),
            })?;
        let Some(mcu_threads) = ctx.mcu_threads else {
            return Err(Error::MetalKernel {
                message: "JPEG Metal fast420 texture MCU dispatch width was missing".to_string(),
            });
        };
        if ctx.decode_params.mcu_rows > 1 {
            for index in 0..ctx.tile_count {
                let texture =
                    ctx.output
                        .texture_trusted(index)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "JPEG Metal batch texture output slot was missing".to_string(),
                        })?;
                let decode_params = JpegFast420TextureBatchParams {
                    tile_index: checked_u32(index, ctx.tile_index_ctx)?,
                    ..ctx.decode_params
                };
                let boundary_encoder = ctx.command_buffer.new_compute_command_encoder();
                boundary_encoder.set_compute_pipeline_state(
                    &runtime.fast420_rgba_texture_vertical_boundary_pipeline,
                );
                boundary_encoder.set_buffer(0, Some(vertical_meta_buffer), 0);
                boundary_encoder.set_buffer(1, Some(vertical_samples_buffer), 0);
                boundary_encoder.set_bytes(
                    2,
                    size_of::<JpegFast420TextureBatchParams>() as u64,
                    (&raw const decode_params).cast(),
                );
                boundary_encoder.set_texture(0, Some(texture));
                dispatch_1d_pipeline(
                    boundary_encoder,
                    &runtime.fast420_rgba_texture_vertical_boundary_pipeline,
                    mcu_threads,
                );
                boundary_encoder.end_encoding();
            }
        }
        if ctx.decode_params.mcus_per_row > 1 && ctx.decode_params.mcu_rows > 1 {
            for index in 0..ctx.tile_count {
                let texture =
                    ctx.output
                        .texture_trusted(index)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "JPEG Metal batch texture output slot was missing".to_string(),
                        })?;
                let decode_params = JpegFast420TextureBatchParams {
                    tile_index: checked_u32(index, ctx.tile_index_ctx)?,
                    ..ctx.decode_params
                };
                let corner_encoder = ctx.command_buffer.new_compute_command_encoder();
                corner_encoder
                    .set_compute_pipeline_state(&runtime.fast420_rgba_texture_corner_pipeline);
                corner_encoder.set_buffer(0, Some(ctx.boundary_meta_buffer), 0);
                corner_encoder.set_buffer(1, Some(vertical_meta_buffer), 0);
                corner_encoder.set_buffer(2, Some(vertical_samples_buffer), 0);
                corner_encoder.set_bytes(
                    3,
                    size_of::<JpegFast420TextureBatchParams>() as u64,
                    (&raw const decode_params).cast(),
                );
                corner_encoder.set_texture(0, Some(texture));
                dispatch_1d_pipeline(
                    corner_encoder,
                    &runtime.fast420_rgba_texture_corner_pipeline,
                    mcu_threads,
                );
                corner_encoder.end_encoding();
            }
        }
        Ok(())
    }
    #[cfg(test)]
    fn split_coeff_idct_pipelines(
        runtime: &MetalRuntime,
    ) -> Option<(&ComputePipelineState, &ComputePipelineState)> {
        Some((
            &runtime.fast420_batch_coeffs_decode_pipeline,
            &runtime.fast420_batch_idct_deposit_pipeline,
        ))
    }
    #[cfg(test)]
    const SPLIT_TEXTURE_SCRATCH_KEYS: (&'static str, &'static str) = (
        "fast420_texture_coeff_blocks",
        "fast420_texture_dc_only_flags",
    );
}

#[cfg(target_os = "macos")]
impl FastSubsampledMetal for JpegFast422PacketV1 {
    const REGION_SCALED_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast422_region_scaled_y",
        cb: "fast422_region_scaled_cb",
        cr: "fast422_region_scaled_cr",
        entropy: "fast422_region_scaled_entropy",
        entropy_offsets: "fast422_region_scaled_entropy_offsets",
        entropy_lens: "fast422_region_scaled_entropy_lens",
        entropy_checkpoints: "fast422_region_scaled_entropy_checkpoints",
        status: "fast422_region_scaled_status",
    };
    const REGION_SCALED_TEXTURE_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast422_region_scaled_texture_y",
        cb: "fast422_region_scaled_texture_cb",
        cr: "fast422_region_scaled_texture_cr",
        entropy: "fast422_region_scaled_texture_entropy",
        entropy_offsets: "fast422_region_scaled_texture_entropy_offsets",
        entropy_lens: "fast422_region_scaled_texture_entropy_lens",
        entropy_checkpoints: "fast422_region_scaled_texture_entropy_checkpoints",
        status: "fast422_region_scaled_texture_status",
    };

    const FULL_BATCH_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast422_full_y",
        cb: "fast422_full_cb",
        cr: "fast422_full_cr",
        entropy: "fast422_full_entropy",
        entropy_offsets: "fast422_full_entropy_offsets",
        entropy_lens: "fast422_full_entropy_lens",
        entropy_checkpoints: "fast422_full_entropy_checkpoints",
        status: "fast422_full_status",
    };
    const TEXTURE_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast422_texture_y",
        cb: "fast422_texture_cb",
        cr: "fast422_texture_cr",
        entropy: "fast422_texture_entropy",
        entropy_offsets: "fast422_texture_entropy_offsets",
        entropy_lens: "fast422_texture_entropy_lens",
        entropy_checkpoints: "fast422_texture_entropy_checkpoints",
        status: "fast422_texture_status",
    };
    const FULL_RGB_BATCH_TIMING_TAG: &'static str = "metal_fast422_batch";
    const TEXTURE_BOUNDARY_META_WORDS: usize = FAST422_TEXTURE_BOUNDARY_META_WORDS;
    const TEXTURE_BOUNDARY_SAMPLE_BYTES: usize = FAST422_TEXTURE_BOUNDARY_SAMPLE_BYTES;
    const TEXTURE_BOUNDARY_META_KEY: &'static str = "fast422_texture_boundary_meta";
    const TEXTURE_BOUNDARY_SAMPLES_KEY: &'static str = "fast422_texture_boundary_samples";
    const TEXTURE_VERTICAL_REPAIR: Option<FastVerticalRepairSpec> = None;
    const USE_FAST444_TEXTURE_PARAMS: bool = false;

    fn from_batched<'a>(packet: &BatchedFastPacket<'a>) -> Option<&'a Self> {
        match packet {
            BatchedFastPacket::Fast422(packet) => Some(packet),
            _ => None,
        }
    }
    fn to_batched(&self) -> BatchedFastPacket<'_> {
        BatchedFastPacket::Fast422(self)
    }
    fn decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_decode_pipeline
    }
    fn region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_region_decode_pipeline
    }
    fn scaled_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_scaled_decode_pipeline
    }
    fn scaled_region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_scaled_region_decode_pipeline
    }
    fn pack_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Option<&ComputePipelineState> {
        pack_422_pipeline_for_format(runtime, fmt)
    }
    fn pack_windowed_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> &ComputePipelineState {
        pack_422_windowed_pipeline_for_format(runtime, fmt)
    }
    fn scaled_region_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_scaled_region_batch_decode_pipeline
    }
    fn pack_windowed_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_422_windowed_rgba_texture_pipeline
    }
    fn full_rgb_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_batch_decode_pipeline
    }
    fn pack_full_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_422_rgb_batch_pipeline
    }
    fn pack_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_422_rgba_texture_pipeline
    }
    fn rgba_texture_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_rgba_texture_batch_decode_pipeline
    }
    fn rgba_texture_boundary_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_rgba_texture_boundary_pipeline
    }
    fn full_rgb_batch_timing_enabled() -> bool {
        false
    }
    fn texture_mcu_dispatch_threads(_total_mcus: usize) -> Result<Option<u32>, Error> {
        Ok(None)
    }
    fn texture_repair_record_count(
        _tile_count: usize,
        _total_mcus: usize,
        total_decode_threads: u32,
    ) -> Result<usize, Error> {
        Ok(total_decode_threads as usize)
    }
    fn horizontal_repair_threads(
        _first: &Self,
        segment_count_u32: u32,
        _mcu_threads: Option<u32>,
    ) -> Option<u32> {
        (segment_count_u32 > 1).then_some(segment_count_u32)
    }
    fn encode_extra_texture_repair_passes(
        _runtime: &MetalRuntime,
        _ctx: &FastTextureRepairCtx<'_>,
    ) -> Result<(), Error> {
        Ok(())
    }
    #[cfg(test)]
    fn split_coeff_idct_pipelines(
        _runtime: &MetalRuntime,
    ) -> Option<(&ComputePipelineState, &ComputePipelineState)> {
        None
    }
    #[cfg(test)]
    const SPLIT_TEXTURE_SCRATCH_KEYS: (&'static str, &'static str) = (
        "fast422_texture_coeff_blocks",
        "fast422_texture_dc_only_flags",
    );
}

#[cfg(target_os = "macos")]
impl FastSubsampledMetal for JpegFast444PacketV1 {
    const REGION_SCALED_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast444_region_scaled_y",
        cb: "fast444_region_scaled_cb",
        cr: "fast444_region_scaled_cr",
        entropy: "fast444_region_scaled_entropy",
        entropy_offsets: "fast444_region_scaled_entropy_offsets",
        entropy_lens: "fast444_region_scaled_entropy_lens",
        entropy_checkpoints: "fast444_region_scaled_entropy_checkpoints",
        status: "fast444_region_scaled_status",
    };
    const REGION_SCALED_TEXTURE_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast444_region_scaled_texture_y",
        cb: "fast444_region_scaled_texture_cb",
        cr: "fast444_region_scaled_texture_cr",
        entropy: "fast444_region_scaled_texture_entropy",
        entropy_offsets: "fast444_region_scaled_texture_entropy_offsets",
        entropy_lens: "fast444_region_scaled_texture_entropy_lens",
        entropy_checkpoints: "fast444_region_scaled_texture_entropy_checkpoints",
        status: "fast444_region_scaled_texture_status",
    };

    const FULL_BATCH_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast444_full_y",
        cb: "fast444_full_cb",
        cr: "fast444_full_cr",
        entropy: "fast444_full_entropy",
        entropy_offsets: "fast444_full_entropy_offsets",
        entropy_lens: "fast444_full_entropy_lens",
        entropy_checkpoints: "fast444_full_entropy_checkpoints",
        status: "fast444_full_status",
    };
    const TEXTURE_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast444_texture_y",
        cb: "fast444_texture_cb",
        cr: "fast444_texture_cr",
        entropy: "fast444_texture_entropy",
        entropy_offsets: "fast444_texture_entropy_offsets",
        entropy_lens: "fast444_texture_entropy_lens",
        entropy_checkpoints: "fast444_texture_entropy_checkpoints",
        status: "fast444_texture_status",
    };
    const FULL_RGB_BATCH_TIMING_TAG: &'static str = "metal_fast444_batch";
    const TEXTURE_BOUNDARY_META_WORDS: usize = 0;
    const TEXTURE_BOUNDARY_SAMPLE_BYTES: usize = 0;
    const TEXTURE_BOUNDARY_META_KEY: &'static str = "fast444_texture_boundary_meta";
    const TEXTURE_BOUNDARY_SAMPLES_KEY: &'static str = "fast444_texture_boundary_samples";
    const TEXTURE_VERTICAL_REPAIR: Option<FastVerticalRepairSpec> = None;
    const USE_FAST444_TEXTURE_PARAMS: bool = true;

    fn from_batched<'a>(packet: &BatchedFastPacket<'a>) -> Option<&'a Self> {
        match packet {
            BatchedFastPacket::Fast444(packet, _) => Some(packet),
            _ => None,
        }
    }
    fn to_batched(&self) -> BatchedFastPacket<'_> {
        BatchedFastPacket::Fast444(self, PlaneMode::YCbCr)
    }
    fn to_batched_with_texture_mode(&self, mode: PlaneMode) -> BatchedFastPacket<'_> {
        BatchedFastPacket::Fast444(self, mode)
    }
    fn texture_plane_mode_from_batched(packet: &BatchedFastPacket<'_>) -> Option<PlaneMode> {
        match packet {
            BatchedFastPacket::Fast444(_, mode) => Some(*mode),
            _ => None,
        }
    }
    fn decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast444_decode_pipeline
    }
    fn region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast444_region_decode_pipeline
    }
    fn scaled_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast444_scaled_decode_pipeline
    }
    fn scaled_region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast444_scaled_region_decode_pipeline
    }
    fn pack_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Option<&ComputePipelineState> {
        match fmt {
            PixelFormat::Rgb8 => Some(&runtime.pack_444_rgb_batch_pipeline),
            PixelFormat::Rgba8 => Some(&runtime.pack_444_rgba_texture_pipeline),
            _ => None,
        }
    }
    fn pack_windowed_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> &ComputePipelineState {
        match fmt {
            PixelFormat::Rgba8 => &runtime.pack_444_rgba_texture_pipeline,
            _ => &runtime.pack_444_rgb_batch_pipeline,
        }
    }
    fn scaled_region_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast444_scaled_region_batch_decode_pipeline
    }
    fn pack_windowed_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_444_rgba_texture_pipeline
    }
    fn full_rgb_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast444_scaled_region_batch_decode_pipeline
    }
    fn pack_full_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_444_rgb_batch_pipeline
    }
    fn pack_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_444_rgba_texture_pipeline
    }
    fn rgba_texture_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast444_rgba_texture_batch_decode_pipeline
    }
    fn rgba_texture_boundary_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast444_rgba_texture_batch_decode_pipeline
    }
    fn full_rgb_batch_timing_enabled() -> bool {
        false
    }
    fn texture_mcu_dispatch_threads(_total_mcus: usize) -> Result<Option<u32>, Error> {
        Ok(None)
    }
    fn texture_repair_record_count(
        _tile_count: usize,
        _total_mcus: usize,
        _total_decode_threads: u32,
    ) -> Result<usize, Error> {
        Ok(0)
    }
    fn horizontal_repair_threads(
        _first: &Self,
        _segment_count_u32: u32,
        _mcu_threads: Option<u32>,
    ) -> Option<u32> {
        None
    }
    fn encode_extra_texture_repair_passes(
        _runtime: &MetalRuntime,
        _ctx: &FastTextureRepairCtx<'_>,
    ) -> Result<(), Error> {
        Ok(())
    }
    #[cfg(test)]
    fn split_coeff_idct_pipelines(
        _runtime: &MetalRuntime,
    ) -> Option<(&ComputePipelineState, &ComputePipelineState)> {
        None
    }
    #[cfg(test)]
    const SPLIT_TEXTURE_SCRATCH_KEYS: (&'static str, &'static str) = (
        "fast444_texture_coeff_blocks",
        "fast444_texture_dc_only_flags",
    );
}

#[cfg(target_os = "macos")]
impl FastRegionScaledMetal for JpegFast420PacketV1 {
    const REGION_SCALED_KEYS: FastScratchKeys = <Self as FastSubsampledMetal>::REGION_SCALED_KEYS;

    fn from_region_scaled_batched<'a>(
        packet: &BatchedFastPacket<'a>,
    ) -> Option<(&'a Self, PlaneMode)> {
        match packet {
            BatchedFastPacket::Fast420(packet) => Some((packet, PlaneMode::YCbCr)),
            _ => None,
        }
    }
    fn to_region_scaled_batched(&self, _mode: PlaneMode) -> BatchedFastPacket<'_> {
        BatchedFastPacket::Fast420(self)
    }
    fn scaled_region_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_scaled_region_batch_decode_pipeline
    }
    fn pack_windowed_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_420_windowed_rgb_batch_pipeline
    }
}

#[cfg(target_os = "macos")]
impl FastRegionScaledMetal for JpegFast422PacketV1 {
    const REGION_SCALED_KEYS: FastScratchKeys = <Self as FastSubsampledMetal>::REGION_SCALED_KEYS;

    fn from_region_scaled_batched<'a>(
        packet: &BatchedFastPacket<'a>,
    ) -> Option<(&'a Self, PlaneMode)> {
        match packet {
            BatchedFastPacket::Fast422(packet) => Some((packet, PlaneMode::YCbCr)),
            _ => None,
        }
    }
    fn to_region_scaled_batched(&self, _mode: PlaneMode) -> BatchedFastPacket<'_> {
        BatchedFastPacket::Fast422(self)
    }
    fn scaled_region_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_scaled_region_batch_decode_pipeline
    }
    fn pack_windowed_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_422_windowed_rgb_batch_pipeline
    }
}

#[cfg(target_os = "macos")]
impl FastRegionScaledMetal for JpegFast444PacketV1 {
    const REGION_SCALED_KEYS: FastScratchKeys = <Self as FastSubsampledMetal>::REGION_SCALED_KEYS;

    fn from_region_scaled_batched<'a>(
        packet: &BatchedFastPacket<'a>,
    ) -> Option<(&'a Self, PlaneMode)> {
        match packet {
            BatchedFastPacket::Fast444(packet, mode) => Some((packet, *mode)),
            _ => None,
        }
    }
    fn to_region_scaled_batched(&self, mode: PlaneMode) -> BatchedFastPacket<'_> {
        BatchedFastPacket::Fast444(self, mode)
    }
    fn scaled_region_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast444_scaled_region_batch_decode_pipeline
    }
    fn pack_windowed_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_444_rgb_batch_pipeline
    }
}
