// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
/// Field access over the sampling-family fast packets, which are
/// field-identical; the sampling factor is carried by the packet type.
trait FastPacketAccess {
    /// Family name used in diagnostics ("fast420" / "fast422" / "fast444").
    const FAMILY_NAME: &'static str;

    fn dimensions(&self) -> (u32, u32);
    fn mcus_per_row(&self) -> u32;
    fn mcu_rows(&self) -> u32;
    fn restart_interval_mcus(&self) -> u32;
    fn restart_offsets(&self) -> &[u32];
    fn entropy_checkpoints(&self) -> &[JpegEntropyCheckpointV1];
    fn entropy_bytes(&self) -> &[u8];
    fn y_quant(&self) -> &[u16; 64];
    fn cb_quant(&self) -> &[u16; 64];
    fn cr_quant(&self) -> &[u16; 64];
    fn y_dc_table(&self) -> &PacketHuffmanTable;
    fn y_ac_table(&self) -> &PacketHuffmanTable;
    fn cb_dc_table(&self) -> &PacketHuffmanTable;
    fn cb_ac_table(&self) -> &PacketHuffmanTable;
    fn cr_dc_table(&self) -> &PacketHuffmanTable;
    fn cr_ac_table(&self) -> &PacketHuffmanTable;
}

macro_rules! impl_fast_packet_access {
    ($packet:ty, $name:literal) => {
        impl FastPacketAccess for $packet {
            const FAMILY_NAME: &'static str = $name;

            fn dimensions(&self) -> (u32, u32) {
                self.dimensions
            }
            fn mcus_per_row(&self) -> u32 {
                self.mcus_per_row
            }
            fn mcu_rows(&self) -> u32 {
                self.mcu_rows
            }
            fn restart_interval_mcus(&self) -> u32 {
                self.restart_interval_mcus
            }
            fn restart_offsets(&self) -> &[u32] {
                &self.restart_offsets
            }
            fn entropy_checkpoints(&self) -> &[JpegEntropyCheckpointV1] {
                &self.entropy_checkpoints
            }
            fn entropy_bytes(&self) -> &[u8] {
                &self.entropy_bytes
            }
            fn y_quant(&self) -> &[u16; 64] {
                &self.y_quant
            }
            fn cb_quant(&self) -> &[u16; 64] {
                &self.cb_quant
            }
            fn cr_quant(&self) -> &[u16; 64] {
                &self.cr_quant
            }
            fn y_dc_table(&self) -> &PacketHuffmanTable {
                &self.y_dc_table
            }
            fn y_ac_table(&self) -> &PacketHuffmanTable {
                &self.y_ac_table
            }
            fn cb_dc_table(&self) -> &PacketHuffmanTable {
                &self.cb_dc_table
            }
            fn cb_ac_table(&self) -> &PacketHuffmanTable {
                &self.cb_ac_table
            }
            fn cr_dc_table(&self) -> &PacketHuffmanTable {
                &self.cr_dc_table
            }
            fn cr_ac_table(&self) -> &PacketHuffmanTable {
                &self.cr_ac_table
            }
        }
    };
}

impl_fast_packet_access!(JpegFast420PacketV1, "fast420");
impl_fast_packet_access!(JpegFast422PacketV1, "fast422");
impl_fast_packet_access!(JpegFast444PacketV1, "fast444");

/// Chroma geometry for the subsampled families that share the
/// `JpegFast420Params` kernel ABI (4:2:0 halves chroma rows, 4:2:2 keeps
/// them; both halve chroma columns).
trait FastSubsampledPacket: FastPacketAccess {
    /// Luma MCU width in pixels.
    const MCU_WIDTH: u32;
    /// Luma MCU height in pixels (4:2:0 MCUs are 16 rows, 4:2:2 are 8).
    const MCU_HEIGHT: u32;
    /// Whether the full-RGB batch path may group restart-interval packets.
    const FULL_RGB_BATCH_SUPPORTS_RESTART: bool;
    const ENTROPY_PAYLOAD_CTX: &'static str;
    const REGION_SCALED_BATCH_OUT_STRIDE_CTX: &'static str;
    const OUTPUT_STRIDE_CTX: &'static str;
    const REGION_OUTPUT_STRIDE_CTX: &'static str;
    const SCALED_ENTROPY_PAYLOAD_CTX: &'static str;
    /// Blocks per MCU when the full-RGB batch path needs block-count
    /// validation for the split coeff/IDCT debug mode (4:2:0 only).
    const FULL_RGB_BATCH_BLOCKS_PER_MCU: Option<usize>;

    fn chroma_height(height: u32) -> u32;
    /// Vertical dispatch extent for the full-frame pack kernels: 4:2:0 packs
    /// 2x2 pixel quads per thread, 4:2:2 packs 2x1 pairs (full-height rows).
    fn packed_height_extent(height: u32) -> u32;
}

impl FastSubsampledPacket for JpegFast420PacketV1 {
    const MCU_WIDTH: u32 = 16;
    const MCU_HEIGHT: u32 = 16;
    const FULL_RGB_BATCH_SUPPORTS_RESTART: bool = true;
    const ENTROPY_PAYLOAD_CTX: &'static str = "fast420 entropy payload";
    const REGION_SCALED_BATCH_OUT_STRIDE_CTX: &'static str =
        "fast420 region scaled batch output stride";
    const OUTPUT_STRIDE_CTX: &'static str = "fast420 output stride";
    const REGION_OUTPUT_STRIDE_CTX: &'static str = "fast420 region output stride";
    const SCALED_ENTROPY_PAYLOAD_CTX: &'static str = "fast420 scaled entropy payload";
    const FULL_RGB_BATCH_BLOCKS_PER_MCU: Option<usize> = Some(6);

    fn chroma_height(height: u32) -> u32 {
        height.div_ceil(2)
    }
    fn packed_height_extent(height: u32) -> u32 {
        height.div_ceil(2).max(1)
    }
}

impl FastSubsampledPacket for JpegFast422PacketV1 {
    const MCU_WIDTH: u32 = 16;
    const MCU_HEIGHT: u32 = 8;
    const FULL_RGB_BATCH_SUPPORTS_RESTART: bool = false;
    const ENTROPY_PAYLOAD_CTX: &'static str = "fast422 entropy payload";
    const REGION_SCALED_BATCH_OUT_STRIDE_CTX: &'static str =
        "fast422 region scaled batch output stride";
    const OUTPUT_STRIDE_CTX: &'static str = "fast422 output stride";
    const REGION_OUTPUT_STRIDE_CTX: &'static str = "fast422 region output stride";
    const SCALED_ENTROPY_PAYLOAD_CTX: &'static str = "fast422 scaled entropy payload";
    const FULL_RGB_BATCH_BLOCKS_PER_MCU: Option<usize> = None;

    fn chroma_height(height: u32) -> u32 {
        height
    }
    fn packed_height_extent(height: u32) -> u32 {
        height
    }
}

/// Scratch-pool cache keys for one batch driver's buffers; keys stay
/// per-family so pooled buffers are never shared across kernel families.
#[cfg(target_os = "macos")]
struct FastScratchKeys {
    y: &'static str,
    cb: &'static str,
    cr: &'static str,
    entropy: &'static str,
    entropy_offsets: &'static str,
    entropy_lens: &'static str,
    entropy_checkpoints: &'static str,
    status: &'static str,
}

/// Per-family vertical chroma-repair scratch layout for the direct-to-texture
/// full-frame path (4:2:0 only; 4:2:2 has no vertical MCU chroma boundary).
#[cfg(target_os = "macos")]
struct FastVerticalRepairSpec {
    meta_words: usize,
    sample_bytes: usize,
    meta_key: &'static str,
    samples_key: &'static str,
}

/// Metal-side hooks for the subsampled families: per-family pipelines,
/// scratch keys, and batched-packet extraction.
#[cfg(target_os = "macos")]
trait FastSubsampledMetal: FastSubsampledPacket {
    const REGION_SCALED_KEYS: FastScratchKeys;
    const REGION_SCALED_TEXTURE_KEYS: FastScratchKeys;
    /// Scratch keys for the full-frame RGB batch driver.
    const FULL_BATCH_KEYS: FastScratchKeys;
    /// Scratch keys for the full-frame RGBA texture batch driver.
    const TEXTURE_KEYS: FastScratchKeys;
    /// Profile tag for the full-RGB batch timing rows.
    const FULL_RGB_BATCH_TIMING_TAG: &'static str;
    /// Boundary chroma-repair record layout for the direct-to-texture path.
    const TEXTURE_BOUNDARY_META_WORDS: usize;
    const TEXTURE_BOUNDARY_SAMPLE_BYTES: usize;
    const TEXTURE_BOUNDARY_META_KEY: &'static str;
    const TEXTURE_BOUNDARY_SAMPLES_KEY: &'static str;
    const TEXTURE_VERTICAL_REPAIR: Option<FastVerticalRepairSpec>;

    fn from_batched<'a>(packet: &BatchedFastPacket<'a>) -> Option<&'a Self>;
    fn to_batched(&self) -> BatchedFastPacket<'_>;
    fn decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn scaled_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn scaled_region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    /// Full-frame pack pipeline for the requested output format, or `None`
    /// when the family has no kernel for that format (4:2:2 packs only RGB
    /// and RGBA; 4:2:0 falls back to its generic pack kernel).
    fn pack_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Option<&ComputePipelineState>;
    fn pack_windowed_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> &ComputePipelineState;
    fn scaled_region_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn pack_windowed_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn pack_windowed_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn full_rgb_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn pack_full_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn pack_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn rgba_texture_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn rgba_texture_boundary_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    /// Whether the full-RGB batch driver should record per-stage timing rows.
    fn full_rgb_batch_timing_enabled() -> bool;
    /// Per-MCU dispatch width for the texture repair passes (4:2:0 only;
    /// 4:2:2 repairs per entropy segment instead).
    fn texture_mcu_dispatch_threads(total_mcus: usize) -> Result<Option<u32>, Error>;
    /// Repair record count for the direct-to-texture boundary scratch.
    fn texture_repair_record_count(
        tile_count: usize,
        total_mcus: usize,
        total_decode_threads: u32,
    ) -> Result<usize, Error>;
    /// Thread count for the horizontal boundary repair pass, or `None` when
    /// the pass is not needed for this batch shape.
    fn horizontal_repair_threads(
        first: &Self,
        segment_count_u32: u32,
        mcu_threads: Option<u32>,
    ) -> Option<u32>;
    /// Encode any family-specific repair passes beyond the shared horizontal
    /// pass (4:2:0 adds vertical and corner passes).
    fn encode_extra_texture_repair_passes(
        runtime: &MetalRuntime,
        ctx: &FastTextureRepairCtx<'_>,
    ) -> Result<(), Error>;
    /// Pipelines for the split coeff/IDCT debug decode mode, when supported.
    #[cfg(test)]
    fn split_coeff_idct_pipelines(
        runtime: &MetalRuntime,
    ) -> Option<(&ComputePipelineState, &ComputePipelineState)>;
    /// Scratch keys (coeff blocks, DC-only flags) for the split decode mode
    /// in the texture driver.
    #[cfg(test)]
    const SPLIT_TEXTURE_SCRATCH_KEYS: (&'static str, &'static str);
}

/// Shared context handed to the family-specific texture repair hooks.
#[cfg(target_os = "macos")]
struct FastTextureRepairCtx<'a> {
    command_buffer: &'a CommandBufferRef,
    output: &'a crate::MetalBatchTextureOutput,
    boundary_meta_buffer: &'a Buffer,
    vertical_buffers: Option<&'a (Buffer, Buffer)>,
    decode_params: JpegFast420TextureBatchParams,
    tile_count: usize,
    mcu_threads: Option<u32>,
    tile_index_ctx: &'a str,
}

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
    fn pack_windowed_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_420_windowed_rgb_batch_pipeline
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
                let texture = ctx
                    .output
                    .texture(index)
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
                let texture = ctx
                    .output
                    .texture(index)
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
    fn pack_windowed_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_422_windowed_rgb_batch_pipeline
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

fn fast_subsampled_params<P: FastSubsampledPacket>(
    packet: &P,
    fmt: PixelFormat,
) -> Result<JpegFast420Params, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!(
            "unsupported JPEG Metal {} pixel format {fmt:?}",
            P::FAMILY_NAME
        ),
    })?;
    let out_stride = packet.dimensions().0 as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420Params {
        width: packet.dimensions().0,
        height: packet.dimensions().1,
        chroma_width: packet.dimensions().0.div_ceil(2),
        chroma_height: P::chroma_height(packet.dimensions().1),
        mcus_per_row: packet.mcus_per_row(),
        mcu_rows: packet.mcu_rows(),
        restart_interval_mcus: packet.restart_interval_mcus(),
        restart_offset_count: checked_entropy_segment_count(
            packet.restart_interval_mcus(),
            packet.restart_offsets().len(),
            packet.entropy_checkpoints().len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes().len(), P::ENTROPY_PAYLOAD_CTX)?,
        out_stride: checked_u32(out_stride, P::OUTPUT_STRIDE_CTX)?,
        alpha: u32::from(u8::MAX),
        out_format,
        origin_x: 0,
        origin_y: 0,
    })
}

fn fast_subsampled_region_params<P: FastSubsampledPacket>(
    packet: &P,
    fmt: PixelFormat,
    source_window: j2k_jpeg::Rect,
) -> Result<JpegFast420Params, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!(
            "unsupported JPEG Metal {} pixel format {fmt:?}",
            P::FAMILY_NAME
        ),
    })?;
    let out_stride = source_window.w as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420Params {
        width: source_window.w,
        height: source_window.h,
        chroma_width: source_window.w.div_ceil(2),
        chroma_height: P::chroma_height(source_window.h),
        mcus_per_row: packet.mcus_per_row(),
        mcu_rows: packet.mcu_rows(),
        restart_interval_mcus: packet.restart_interval_mcus(),
        restart_offset_count: checked_entropy_segment_count(
            packet.restart_interval_mcus(),
            packet.restart_offsets().len(),
            packet.entropy_checkpoints().len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes().len(), P::ENTROPY_PAYLOAD_CTX)?,
        out_stride: checked_u32(out_stride, P::REGION_OUTPUT_STRIDE_CTX)?,
        alpha: u32::from(u8::MAX),
        out_format,
        origin_x: source_window.x,
        origin_y: source_window.y,
    })
}

fn fast_subsampled_scaled_params<P: FastSubsampledPacket>(
    packet: &P,
    scale: j2k_core::Downscale,
) -> Option<JpegFast420ScaledParams> {
    let scale_shift = match scale {
        j2k_core::Downscale::Half => 1,
        j2k_core::Downscale::Quarter => 2,
        j2k_core::Downscale::Eighth => 3,
        _ => return None,
    };
    let denom = 1u32 << scale_shift;
    let scaled_width = packet.dimensions().0.div_ceil(denom);
    let scaled_height = packet.dimensions().1.div_ceil(denom);
    Some(JpegFast420ScaledParams {
        scaled_width,
        scaled_height,
        chroma_width: scaled_width.div_ceil(2),
        chroma_height: P::chroma_height(scaled_height),
        mcus_per_row: packet.mcus_per_row(),
        mcu_rows: packet.mcu_rows(),
        restart_interval_mcus: packet.restart_interval_mcus(),
        restart_offset_count: optional_entropy_segment_count(
            packet.restart_interval_mcus(),
            packet.restart_offsets().len(),
            packet.entropy_checkpoints().len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes().len(), P::SCALED_ENTROPY_PAYLOAD_CTX)
            .ok()?,
        scale_shift,
        origin_x: 0,
        origin_y: 0,
    })
}

fn fast_subsampled_scaled_region_params<P: FastSubsampledPacket>(
    packet: &P,
    scale: j2k_core::Downscale,
    source_window: j2k_jpeg::Rect,
) -> Option<JpegFast420ScaledParams> {
    let full = fast_subsampled_scaled_params(packet, scale)?;
    Some(JpegFast420ScaledParams {
        scaled_width: source_window.w,
        scaled_height: source_window.h,
        chroma_width: source_window.w.div_ceil(2),
        chroma_height: P::chroma_height(source_window.h),
        origin_x: source_window.x,
        origin_y: source_window.y,
        ..full
    })
}

fn fast_subsampled_full_mcu_window<P: FastSubsampledPacket>(
    dims: (u32, u32),
    roi: j2k_jpeg::Rect,
) -> j2k_jpeg::Rect {
    let x0 = (roi.x / P::MCU_WIDTH) * P::MCU_WIDTH;
    let y0 = (roi.y / P::MCU_HEIGHT) * P::MCU_HEIGHT;
    let x1 = (roi.x + roi.w).div_ceil(P::MCU_WIDTH) * P::MCU_WIDTH;
    let y1 = (roi.y + roi.h).div_ceil(P::MCU_HEIGHT) * P::MCU_HEIGHT;
    j2k_jpeg::Rect {
        x: x0,
        y: y0,
        w: x1.min(dims.0).saturating_sub(x0),
        h: y1.min(dims.1).saturating_sub(y0),
    }
}

fn fast_subsampled_full_mcu_scaled_window<P: FastSubsampledPacket>(
    scaled_dims: (u32, u32),
    roi: j2k_jpeg::Rect,
    scale_shift: u32,
) -> j2k_jpeg::Rect {
    let mcu_width = P::MCU_WIDTH >> scale_shift;
    let mcu_height = P::MCU_HEIGHT >> scale_shift;
    let x0 = (roi.x / mcu_width) * mcu_width;
    let y0 = (roi.y / mcu_height) * mcu_height;
    let x1 = (roi.x + roi.w).div_ceil(mcu_width) * mcu_width;
    let y1 = (roi.y + roi.h).div_ceil(mcu_height) * mcu_height;
    j2k_jpeg::Rect {
        x: x0,
        y: y0,
        w: x1.min(scaled_dims.0).saturating_sub(x0),
        h: y1.min(scaled_dims.1).saturating_sub(y0),
    }
}

#[cfg(target_os = "macos")]
fn fast444_params(packet: &JpegFast444PacketV1) -> Result<JpegFast444Params, Error> {
    Ok(JpegFast444Params {
        width: packet.dimensions.0,
        height: packet.dimensions.1,
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: checked_entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes.len(), "fast444 entropy payload")?,
        origin_x: 0,
        origin_y: 0,
    })
}

#[cfg(target_os = "macos")]
fn fast444_region_params(
    packet: &JpegFast444PacketV1,
    roi: j2k_jpeg::Rect,
) -> Result<JpegFast444Params, Error> {
    Ok(JpegFast444Params {
        width: roi.w,
        height: roi.h,
        origin_x: roi.x,
        origin_y: roi.y,
        ..fast444_params(packet)?
    })
}

#[cfg(target_os = "macos")]
fn fast444_scaled_params(
    packet: &JpegFast444PacketV1,
    scale: j2k_core::Downscale,
) -> Option<JpegFast444ScaledParams> {
    let scale_shift = match scale {
        j2k_core::Downscale::Half => 1,
        j2k_core::Downscale::Quarter => 2,
        j2k_core::Downscale::Eighth => 3,
        _ => return None,
    };
    let denom = 1u32 << scale_shift;
    Some(JpegFast444ScaledParams {
        scaled_width: packet.dimensions.0.div_ceil(denom),
        scaled_height: packet.dimensions.1.div_ceil(denom),
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: optional_entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes.len(), "fast444 scaled entropy payload")
            .ok()?,
        scale_shift,
        origin_x: 0,
        origin_y: 0,
    })
}

#[cfg(target_os = "macos")]
fn fast444_scaled_region_params(
    packet: &JpegFast444PacketV1,
    scale: j2k_core::Downscale,
    roi: j2k_jpeg::Rect,
) -> Option<JpegFast444ScaledParams> {
    Some(JpegFast444ScaledParams {
        scaled_width: roi.w,
        scaled_height: roi.h,
        origin_x: roi.x,
        origin_y: roi.y,
        ..fast444_scaled_params(packet, scale)?
    })
}

#[cfg(target_os = "macos")]
fn fast_subsampled_windowed_pack_params_for_dims<P: FastSubsampledPacket>(
    dims: (u32, u32),
    fmt: PixelFormat,
    roi: j2k_jpeg::Rect,
) -> Result<JpegFast420WindowedPackParams, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!(
            "unsupported JPEG Metal {} pixel format {fmt:?}",
            P::FAMILY_NAME
        ),
    })?;
    let out_stride = roi.w as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420WindowedPackParams {
        src_width: dims.0,
        src_height: dims.1,
        chroma_width: dims.0.div_ceil(2),
        chroma_height: P::chroma_height(dims.1),
        src_x: roi.x,
        src_y: roi.y,
        width: roi.w,
        height: roi.h,
        out_stride: checked_u32(
            out_stride,
            &format!("{} windowed output stride", P::FAMILY_NAME),
        )?,
        alpha: u32::from(u8::MAX),
        out_format,
    })
}

#[cfg(target_os = "macos")]
fn restart_offsets_buffer(device: &Device, restart_offsets: &[u32]) -> Result<Buffer, Error> {
    if restart_offsets.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal restart offsets must contain at least one entry".to_string(),
        });
    }
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let bytes = unsafe {
        core::slice::from_raw_parts(
            restart_offsets.as_ptr().cast::<u8>(),
            size_of_val(restart_offsets),
        )
    };
    Ok(new_shared_buffer_with_data(device, bytes))
}

#[cfg(target_os = "macos")]
fn entropy_checkpoints_buffer(
    device: &Device,
    entropy_checkpoints: &[JpegEntropyCheckpointV1],
) -> Result<Buffer, Error> {
    if entropy_checkpoints.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal entropy checkpoints must contain at least one entry".to_string(),
        });
    }
    let checkpoints = entropy_checkpoint_hosts(entropy_checkpoints)?;
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let bytes = unsafe {
        core::slice::from_raw_parts(
            checkpoints.as_ptr().cast::<u8>(),
            size_of_val(checkpoints.as_slice()),
        )
    };
    Ok(new_shared_buffer_with_data(device, bytes))
}

#[cfg(target_os = "macos")]
fn entropy_checkpoint_hosts(
    entropy_checkpoints: &[JpegEntropyCheckpointV1],
) -> Result<Vec<JpegEntropyCheckpointHost>, Error> {
    if entropy_checkpoints.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal entropy checkpoints must contain at least one entry".to_string(),
        });
    }
    Ok(entropy_checkpoints
        .iter()
        .copied()
        .map(JpegEntropyCheckpointHost::from)
        .collect::<Vec<_>>())
}

#[cfg(target_os = "macos")]
fn entropy_segment_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> u32 {
    let len = if restart_interval_mcus == 0 {
        entropy_checkpoints_len
    } else {
        restart_offsets_len
    };
    u32::try_from(len)
        .expect("JPEG Metal entropy segment count fits in u32")
        .max(1)
}

#[cfg(target_os = "macos")]
fn optional_entropy_segment_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> Option<u32> {
    let len = if restart_interval_mcus == 0 {
        entropy_checkpoints_len
    } else {
        restart_offsets_len
    };
    u32::try_from(len).ok().map(|count| count.max(1))
}

#[cfg(target_os = "macos")]
fn checked_entropy_segment_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> Result<u32, Error> {
    optional_entropy_segment_count(
        restart_interval_mcus,
        restart_offsets_len,
        entropy_checkpoints_len,
    )
    .ok_or_else(|| Error::MetalKernel {
        message: "JPEG Metal entropy segment count does not fit in u32".to_string(),
    })
}

#[cfg(target_os = "macos")]
fn restart_work_for_mcu_range(
    restart_offsets: &[u32],
    restart_interval_mcus: u32,
    total_mcus: u32,
    first_mcu: u32,
    end_mcu: u32,
) -> (u32, &[u32]) {
    if restart_interval_mcus == 0 || restart_offsets.len() <= 1 {
        return (0, restart_offsets);
    }

    let first_mcu = first_mcu.min(total_mcus);
    let end_mcu = end_mcu.min(total_mcus).max(first_mcu + 1);
    let restart_offset_count =
        u32::try_from(restart_offsets.len()).expect("JPEG Metal restart offsets fit in u32");
    let first_segment = (first_mcu / restart_interval_mcus).min(restart_offset_count - 1);
    let end_segment = end_mcu
        .div_ceil(restart_interval_mcus)
        .min(restart_offset_count)
        .max(first_segment + 1);
    (
        first_segment * restart_interval_mcus,
        &restart_offsets[first_segment as usize..end_segment as usize],
    )
}

#[cfg(target_os = "macos")]
fn mcu_range_for_rect(
    rect: j2k_jpeg::Rect,
    mcus_per_row: u32,
    mcu_rows: u32,
    mcu_width: u32,
    mcu_height: u32,
) -> (u32, u32) {
    if rect.w == 0 || rect.h == 0 || mcus_per_row == 0 || mcu_rows == 0 {
        return (0, 0);
    }

    let max_col = mcus_per_row - 1;
    let max_row = mcu_rows - 1;
    let last_x = rect.x.saturating_add(rect.w).saturating_sub(1);
    let last_y = rect.y.saturating_add(rect.h).saturating_sub(1);
    let first_col = (rect.x / mcu_width).min(max_col);
    let last_col = (last_x / mcu_width).min(max_col);
    let first_row = (rect.y / mcu_height).min(max_row);
    let last_row = (last_y / mcu_height).min(max_row);
    let first_mcu = first_row * mcus_per_row + first_col;
    let end_mcu = last_row * mcus_per_row + last_col + 1;
    (first_mcu, end_mcu)
}

#[cfg(target_os = "macos")]
fn entropy_decode_thread_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> u32 {
    entropy_segment_count(
        restart_interval_mcus,
        restart_offsets_len,
        entropy_checkpoints_len,
    )
}

