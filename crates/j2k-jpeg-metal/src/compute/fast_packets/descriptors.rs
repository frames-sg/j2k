// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    BatchedFastPacket, Buffer, CommandBufferRef, ComputePipelineState, Error,
    JpegEntropyCheckpointV1, JpegFast420PacketV1, JpegFast420TextureBatchParams,
    JpegFast422PacketV1, JpegFast444PacketV1, JpegHuffmanTable, MetalRuntime, PixelFormat,
    PlaneMode,
};
use crate::abi::{CHROMA_FANCY, CHROMA_REPLICATE, CHROMA_UNSAMPLED};

/// libjpeg-turbo's chroma geometry for a luma extent at one decode scale:
/// the chroma plane extent the decode kernels write and the pack-kernel chroma
/// mode that maps it onto output pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::compute) struct ScaledChroma {
    pub(in crate::compute) width: u32,
    pub(in crate::compute) height: u32,
    /// `CHROMA_FANCY`, `CHROMA_REPLICATE` or `CHROMA_UNSAMPLED`.
    pub(in crate::compute) mode: u32,
}

impl ScaledChroma {
    pub(in crate::compute) fn plane_len(self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// Chroma geometry for the subsampled families that share the
/// `JpegFast420Params` kernel ABI (4:2:0 halves chroma rows, 4:2:2 keeps
/// them; both halve chroma columns).
pub(in crate::compute) trait FastSubsampledPacket {
    /// Family name used in backend diagnostics.
    const FAMILY_NAME: &'static str;
    /// Luma MCU width in pixels.
    const MCU_WIDTH: u32;
    /// Luma MCU height in pixels (4:2:0 MCUs are 16 rows, 4:2:2 are 8).
    const MCU_HEIGHT: u32;
    /// Whether the full-RGB batch path may group restart-interval packets.
    const FULL_RGB_BATCH_SUPPORTS_RESTART: bool;
    /// Whether region windows need a neighbouring pixel on each side for
    /// chroma smoothing (the subsampled families).
    const REGION_CHROMA_CONTEXT: bool;
    const ENTROPY_PAYLOAD_CTX: &'static str;
    const REGION_SCALED_BATCH_OUT_STRIDE_CTX: &'static str;
    const OUTPUT_STRIDE_CTX: &'static str;
    const REGION_OUTPUT_STRIDE_CTX: &'static str;
    const SCALED_ENTROPY_PAYLOAD_CTX: &'static str;
    /// Blocks per MCU when the full-RGB batch path needs block-count
    /// validation for the split coeff/IDCT debug mode (4:2:0 only).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the split coefficient/IDCT diagnostic route is test-only"
        )
    )]
    const FULL_RGB_BATCH_BLOCKS_PER_MCU: Option<usize>;

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
    fn y_dc_table(&self) -> &JpegHuffmanTable;
    fn y_ac_table(&self) -> &JpegHuffmanTable;
    fn cb_dc_table(&self) -> &JpegHuffmanTable;
    fn cb_ac_table(&self) -> &JpegHuffmanTable;
    fn cr_dc_table(&self) -> &JpegHuffmanTable;
    fn cr_ac_table(&self) -> &JpegHuffmanTable;

    fn chroma_width(width: u32) -> u32;
    fn chroma_height(height: u32) -> u32;
    /// Chroma geometry for the luma `extent` of an image `image_width` pixels
    /// wide decoded at `scale_shift` (0 full size, 3 = 1/8), following
    /// libjpeg-turbo's per-component IDCT size and upsampler choice.
    fn scaled_chroma(image_width: u32, extent: (u32, u32), scale_shift: u32) -> ScaledChroma;
    /// Vertical dispatch extent for the full-frame pack kernels: 4:2:0 packs
    /// 2x2 pixel quads per thread, 4:2:2 packs 2x1 pairs (full-height rows).
    fn packed_height_extent(height: u32) -> u32;
    /// Horizontal dispatch extent for the full-frame pack kernels: the
    /// subsampled families pack horizontal pairs, 4:4:4 one pixel per thread.
    fn packed_width_extent(width: u32) -> u32 {
        width.div_ceil(2).max(1)
    }
}

macro_rules! impl_fast_subsampled_packet_accessors {
    () => {
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

        fn y_dc_table(&self) -> &JpegHuffmanTable {
            &self.y_dc_table
        }

        fn y_ac_table(&self) -> &JpegHuffmanTable {
            &self.y_ac_table
        }

        fn cb_dc_table(&self) -> &JpegHuffmanTable {
            &self.cb_dc_table
        }

        fn cb_ac_table(&self) -> &JpegHuffmanTable {
            &self.cb_ac_table
        }

        fn cr_dc_table(&self) -> &JpegHuffmanTable {
            &self.cr_dc_table
        }

        fn cr_ac_table(&self) -> &JpegHuffmanTable {
            &self.cr_ac_table
        }
    };
}

impl FastSubsampledPacket for JpegFast420PacketV1 {
    const FAMILY_NAME: &'static str = "fast420";
    const REGION_CHROMA_CONTEXT: bool = true;
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

    impl_fast_subsampled_packet_accessors!();

    fn chroma_width(width: u32) -> u32 {
        width.div_ceil(2)
    }
    fn chroma_height(height: u32) -> u32 {
        height.div_ceil(2)
    }
    fn scaled_chroma(image_width: u32, extent: (u32, u32), scale_shift: u32) -> ScaledChroma {
        if scale_shift > 0 {
            // Below full size libjpeg-turbo decodes 4:2:0 chroma with the next
            // larger IDCT, at luma resolution.
            return ScaledChroma {
                width: extent.0,
                height: extent.1,
                mode: CHROMA_UNSAMPLED,
            };
        }
        ScaledChroma {
            width: Self::chroma_width(extent.0),
            height: Self::chroma_height(extent.1),
            mode: subsampled_chroma_mode(image_width.div_ceil(2), true),
        }
    }
    fn packed_height_extent(height: u32) -> u32 {
        height.div_ceil(2).max(1)
    }
}

impl FastSubsampledPacket for JpegFast422PacketV1 {
    const FAMILY_NAME: &'static str = "fast422";
    const REGION_CHROMA_CONTEXT: bool = true;
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

    impl_fast_subsampled_packet_accessors!();

    fn chroma_width(width: u32) -> u32 {
        width.div_ceil(2)
    }
    fn chroma_height(height: u32) -> u32 {
        height
    }
    fn scaled_chroma(image_width: u32, extent: (u32, u32), scale_shift: u32) -> ScaledChroma {
        // `downsampled_width` of 4:2:2 chroma: ceil(width * (8 >> shift) / 16).
        let image_chroma_width = image_width.div_ceil(2 << scale_shift);
        ScaledChroma {
            width: Self::chroma_width(extent.0),
            height: Self::chroma_height(extent.1),
            mode: subsampled_chroma_mode(image_chroma_width, scale_shift < 3),
        }
    }
    fn packed_height_extent(height: u32) -> u32 {
        height
    }
}

impl FastSubsampledPacket for JpegFast444PacketV1 {
    const FAMILY_NAME: &'static str = "fast444";
    const REGION_CHROMA_CONTEXT: bool = false;
    const MCU_WIDTH: u32 = 8;
    const MCU_HEIGHT: u32 = 8;
    const FULL_RGB_BATCH_SUPPORTS_RESTART: bool = false;
    const ENTROPY_PAYLOAD_CTX: &'static str = "fast444 entropy payload";
    const REGION_SCALED_BATCH_OUT_STRIDE_CTX: &'static str =
        "fast444 region scaled batch output stride";
    const OUTPUT_STRIDE_CTX: &'static str = "fast444 output stride";
    const REGION_OUTPUT_STRIDE_CTX: &'static str = "fast444 region output stride";
    const SCALED_ENTROPY_PAYLOAD_CTX: &'static str = "fast444 scaled entropy payload";
    const FULL_RGB_BATCH_BLOCKS_PER_MCU: Option<usize> = None;

    impl_fast_subsampled_packet_accessors!();

    fn chroma_width(width: u32) -> u32 {
        width
    }
    fn chroma_height(height: u32) -> u32 {
        height
    }
    fn scaled_chroma(_image_width: u32, extent: (u32, u32), _scale_shift: u32) -> ScaledChroma {
        ScaledChroma {
            width: extent.0,
            height: extent.1,
            mode: CHROMA_UNSAMPLED,
        }
    }
    fn packed_height_extent(height: u32) -> u32 {
        height
    }
    fn packed_width_extent(width: u32) -> u32 {
        width
    }
}

/// `jdsample.c`: 2:1 chroma is smoothed only above 1/8 scale and when it is
/// more than two samples wide; otherwise each sample is replicated.
fn subsampled_chroma_mode(image_chroma_width: u32, smoothing_scale: bool) -> u32 {
    if smoothing_scale && image_chroma_width > 2 {
        CHROMA_FANCY
    } else {
        CHROMA_REPLICATE
    }
}

/// Scratch-pool cache keys for one batch driver's buffers; keys stay
/// per-family so pooled buffers are never shared across kernel families.
#[cfg(target_os = "macos")]
pub(in crate::compute) struct FastScratchKeys {
    pub(in crate::compute) y: &'static str,
    pub(in crate::compute) cb: &'static str,
    pub(in crate::compute) cr: &'static str,
    pub(in crate::compute) entropy: &'static str,
    pub(in crate::compute) entropy_offsets: &'static str,
    pub(in crate::compute) entropy_lens: &'static str,
    pub(in crate::compute) entropy_checkpoints: &'static str,
    pub(in crate::compute) status: &'static str,
}

/// Per-family vertical chroma-repair scratch layout for the direct-to-texture
/// full-frame path (4:2:0 only; 4:2:2 has no vertical MCU chroma boundary).
#[cfg(target_os = "macos")]
pub(in crate::compute) struct FastVerticalRepairSpec {
    pub(in crate::compute) meta_words: usize,
    pub(in crate::compute) sample_bytes: usize,
    pub(in crate::compute) meta_key: &'static str,
    pub(in crate::compute) samples_key: &'static str,
}

/// Metal-side hooks for the subsampled families: per-family pipelines,
/// scratch keys, and batched-packet extraction.
#[cfg(target_os = "macos")]
pub(in crate::compute) trait FastSubsampledMetal: FastSubsampledPacket {
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
    /// The 4:4:4 direct texture kernel has a distinct params ABI because it
    /// carries color-space mode and has no chroma dimensions.
    const USE_FAST444_TEXTURE_PARAMS: bool;

    fn from_batched<'a>(packet: &BatchedFastPacket<'a>) -> Option<&'a Self>;
    fn to_batched(&self) -> BatchedFastPacket<'_>;
    fn texture_plane_mode_from_batched(packet: &BatchedFastPacket<'_>) -> Option<PlaneMode> {
        Self::from_batched(packet).map(|_| PlaneMode::YCbCr)
    }
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

#[cfg(target_os = "macos")]
pub(in crate::compute) trait FastRegionScaledMetal:
    FastSubsampledPacket
{
    const REGION_SCALED_KEYS: FastScratchKeys;

    fn from_region_scaled_batched<'a>(
        packet: &BatchedFastPacket<'a>,
    ) -> Option<(&'a Self, PlaneMode)>;
    fn to_region_scaled_batched(&self, mode: PlaneMode) -> BatchedFastPacket<'_>;
    fn scaled_region_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn pack_windowed_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
}

/// Shared context handed to the family-specific texture repair hooks.
#[cfg(target_os = "macos")]
pub(in crate::compute) struct FastTextureRepairCtx<'a> {
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) output: &'a crate::MetalBatchTextureOutput,
    pub(in crate::compute) boundary_meta_buffer: &'a Buffer,
    pub(in crate::compute) vertical_buffers: Option<&'a (Buffer, Buffer)>,
    pub(in crate::compute) decode_params: JpegFast420TextureBatchParams,
    pub(in crate::compute) tile_count: usize,
    pub(in crate::compute) mcu_threads: Option<u32>,
    pub(in crate::compute) tile_index_ctx: &'a str,
}
