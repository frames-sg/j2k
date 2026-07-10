// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, Buffer, CommandBuffer, ComputePipelineState, EncodeProgressionOrder, HybridSignpostName,
    J2kClassicEncodeBatchJob, J2kHtEncodeBatchJob, J2kPacketizationPacketDescriptor,
    J2kResidentEncodeGpuStageCommandBuffer, J2kResidentEncodeStageStats, MetalRuntime,
};

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
    pub(crate) component_count: u8,
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
    pub(in crate::compute) coefficient_buffer: Buffer,
    pub(in crate::compute) coefficient_byte_offset: usize,
    pub(in crate::compute) coefficient_byte_len: usize,
    pub(in crate::compute) coefficient_buffer_is_batch_shared: bool,
    pub(in crate::compute) code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    pub(in crate::compute) recyclable_private_buffers: Vec<(usize, Buffer)>,
    pub(in crate::compute) _prepare_command_buffer: CommandBuffer,
    pub(in crate::compute) _prepare_deinterleave_rct_command_buffer: Option<CommandBuffer>,
    pub(in crate::compute) _prepare_dwt53_command_buffer: Option<CommandBuffer>,
    pub(in crate::compute) _prepare_dwt53_vertical_command_buffers: Vec<CommandBuffer>,
    pub(in crate::compute) _prepare_dwt53_horizontal_command_buffers: Vec<CommandBuffer>,
    pub(in crate::compute) _prepare_coefficient_extract_command_buffer: Option<CommandBuffer>,
    pub(in crate::compute) _deinterleave_status_buffer: Buffer,
    pub(in crate::compute) _plane_buffers: Vec<Buffer>,
    pub(in crate::compute) _scratch_buffers: Vec<Buffer>,
    pub(in crate::compute) _coefficient_job_buffer: Buffer,
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
    pub(crate) component_count: u8,
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
    pub(crate) component_count: u8,
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
    pub(in crate::compute) output_buffer: Buffer,
    pub(in crate::compute) status_buffer: Buffer,
    pub(in crate::compute) job_buffer: Buffer,
    pub(in crate::compute) batch_jobs: Vec<J2kClassicEncodeBatchJob>,
    pub(in crate::compute) code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    pub(in crate::compute) output_capacity_total: usize,
    pub(in crate::compute) _segment_buffer: Buffer,
    pub(in crate::compute) tier1_command_buffer: CommandBuffer,
    pub(in crate::compute) _coefficient_buffer: Buffer,
    pub(in crate::compute) prepare_command_buffer: CommandBuffer,
    pub(in crate::compute) _deinterleave_status_buffer: Buffer,
    pub(in crate::compute) _plane_buffers: Vec<Buffer>,
    pub(in crate::compute) _scratch_buffers: Vec<Buffer>,
    pub(in crate::compute) _coefficient_job_buffer: Buffer,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kResidentLosslessHtCodeBlocks {
    pub(in crate::compute) output_buffer: Buffer,
    pub(in crate::compute) status_buffer: Buffer,
    pub(in crate::compute) job_buffer: Buffer,
    pub(in crate::compute) batch_jobs: Vec<J2kHtEncodeBatchJob>,
    pub(in crate::compute) code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    pub(in crate::compute) output_capacity_total: usize,
    pub(in crate::compute) tier1_command_buffer: CommandBuffer,
    pub(in crate::compute) _coefficient_buffer: Buffer,
    pub(in crate::compute) prepare_command_buffer: CommandBuffer,
    pub(in crate::compute) _deinterleave_status_buffer: Buffer,
    pub(in crate::compute) _plane_buffers: Vec<Buffer>,
    pub(in crate::compute) _scratch_buffers: Vec<Buffer>,
    pub(in crate::compute) _coefficient_job_buffer: Buffer,
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
pub(in crate::compute) enum J2kResidentTier1StatusKind {
    Classic,
    HighThroughput,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct J2kResidentTier1StatusReadback {
    pub(in crate::compute) buffer: Buffer,
    pub(in crate::compute) kind: J2kResidentTier1StatusKind,
    pub(in crate::compute) classic_style_flags: u32,
    pub(in crate::compute) classic_jobs: Option<Vec<J2kClassicEncodeBatchJob>>,
    pub(in crate::compute) count: usize,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct J2kResidentClassicTier1DensityReadback {
    pub(in crate::compute) buffer: Buffer,
    pub(in crate::compute) count: usize,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct J2kResidentClassicTier1SymbolPlanReadback {
    pub(in crate::compute) buffer: Buffer,
    pub(in crate::compute) count: usize,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct J2kResidentClassicTier1PassPlanReadback {
    pub(in crate::compute) buffer: Buffer,
    pub(in crate::compute) count: usize,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct J2kResidentClassicTier1TokenEmitReadback {
    pub(in crate::compute) counter_buffer: Buffer,
    pub(in crate::compute) token_buffer: Option<Buffer>,
    pub(in crate::compute) segment_buffer: Option<Buffer>,
    pub(in crate::compute) token_stride_bytes: usize,
    pub(in crate::compute) token_segment_stride: usize,
    pub(in crate::compute) count: usize,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct J2kResidentClassicTier1GpuTokenBuffers {
    pub(in crate::compute) counter_buffer: Buffer,
    pub(in crate::compute) token_buffer: Buffer,
    pub(in crate::compute) segment_buffer: Buffer,
    pub(in crate::compute) job_count: u32,
    pub(in crate::compute) token_stride_bytes: u32,
    pub(in crate::compute) token_segment_stride: u32,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct J2kResidentClassicTier1SplitTokenBuffers {
    pub(in crate::compute) counter_buffer: Buffer,
    pub(in crate::compute) mq_token_buffer: Buffer,
    pub(in crate::compute) raw_token_buffer: Buffer,
    pub(in crate::compute) segment_buffer: Buffer,
    pub(in crate::compute) job_count: u32,
    pub(in crate::compute) mq_token_stride_bytes: u32,
    pub(in crate::compute) raw_token_stride_bytes: u32,
    pub(in crate::compute) token_segment_stride: u32,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kPendingResidentLosslessCodestreamBatch {
    pub(in crate::compute) runtime: Arc<MetalRuntime>,
    pub(in crate::compute) buffer: Buffer,
    pub(in crate::compute) byte_offsets: Vec<usize>,
    pub(in crate::compute) capacities: Vec<usize>,
    pub(in crate::compute) status_buffer: Buffer,
    pub(in crate::compute) packet_status_buffer: Buffer,
    pub(in crate::compute) tier1_status_readback: Option<J2kResidentTier1StatusReadback>,
    pub(in crate::compute) classic_tier1_density_readback:
        Option<J2kResidentClassicTier1DensityReadback>,
    pub(in crate::compute) classic_tier1_symbol_plan_readback:
        Option<J2kResidentClassicTier1SymbolPlanReadback>,
    pub(in crate::compute) classic_tier1_pass_plan_readback:
        Option<J2kResidentClassicTier1PassPlanReadback>,
    pub(in crate::compute) classic_tier1_token_emit_readback:
        Option<J2kResidentClassicTier1TokenEmitReadback>,
    pub(in crate::compute) classic_tier1_split_token_emit_readback:
        Option<J2kResidentClassicTier1SplitTokenBuffers>,
    pub(in crate::compute) classic_gpu_token_pack_used: bool,
    pub(in crate::compute) command_buffer: CommandBuffer,
    pub(in crate::compute) retained_command_buffers: Vec<CommandBuffer>,
    pub(in crate::compute) _retained_buffers: Vec<Buffer>,
    pub(in crate::compute) recyclable_private_buffers: Vec<(usize, Buffer)>,
    pub(in crate::compute) recyclable_shared_buffers: Vec<(usize, Buffer)>,
    pub(in crate::compute) gpu_stage_command_buffers: Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    pub(in crate::compute) stage_stats: J2kResidentEncodeStageStats,
    pub(in crate::compute) codestream_payload_copy_dispatched: bool,
    pub(in crate::compute) status_stage: &'static str,
    pub(in crate::compute) length_error: &'static str,
    pub(in crate::compute) capacity_error: &'static str,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct J2kBatchedPacketPayloadCopyDispatch<'a> {
    pub(in crate::compute) payload_buffer: &'a Buffer,
    pub(in crate::compute) packet_output_buffer: &'a Buffer,
    pub(in crate::compute) packet_job_buffer: &'a Buffer,
    pub(in crate::compute) packet_status_buffer: &'a Buffer,
    pub(in crate::compute) packet_payload_copy_job_buffer: &'a Buffer,
    pub(in crate::compute) tile_count: u64,
    pub(in crate::compute) max_payload_copy_jobs_per_tile: u64,
    pub(in crate::compute) label: &'a str,
    pub(in crate::compute) signpost_name: HybridSignpostName,
}
