// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::test_counters;
use super::{
    accumulate_classic_tier1_scan_estimates, checked_buffer_read, checked_buffer_slice,
    classic_encode_code_blocks_pipeline_kind, classic_tier1_gpu_token_pack_supported,
    classic_tier1_pass_class_counts, completed_command_buffers_gpu_duration,
    completed_command_buffers_gpu_duration_and_elapsed_window, dispatch_1d_pipeline,
    duration_share, encode_status_error, hybrid_stage_signpost, label_compute_encoder,
    metal_profile_classic_tier1_arithmetic_pack_enabled,
    metal_profile_classic_tier1_density_enabled, metal_profile_classic_tier1_pass_plan_enabled,
    metal_profile_classic_tier1_raw_pack_enabled,
    metal_profile_classic_tier1_split_token_emit_enabled,
    metal_profile_classic_tier1_symbol_plan_enabled,
    metal_profile_classic_tier1_token_emit_enabled, metal_profile_classic_tier1_token_pack_enabled,
    metal_profile_stages_enabled, pack_j2k_code_block_scalar_from_tier1_tokens,
    packet_encode_status_error, record_completed_resident_encode_gpu_stages,
    recycle_private_buffers, recycle_shared_buffers, size_of, take_recyclable_private_buffer,
    wait_for_completion_metal, Arc, Buffer, CommandBuffer, CommandBufferRef, ComputePipelineState,
    EncodeProgressionOrder, Error, HybridSignpostName, Instant, IntoParallelIterator,
    J2kClassicEncodeBatchJob, J2kClassicEncodePipelineKind, J2kClassicEncodeStatus,
    J2kClassicTier1DensityCounters, J2kClassicTier1PassPlanCounters,
    J2kClassicTier1SymbolPlanCounters, J2kClassicTier1TokenSegment, J2kCodestreamAssemblyStatus,
    J2kHtEncodeBatchJob, J2kHtEncodeStatus, J2kPacketEncodeStatus,
    J2kPacketizationPacketDescriptor, J2kPendingResidentLosslessCodestream,
    J2kResidentEncodeGpuStageCommandBuffer, J2kResidentEncodeStageStats,
    J2kResidentLosslessCodestream, J2kResidentLosslessCodestreamBatchResult, J2kTier1TokenSegment,
    MTLResourceOptions, MTLSize, MetalRuntime, ParallelIterator,
    CLASSIC_TIER1_MQ_BYTE_TOKEN_ARENA_BYTES, CLASSIC_TIER1_TOKEN_ARENA_BYTES,
    CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY, J2K_ENCODE_STATUS_OK,
    PACKET_PAYLOAD_COPY_STRIPES_PER_JOB, SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT,
    SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST,
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
    pub(super) coefficient_buffer: Buffer,
    pub(super) coefficient_byte_offset: usize,
    pub(super) coefficient_byte_len: usize,
    pub(super) coefficient_buffer_is_batch_shared: bool,
    pub(super) code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    pub(super) recyclable_private_buffers: Vec<(usize, Buffer)>,
    pub(super) _prepare_command_buffer: CommandBuffer,
    pub(super) _prepare_deinterleave_rct_command_buffer: Option<CommandBuffer>,
    pub(super) _prepare_dwt53_command_buffer: Option<CommandBuffer>,
    pub(super) _prepare_dwt53_vertical_command_buffers: Vec<CommandBuffer>,
    pub(super) _prepare_dwt53_horizontal_command_buffers: Vec<CommandBuffer>,
    pub(super) _prepare_coefficient_extract_command_buffer: Option<CommandBuffer>,
    pub(super) _deinterleave_status_buffer: Buffer,
    pub(super) _plane_buffers: Vec<Buffer>,
    pub(super) _scratch_buffers: Vec<Buffer>,
    pub(super) _coefficient_job_buffer: Buffer,
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
    pub(super) output_buffer: Buffer,
    pub(super) status_buffer: Buffer,
    pub(super) job_buffer: Buffer,
    pub(super) batch_jobs: Vec<J2kClassicEncodeBatchJob>,
    pub(super) code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    pub(super) output_capacity_total: usize,
    pub(super) _segment_buffer: Buffer,
    pub(super) tier1_command_buffer: CommandBuffer,
    pub(super) _coefficient_buffer: Buffer,
    pub(super) prepare_command_buffer: CommandBuffer,
    pub(super) _deinterleave_status_buffer: Buffer,
    pub(super) _plane_buffers: Vec<Buffer>,
    pub(super) _scratch_buffers: Vec<Buffer>,
    pub(super) _coefficient_job_buffer: Buffer,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kResidentLosslessHtCodeBlocks {
    pub(super) output_buffer: Buffer,
    pub(super) status_buffer: Buffer,
    pub(super) job_buffer: Buffer,
    pub(super) batch_jobs: Vec<J2kHtEncodeBatchJob>,
    pub(super) code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    pub(super) output_capacity_total: usize,
    pub(super) tier1_command_buffer: CommandBuffer,
    pub(super) _coefficient_buffer: Buffer,
    pub(super) prepare_command_buffer: CommandBuffer,
    pub(super) _deinterleave_status_buffer: Buffer,
    pub(super) _plane_buffers: Vec<Buffer>,
    pub(super) _scratch_buffers: Vec<Buffer>,
    pub(super) _coefficient_job_buffer: Buffer,
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
pub(super) enum J2kResidentTier1StatusKind {
    Classic,
    HighThroughput,
}

#[cfg(target_os = "macos")]
pub(super) struct J2kResidentTier1StatusReadback {
    pub(super) buffer: Buffer,
    pub(super) kind: J2kResidentTier1StatusKind,
    pub(super) classic_style_flags: u32,
    pub(super) classic_jobs: Option<Vec<J2kClassicEncodeBatchJob>>,
    pub(super) count: usize,
}

#[cfg(target_os = "macos")]
pub(super) struct J2kResidentClassicTier1DensityReadback {
    pub(super) buffer: Buffer,
    pub(super) count: usize,
}

#[cfg(target_os = "macos")]
pub(super) struct J2kResidentClassicTier1SymbolPlanReadback {
    pub(super) buffer: Buffer,
    pub(super) count: usize,
}

#[cfg(target_os = "macos")]
pub(super) struct J2kResidentClassicTier1PassPlanReadback {
    pub(super) buffer: Buffer,
    pub(super) count: usize,
}

#[cfg(target_os = "macos")]
pub(super) struct J2kResidentClassicTier1TokenEmitReadback {
    pub(super) counter_buffer: Buffer,
    pub(super) token_buffer: Option<Buffer>,
    pub(super) segment_buffer: Option<Buffer>,
    pub(super) token_stride_bytes: usize,
    pub(super) token_segment_stride: usize,
    pub(super) count: usize,
}

#[cfg(target_os = "macos")]
pub(super) struct J2kResidentClassicTier1GpuTokenBuffers {
    pub(super) counter_buffer: Buffer,
    pub(super) token_buffer: Buffer,
    pub(super) segment_buffer: Buffer,
    pub(super) job_count: u32,
    pub(super) token_stride_bytes: u32,
    pub(super) token_segment_stride: u32,
}

#[cfg(target_os = "macos")]
pub(super) struct J2kResidentClassicTier1SplitTokenBuffers {
    pub(super) counter_buffer: Buffer,
    pub(super) mq_token_buffer: Buffer,
    pub(super) raw_token_buffer: Buffer,
    pub(super) segment_buffer: Buffer,
    pub(super) job_count: u32,
    pub(super) mq_token_stride_bytes: u32,
    pub(super) raw_token_stride_bytes: u32,
    pub(super) token_segment_stride: u32,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kPendingResidentLosslessCodestreamBatch {
    pub(super) runtime: Arc<MetalRuntime>,
    pub(super) buffer: Buffer,
    pub(super) byte_offsets: Vec<usize>,
    pub(super) capacities: Vec<usize>,
    pub(super) status_buffer: Buffer,
    pub(super) packet_status_buffer: Buffer,
    pub(super) tier1_status_readback: Option<J2kResidentTier1StatusReadback>,
    pub(super) classic_tier1_density_readback: Option<J2kResidentClassicTier1DensityReadback>,
    pub(super) classic_tier1_symbol_plan_readback:
        Option<J2kResidentClassicTier1SymbolPlanReadback>,
    pub(super) classic_tier1_pass_plan_readback: Option<J2kResidentClassicTier1PassPlanReadback>,
    pub(super) classic_tier1_token_emit_readback: Option<J2kResidentClassicTier1TokenEmitReadback>,
    pub(super) classic_tier1_split_token_emit_readback:
        Option<J2kResidentClassicTier1SplitTokenBuffers>,
    pub(super) classic_gpu_token_pack_used: bool,
    pub(super) command_buffer: CommandBuffer,
    pub(super) retained_command_buffers: Vec<CommandBuffer>,
    pub(super) _retained_buffers: Vec<Buffer>,
    pub(super) recyclable_private_buffers: Vec<(usize, Buffer)>,
    pub(super) recyclable_shared_buffers: Vec<(usize, Buffer)>,
    pub(super) gpu_stage_command_buffers: Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    pub(super) stage_stats: J2kResidentEncodeStageStats,
    pub(super) codestream_payload_copy_dispatched: bool,
    pub(super) status_stage: &'static str,
    pub(super) length_error: &'static str,
    pub(super) capacity_error: &'static str,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct J2kBatchedPacketPayloadCopyDispatch<'a> {
    pub(super) payload_buffer: &'a Buffer,
    pub(super) packet_output_buffer: &'a Buffer,
    pub(super) packet_job_buffer: &'a Buffer,
    pub(super) packet_status_buffer: &'a Buffer,
    pub(super) packet_payload_copy_job_buffer: &'a Buffer,
    pub(super) tile_count: u64,
    pub(super) max_payload_copy_jobs_per_tile: u64,
    pub(super) label: &'a str,
    pub(super) signpost_name: HybridSignpostName,
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
    let status = checked_buffer_read::<J2kCodestreamAssemblyStatus>(
        &pending.status_buffer,
        "resident codestream assembly status",
    )?;
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
#[derive(Clone, Copy)]
pub(super) struct ResidentTier1StatusReadbackRequest<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a CommandBufferRef,
    pub(super) status_buffer: &'a Buffer,
    pub(super) kind: J2kResidentTier1StatusKind,
    pub(super) classic_style_flags: u32,
    pub(super) classic_jobs: Option<&'a [J2kClassicEncodeBatchJob]>,
    pub(super) count: usize,
    pub(super) status_size: usize,
    pub(super) profile_stages: bool,
}

#[cfg(target_os = "macos")]
impl<'a> ResidentTier1StatusReadbackRequest<'a> {
    pub(super) fn high_throughput(
        runtime: &'a MetalRuntime,
        command_buffer: &'a CommandBufferRef,
        status_buffer: &'a Buffer,
        count: usize,
        profile_stages: bool,
    ) -> Self {
        Self {
            runtime,
            command_buffer,
            status_buffer,
            kind: J2kResidentTier1StatusKind::HighThroughput,
            classic_style_flags: 0,
            classic_jobs: None,
            count,
            status_size: size_of::<J2kHtEncodeStatus>(),
            profile_stages,
        }
    }

    pub(super) fn classic(
        runtime: &'a MetalRuntime,
        command_buffer: &'a CommandBufferRef,
        status_buffer: &'a Buffer,
        classic_style_flags: u32,
        classic_jobs: &'a [J2kClassicEncodeBatchJob],
        profile_stages: bool,
    ) -> Self {
        Self {
            runtime,
            command_buffer,
            status_buffer,
            kind: J2kResidentTier1StatusKind::Classic,
            classic_style_flags,
            classic_jobs: Some(classic_jobs),
            count: classic_jobs.len(),
            status_size: size_of::<J2kClassicEncodeStatus>(),
            profile_stages,
        }
    }
}

#[cfg(target_os = "macos")]
pub(super) fn schedule_resident_tier1_status_readback(
    request: ResidentTier1StatusReadbackRequest<'_>,
) -> Result<Option<J2kResidentTier1StatusReadback>, Error> {
    let ResidentTier1StatusReadbackRequest {
        runtime,
        command_buffer,
        status_buffer,
        kind,
        classic_style_flags,
        classic_jobs,
        count,
        status_size,
        profile_stages,
    } = request;
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
pub(super) fn dispatch_classic_tier1_density_profile(
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
pub(super) fn dispatch_classic_tier1_raw_pack_profile(
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
pub(super) fn dispatch_classic_tier1_arithmetic_pack_profile(
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
pub(super) fn dispatch_classic_tier1_symbol_plan_profile(
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
pub(super) fn dispatch_classic_tier1_pass_plan_profile(
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
pub(super) fn dispatch_classic_tier1_token_emit_profile(
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
pub(super) fn dispatch_classic_tier1_split_token_emit_for_cpu_pack(
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
pub(super) fn dispatch_classic_tier1_split_token_emit_profile(
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
pub(super) fn dispatch_classic_tier1_split_token_emit_for_gpu_pack(
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
pub(super) fn dispatch_classic_tier1_token_emit_for_gpu_pack(
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
pub(super) fn dispatch_classic_tier1_token_pack_from_gpu_tokens(
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
pub(super) fn dispatch_classic_tier1_split_token_pack_from_gpu_tokens(
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
pub(super) fn schedule_classic_tier1_gpu_token_pack_readback(
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
pub(super) fn record_classic_tier1_density_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1DensityReadback,
) -> Result<(), Error> {
    let counters = checked_buffer_slice::<J2kClassicTier1DensityCounters>(
        &readback.buffer,
        readback.count,
        "classic Tier-1 density counters",
    )?;
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
pub(super) fn record_classic_tier1_symbol_plan_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1SymbolPlanReadback,
) -> Result<(), Error> {
    let counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
        &readback.buffer,
        readback.count,
        "classic Tier-1 symbol-plan counters",
    )?;
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
pub(super) fn record_classic_tier1_pass_plan_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1PassPlanReadback,
) -> Result<(), Error> {
    let counters = checked_buffer_slice::<J2kClassicTier1PassPlanCounters>(
        &readback.buffer,
        readback.count,
        "classic Tier-1 pass-plan counters",
    )?;
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
pub(super) fn compare_classic_tier1_symbol_plan_and_pass_plan_counters(
    symbol_plan: &J2kResidentClassicTier1SymbolPlanReadback,
    pass_plan: &J2kResidentClassicTier1PassPlanReadback,
) -> Result<(), Error> {
    if symbol_plan.count != pass_plan.count {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 pass-plan comparison count mismatch".to_string(),
        });
    }
    let symbol_plan_counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
        &symbol_plan.buffer,
        symbol_plan.count,
        "classic Tier-1 symbol-plan comparison counters",
    )?;
    let pass_plan_counters = checked_buffer_slice::<J2kClassicTier1PassPlanCounters>(
        &pass_plan.buffer,
        pass_plan.count,
        "classic Tier-1 pass-plan comparison counters",
    )?;
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
pub(super) fn record_classic_tier1_token_emit_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1TokenEmitReadback,
) -> Result<(), Error> {
    let counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
        &readback.counter_buffer,
        readback.count,
        "classic Tier-1 token-emit counters",
    )?;
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
pub(super) fn compare_classic_tier1_symbol_plan_and_token_emit_counters(
    symbol_plan: &J2kResidentClassicTier1SymbolPlanReadback,
    token_emit: &J2kResidentClassicTier1TokenEmitReadback,
) -> Result<(), Error> {
    if symbol_plan.count != token_emit.count {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token-emitter comparison count mismatch".to_string(),
        });
    }
    let symbol_plan_counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
        &symbol_plan.buffer,
        symbol_plan.count,
        "classic Tier-1 symbol-token comparison counters",
    )?;
    let token_emit_counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
        &token_emit.counter_buffer,
        token_emit.count,
        "classic Tier-1 token-emit comparison counters",
    )?;
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
pub(super) fn validate_classic_tier1_split_token_emit_counters(
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
    let counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
        &readback.counter_buffer,
        count,
        "classic Tier-1 split-token counters",
    )?;
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
pub(super) fn compare_classic_tier1_symbol_plan_and_split_token_emit_counters(
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
    let symbol_plan_counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
        &symbol_plan.buffer,
        symbol_plan.count,
        "classic Tier-1 split-token symbol comparison counters",
    )?;
    let split_emit_counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
        &split_emit.counter_buffer,
        split_count,
        "classic Tier-1 split-token emit comparison counters",
    )?;
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
pub(super) fn profile_classic_tier1_token_pack(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1TokenEmitReadback,
) -> Result<(), Error> {
    if !metal_profile_classic_tier1_token_pack_enabled() {
        return Ok(());
    }
    let counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
        &readback.counter_buffer,
        readback.count,
        "classic Tier-1 token-pack counters",
    )?;
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
    let token_bytes = checked_buffer_slice::<u8>(
        token_buffer,
        readback.count.saturating_mul(readback.token_stride_bytes),
        "classic Tier-1 token-pack bytes",
    )?;
    let token_segments = checked_buffer_slice::<J2kClassicTier1TokenSegment>(
        segment_buffer,
        readback.count.saturating_mul(readback.token_segment_stride),
        "classic Tier-1 token-pack segments",
    )?;
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
pub(super) fn record_resident_tier1_output_usage(
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
            let statuses = checked_buffer_slice::<J2kClassicEncodeStatus>(
                &readback.buffer,
                readback.count,
                "resident classic Tier-1 statuses",
            )?;
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
            let statuses = checked_buffer_slice::<J2kHtEncodeStatus>(
                &readback.buffer,
                readback.count,
                "resident HT Tier-1 statuses",
            )?;
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
pub(super) fn wait_resident_codestream_command_buffer(
    command_buffer: &CommandBufferRef,
) -> Result<(), Error> {
    #[cfg(test)]
    test_counters::record_resident_codestream_command_buffer_wait();
    let _signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT);
    wait_for_completion_metal(command_buffer)
}

#[cfg(target_os = "macos")]
pub(super) fn finish_completed_resident_lossless_codestream_batch(
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
    let statuses = checked_buffer_slice::<J2kCodestreamAssemblyStatus>(
        &pending.status_buffer,
        pending.capacities.len(),
        "resident codestream assembly statuses",
    )?
    .to_vec();
    let packet_statuses = checked_buffer_slice::<J2kPacketEncodeStatus>(
        &pending.packet_status_buffer,
        pending.capacities.len(),
        "resident packet encode statuses",
    )?
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
