// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::{Duration, Instant};

use metal::CommandBuffer;

use crate::profile_env::label_command_buffer;

use super::{
    completed_command_buffer_gpu_duration, new_command_buffer, Error, J2kResidentEncodeStageStats,
    MetalRuntime,
};

pub(super) struct J2kResidentEncodeGpuStageCommandBuffer {
    pub(super) stage: J2kResidentEncodeGpuStage,
    pub(super) command_buffer: CommandBuffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum J2kResidentEncodeGpuStage {
    CoefficientPrep,
    CoefficientDeinterleaveRct,
    CoefficientDwt53,
    CoefficientDwt53Vertical,
    CoefficientDwt53Horizontal,
    CoefficientExtract,
    CoefficientCopy,
    ClassicBlock,
    ClassicTier1Density,
    ClassicTier1RawPack,
    ClassicTier1ArithmeticPack,
    ClassicTier1SymbolPlan,
    ClassicTier1PassPlan,
    ClassicTier1TokenEmit,
    ClassicTier1SplitTokenEmit,
    ClassicTier1TokenPack,
    HtBlock,
    PacketBlockPrep,
    Packetization,
    PacketPayloadCopy,
    CodestreamAssembly,
    CodestreamPayloadCopy,
}

pub(super) fn duration_share(duration: Duration, count: usize) -> Duration {
    if count == 0 {
        return Duration::ZERO;
    }
    let nanos = duration.as_nanos() / count as u128;
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}

#[expect(
    clippy::too_many_lines,
    reason = "exhaustive stage-to-counter mapping is clearer in one match"
)]
pub(super) fn record_completed_resident_encode_gpu_stages(
    stats: &mut J2kResidentEncodeStageStats,
    command_buffers: &[J2kResidentEncodeGpuStageCommandBuffer],
) {
    for stage_command_buffer in command_buffers {
        let Some(duration) =
            completed_command_buffer_gpu_duration(&stage_command_buffer.command_buffer)
        else {
            continue;
        };
        match stage_command_buffer.stage {
            J2kResidentEncodeGpuStage::CoefficientPrep => {
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientDeinterleaveRct => {
                stats.coefficient_deinterleave_rct_gpu_duration = stats
                    .coefficient_deinterleave_rct_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientDwt53 => {
                stats.coefficient_dwt53_gpu_duration = stats
                    .coefficient_dwt53_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientDwt53Vertical => {
                stats.coefficient_dwt53_vertical_gpu_duration = stats
                    .coefficient_dwt53_vertical_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_dwt53_gpu_duration = stats
                    .coefficient_dwt53_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientDwt53Horizontal => {
                stats.coefficient_dwt53_horizontal_gpu_duration = stats
                    .coefficient_dwt53_horizontal_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_dwt53_gpu_duration = stats
                    .coefficient_dwt53_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientExtract => {
                stats.coefficient_extract_gpu_duration = stats
                    .coefficient_extract_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientCopy => {
                stats.coefficient_copy_gpu_duration =
                    stats.coefficient_copy_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicBlock => {
                stats.classic_block_gpu_duration =
                    stats.classic_block_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1Density => {
                stats.classic_tier1_density_gpu_duration = stats
                    .classic_tier1_density_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1RawPack => {
                stats.classic_tier1_raw_pack_gpu_duration = stats
                    .classic_tier1_raw_pack_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1ArithmeticPack => {
                stats.classic_tier1_arithmetic_pack_gpu_duration = stats
                    .classic_tier1_arithmetic_pack_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1SymbolPlan => {
                stats.classic_tier1_symbol_plan_gpu_duration = stats
                    .classic_tier1_symbol_plan_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1PassPlan => {
                stats.classic_tier1_pass_plan_gpu_duration = stats
                    .classic_tier1_pass_plan_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1TokenEmit => {
                stats.classic_tier1_token_emit_gpu_duration = stats
                    .classic_tier1_token_emit_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1SplitTokenEmit => {
                stats.classic_tier1_split_token_emit_gpu_duration = stats
                    .classic_tier1_split_token_emit_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1TokenPack => {
                stats.classic_tier1_token_pack_gpu_duration = stats
                    .classic_tier1_token_pack_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::HtBlock => {
                stats.ht_block_gpu_duration = stats.ht_block_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::PacketBlockPrep => {
                stats.packet_block_prep_gpu_duration = stats
                    .packet_block_prep_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::Packetization => {
                stats.packetization_gpu_duration =
                    stats.packetization_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::PacketPayloadCopy => {
                stats.packet_payload_copy_gpu_duration = stats
                    .packet_payload_copy_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CodestreamAssembly => {
                stats.codestream_assembly_gpu_duration = stats
                    .codestream_assembly_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CodestreamPayloadCopy => {
                stats.codestream_payload_copy_gpu_duration = stats
                    .codestream_payload_copy_gpu_duration
                    .saturating_add(duration);
            }
        }
    }
}

pub(super) fn new_resident_encode_command_buffer(
    runtime: &MetalRuntime,
    label: &str,
) -> Result<CommandBuffer, Error> {
    let command_buffer = new_command_buffer(&runtime.queue)?;
    label_command_buffer(&command_buffer, label);
    Ok(command_buffer)
}

pub(super) fn finish_resident_encode_split_command_buffer(
    command_buffer: CommandBuffer,
    runtime: &MetalRuntime,
    stage: J2kResidentEncodeGpuStage,
    next_label: &str,
    command_buffers: &mut Vec<J2kResidentEncodeGpuStageCommandBuffer>,
) -> Result<CommandBuffer, Error> {
    command_buffer.commit();
    command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
        stage,
        command_buffer,
    });
    new_resident_encode_command_buffer(runtime, next_label)
}

pub(super) fn finish_resident_encode_split_command_buffer_timed(
    command_buffer: CommandBuffer,
    runtime: &MetalRuntime,
    stage: J2kResidentEncodeGpuStage,
    next_label: &str,
    command_buffers: &mut Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    profile_stages: bool,
    accumulated: &mut Duration,
) -> Result<CommandBuffer, Error> {
    let started = profile_stages.then(Instant::now);
    let next = finish_resident_encode_split_command_buffer(
        command_buffer,
        runtime,
        stage,
        next_label,
        command_buffers,
    )?;
    if let Some(started) = started {
        *accumulated = accumulated.saturating_add(started.elapsed());
    }
    Ok(next)
}
