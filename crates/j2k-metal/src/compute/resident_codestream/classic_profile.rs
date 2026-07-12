// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::CommandBuffer;

use super::super::resident_tier1::{
    J2kResidentClassicTier1DensityReadback, J2kResidentClassicTier1PassPlanReadback,
    J2kResidentClassicTier1SplitTokenBuffers, J2kResidentClassicTier1SymbolPlanReadback,
    J2kResidentClassicTier1TokenEmitReadback,
};
use super::{
    dispatch_classic_tier1_arithmetic_pack_profile, dispatch_classic_tier1_density_profile,
    dispatch_classic_tier1_pass_plan_profile, dispatch_classic_tier1_raw_pack_profile,
    dispatch_classic_tier1_split_token_emit_profile, dispatch_classic_tier1_symbol_plan_profile,
    dispatch_classic_tier1_token_emit_profile, finish_resident_encode_split_command_buffer_timed,
    metal_profile_classic_tier1_arithmetic_pack_enabled,
    metal_profile_classic_tier1_density_enabled, metal_profile_classic_tier1_pass_plan_enabled,
    metal_profile_classic_tier1_raw_pack_enabled,
    metal_profile_classic_tier1_split_token_emit_enabled,
    metal_profile_classic_tier1_symbol_plan_enabled,
    metal_profile_classic_tier1_token_emit_enabled, next_enabled_classic_stage_label, Buffer,
    Duration, Error, J2kClassicEncodeBatchJob, J2kResidentEncodeGpuStage,
    J2kResidentEncodeGpuStageCommandBuffer, MetalRuntime, CLASSIC_TIER1_ARITHMETIC_PACK_LABEL,
    CLASSIC_TIER1_DENSITY_LABEL, CLASSIC_TIER1_PASS_PLAN_LABEL, CLASSIC_TIER1_RAW_PACK_LABEL,
    CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL, CLASSIC_TIER1_SYMBOL_PLAN_LABEL,
    CLASSIC_TIER1_TOKEN_EMIT_LABEL,
};

#[derive(Clone, Copy)]
pub(super) struct ClassicProfileStages {
    enabled: [bool; 6],
    pub(super) token_pack_next_label: &'static str,
}

pub(super) fn classic_profile_stages_from_env() -> ClassicProfileStages {
    let profile_classic_tier1_density = metal_profile_classic_tier1_density_enabled();
    let profile_classic_tier1_raw_pack = metal_profile_classic_tier1_raw_pack_enabled();
    let profile_classic_tier1_arithmetic_pack =
        metal_profile_classic_tier1_arithmetic_pack_enabled();
    let profile_classic_tier1_pass_plan = metal_profile_classic_tier1_pass_plan_enabled();
    let profile_classic_tier1_symbol_plan = metal_profile_classic_tier1_symbol_plan_enabled();
    let profile_classic_tier1_token_emit = metal_profile_classic_tier1_token_emit_enabled();
    let profile_classic_tier1_split_token_emit =
        metal_profile_classic_tier1_split_token_emit_enabled();
    let classic_token_pack_next_label = next_enabled_classic_stage_label(&[
        (profile_classic_tier1_density, CLASSIC_TIER1_DENSITY_LABEL),
        (profile_classic_tier1_raw_pack, CLASSIC_TIER1_RAW_PACK_LABEL),
        (
            profile_classic_tier1_arithmetic_pack,
            CLASSIC_TIER1_ARITHMETIC_PACK_LABEL,
        ),
        (
            profile_classic_tier1_symbol_plan,
            CLASSIC_TIER1_SYMBOL_PLAN_LABEL,
        ),
        (
            profile_classic_tier1_token_emit,
            CLASSIC_TIER1_TOKEN_EMIT_LABEL,
        ),
        (
            profile_classic_tier1_split_token_emit,
            CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
        ),
    ]);

    ClassicProfileStages {
        enabled: [
            profile_classic_tier1_raw_pack,
            profile_classic_tier1_arithmetic_pack,
            profile_classic_tier1_pass_plan,
            profile_classic_tier1_symbol_plan,
            profile_classic_tier1_token_emit,
            profile_classic_tier1_split_token_emit,
        ],
        token_pack_next_label: classic_token_pack_next_label,
    }
}

pub(super) struct ClassicTier1ProfileResult {
    pub(super) command_buffer: CommandBuffer,
    pub(super) classic_tier1_density_readback: Option<J2kResidentClassicTier1DensityReadback>,
    pub(super) classic_tier1_raw_pack_buffer: Option<Buffer>,
    pub(super) classic_tier1_arithmetic_pack_buffer: Option<Buffer>,
    pub(super) classic_tier1_symbol_plan_readback:
        Option<J2kResidentClassicTier1SymbolPlanReadback>,
    pub(super) classic_tier1_pass_plan_readback: Option<J2kResidentClassicTier1PassPlanReadback>,
    pub(super) classic_tier1_token_emit_readback: Option<J2kResidentClassicTier1TokenEmitReadback>,
    pub(super) classic_tier1_split_token_emit_readback:
        Option<J2kResidentClassicTier1SplitTokenBuffers>,
}

pub(super) struct ClassicTier1ProfileRequest<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: CommandBuffer,
    pub(super) coefficient_buffer: &'a Buffer,
    pub(super) tier1_job_buffer: &'a Buffer,
    pub(super) tier1_jobs: &'a [J2kClassicEncodeBatchJob],
    pub(super) tier1_job_count: u32,
    pub(super) tier1_output_capacity_total: usize,
    pub(super) classic_gpu_token_pack_readback: Option<J2kResidentClassicTier1TokenEmitReadback>,
    pub(super) profile_stages: bool,
    pub(super) stages: ClassicProfileStages,
    pub(super) gpu_stage_command_buffers: &'a mut Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    pub(super) classic_command_buffer_commit_duration: &'a mut Duration,
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "profile mode dispatch is an exhaustive ordered Metal pipeline"
)]
pub(super) fn dispatch_classic_tier1_profiles(
    request: ClassicTier1ProfileRequest<'_>,
) -> Result<ClassicTier1ProfileResult, Error> {
    let ClassicTier1ProfileRequest {
        runtime,
        mut command_buffer,
        coefficient_buffer,
        tier1_job_buffer,
        tier1_jobs,
        tier1_job_count,
        tier1_output_capacity_total,
        classic_gpu_token_pack_readback,
        profile_stages,
        stages,
        gpu_stage_command_buffers,
        classic_command_buffer_commit_duration,
    } = request;
    let split_command_buffers = true;
    let ClassicProfileStages {
        enabled:
            [profile_classic_tier1_raw_pack, profile_classic_tier1_arithmetic_pack, profile_classic_tier1_pass_plan, profile_classic_tier1_symbol_plan, profile_classic_tier1_token_emit, profile_classic_tier1_split_token_emit],
        token_pack_next_label: _,
    } = stages;
    let classic_tier1_density_readback = if tier1_job_count > 0 {
        let readback = dispatch_classic_tier1_density_profile(
            runtime,
            &command_buffer,
            coefficient_buffer,
            tier1_job_buffer,
            tier1_jobs,
        )?;
        if readback.is_some() && split_command_buffers {
            let next_label = next_enabled_classic_stage_label(&[
                (profile_classic_tier1_raw_pack, CLASSIC_TIER1_RAW_PACK_LABEL),
                (
                    profile_classic_tier1_arithmetic_pack,
                    CLASSIC_TIER1_ARITHMETIC_PACK_LABEL,
                ),
                (
                    profile_classic_tier1_symbol_plan,
                    CLASSIC_TIER1_SYMBOL_PLAN_LABEL,
                ),
                (
                    profile_classic_tier1_token_emit,
                    CLASSIC_TIER1_TOKEN_EMIT_LABEL,
                ),
                (
                    profile_classic_tier1_split_token_emit,
                    CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
                ),
            ]);
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::ClassicTier1Density,
                next_label,
                gpu_stage_command_buffers,
                profile_stages,
                classic_command_buffer_commit_duration,
            )?;
        }
        readback
    } else {
        None
    };
    let classic_tier1_raw_pack_buffer = if tier1_job_count > 0 {
        let buffer = dispatch_classic_tier1_raw_pack_profile(
            runtime,
            &command_buffer,
            coefficient_buffer,
            tier1_job_buffer,
            tier1_jobs,
            tier1_output_capacity_total,
        )?;
        if buffer.is_some() && split_command_buffers {
            let next_label = next_enabled_classic_stage_label(&[
                (
                    profile_classic_tier1_arithmetic_pack,
                    CLASSIC_TIER1_ARITHMETIC_PACK_LABEL,
                ),
                (
                    profile_classic_tier1_symbol_plan,
                    CLASSIC_TIER1_SYMBOL_PLAN_LABEL,
                ),
                (
                    profile_classic_tier1_token_emit,
                    CLASSIC_TIER1_TOKEN_EMIT_LABEL,
                ),
                (
                    profile_classic_tier1_split_token_emit,
                    CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
                ),
            ]);
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::ClassicTier1RawPack,
                next_label,
                gpu_stage_command_buffers,
                profile_stages,
                classic_command_buffer_commit_duration,
            )?;
        }
        buffer
    } else {
        None
    };
    let classic_tier1_arithmetic_pack_buffer = if tier1_job_count > 0 {
        let buffer = dispatch_classic_tier1_arithmetic_pack_profile(
            runtime,
            &command_buffer,
            coefficient_buffer,
            tier1_job_buffer,
            tier1_jobs,
            tier1_output_capacity_total,
        )?;
        if buffer.is_some() && split_command_buffers {
            let next_label = next_enabled_classic_stage_label(&[
                (
                    profile_classic_tier1_symbol_plan,
                    CLASSIC_TIER1_SYMBOL_PLAN_LABEL,
                ),
                (
                    profile_classic_tier1_token_emit,
                    CLASSIC_TIER1_TOKEN_EMIT_LABEL,
                ),
                (
                    profile_classic_tier1_split_token_emit,
                    CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
                ),
            ]);
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::ClassicTier1ArithmeticPack,
                next_label,
                gpu_stage_command_buffers,
                profile_stages,
                classic_command_buffer_commit_duration,
            )?;
        }
        buffer
    } else {
        None
    };
    let classic_tier1_symbol_plan_readback = if tier1_job_count > 0 {
        let readback = dispatch_classic_tier1_symbol_plan_profile(
            runtime,
            &command_buffer,
            coefficient_buffer,
            tier1_job_buffer,
            tier1_jobs,
        )?;
        if readback.is_some() && split_command_buffers {
            let next_label = next_enabled_classic_stage_label(&[
                (
                    profile_classic_tier1_pass_plan,
                    CLASSIC_TIER1_PASS_PLAN_LABEL,
                ),
                (
                    profile_classic_tier1_token_emit,
                    CLASSIC_TIER1_TOKEN_EMIT_LABEL,
                ),
                (
                    profile_classic_tier1_split_token_emit,
                    CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
                ),
            ]);
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::ClassicTier1SymbolPlan,
                next_label,
                gpu_stage_command_buffers,
                profile_stages,
                classic_command_buffer_commit_duration,
            )?;
        }
        readback
    } else {
        None
    };
    let classic_tier1_pass_plan_readback = if tier1_job_count > 0 {
        let readback = dispatch_classic_tier1_pass_plan_profile(
            runtime,
            &command_buffer,
            coefficient_buffer,
            tier1_job_buffer,
            tier1_jobs,
        )?;
        if readback.is_some() && split_command_buffers {
            let next_label = next_enabled_classic_stage_label(&[
                (
                    profile_classic_tier1_token_emit,
                    CLASSIC_TIER1_TOKEN_EMIT_LABEL,
                ),
                (
                    profile_classic_tier1_split_token_emit,
                    CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
                ),
            ]);
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::ClassicTier1PassPlan,
                next_label,
                gpu_stage_command_buffers,
                profile_stages,
                classic_command_buffer_commit_duration,
            )?;
        }
        readback
    } else {
        None
    };
    let classic_tier1_token_emit_readback = if classic_gpu_token_pack_readback.is_some() {
        classic_gpu_token_pack_readback
    } else if tier1_job_count > 0 {
        let readback = dispatch_classic_tier1_token_emit_profile(
            runtime,
            &command_buffer,
            coefficient_buffer,
            tier1_job_buffer,
            tier1_jobs,
        )?;
        if readback.is_some() && split_command_buffers {
            let next_label = next_enabled_classic_stage_label(&[(
                profile_classic_tier1_split_token_emit,
                CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL,
            )]);
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::ClassicTier1TokenEmit,
                next_label,
                gpu_stage_command_buffers,
                profile_stages,
                classic_command_buffer_commit_duration,
            )?;
        }
        readback
    } else {
        None
    };
    let classic_tier1_split_token_emit_readback = if tier1_job_count > 0 {
        let readback = dispatch_classic_tier1_split_token_emit_profile(
            runtime,
            &command_buffer,
            coefficient_buffer,
            tier1_job_buffer,
            tier1_jobs,
        )?;
        if readback.is_some() && split_command_buffers {
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::ClassicTier1SplitTokenEmit,
                "j2k classic resident packetization",
                gpu_stage_command_buffers,
                profile_stages,
                classic_command_buffer_commit_duration,
            )?;
        }
        readback
    } else {
        None
    };

    Ok(ClassicTier1ProfileResult {
        command_buffer,
        classic_tier1_density_readback,
        classic_tier1_raw_pack_buffer,
        classic_tier1_arithmetic_pack_buffer,
        classic_tier1_symbol_plan_readback,
        classic_tier1_pass_plan_readback,
        classic_tier1_token_emit_readback,
        classic_tier1_split_token_emit_readback,
    })
}
