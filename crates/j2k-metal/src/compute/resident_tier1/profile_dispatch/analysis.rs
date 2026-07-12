// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    classic_encode_code_blocks_pipeline_kind, label_compute_encoder,
    metal_profile_classic_tier1_arithmetic_pack_enabled,
    metal_profile_classic_tier1_density_enabled, metal_profile_classic_tier1_pass_plan_enabled,
    metal_profile_classic_tier1_raw_pack_enabled, metal_profile_classic_tier1_symbol_plan_enabled,
    metal_profile_classic_tier1_token_emit_enabled, new_compute_command_encoder,
    new_private_buffer, new_shared_buffer, size_of, Buffer, CommandBufferRef, Error,
    J2kClassicEncodeBatchJob, J2kClassicEncodePipelineKind, J2kClassicTier1DensityCounters,
    J2kClassicTier1PassPlanCounters, J2kClassicTier1SymbolPlanCounters,
    J2kClassicTier1TokenSegment, J2kResidentClassicTier1DensityReadback,
    J2kResidentClassicTier1PassPlanReadback, J2kResidentClassicTier1SymbolPlanReadback,
    J2kResidentClassicTier1TokenEmitReadback, MTLSize, MetalRuntime,
    CLASSIC_TIER1_TOKEN_ARENA_BYTES, CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_classic_tier1_density_profile(
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

    let counter_bytes = tier1_jobs
        .len()
        .max(1)
        .checked_mul(size_of::<J2kClassicTier1DensityCounters>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 density counter size overflow".to_string(),
        })?;
    let counter_buffer = new_shared_buffer(&runtime.device, counter_bytes)?;
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 density job count exceeds u32".to_string(),
    })?;
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K classic Tier-1 density profile");
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
pub(in crate::compute) fn dispatch_classic_tier1_raw_pack_profile(
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

    let raw_output_buffer =
        new_private_buffer(&runtime.device, tier1_output_capacity_total.max(1))?;
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 raw-pack job count exceeds u32".to_string(),
    })?;
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K classic Tier-1 raw-pack profile");
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
pub(in crate::compute) fn dispatch_classic_tier1_arithmetic_pack_profile(
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

    let arithmetic_output_buffer =
        new_private_buffer(&runtime.device, tier1_output_capacity_total.max(1))?;
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 arithmetic-pack job count exceeds u32".to_string(),
    })?;
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K classic Tier-1 arithmetic-pack profile");
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
pub(in crate::compute) fn dispatch_classic_tier1_symbol_plan_profile(
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

    let counter_bytes = tier1_jobs
        .len()
        .max(1)
        .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 symbol counter size overflow".to_string(),
        })?;
    let counter_buffer = new_shared_buffer(&runtime.device, counter_bytes)?;
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 symbol-plan job count exceeds u32".to_string(),
    })?;
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K classic Tier-1 symbol plan");
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
pub(in crate::compute) fn dispatch_classic_tier1_pass_plan_profile(
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

    let counter_bytes = tier1_jobs
        .len()
        .max(1)
        .checked_mul(size_of::<J2kClassicTier1PassPlanCounters>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 pass counter size overflow".to_string(),
        })?;
    let counter_buffer = new_shared_buffer(&runtime.device, counter_bytes)?;
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 pass-plan job count exceeds u32".to_string(),
    })?;
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K classic Tier-1 pass plan");
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
pub(in crate::compute) fn dispatch_classic_tier1_token_emit_profile(
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

    let counter_bytes = tier1_jobs
        .len()
        .max(1)
        .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token counter size overflow".to_string(),
        })?;
    let counter_buffer = new_shared_buffer(&runtime.device, counter_bytes)?;
    let token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token buffer size overflow".to_string(),
        })?;
    let token_buffer = new_shared_buffer(&runtime.device, token_buffer_len)?;
    let segment_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY)
        .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token segment buffer size overflow".to_string(),
        })?;
    let segment_buffer = new_shared_buffer(&runtime.device, segment_buffer_len)?;
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

    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K classic Tier-1 token emit");
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
