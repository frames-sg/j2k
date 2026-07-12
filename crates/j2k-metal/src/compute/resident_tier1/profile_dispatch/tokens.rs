// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::super::test_counters;
use super::super::{
    classic_tier1_gpu_token_pack_supported, dispatch_1d_pipeline, label_compute_encoder,
    metal_profile_classic_tier1_split_token_emit_enabled,
    metal_profile_classic_tier1_token_pack_enabled, new_blit_command_encoder,
    new_compute_command_encoder, new_shared_buffer, size_of, take_recyclable_private_buffer,
    Buffer, CommandBufferRef, Error, J2kClassicEncodeBatchJob, J2kClassicTier1SymbolPlanCounters,
    J2kClassicTier1TokenSegment, J2kResidentClassicTier1GpuTokenBuffers,
    J2kResidentClassicTier1SplitTokenBuffers, J2kResidentClassicTier1TokenEmitReadback, MTLSize,
    MetalRuntime, CLASSIC_TIER1_MQ_BYTE_TOKEN_ARENA_BYTES, CLASSIC_TIER1_TOKEN_ARENA_BYTES,
    CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY,
};

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "profile dispatch keeps Metal bindings and retained buffers in ABI order"
)]
pub(in crate::compute) fn dispatch_classic_tier1_split_token_emit_for_cpu_pack(
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

    let counter_bytes = tier1_jobs
        .len()
        .max(1)
        .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split-token counter size overflow".to_string(),
        })?;
    let counter_buffer = new_shared_buffer(&runtime.device, counter_bytes)?;
    let mq_token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split-token MQ buffer size overflow".to_string(),
        })?;
    let mq_token_buffer = new_shared_buffer(&runtime.device, mq_token_buffer_len)?;
    let raw_token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split-token raw buffer size overflow".to_string(),
        })?;
    let raw_token_buffer = new_shared_buffer(&runtime.device, raw_token_buffer_len)?;
    let segment_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY)
        .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split-token segment buffer size overflow".to_string(),
        })?;
    let segment_buffer = new_shared_buffer(&runtime.device, segment_buffer_len)?;
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

    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K classic Tier-1 split token emit");
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
pub(in crate::compute) fn dispatch_classic_tier1_split_token_emit_profile(
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
#[expect(
    clippy::too_many_lines,
    reason = "profile dispatch keeps Metal bindings and retained buffers in ABI order"
)]
pub(in crate::compute) fn dispatch_classic_tier1_split_token_emit_for_gpu_pack(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
    recyclable_private_buffers: &mut Vec<crate::buffer_pool::PooledBuffer>,
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

    let encoder = new_compute_command_encoder(command_buffer)?;
    if use_mq_byte_emit {
        label_compute_encoder(&encoder, "J2K classic Tier-1 split MQ-byte token emit");
    } else {
        label_compute_encoder(&encoder, "J2K classic Tier-1 split token emit");
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
    dispatch_1d_pipeline(&encoder, emit_pipeline, u64::from(job_count));
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
pub(in crate::compute) fn dispatch_classic_tier1_token_emit_for_gpu_pack(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
    recyclable_private_buffers: &mut Vec<crate::buffer_pool::PooledBuffer>,
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
#[derive(Clone, Copy)]
enum ClassicTier1TokenPackBuffers<'a> {
    Combined(&'a J2kResidentClassicTier1GpuTokenBuffers),
    Split(&'a J2kResidentClassicTier1SplitTokenBuffers),
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_token_pack_from_buffers(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    tier1_job_buffer: &Buffer,
    token_buffers: ClassicTier1TokenPackBuffers<'_>,
    tier1_output_buffer: &Buffer,
    tier1_status_buffer: &Buffer,
    tier1_segment_buffer: &Buffer,
) -> Result<(), Error> {
    #[cfg(test)]
    test_counters::record_classic_gpu_token_pack_dispatch();

    let encoder = new_compute_command_encoder(command_buffer)?;
    let (pipeline, job_count) = match token_buffers {
        ClassicTier1TokenPackBuffers::Combined(token_buffers) => {
            let pipeline = &runtime.classic_tier1_token_pack_bypass_u16_32;
            label_compute_encoder(&encoder, "J2K classic Tier-1 token pack");
            encoder.set_compute_pipeline_state(pipeline);
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
            (pipeline, token_buffers.job_count)
        }
        ClassicTier1TokenPackBuffers::Split(token_buffers) => {
            let pipeline = &runtime.classic_tier1_split_token_pack_bypass_u16_32;
            label_compute_encoder(&encoder, "J2K classic Tier-1 split token pack");
            encoder.set_compute_pipeline_state(pipeline);
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
            (pipeline, token_buffers.job_count)
        }
    };
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: pipeline.thread_execution_width().max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_classic_tier1_token_pack_from_gpu_tokens(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    tier1_job_buffer: &Buffer,
    token_buffers: &J2kResidentClassicTier1GpuTokenBuffers,
    tier1_output_buffer: &Buffer,
    tier1_status_buffer: &Buffer,
    tier1_segment_buffer: &Buffer,
) -> Result<(), Error> {
    dispatch_classic_tier1_token_pack_from_buffers(
        runtime,
        command_buffer,
        tier1_job_buffer,
        ClassicTier1TokenPackBuffers::Combined(token_buffers),
        tier1_output_buffer,
        tier1_status_buffer,
        tier1_segment_buffer,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_classic_tier1_split_token_pack_from_gpu_tokens(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    tier1_job_buffer: &Buffer,
    token_buffers: &J2kResidentClassicTier1SplitTokenBuffers,
    tier1_output_buffer: &Buffer,
    tier1_status_buffer: &Buffer,
    tier1_segment_buffer: &Buffer,
) -> Result<(), Error> {
    dispatch_classic_tier1_token_pack_from_buffers(
        runtime,
        command_buffer,
        tier1_job_buffer,
        ClassicTier1TokenPackBuffers::Split(token_buffers),
        tier1_output_buffer,
        tier1_status_buffer,
        tier1_segment_buffer,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn schedule_classic_tier1_gpu_token_pack_readback(
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
    let counter_readback = new_shared_buffer(&runtime.device, counter_byte_len.max(1))?;

    let copy_token_payloads = metal_profile_classic_tier1_token_pack_enabled();
    let (token_readback, token_byte_len) = if copy_token_payloads {
        let byte_len = count
            .checked_mul(token_stride_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal classic GPU token-pack token readback size overflow"
                    .to_string(),
            })?;
        (
            Some(new_shared_buffer(&runtime.device, byte_len.max(1))?),
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
            Some(new_shared_buffer(&runtime.device, byte_len.max(1))?),
            byte_len,
        )
    } else {
        (None, 0)
    };

    let blit = new_blit_command_encoder(command_buffer)?;
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
