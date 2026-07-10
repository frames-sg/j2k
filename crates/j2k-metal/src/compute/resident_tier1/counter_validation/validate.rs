// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_buffer_slice, encode_status_error, metal_profile_classic_tier1_token_pack_enabled,
    pack_j2k_code_block_scalar_from_tier1_tokens, Error, Instant, IntoParallelIterator,
    J2kClassicTier1PassPlanCounters, J2kClassicTier1SymbolPlanCounters,
    J2kClassicTier1TokenSegment, J2kResidentClassicTier1PassPlanReadback,
    J2kResidentClassicTier1SplitTokenBuffers, J2kResidentClassicTier1SymbolPlanReadback,
    J2kResidentClassicTier1TokenEmitReadback, J2kResidentEncodeStageStats, J2kTier1TokenSegment,
    ParallelIterator, CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY, J2K_ENCODE_STATUS_OK,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn compare_classic_tier1_symbol_plan_and_pass_plan_counters(
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
pub(in crate::compute) fn record_classic_tier1_token_emit_counters(
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
pub(in crate::compute) fn compare_classic_tier1_symbol_plan_and_token_emit_counters(
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
pub(in crate::compute) fn validate_classic_tier1_split_token_emit_counters(
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
pub(in crate::compute) fn compare_classic_tier1_symbol_plan_and_split_token_emit_counters(
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
pub(in crate::compute) fn profile_classic_tier1_token_pack(
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
