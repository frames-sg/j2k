// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    add_resident_prep_wall_duration, compute, resident_codestream_assembly_job_for_metadata,
    resident_lossless_chunk_ranges_from_code_blocks, resident_lossless_code_block_chunk_cap,
    Duration, Instant, J2kBlockCodingMode, MetalLosslessEncodeBatchStats,
    PlannedResidentLosslessBufferEncode, PreparedResidentLosslessBufferEncode,
    SubmittedResidentLosslessMetalBufferEncodeBatchKind,
    SubmittedResidentLosslessMetalBufferEncodeChunk,
};

#[cfg(target_os = "macos")]
pub(super) fn submit_planned_resident_lossless_tiles(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<SubmittedResidentLosslessMetalBufferEncodeBatchKind, crate::Error> {
    if planned.is_empty() {
        return Ok(SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty);
    }
    if planned.iter().all(|planned| {
        planned.metadata.plan.block_coding_mode == J2kBlockCodingMode::HighThroughput
    }) {
        return submit_planned_resident_ht_lossless_tiles_batch(
            planned,
            session,
            inflight_tiles,
            stats,
        );
    }
    if planned
        .iter()
        .all(|planned| planned.metadata.plan.block_coding_mode == J2kBlockCodingMode::Classic)
    {
        return submit_planned_resident_classic_lossless_tiles_batch(
            planned,
            session,
            inflight_tiles,
            stats,
        );
    }
    Ok(SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty)
}

#[cfg(target_os = "macos")]
struct PreparedResidentLosslessBatchItem {
    prepared: PreparedResidentLosslessBufferEncode,
    prepare_duration: Duration,
}

#[cfg(target_os = "macos")]
fn prepare_planned_resident_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
) -> Result<Vec<PreparedResidentLosslessBatchItem>, crate::Error> {
    struct BatchPlanInfo {
        index: usize,
        coefficient_count: usize,
        bytes_per_sample: u8,
        code_blocks: Vec<compute::J2kLosslessDeviceCodeBlock>,
    }

    let started = Instant::now();
    let mut metadatas = Vec::with_capacity(planned.len());
    let mut plan_infos = Vec::with_capacity(planned.len());
    for planned in planned {
        #[cfg(test)]
        if planned.failure_injection_index == Some(planned.index) {
            return Err(crate::Error::MetalKernel {
                message: format!(
                    "injected J2K Metal resident encode failure at tile {}",
                    planned.index
                ),
            });
        }

        plan_infos.push(BatchPlanInfo {
            index: planned.index,
            coefficient_count: planned.coefficient_count,
            bytes_per_sample: planned.bytes_per_sample,
            code_blocks: planned.metadata.plan.code_blocks.clone(),
        });
        metadatas.push(planned.metadata);
    }

    let mut batch_items = Vec::with_capacity(metadatas.len());
    for (metadata, plan_info) in metadatas.iter().zip(plan_infos) {
        let tile = metadata.tile.as_tile();
        batch_items.push(compute::J2kLosslessDeviceBatchPrepareItem {
            tile_index: plan_info.index,
            job: compute::J2kLosslessDevicePrepareJob {
                input: tile.buffer,
                input_byte_offset: tile.byte_offset,
                input_width: tile.width,
                input_height: tile.height,
                input_pitch_bytes: tile.pitch_bytes,
                output_width: tile.output_width,
                output_height: tile.output_height,
                component_count: metadata.components,
                bytes_per_sample: plan_info.bytes_per_sample,
                bit_depth: metadata.bit_depth,
                num_decomposition_levels: metadata.plan.num_decomposition_levels,
                coefficient_count: plan_info.coefficient_count,
            },
            code_blocks: plan_info.code_blocks,
        });
    }

    let prepared = compute::prepare_lossless_device_code_blocks_batch(session, batch_items)?;
    let prepare_duration = duration_share(started.elapsed(), prepared.len());
    Ok(metadatas
        .into_iter()
        .zip(prepared)
        .map(|(metadata, prepared)| PreparedResidentLosslessBatchItem {
            prepared: PreparedResidentLosslessBufferEncode { metadata, prepared },
            prepare_duration,
        })
        .collect())
}

#[cfg(target_os = "macos")]
fn submit_planned_resident_ht_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<SubmittedResidentLosslessMetalBufferEncodeBatchKind, crate::Error> {
    let code_block_counts = planned
        .iter()
        .map(|planned| planned.metadata.plan.code_blocks.len())
        .collect::<Vec<_>>();
    let chunk_ranges = resident_lossless_chunk_ranges_from_code_blocks(
        &code_block_counts,
        inflight_tiles,
        resident_lossless_code_block_chunk_cap(&code_block_counts),
    );
    submit_planned_resident_lossless_tiles_chunked(
        planned,
        session,
        stats,
        "HT",
        chunk_ranges,
        true,
        |session, batch_items| {
            compute::submit_lossless_codestream_buffers_from_prepared_ht_batch(
                session,
                batch_items,
                compute::ht_packet_output_capacity_mode_from_env(),
            )
        },
    )
}

#[cfg(target_os = "macos")]
fn submit_planned_resident_classic_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<SubmittedResidentLosslessMetalBufferEncodeBatchKind, crate::Error> {
    let batch_limit = inflight_tiles.max(1);
    let chunk_ranges = (0..planned.len())
        .step_by(batch_limit)
        .map(|start| start..(start + batch_limit).min(planned.len()))
        .collect::<Vec<_>>();
    submit_planned_resident_lossless_tiles_chunked(
        planned,
        session,
        stats,
        "classic",
        chunk_ranges,
        false,
        |session, batch_items| {
            compute::submit_lossless_codestream_buffers_from_prepared_classic_batch(
                session,
                batch_items,
                compute::J2kClassicEncodeOutputCapacityMode::Tight,
            )
        },
    )
}

/// Shared chunked submit driver for the per-family resident lossless batch
/// paths. `time_prepare_in_submit` preserves each family's historical
/// `prepare_submit_duration` semantics: HT (`true`) measures prepare + item
/// build + submit, classic (`false`) measures only the submit call.
#[cfg(target_os = "macos")]
fn submit_planned_resident_lossless_tiles_chunked(
    mut planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    stats: &mut MetalLosslessEncodeBatchStats,
    family_name: &str,
    chunk_ranges: Vec<std::ops::Range<usize>>,
    time_prepare_in_submit: bool,
    submit_chunk: impl Fn(
        &crate::MetalBackendSession,
        Vec<compute::J2kResidentBatchEncodeItem>,
    )
        -> Result<compute::J2kPendingResidentLosslessCodestreamBatch, crate::Error>,
) -> Result<SubmittedResidentLosslessMetalBufferEncodeBatchKind, crate::Error> {
    let planned_len = planned.len();
    let profile_stages = compute::metal_profile_stages_enabled();
    if profile_stages {
        stats.stage_stats.chunk_count = stats
            .stage_stats
            .chunk_count
            .saturating_add(chunk_ranges.len());
        stats.stage_stats.tile_count = stats.stage_stats.tile_count.saturating_add(planned_len);
    }
    stats.max_observed_inflight_tiles = stats.max_observed_inflight_tiles.max(
        chunk_ranges
            .iter()
            .map(std::ops::Range::len)
            .max()
            .unwrap_or(0),
    );

    let mut chunks = Vec::with_capacity(chunk_ranges.len());
    for range in chunk_ranges {
        let take = range.len();
        let chunk_planned = planned.drain(..take).collect::<Vec<_>>();
        let early_prepare_submit_started =
            (profile_stages && time_prepare_in_submit).then(Instant::now);
        let prep_wall_started = profile_stages.then(Instant::now);
        let prepared = prepare_planned_resident_lossless_tiles_batch(chunk_planned, session)
            .map_err(|err| crate::Error::MetalKernel {
                message: format!("J2K Metal resident {family_name} batch encode failed: {err}"),
            })?;
        if let Some(started) = prep_wall_started {
            add_resident_prep_wall_duration(stats, started.elapsed(), profile_stages);
        }

        let mut metadatas = Vec::with_capacity(prepared.len());
        let mut prepare_durations = Vec::with_capacity(prepared.len());
        let mut batch_items = Vec::with_capacity(prepared.len());
        for item in prepared {
            let PreparedResidentLosslessBatchItem {
                prepared,
                prepare_duration,
            } = item;
            let metadata = prepared.metadata;
            let codestream = resident_codestream_assembly_job_for_metadata(&metadata);
            batch_items.push(compute::J2kResidentBatchEncodeItem {
                prepared: prepared.prepared,
                resolution_count: u32::try_from(metadata.plan.resolutions.len()).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode resolution count exceeds u32"
                            .to_string(),
                    }
                })?,
                num_layers: 1,
                component_count: metadata.plan.components,
                code_block_count: u32::try_from(metadata.plan.code_blocks.len()).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode code-block count exceeds u32"
                            .to_string(),
                    }
                })?,
                packet_descriptors: metadata.packet_descriptors.clone(),
                resolutions: metadata.packetization_resolutions.clone(),
                codestream,
            });
            prepare_durations.push(prepare_duration);
            metadatas.push(metadata);
        }

        let batch_started = Instant::now();
        let prepare_submit_started = if time_prepare_in_submit {
            early_prepare_submit_started
        } else {
            profile_stages.then(Instant::now)
        };
        let pending = submit_chunk(session, batch_items)?;
        if let Some(started) = prepare_submit_started {
            stats.stage_stats.prepare_submit_duration = stats
                .stage_stats
                .prepare_submit_duration
                .saturating_add(started.elapsed());
        }
        chunks.push(SubmittedResidentLosslessMetalBufferEncodeChunk {
            metadatas,
            prepare_durations,
            pending,
            batch_started,
        });
    }

    if !planned.is_empty() {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident {family_name} batch chunking left unsubmitted tiles"
            ),
        });
    }

    if chunks.is_empty() && planned_len > 0 {
        return Err(crate::Error::MetalKernel {
            message: format!("J2K Metal resident {family_name} batch chunking produced no chunks"),
        });
    }

    Ok(SubmittedResidentLosslessMetalBufferEncodeBatchKind::Chunks(
        chunks,
    ))
}

#[cfg(target_os = "macos")]
pub(super) fn duration_share(duration: Duration, count: usize) -> Duration {
    if count == 0 {
        return Duration::ZERO;
    }
    let nanos = duration.as_nanos() / count as u128;
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}
