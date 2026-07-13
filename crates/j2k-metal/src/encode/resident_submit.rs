// SPDX-License-Identifier: MIT OR Apache-2.0

use super::resident_prepare::{
    prepare_planned_resident_lossless_tiles_batch, PreparedResidentLosslessBatchItem,
};

struct ResidentChunkItems {
    metadatas: Vec<super::ResidentLosslessBufferEncodeMetadata>,
    prepare_durations: Vec<Duration>,
    batch_items: Vec<compute::J2kResidentBatchEncodeItem>,
}

fn build_resident_chunk_items(
    prepared: Vec<PreparedResidentLosslessBatchItem>,
    budget: &mut crate::batch_allocation::BatchMetadataBudget,
) -> Result<ResidentChunkItems, crate::Error> {
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<
            super::ResidentLosslessBufferEncodeMetadata,
        >(prepared.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<Duration>(prepared.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<compute::J2kResidentBatchEncodeItem>(
            prepared.len(),
        ),
    ])?;
    let mut metadatas = budget.try_vec(prepared.len(), "J2K Metal resident chunk metadata")?;
    let mut prepare_durations = budget.try_vec(
        prepared.len(),
        "J2K Metal resident chunk preparation durations",
    )?;
    let mut batch_items =
        budget.try_vec(prepared.len(), "J2K Metal resident chunk encode items")?;
    for item in prepared {
        let PreparedResidentLosslessBatchItem {
            prepared,
            prepare_duration,
        } = item;
        let mut metadata = prepared.metadata;
        let codestream = resident_codestream_assembly_job_for_metadata(&metadata);
        let packet_descriptors = metadata.take_packet_descriptors();
        let resolutions = metadata.take_packetization_resolutions();
        batch_items.push(compute::J2kResidentBatchEncodeItem {
            prepared: prepared.prepared,
            resolution_count: u32::try_from(metadata.resolution_count).map_err(|_| {
                crate::Error::MetalKernel {
                    message: "J2K Metal resident encode resolution count exceeds u32".to_string(),
                }
            })?,
            num_layers: 1,
            component_count: metadata.plan.components,
            code_block_count: u32::try_from(metadata.code_block_count).map_err(|_| {
                crate::Error::MetalKernel {
                    message: "J2K Metal resident encode code-block count exceeds u32".to_string(),
                }
            })?,
            packet_descriptors,
            resolutions,
            codestream,
        });
        prepare_durations.push(prepare_duration);
        metadatas.push(metadata);
    }
    Ok(ResidentChunkItems {
        metadatas,
        prepare_durations,
        batch_items,
    })
}
use super::{
    add_resident_prep_wall_duration, compute, resident_codestream_assembly_job_for_metadata,
    resident_lossless_chunk_ranges_from_code_blocks, resident_lossless_code_block_chunk_cap,
    Duration, Instant, J2kBlockCodingMode, MetalLosslessEncodeBatchStats,
    PlannedResidentLosslessBufferEncode, SubmittedResidentLosslessMetalBufferEncodeBatchKind,
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
fn submit_planned_resident_ht_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<SubmittedResidentLosslessMetalBufferEncodeBatchKind, crate::Error> {
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal resident HT chunk plan");
    let mut code_block_counts =
        budget.try_vec(planned.len(), "J2K Metal resident HT code-block counts")?;
    code_block_counts.extend(
        planned
            .iter()
            .map(|planned| planned.metadata.plan.code_blocks.len()),
    );
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
    let chunk_count = planned.len().div_ceil(batch_limit);
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal resident classic chunk plan");
    let mut chunk_ranges =
        budget.try_vec(chunk_count, "J2K Metal resident classic chunk ranges")?;
    chunk_ranges.extend(
        (0..planned.len())
            .step_by(batch_limit)
            .map(|start| start..(start + batch_limit).min(planned.len())),
    );
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

    let mut chunk_budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal resident submitted chunks");
    let mut chunks = chunk_budget.try_vec(
        chunk_ranges.len(),
        "J2K Metal resident submitted chunk records",
    )?;
    for range in chunk_ranges {
        let take = range.len();
        let mut iteration_budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal resident chunk preparation",
        );
        let mut chunk_planned =
            iteration_budget.try_vec(take, "J2K Metal resident chunk planned tiles")?;
        chunk_planned.extend(planned.drain(..take));
        let early_prepare_submit_started =
            (profile_stages && time_prepare_in_submit).then(Instant::now);
        let prep_wall_started = profile_stages.then(Instant::now);
        let prepared = prepare_planned_resident_lossless_tiles_batch(chunk_planned, session)?;
        if let Some(started) = prep_wall_started {
            add_resident_prep_wall_duration(stats, started.elapsed(), profile_stages);
        }

        let ResidentChunkItems {
            metadatas,
            prepare_durations,
            batch_items,
        } = build_resident_chunk_items(prepared, &mut iteration_budget)?;

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
