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
    Duration, Instant, MetalLosslessEncodeBatchStats, PlannedResidentLosslessBufferEncode,
    SubmittedResidentLosslessMetalBufferEncodeChunk,
};

#[derive(Clone, Copy)]
pub(super) enum ResidentSubmissionFamily {
    HighThroughput(compute::J2kHtPacketOutputCapacityMode),
    Classic,
}

impl ResidentSubmissionFamily {
    const fn time_prepare_in_submit(self) -> bool {
        matches!(self, Self::HighThroughput(_))
    }
}

#[cfg(target_os = "macos")]
pub(super) fn submit_resident_lossless_chunk(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    stats: &mut MetalLosslessEncodeBatchStats,
    family: ResidentSubmissionFamily,
) -> Result<SubmittedResidentLosslessMetalBufferEncodeChunk, crate::Error> {
    let tile_count = planned.len();
    if tile_count == 0 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident scheduler produced an empty chunk".to_string(),
        });
    }
    let profile_stages = compute::metal_profile_stages_enabled();
    if profile_stages {
        stats.stage_stats.chunk_count = stats.stage_stats.chunk_count.saturating_add(1);
        stats.stage_stats.tile_count = stats.stage_stats.tile_count.saturating_add(tile_count);
    }
    stats.max_observed_inflight_tiles = stats.max_observed_inflight_tiles.max(tile_count);

    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal resident chunk preparation");
    let early_prepare_submit_started =
        (profile_stages && family.time_prepare_in_submit()).then(Instant::now);
    let prep_wall_started = profile_stages.then(Instant::now);
    let prepared = prepare_planned_resident_lossless_tiles_batch(planned, session)?;
    if let Some(started) = prep_wall_started {
        add_resident_prep_wall_duration(stats, started.elapsed(), profile_stages);
    }

    let ResidentChunkItems {
        metadatas,
        prepare_durations,
        batch_items,
    } = build_resident_chunk_items(prepared, &mut budget)?;
    let batch_started = Instant::now();
    let prepare_submit_started = if family.time_prepare_in_submit() {
        early_prepare_submit_started
    } else {
        profile_stages.then(Instant::now)
    };
    let pending = match family {
        ResidentSubmissionFamily::HighThroughput(mode) => {
            compute::submit_lossless_codestream_buffers_from_prepared_ht_batch(
                session,
                batch_items,
                mode,
            )?
        }
        ResidentSubmissionFamily::Classic => {
            compute::submit_lossless_codestream_buffers_from_prepared_classic_batch(
                session,
                batch_items,
                compute::J2kClassicEncodeOutputCapacityMode::Tight,
            )?
        }
    };
    if let Some(started) = prepare_submit_started {
        stats.stage_stats.prepare_submit_duration = stats
            .stage_stats
            .prepare_submit_duration
            .saturating_add(started.elapsed());
    }
    Ok(SubmittedResidentLosslessMetalBufferEncodeChunk {
        metadatas,
        prepare_durations,
        pending,
        batch_started,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn duration_share(duration: Duration, count: usize) -> Duration {
    if count == 0 {
        return Duration::ZERO;
    }
    let nanos = duration.as_nanos() / count as u128;
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}
