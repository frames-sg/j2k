// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::Range;

use super::{
    resident_lossless_chunk_ranges_from_code_blocks, resident_lossless_code_block_chunk_cap,
    resident_submit::{submit_resident_lossless_chunk, ResidentSubmissionFamily},
    J2kBlockCodingMode, MetalLosslessEncodeBatchStats, PlannedResidentLosslessBufferEncode,
    SubmittedResidentLosslessMetalBufferEncodeChunk,
};

pub(super) fn submit_planned_resident_lossless_tiles(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<Option<SubmittedResidentLosslessChunkPipeline>, crate::Error> {
    if planned.is_empty() {
        return Ok(None);
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
    Ok(None)
}

fn submit_planned_resident_ht_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<Option<SubmittedResidentLosslessChunkPipeline>, crate::Error> {
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
    let family = ResidentSubmissionFamily::HighThroughput(
        crate::compute::ht_packet_output_capacity_mode_from_env(),
    );
    SubmittedResidentLosslessChunkPipeline::new(planned, chunk_ranges, family, session, stats)
        .map(Some)
}

fn submit_planned_resident_classic_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<Option<SubmittedResidentLosslessChunkPipeline>, crate::Error> {
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
    SubmittedResidentLosslessChunkPipeline::new(
        planned,
        chunk_ranges,
        ResidentSubmissionFamily::Classic,
        session,
        stats,
    )
    .map(Some)
}

pub(super) struct SubmittedResidentLosslessChunkPipeline {
    remaining: std::vec::IntoIter<PlannedResidentLosslessBufferEncode>,
    ranges: std::vec::IntoIter<Range<usize>>,
    active: Option<SubmittedResidentLosslessMetalBufferEncodeChunk>,
    total_tiles: usize,
    family: ResidentSubmissionFamily,
}

impl SubmittedResidentLosslessChunkPipeline {
    pub(super) fn new(
        planned: Vec<PlannedResidentLosslessBufferEncode>,
        ranges: Vec<Range<usize>>,
        family: ResidentSubmissionFamily,
        session: &crate::MetalBackendSession,
        stats: &mut MetalLosslessEncodeBatchStats,
    ) -> Result<Self, crate::Error> {
        validate_chunk_ranges(planned.len(), &ranges)?;
        let total_tiles = planned.len();
        let mut pipeline = Self {
            remaining: planned.into_iter(),
            ranges: ranges.into_iter(),
            active: None,
            total_tiles,
            family,
        };
        pipeline.submit_next(session, stats)?;
        Ok(pipeline)
    }

    pub(super) const fn total_tiles(&self) -> usize {
        self.total_tiles
    }

    pub(super) fn take_active(
        &mut self,
    ) -> Option<SubmittedResidentLosslessMetalBufferEncodeChunk> {
        self.active.take()
    }

    pub(super) fn submit_next(
        &mut self,
        session: &crate::MetalBackendSession,
        stats: &mut MetalLosslessEncodeBatchStats,
    ) -> Result<(), crate::Error> {
        if self.active.is_some() {
            return Err(crate::Error::MetalKernel {
                message: "J2K Metal resident scheduler cannot submit over an active chunk"
                    .to_string(),
            });
        }
        let Some(range) = self.ranges.next() else {
            if self.remaining.len() != 0 {
                return Err(crate::Error::MetalKernel {
                    message: "J2K Metal resident scheduler exhausted ranges before planned tiles"
                        .to_string(),
                });
            }
            return Ok(());
        };
        let mut budget =
            crate::batch_allocation::BatchMetadataBudget::new("J2K Metal resident scheduled chunk");
        let mut chunk = budget.try_vec(range.len(), "J2K Metal resident scheduled tiles")?;
        for _ in range {
            chunk.push(
                self.remaining
                    .next()
                    .ok_or_else(|| crate::Error::MetalKernel {
                        message: "J2K Metal resident chunk range exceeds planned tiles".to_string(),
                    })?,
            );
        }
        let pending = submit_resident_lossless_chunk(chunk, session, stats, self.family)?;
        record_submitted_chunk(pending.metadatas.len());
        self.active = Some(pending);
        Ok(())
    }

    pub(super) fn record_active_completed(tile_count: usize) {
        record_completed_chunk(tile_count);
    }
}

impl Drop for SubmittedResidentLosslessChunkPipeline {
    fn drop(&mut self) {
        if let Some(active) = &self.active {
            record_completed_chunk(active.metadatas.len());
        }
    }
}

fn validate_chunk_ranges(planned_len: usize, ranges: &[Range<usize>]) -> Result<(), crate::Error> {
    let mut expected_start = 0usize;
    for range in ranges {
        if range.start != expected_start || range.end <= range.start || range.end > planned_len {
            return Err(crate::Error::MetalKernel {
                message:
                    "J2K Metal resident chunk ranges must be non-empty, contiguous, and in bounds"
                        .to_string(),
            });
        }
        expected_start = range.end;
    }
    if expected_start != planned_len {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident chunk ranges do not cover every planned tile".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
std::thread_local! {
    static PENDING_TILES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static PEAK_PENDING_TILES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SUBMITTED_CHUNKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_submitted_chunk(tile_count: usize) {
    PENDING_TILES.with(|pending| pending.set(tile_count));
    PEAK_PENDING_TILES.with(|peak| peak.set(peak.get().max(tile_count)));
    SUBMITTED_CHUNKS.with(|chunks| chunks.set(chunks.get().saturating_add(1)));
}

#[cfg(not(test))]
fn record_submitted_chunk(_tile_count: usize) {}

#[cfg(test)]
fn record_completed_chunk(tile_count: usize) {
    let previous = PENDING_TILES.with(|pending| pending.replace(0));
    debug_assert_eq!(previous, tile_count);
}

#[cfg(not(test))]
fn record_completed_chunk(_tile_count: usize) {}

#[cfg(test)]
pub(super) fn reset_resident_schedule_counters_for_test() {
    PENDING_TILES.with(|pending| pending.set(0));
    PEAK_PENDING_TILES.with(|peak| peak.set(0));
    SUBMITTED_CHUNKS.with(|chunks| chunks.set(0));
}

#[cfg(test)]
pub(super) fn resident_schedule_counters_for_test() -> (usize, usize, usize) {
    let pending = PENDING_TILES.with(std::cell::Cell::get);
    let peak = PEAK_PENDING_TILES.with(std::cell::Cell::get);
    let submitted = SUBMITTED_CHUNKS.with(std::cell::Cell::get);
    (pending, peak, submitted)
}

#[cfg(test)]
mod tests {
    use super::validate_chunk_ranges;

    #[test]
    fn chunk_ranges_are_validated_before_submission() {
        assert!(validate_chunk_ranges(4, &[0..2, 2..4]).is_ok());
        assert!(validate_chunk_ranges(4, &[]).is_err());
        assert!(validate_chunk_ranges(4, &[0..0, 0..4]).is_err());
        assert!(validate_chunk_ranges(4, &[1..2, 2..4]).is_err());
        assert!(validate_chunk_ranges(4, &[0..2, 3..4]).is_err());
        assert!(validate_chunk_ranges(4, &[0..5, 5..6]).is_err());
    }
}
