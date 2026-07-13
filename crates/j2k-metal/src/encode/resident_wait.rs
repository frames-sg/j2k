// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    compute, duration_share, encode_owned_lossless_tiles_to_metal_buffer_fallback_batch,
    resident_classic_batch_encode_should_retry_conservative,
    resident_ht_batch_encode_should_retry_conservative,
    validate_lossless_roundtrip_on_metal_region_with_session,
    validate_lossless_roundtrip_on_metal_tile_with_session, Duration,
    FinishedResidentLosslessBufferEncode, Instant, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, MetalEncodedJ2k, MetalLosslessBufferEncodeBatchOutcome,
    MetalLosslessBufferEncodeOutcome, MetalLosslessEncodeResidency, MetalLosslessEncodeStageStats,
    ResidentLosslessBufferEncodeMetadata, SubmittedResidentLosslessMetalBufferEncodeBatch,
    SubmittedResidentLosslessMetalBufferEncodeBatchKind,
};

#[cfg(target_os = "macos")]
pub(super) fn wait_submitted_resident_lossless_buffer_encode_batch(
    mut submitted: SubmittedResidentLosslessMetalBufferEncodeBatch,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let result = wait_submitted_resident_lossless_buffer_encode_batch_once(&mut submitted);
    match result {
        Ok(outcome) => Ok(outcome),
        Err(err) => {
            if submitted.options.block_coding_mode == J2kBlockCodingMode::Classic
                && !submitted.tiles.is_empty()
                && resident_classic_batch_encode_should_retry_conservative(&err)
            {
                return encode_owned_lossless_tiles_to_metal_buffer_fallback_batch(
                    &submitted.tiles,
                    submitted.options,
                    &submitted.session,
                    submitted.staging,
                )
                .map_err(|retry_err| crate::Error::MetalKernel {
                    message: format!(
                        "J2K Metal resident classic batch conservative retry failed after tight resident capacity failure ({err}); retry error: {retry_err}"
                    ),
                });
            }
            if submitted.options.block_coding_mode == J2kBlockCodingMode::HighThroughput
                && !submitted.tiles.is_empty()
                && resident_ht_batch_encode_should_retry_conservative(&err)
            {
                return encode_owned_lossless_tiles_to_metal_buffer_fallback_batch(
                    &submitted.tiles,
                    submitted.options,
                    &submitted.session,
                    submitted.staging,
                )
                .map_err(|retry_err| crate::Error::MetalKernel {
                    message: format!(
                        "J2K Metal resident HT batch conservative retry failed after tight packet capacity failure ({err}); retry error: {retry_err}"
                    ),
                });
            }
            Err(err)
        }
    }
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "wait and harvest preserve chunk outcome and timing order"
)]
fn wait_submitted_resident_lossless_buffer_encode_batch_once(
    submitted: &mut SubmittedResidentLosslessMetalBufferEncodeBatch,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let outcome_count = match &submitted.kind {
        SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty => 0,
        SubmittedResidentLosslessMetalBufferEncodeBatchKind::Chunks(chunks) => chunks
            .iter()
            .try_fold(0usize, |total, chunk| {
                total.checked_add(chunk.metadatas.len())
            })
            .ok_or(j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "J2K Metal resident encode outcome collection",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?,
    };
    let mut outcome_budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal resident encode outcome collection",
    );
    let mut outcomes =
        outcome_budget.try_vec(outcome_count, "J2K Metal resident encode outcome slots")?;
    match std::mem::replace(
        &mut submitted.kind,
        SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty,
    ) {
        SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty => {}
        SubmittedResidentLosslessMetalBufferEncodeBatchKind::Chunks(chunks) => {
            if submitted.options.validation == J2kEncodeValidation::External
                && submitted.options.block_coding_mode == J2kBlockCodingMode::HighThroughput
                && chunks.len() > 1
            {
                let wait_started = compute::metal_profile_stages_enabled().then(Instant::now);
                let mut chunk_budget = crate::batch_allocation::BatchMetadataBudget::new(
                    "J2K Metal resident encode chunk wait plan",
                );
                let mut chunk_metadatas = chunk_budget.try_vec(
                    chunks.len(),
                    "J2K Metal resident encode chunk metadata groups",
                )?;
                let mut pending_batches = chunk_budget
                    .try_vec(chunks.len(), "J2K Metal resident encode pending batches")?;
                for chunk in chunks {
                    chunk_metadatas.push((
                        chunk.metadatas,
                        chunk.prepare_durations,
                        chunk.batch_started,
                    ));
                    pending_batches.push(chunk.pending);
                }
                let batches = compute::wait_resident_lossless_codestream_batches(pending_batches)?;
                if let Some(started) = wait_started {
                    let elapsed = started.elapsed();
                    submitted.stats.stage_stats.codestream_wait_duration = submitted
                        .stats
                        .stage_stats
                        .codestream_wait_duration
                        .saturating_add(elapsed);
                    submitted.stats.stage_stats.sync_wait_duration = submitted
                        .stats
                        .stage_stats
                        .sync_wait_duration
                        .saturating_add(elapsed);
                }
                for ((metadatas, prepare_durations, batch_started), batch) in
                    chunk_metadatas.into_iter().zip(batches)
                {
                    if compute::metal_profile_stages_enabled() {
                        submitted
                            .stats
                            .stage_stats
                            .add_assign(MetalLosslessEncodeStageStats::from(batch.stage_stats));
                    }
                    let codestreams = batch.codestreams;
                    let batch_duration = duration_share(batch_started.elapsed(), codestreams.len());
                    for ((metadata, prepare_duration), codestream) in metadatas
                        .into_iter()
                        .zip(prepare_durations)
                        .zip(codestreams)
                    {
                        let finished = finished_resident_lossless_buffer_encode(
                            metadata,
                            codestream,
                            prepare_duration.saturating_add(batch_duration),
                        )?;
                        outcomes.push(validate_finished_resident_lossless_buffer_encode(
                            finished,
                            submitted.options,
                            &submitted.session,
                        )?);
                    }
                }
            } else {
                for chunk in chunks {
                    let wait_started = compute::metal_profile_stages_enabled().then(Instant::now);
                    let batch = compute::wait_resident_lossless_codestream_batch(chunk.pending)?;
                    if let Some(started) = wait_started {
                        let elapsed = started.elapsed();
                        submitted.stats.stage_stats.codestream_wait_duration = submitted
                            .stats
                            .stage_stats
                            .codestream_wait_duration
                            .saturating_add(elapsed);
                        submitted.stats.stage_stats.sync_wait_duration = submitted
                            .stats
                            .stage_stats
                            .sync_wait_duration
                            .saturating_add(elapsed);
                        submitted
                            .stats
                            .stage_stats
                            .add_assign(MetalLosslessEncodeStageStats::from(batch.stage_stats));
                    }
                    let codestreams = batch.codestreams;
                    let batch_duration =
                        duration_share(chunk.batch_started.elapsed(), codestreams.len());
                    for ((metadata, prepare_duration), codestream) in chunk
                        .metadatas
                        .into_iter()
                        .zip(chunk.prepare_durations)
                        .zip(codestreams)
                    {
                        let finished = finished_resident_lossless_buffer_encode(
                            metadata,
                            codestream,
                            prepare_duration.saturating_add(batch_duration),
                        )?;
                        outcomes.push(validate_finished_resident_lossless_buffer_encode(
                            finished,
                            submitted.options,
                            &submitted.session,
                        )?);
                    }
                }
            }
        }
    }
    submitted.stats.encode_wall_duration = submitted.encode_started.elapsed();
    Ok(MetalLosslessBufferEncodeBatchOutcome {
        outcomes,
        stats: submitted.stats,
    })
}

#[cfg(target_os = "macos")]
fn finished_resident_lossless_buffer_encode(
    metadata: ResidentLosslessBufferEncodeMetadata,
    codestream: compute::J2kResidentLosslessCodestream,
    encode_duration: Duration,
) -> Result<FinishedResidentLosslessBufferEncode, crate::Error> {
    let codestream_end = codestream
        .byte_offset
        .checked_add(codestream.byte_len)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal codestream byte range overflows usize".to_string(),
        })?;
    let encoded = MetalEncodedJ2k::from_completed_buffer(
        codestream.buffer,
        codestream.byte_offset..codestream_end,
        codestream.capacity,
        (metadata.tile.output_width, metadata.tile.output_height),
        metadata.components,
        metadata.bit_depth,
        false,
    )?;

    Ok(FinishedResidentLosslessBufferEncode {
        metadata,
        encoded,
        encode_duration,
        gpu_duration: codestream.gpu_duration,
    })
}

#[cfg(target_os = "macos")]
fn validate_finished_resident_lossless_buffer_encode(
    finished: FinishedResidentLosslessBufferEncode,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let FinishedResidentLosslessBufferEncode {
        metadata,
        encoded,
        encode_duration,
        gpu_duration,
    } = finished;

    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        let tile = metadata.tile.as_tile();
        if tile.width == tile.output_width
            && tile.height == tile.output_height
            && tile.pitch_bytes == tile.output_width as usize * metadata.bytes_per_pixel
        {
            validate_lossless_roundtrip_on_metal_tile_with_session(
                tile,
                &encoded.codestream_bytes()?,
                session,
            )?;
        } else {
            validate_lossless_roundtrip_on_metal_region_with_session(
                tile,
                tile.output_width,
                tile.output_height,
                metadata.bytes_per_pixel,
                &encoded.codestream_bytes()?,
                session,
            )?;
        }
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };

    Ok(MetalLosslessBufferEncodeOutcome {
        encoded,
        input_copy_used: false,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: true,
            packetization_used: true,
            codestream_assembly_used: true,
        },
        input_copy_duration: Duration::ZERO,
        encode_duration,
        gpu_duration,
        validation_duration,
    })
}
