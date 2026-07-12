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
fn wait_submitted_resident_lossless_buffer_encode_batch_once(
    submitted: &mut SubmittedResidentLosslessMetalBufferEncodeBatch,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let outcome_count = submitted.pipeline.as_ref().map_or(
        0,
        super::resident_schedule::SubmittedResidentLosslessChunkPipeline::total_tiles,
    );
    let mut outcome_budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal resident encode outcome collection",
    );
    let mut outcomes =
        outcome_budget.try_vec(outcome_count, "J2K Metal resident encode outcome slots")?;
    if let Some(mut pipeline) = submitted.pipeline.take() {
        while let Some(chunk) = pipeline.take_active() {
            let tile_count = chunk.metadatas.len();
            super::resident_schedule::SubmittedResidentLosslessChunkPipeline::record_active_completed(
                tile_count,
            );
            wait_and_harvest_resident_chunk(submitted, chunk, &mut outcomes)?;
            pipeline.submit_next(&submitted.session, &mut submitted.stats)?;
        }
    }
    if outcomes.len() != outcome_count {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident encode produced an unexpected outcome count".to_string(),
        });
    }
    submitted.stats.encode_wall_duration = submitted.encode_started.elapsed();
    Ok(MetalLosslessBufferEncodeBatchOutcome {
        outcomes,
        stats: submitted.stats,
    })
}

#[cfg(target_os = "macos")]
fn wait_and_harvest_resident_chunk(
    submitted: &mut SubmittedResidentLosslessMetalBufferEncodeBatch,
    chunk: super::SubmittedResidentLosslessMetalBufferEncodeChunk,
    outcomes: &mut Vec<MetalLosslessBufferEncodeOutcome>,
) -> Result<(), crate::Error> {
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
    validate_resident_chunk_result_counts(
        chunk.metadatas.len(),
        chunk.prepare_durations.len(),
        batch.codestreams.len(),
    )?;
    let batch_duration = duration_share(chunk.batch_started.elapsed(), batch.codestreams.len());
    for ((metadata, prepare_duration), codestream) in chunk
        .metadatas
        .into_iter()
        .zip(chunk.prepare_durations)
        .zip(batch.codestreams)
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
    Ok(())
}

fn validate_resident_chunk_result_counts(
    metadata_count: usize,
    prepare_duration_count: usize,
    codestream_count: usize,
) -> Result<(), crate::Error> {
    if metadata_count == prepare_duration_count && metadata_count == codestream_count {
        return Ok(());
    }
    Err(crate::Error::MetalKernel {
        message: "J2K Metal resident chunk result count does not match submitted tiles".to_string(),
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

#[cfg(test)]
mod tests {
    use super::validate_resident_chunk_result_counts;

    #[test]
    fn resident_chunk_result_counts_must_match_before_zip() {
        assert!(validate_resident_chunk_result_counts(2, 2, 2).is_ok());
        for counts in [(2, 1, 2), (2, 2, 1), (1, 2, 2)] {
            assert!(
                validate_resident_chunk_result_counts(counts.0, counts.1, counts.2).is_err(),
                "accepted mismatched resident counts {counts:?}"
            );
        }
    }
}
