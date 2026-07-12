// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    compute, encode_lossless_tile_to_metal_buffer_with_report, encode_lossless_tile_with_report,
    lossless_sample_shape, plan_resident_lossless_buffer_encode,
    resident_lossless_encode_config_for_mode, resolve_lossless_encode_config,
    should_try_resident_lossless_host_encode, should_try_resident_lossless_host_encode_for_tiles,
    submit_planned_resident_lossless_tiles, try_encode_lossless_tile_device_resident_with_report,
    validate_metal_encode_tile, validate_padded_contiguous_metal_encode_tile,
    wait_submitted_resident_lossless_buffer_encode_batch, EncodeBackendPreference, Instant,
    J2kBlockCodingMode, J2kLosslessEncodeOptions, MetalEncodeInputStaging,
    MetalEncodeStageAccelerator, MetalLosslessBufferEncodeBatchOutcome,
    MetalLosslessBufferEncodeOutcome, MetalLosslessEncodeBatchRequest,
    MetalLosslessEncodeBatchStats, MetalLosslessEncodeConfig, MetalLosslessEncodeOutcome,
    MetalLosslessEncodeTile, OwnedMetalLosslessEncodeTile, PlannedResidentLosslessBufferEncode,
    SubmittedJ2kLosslessMetalBufferEncodeBatch, SubmittedJ2kLosslessMetalBufferEncodeBatchState,
    SubmittedJ2kLosslessMetalEncodeBatch, SubmittedJ2kLosslessMetalEncodeBatchState,
    SubmittedResidentLosslessMetalBufferEncodeBatch,
    SubmittedResidentLosslessMetalBufferEncodeBatchKind,
};

#[cfg(target_os = "macos")]
/// Submit a lossless tile batch that resolves to host codestream bytes.
pub fn submit_lossless_batch(
    request: MetalLosslessEncodeBatchRequest<'_, '_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    submit_lossless_tiles(
        request.tiles,
        *options,
        session,
        request.staging,
        request.config,
    )
}

#[cfg(target_os = "macos")]
/// Submit a lossless tile batch that resolves to Metal-backed codestreams.
pub fn submit_lossless_batch_to_metal(
    request: MetalLosslessEncodeBatchRequest<'_, '_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalBufferEncodeBatch, crate::Error> {
    submit_lossless_tiles_to_metal_buffer_batch(
        request.tiles,
        *options,
        session,
        request.staging,
        request.config,
    )
}

#[cfg(target_os = "macos")]
/// Encode a lossless tile batch and return host-byte timing reports.
#[doc(hidden)]
pub fn encode_lossless_batch_with_report(
    request: MetalLosslessEncodeBatchRequest<'_, '_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    encode_lossless_tiles_with_report(
        request.tiles,
        *options,
        session,
        request.staging,
        request.config,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn host_outcome_from_buffer_outcome(
    outcome: MetalLosslessBufferEncodeOutcome,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    let (encoded, host_readback_duration) =
        outcome.encoded.to_encoded_j2k_with_readback_duration()?;
    Ok(MetalLosslessEncodeOutcome {
        encoded,
        input_copy_used: outcome.input_copy_used,
        resident: outcome.resident,
        input_copy_duration: outcome.input_copy_duration,
        encode_duration: outcome.encode_duration,
        gpu_duration: outcome.gpu_duration,
        validation_duration: outcome.validation_duration,
        host_readback_duration,
    })
}

#[cfg(target_os = "macos")]
fn encode_lossless_tiles_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    if should_try_resident_lossless_host_encode_for_tiles(tiles, options, staging) {
        let batch = try_encode_resident_lossless_tiles_to_metal_buffer_batch(
            tiles, options, session, staging, config,
        )?;
        if let Some(outcomes) = batch {
            let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal resident host encode outcomes",
            );
            let mut results = budget.try_vec(
                outcomes.outcomes.len(),
                "J2K Metal resident host encode outcome slots",
            )?;
            for outcome in outcomes.outcomes {
                results.push(host_outcome_from_buffer_outcome(outcome)?);
            }
            return Ok(results);
        }
    }

    let mut accelerator = MetalEncodeStageAccelerator::for_host_output(options);
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal host encode outcomes");
    let mut results = budget.try_vec(tiles.len(), "J2K Metal host encode outcome slots")?;
    for &tile in tiles {
        results.push(encode_lossless_tile_with_report(
            tile,
            options,
            session,
            staging,
            &mut accelerator,
        )?);
    }
    Ok(results)
}

#[cfg(target_os = "macos")]
pub(super) fn encode_lossless_owned_tiles_with_report(
    tiles: &[OwnedMetalLosslessEncodeTile],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal owned host encode batch");
    let mut borrowed = budget.try_vec(tiles.len(), "J2K Metal borrowed owned encode tiles")?;
    borrowed.extend(tiles.iter().map(OwnedMetalLosslessEncodeTile::as_tile));
    if should_try_resident_lossless_host_encode_for_tiles(&borrowed, options, staging) {
        let batch = try_encode_resident_lossless_tiles_to_metal_buffer_batch(
            &borrowed, options, session, staging, config,
        )?;
        if let Some(outcomes) = batch {
            let mut results = budget.try_vec(
                outcomes.outcomes.len(),
                "J2K Metal owned resident host encode outcomes",
            )?;
            for outcome in outcomes.outcomes {
                results.push(host_outcome_from_buffer_outcome(outcome)?);
            }
            return Ok(results);
        }
    }

    let mut accelerator = MetalEncodeStageAccelerator::for_host_output(options);
    let mut results =
        budget.try_vec(borrowed.len(), "J2K Metal owned host encode outcome slots")?;
    for &tile in &borrowed {
        results.push(encode_lossless_tile_with_report(
            tile,
            options,
            session,
            staging,
            &mut accelerator,
        )?);
    }
    Ok(results)
}

#[cfg(target_os = "macos")]
fn submit_lossless_tiles_to_metal_buffer_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalBufferEncodeBatch, crate::Error> {
    if options.backend != EncodeBackendPreference::CpuOnly {
        if let Some(submitted) = try_submit_resident_lossless_tiles_to_metal_buffer_batch(
            tiles, options, session, staging, config,
        )? {
            return Ok(SubmittedJ2kLosslessMetalBufferEncodeBatch {
                state: SubmittedJ2kLosslessMetalBufferEncodeBatchState::Resident(Box::new(
                    submitted,
                )),
            });
        }
    }

    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal deferred buffer encode batch");
    let mut owned = budget.try_vec(tiles.len(), "J2K Metal owned deferred buffer encode tiles")?;
    for &tile in tiles {
        validate_metal_encode_tile(tile)?;
        if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
            lossless_sample_shape(tile.format)?;
            validate_padded_contiguous_metal_encode_tile(tile, tile.format.bytes_per_pixel())?;
        }
        owned.push(OwnedMetalLosslessEncodeTile::from_tile(tile));
    }
    Ok(SubmittedJ2kLosslessMetalBufferEncodeBatch {
        state: SubmittedJ2kLosslessMetalBufferEncodeBatchState::Deferred {
            tiles: owned,
            options,
            session: session.clone(),
            staging,
        },
    })
}

#[cfg(target_os = "macos")]
pub(super) fn encode_owned_lossless_tiles_to_metal_buffer_fallback_batch(
    tiles: &[OwnedMetalLosslessEncodeTile],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal fallback buffer encode outcomes",
    );
    let mut outcomes = budget.try_vec(
        tiles.len(),
        "J2K Metal fallback buffer encode outcome slots",
    )?;
    for tile in tiles {
        outcomes.push(encode_lossless_tile_to_metal_buffer_with_report(
            tile.as_tile(),
            options,
            session,
            staging,
        )?);
    }
    Ok(MetalLosslessBufferEncodeBatchOutcome {
        outcomes,
        stats: MetalLosslessEncodeBatchStats::default(),
    })
}

#[cfg(target_os = "macos")]
fn try_submit_resident_lossless_tiles_to_metal_buffer_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<Option<SubmittedResidentLosslessMetalBufferEncodeBatch>, crate::Error> {
    let profile_stages = compute::metal_profile_stages_enabled();
    if tiles.is_empty() {
        return Ok(Some(SubmittedResidentLosslessMetalBufferEncodeBatch {
            options,
            session: session.clone(),
            stats: resolve_lossless_encode_config(0, 1, config)?,
            encode_started: Instant::now(),
            tiles: Vec::new(),
            staging,
            kind: SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty,
        }));
    }

    let plan_started = profile_stages.then(Instant::now);
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal resident encode batch plan");
    let mut planned = budget.try_vec(tiles.len(), "J2K Metal resident encode planned tiles")?;
    for (index, &tile) in tiles.iter().enumerate() {
        let Some(item) = plan_resident_lossless_buffer_encode(index, tile, options, staging)?
        else {
            return Ok(None);
        };
        planned.push(item);
    }
    let estimated_peak_bytes_per_tile = planned
        .iter()
        .map(PlannedResidentLosslessBufferEncode::estimated_peak_bytes)
        .max()
        .unwrap_or(1);
    let classic_resident_mode = planned
        .iter()
        .all(|planned| planned.metadata.plan.block_coding_mode == J2kBlockCodingMode::Classic);
    let ht_resident_mode = planned.iter().all(|planned| {
        planned.metadata.plan.block_coding_mode == J2kBlockCodingMode::HighThroughput
    });
    if !(classic_resident_mode || ht_resident_mode) {
        return Ok(None);
    }
    let resolved_config =
        resident_lossless_encode_config_for_mode(config, classic_resident_mode, tiles.len());
    let mut stats = resolve_lossless_encode_config(
        tiles.len(),
        estimated_peak_bytes_per_tile,
        resolved_config,
    )?;
    if let Some(started) = plan_started {
        stats.stage_stats.plan_duration = started.elapsed();
    }
    let encode_started = Instant::now();
    let kind = submit_planned_resident_lossless_tiles(
        planned,
        session,
        stats.effective_inflight_tiles,
        &mut stats,
    )?;
    let mut owned_tiles =
        budget.try_vec(tiles.len(), "J2K Metal retained resident encode tiles")?;
    owned_tiles.extend(
        tiles
            .iter()
            .map(|&tile| OwnedMetalLosslessEncodeTile::from_tile(tile)),
    );
    Ok(Some(SubmittedResidentLosslessMetalBufferEncodeBatch {
        options,
        session: session.clone(),
        stats,
        encode_started,
        tiles: owned_tiles,
        staging,
        kind,
    }))
}

#[cfg(target_os = "macos")]
fn try_encode_resident_lossless_tiles_to_metal_buffer_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<Option<MetalLosslessBufferEncodeBatchOutcome>, crate::Error> {
    let Some(submitted) = try_submit_resident_lossless_tiles_to_metal_buffer_batch(
        tiles, options, session, staging, config,
    )?
    else {
        return Ok(None);
    };
    wait_submitted_resident_lossless_buffer_encode_batch(submitted).map(Some)
}

#[cfg(target_os = "macos")]
fn submit_lossless_tiles(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous)
        && should_try_resident_lossless_host_encode(options)
    {
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal ready resident encode batch",
        );
        let mut ready = budget.try_vec(tiles.len(), "J2K Metal ready resident encode results")?;
        let mut all_ready = true;
        for &tile in tiles {
            validate_metal_encode_tile(tile)?;
            lossless_sample_shape(tile.format)?;
            validate_padded_contiguous_metal_encode_tile(tile, tile.format.bytes_per_pixel())?;
            if let Some(outcome) = try_encode_lossless_tile_device_resident_with_report(
                tile, options, session, staging,
            )? {
                ready.push(outcome.encoded);
            } else {
                all_ready = false;
                break;
            }
        }
        if all_ready {
            return Ok(SubmittedJ2kLosslessMetalEncodeBatch {
                state: SubmittedJ2kLosslessMetalEncodeBatchState::Ready(ready),
            });
        }
        if options.backend == EncodeBackendPreference::RequireDevice {
            return Err(crate::Error::UnsupportedMetalRequest {
                reason: "J2K Metal resident encode requires classic padded contiguous Gray/RGB lossless input with at most one DWT level",
            });
        }
    }

    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal deferred encode batch");
    let mut owned = budget.try_vec(tiles.len(), "J2K Metal owned deferred encode tiles")?;
    for &tile in tiles {
        validate_metal_encode_tile(tile)?;
        if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
            lossless_sample_shape(tile.format)?;
            validate_padded_contiguous_metal_encode_tile(tile, tile.format.bytes_per_pixel())?;
        }
        owned.push(OwnedMetalLosslessEncodeTile::from_tile(tile));
    }
    Ok(SubmittedJ2kLosslessMetalEncodeBatch {
        state: SubmittedJ2kLosslessMetalEncodeBatchState::Deferred {
            tiles: owned,
            options,
            session: session.clone(),
            staging,
            config,
        },
    })
}
