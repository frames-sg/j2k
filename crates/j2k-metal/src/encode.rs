// SPDX-License-Identifier: Apache-2.0

#[cfg(any(test, target_os = "macos"))]
mod config;
mod encoded;
#[cfg(target_os = "macos")]
mod packet_plan;
#[cfg(target_os = "macos")]
mod plan;
#[cfg(target_os = "macos")]
mod resident_estimate;
#[cfg(target_os = "macos")]
mod resident_types;
mod roundtrip_validation;
mod stage_accelerator;
mod stats;
mod submitted;
#[cfg(all(test, target_os = "macos"))]
mod test_helpers;
mod types;
#[cfg(target_os = "macos")]
mod validation;

#[cfg(target_os = "macos")]
use crate::compute;
use j2k::J2kLosslessEncodeOptions;
#[cfg(target_os = "macos")]
use j2k::{EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, ReversibleTransform};
#[cfg(target_os = "macos")]
use j2k::{EncodedJ2k, J2kLosslessSamples};
#[cfg(target_os = "macos")]
use j2k_core::{BackendKind, DeviceSurface, PixelFormat};
#[cfg(target_os = "macos")]
use j2k_native::{EncodeOptions, EncodeProgressionOrder};
#[cfg(target_os = "macos")]
use j2k_native::{J2kEncodeStageAccelerator, J2kHtj2kTileEncodeJob, J2kPacketizationEncodeJob};
#[cfg(target_os = "macos")]
use metal::Buffer;
#[cfg(target_os = "macos")]
use std::time::Duration;
#[cfg(target_os = "macos")]
use std::time::Instant;

#[cfg(test)]
use self::config::{
    default_gpu_encode_memory_budget_bytes_for_hw_mem, resident_lossless_chunk_ranges_for_test,
    resolve_lossless_encode_config_for_test,
};
#[cfg(any(test, target_os = "macos"))]
use self::config::{
    resident_lossless_chunk_ranges_from_code_blocks, resident_lossless_code_block_chunk_cap,
    resident_lossless_encode_config_for_mode, resolve_lossless_encode_config,
};
pub use self::encoded::MetalEncodedJ2k;
#[cfg(target_os = "macos")]
use self::packet_plan::{
    cpu_packetization_resolutions_from_lossless_device_plan,
    lossless_options_for_resident_htj2k_tile_job, packet_descriptors_for_lossless_device_order,
    packetization_progression_order, resident_packetization_resolutions_from_lossless_device_plan,
    should_use_resident_htj2k_host_tile_for_auto,
};
#[cfg(target_os = "macos")]
use self::plan::{
    lossless_device_encode_plan, LosslessDeviceEncodePlan, RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
};
#[cfg(all(test, target_os = "macos"))]
use self::resident_estimate::estimated_tier1_output_bytes;
#[cfg(target_os = "macos")]
use self::resident_estimate::{
    checked_mul_bytes, estimate_resident_lossless_encode_peak_bytes,
    resident_classic_batch_encode_should_retry_conservative,
    resident_codestream_assembly_job_for_metadata,
    resident_ht_batch_encode_should_retry_conservative,
};
#[cfg(target_os = "macos")]
use self::resident_types::{
    FinishedResidentLosslessBufferEncode, PlannedResidentLosslessBufferEncode,
    PreparedResidentLosslessBufferEncode, ResidentLosslessBufferEncodeMetadata,
    SubmittedResidentLosslessMetalBufferEncodeBatch,
    SubmittedResidentLosslessMetalBufferEncodeBatchKind,
    SubmittedResidentLosslessMetalBufferEncodeChunk,
};
pub use self::roundtrip_validation::{
    validate_lossless_roundtrip_on_metal, validate_lossless_roundtrip_on_metal_with_session,
};
#[cfg(all(test, target_os = "macos"))]
use self::stage_accelerator::metal_dispatch_option;
pub use self::stage_accelerator::MetalEncodeStageAccelerator;
#[cfg(test)]
use self::stats::add_resident_prep_duration;
#[cfg(any(test, target_os = "macos"))]
use self::stats::add_resident_prep_wall_duration;
pub use self::stats::{
    MetalLosslessBufferEncodeBatchOutcome, MetalLosslessEncodeBatchStats,
    MetalLosslessEncodeStageStats,
};
#[cfg(target_os = "macos")]
use self::submitted::{
    OwnedMetalLosslessEncodeTile, SubmittedJ2kLosslessMetalBufferEncodeBatchState,
    SubmittedJ2kLosslessMetalEncodeBatchState,
};
pub use self::submitted::{
    SubmittedJ2kLosslessMetalBufferEncodeBatch, SubmittedJ2kLosslessMetalEncodeBatch,
};
#[cfg(all(test, target_os = "macos"))]
use self::test_helpers::{
    collect_inflight_limited_ordered, encode_lossless_from_metal_buffer,
    encode_lossless_from_metal_buffer_to_metal_with_report,
    encode_lossless_from_metal_buffers_to_metal_with_report,
    encode_lossless_from_padded_metal_buffer_to_metal_with_report,
    encode_lossless_from_padded_metal_buffer_with_report,
    encode_lossless_from_padded_metal_buffers_to_metal_batch,
    encode_lossless_from_padded_metal_buffers_to_metal_with_report,
    encode_lossless_from_padded_metal_buffers_with_report, set_test_resident_encode_failure_index,
    submit_lossless_from_metal_buffer, submit_lossless_from_padded_metal_buffer,
    test_resident_encode_failure_index,
};
pub use self::types::{
    MetalEncodeInputStaging, MetalLosslessBufferEncodeOutcome, MetalLosslessEncodeBatchRequest,
    MetalLosslessEncodeConfig, MetalLosslessEncodeOutcome, MetalLosslessEncodeResidency,
    MetalLosslessEncodeTile,
};
#[cfg(target_os = "macos")]
use self::validation::{
    lossless_sample_shape, validate_metal_encode_tile, validate_padded_contiguous_metal_encode_tile,
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
fn host_outcome_from_buffer_outcome(
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
    if should_try_resident_lossless_host_encode(options) {
        let batch = try_encode_resident_lossless_tiles_to_metal_buffer_batch(
            tiles, options, session, staging, config,
        )?;
        if let Some(outcomes) = batch {
            return outcomes
                .outcomes
                .into_iter()
                .map(host_outcome_from_buffer_outcome)
                .collect();
        }
    }

    let mut accelerator = MetalEncodeStageAccelerator::for_host_output(options);
    tiles
        .iter()
        .map(|&tile| {
            encode_lossless_tile_with_report(tile, options, session, staging, &mut accelerator)
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn encode_lossless_owned_tiles_with_report(
    tiles: &[OwnedMetalLosslessEncodeTile],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    let borrowed = tiles
        .iter()
        .map(OwnedMetalLosslessEncodeTile::as_tile)
        .collect::<Vec<_>>();
    if should_try_resident_lossless_host_encode(options) {
        let batch = try_encode_resident_lossless_tiles_to_metal_buffer_batch(
            &borrowed, options, session, staging, config,
        )?;
        if let Some(outcomes) = batch {
            return outcomes
                .outcomes
                .into_iter()
                .map(host_outcome_from_buffer_outcome)
                .collect();
        }
    }

    let mut accelerator = MetalEncodeStageAccelerator::for_host_output(options);
    borrowed
        .iter()
        .map(|&tile| {
            encode_lossless_tile_with_report(tile, options, session, staging, &mut accelerator)
        })
        .collect()
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

    let mut owned = Vec::with_capacity(tiles.len());
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
fn encode_owned_lossless_tiles_to_metal_buffer_fallback_batch(
    tiles: &[OwnedMetalLosslessEncodeTile],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let mut outcomes = Vec::with_capacity(tiles.len());
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
    let mut planned = Vec::with_capacity(tiles.len());
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
    let tiles = tiles
        .iter()
        .map(|&tile| OwnedMetalLosslessEncodeTile::from_tile(tile))
        .collect();
    Ok(Some(SubmittedResidentLosslessMetalBufferEncodeBatch {
        options,
        session: session.clone(),
        stats,
        encode_started,
        tiles,
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
        let mut ready = Vec::with_capacity(tiles.len());
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

    let mut owned = Vec::with_capacity(tiles.len());
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

#[cfg(target_os = "macos")]
fn should_try_resident_lossless_host_encode(options: J2kLosslessEncodeOptions) -> bool {
    options.backend == EncodeBackendPreference::RequireDevice
}

#[cfg(target_os = "macos")]
fn host_output_encode_options(mut options: J2kLosslessEncodeOptions) -> J2kLosslessEncodeOptions {
    options.validation = J2kEncodeValidation::External;
    options
}

#[cfg(target_os = "macos")]
fn borrow_padded_metal_buffer_from_bytes(
    session: &crate::MetalBackendSession,
    bytes: &[u8],
) -> Result<Buffer, crate::Error> {
    if bytes.is_empty() {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal hybrid encode input is empty".to_string(),
        });
    }
    Ok(session.device().new_buffer_with_bytes_no_copy(
        bytes.as_ptr().cast(),
        bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
        None,
    ))
}

#[cfg(target_os = "macos")]
struct ResidentHybridHtTileBody {
    tile_data: Vec<u8>,
    components: u8,
    bit_depth: u8,
    bytes_per_pixel: usize,
    code_block_count: usize,
    code_block_width_exp: u8,
    code_block_height_exp: u8,
    num_decomposition_levels: u8,
    used_fused_rct: bool,
    guard_bits: u8,
    progression_order: EncodeProgressionOrder,
    write_tlm: bool,
    forward_dwt53_dispatches: usize,
    ht_code_block_dispatches: usize,
}

#[cfg(target_os = "macos")]
fn encode_resident_ht_tile_body_with_cpu_packetization(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    code_block_width: u32,
    code_block_height: u32,
) -> Result<Option<ResidentHybridHtTileBody>, crate::Error> {
    if !should_try_resident_lossless_ht_cpu_packetization(tile, options, staging) {
        return Ok(None);
    }
    validate_metal_encode_tile(tile)?;
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    let bytes_per_sample =
        u8::try_from(tile.format.bytes_per_sample()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal resident hybrid bytes per sample exceeds u8".to_string(),
        })?;
    validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    let Some(plan) = lossless_device_encode_plan(
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        options,
        code_block_width,
        code_block_height,
    )?
    else {
        return Ok(None);
    };
    if plan.block_coding_mode != J2kBlockCodingMode::HighThroughput {
        return Ok(None);
    }

    let coefficient_count = lossless_device_coefficient_count(&plan.code_blocks)?;
    let prepared = compute::prepare_lossless_device_code_blocks(
        session,
        compute::J2kLosslessDevicePrepareJob {
            input: tile.buffer,
            input_byte_offset: tile.byte_offset,
            input_width: tile.width,
            input_height: tile.height,
            input_pitch_bytes: tile.pitch_bytes,
            output_width: tile.output_width,
            output_height: tile.output_height,
            components,
            bytes_per_sample,
            bit_depth,
            num_decomposition_levels: plan.num_decomposition_levels,
            coefficient_count,
        },
        plan.code_blocks.clone(),
    )?;
    let resident_tier1 =
        compute::encode_ht_prepared_device_code_blocks_resident(session, prepared)?;
    let encoded_blocks = compute::read_resident_ht_tier1_code_blocks_for_cpu_packetization(
        session,
        &resident_tier1,
    )?;
    let packetization_resolutions =
        cpu_packetization_resolutions_from_lossless_device_plan(&plan, &encoded_blocks)?;
    let packet_descriptors = packet_descriptors_for_lossless_device_order(
        plan.resolutions.len(),
        plan.components,
        plan.progression_order,
    )?;
    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: u32::try_from(plan.resolutions.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident hybrid resolution count exceeds u32".to_string(),
            }
        })?,
        num_layers: 1,
        num_components: plan.components,
        code_block_count: u32::try_from(plan.code_blocks.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident hybrid code-block count exceeds u32".to_string(),
            }
        })?,
        progression_order: packetization_progression_order(plan.progression_order),
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let tile_data =
        j2k_native::encode_j2k_packetization_scalar(packetization_job).map_err(|reason| {
            crate::Error::MetalKernel {
                message: format!("J2K Metal resident hybrid CPU packetization failed: {reason}"),
            }
        })?;

    Ok(Some(ResidentHybridHtTileBody {
        tile_data,
        components,
        bit_depth,
        bytes_per_pixel,
        code_block_count: plan.code_blocks.len(),
        code_block_width_exp: plan.code_block_width_exp,
        code_block_height_exp: plan.code_block_height_exp,
        num_decomposition_levels: plan.num_decomposition_levels,
        used_fused_rct: plan.use_mct && tile.format == PixelFormat::Rgb8,
        guard_bits: plan.guard_bits,
        progression_order: plan.progression_order,
        write_tlm: plan.write_tlm,
        forward_dwt53_dispatches: if plan.num_decomposition_levels > 0 {
            usize::from(plan.components)
        } else {
            0
        },
        ht_code_block_dispatches: usize::from(!plan.code_blocks.is_empty()),
    }))
}

#[cfg(target_os = "macos")]
#[derive(Debug, Default)]
struct PrepacketizedHtj2kTileAccelerator {
    tile_data: Option<Vec<u8>>,
}

#[cfg(target_os = "macos")]
impl J2kEncodeStageAccelerator for PrepacketizedHtj2kTileAccelerator {
    fn encode_htj2k_tile(
        &mut self,
        _job: J2kHtj2kTileEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        Ok(self.tile_data.take())
    }
}

#[cfg(target_os = "macos")]
fn lossless_device_coefficient_count(
    code_blocks: &[compute::J2kLosslessDeviceCodeBlock],
) -> Result<usize, crate::Error> {
    let mut count = 0usize;
    for block in code_blocks {
        let offset =
            usize::try_from(block.coefficient_offset).map_err(|_| crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient offset exceeds usize".to_string(),
            })?;
        let block_count = (block.width as usize)
            .checked_mul(block.height as usize)
            .ok_or_else(|| crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient count overflow".to_string(),
            })?;
        count = count.max(offset.checked_add(block_count).ok_or_else(|| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient count overflow".to_string(),
            }
        })?);
    }
    Ok(count)
}

#[cfg(target_os = "macos")]
fn plan_resident_lossless_buffer_encode(
    index: usize,
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    staging: MetalEncodeInputStaging,
) -> Result<Option<PlannedResidentLosslessBufferEncode>, crate::Error> {
    validate_metal_encode_tile(tile)?;
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Ok(None);
    }
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    let bytes_per_sample =
        u8::try_from(tile.format.bytes_per_sample()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal resident encode bytes per sample exceeds u8".to_string(),
        })?;
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    }
    let Some(plan) = lossless_device_encode_plan(
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        options,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
    )?
    else {
        return Ok(None);
    };
    let coefficient_count = lossless_device_coefficient_count(&plan.code_blocks)?;
    let packetization_resolutions =
        resident_packetization_resolutions_from_lossless_device_plan(&plan)?;
    let packet_descriptors = packet_descriptors_for_lossless_device_order(
        plan.resolutions.len(),
        plan.components,
        plan.progression_order,
    )?;
    let metadata = ResidentLosslessBufferEncodeMetadata {
        tile: OwnedMetalLosslessEncodeTile::from_tile(tile),
        components,
        bit_depth,
        bytes_per_pixel,
        plan,
        packet_descriptors,
        packetization_resolutions,
    };
    let estimated_peak_bytes =
        estimate_resident_lossless_encode_peak_bytes(&metadata, coefficient_count, staging);
    Ok(Some(PlannedResidentLosslessBufferEncode {
        index,
        metadata,
        coefficient_count,
        bytes_per_sample,
        estimated_peak_bytes,
        #[cfg(test)]
        failure_injection_index: test_resident_encode_failure_index(),
    }))
}

#[cfg(target_os = "macos")]
fn wait_submitted_resident_lossless_buffer_encode_batch(
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
    let mut outcomes = Vec::new();
    match std::mem::replace(
        &mut submitted.kind,
        SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty,
    ) {
        SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty => {}
        SubmittedResidentLosslessMetalBufferEncodeBatchKind::Chunks(chunks) => {
            outcomes.reserve(chunks.iter().map(|chunk| chunk.metadatas.len()).sum());
            if submitted.options.validation == J2kEncodeValidation::External
                && submitted.options.block_coding_mode == J2kBlockCodingMode::HighThroughput
                && chunks.len() > 1
            {
                let wait_started = compute::metal_profile_stages_enabled().then(Instant::now);
                let mut chunk_metadatas = Vec::with_capacity(chunks.len());
                let mut pending_batches = Vec::with_capacity(chunks.len());
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
                        );
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
                        );
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
) -> FinishedResidentLosslessBufferEncode {
    let encoded = MetalEncodedJ2k {
        codestream_buffer: codestream.buffer,
        byte_offset: codestream.byte_offset,
        byte_len: codestream.byte_len,
        capacity: codestream.capacity,
        width: metadata.tile.output_width,
        height: metadata.tile.output_height,
        components: metadata.components,
        bit_depth: metadata.bit_depth,
        signed: false,
    };

    FinishedResidentLosslessBufferEncode {
        metadata,
        encoded,
        encode_duration,
        gpu_duration: codestream.gpu_duration,
    }
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
                encoded.codestream_bytes()?,
                session,
            )?;
        } else {
            validate_lossless_roundtrip_on_metal_region_with_session(
                tile,
                tile.output_width,
                tile.output_height,
                metadata.bytes_per_pixel,
                encoded.codestream_bytes()?,
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

#[cfg(target_os = "macos")]
fn submit_planned_resident_lossless_tiles(
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
                components: metadata.components,
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
/// prepare_submit_duration semantics: HT (true) measures prepare + item
/// build + submit, classic (false) measures only the submit call.
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
                num_components: metadata.plan.components,
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
fn duration_share(duration: Duration, count: usize) -> Duration {
    if count == 0 {
        return Duration::ZERO;
    }
    let nanos = duration.as_nanos() / count as u128;
    Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64)
}

#[cfg(target_os = "macos")]
fn validate_lossless_roundtrip_on_metal_tile_with_session(
    tile: MetalLosslessEncodeTile<'_>,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let mut decoder = crate::J2kDecoder::new(codestream)?;
    let surface = decoder.decode_to_device_with_session(tile.format, session)?;
    if surface.dimensions() != (tile.output_width, tile.output_height) {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident validation geometry mismatch: expected {}x{}, got {}x{}",
                tile.output_width,
                tile.output_height,
                surface.dimensions().0,
                surface.dimensions().1
            ),
        });
    }
    if surface.pixel_format() != tile.format {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident validation format mismatch: expected {:?}, got {:?}",
                tile.format,
                surface.pixel_format()
            ),
        });
    }
    let expected_pitch = tile.output_width as usize * tile.format.bytes_per_pixel();
    if surface.pitch_bytes() != expected_pitch || tile.pitch_bytes != expected_pitch {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident validation requires contiguous source and decoded rows"
                .to_string(),
        });
    }
    let byte_len = expected_pitch
        .checked_mul(tile.output_height as usize)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal resident validation byte length overflow".to_string(),
        })?;
    let (decoded_buffer, decoded_offset) =
        surface
            .metal_buffer()
            .ok_or(crate::Error::UnsupportedMetalRequest {
                reason: "J2K Metal resident validation decode did not return a Metal buffer",
            })?;
    compute::validate_metal_buffers_match(
        tile.buffer,
        tile.byte_offset,
        decoded_buffer,
        decoded_offset,
        byte_len,
        session,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn validate_lossless_roundtrip_on_metal_region_with_session(
    source: MetalLosslessEncodeTile<'_>,
    output_width: u32,
    output_height: u32,
    bytes_per_pixel: usize,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let staged_buffer = compute::copy_interleaved_padded_to_shared_buffer(
        source.buffer,
        source.byte_offset,
        source.width,
        source.height,
        source.pitch_bytes,
        output_width,
        output_height,
        bytes_per_pixel,
        session,
    )?;
    let staged_tile = MetalLosslessEncodeTile {
        buffer: &staged_buffer,
        byte_offset: 0,
        width: output_width,
        height: output_height,
        pitch_bytes: output_width as usize * bytes_per_pixel,
        output_width,
        output_height,
        format: source.format,
    };
    validate_lossless_roundtrip_on_metal_tile_with_session(staged_tile, codestream, session)
}

#[cfg(target_os = "macos")]
fn should_try_resident_lossless_ht_cpu_packetization(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    staging: MetalEncodeInputStaging,
) -> bool {
    options.backend == EncodeBackendPreference::Auto
        && options.block_coding_mode == J2kBlockCodingMode::HighThroughput
        && options.reversible_transform == ReversibleTransform::Rct53
        && matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous)
        && tile.format == PixelFormat::Rgb8
}

#[cfg(target_os = "macos")]
fn encode_cpu_codestream_from_prepacketized_ht_tile(
    tile_body: ResidentHybridHtTileBody,
    tile: MetalLosslessEncodeTile<'_>,
) -> Result<EncodedJ2k, crate::Error> {
    let dummy_len = checked_mul_bytes(
        checked_mul_bytes(tile.output_width as usize, tile.output_height as usize),
        tile_body.bytes_per_pixel,
    );
    let dummy = vec![0u8; dummy_len];
    let samples = J2kLosslessSamples::new(
        &dummy,
        tile.output_width,
        tile.output_height,
        tile_body.components,
        tile_body.bit_depth,
        false,
    )
    .map_err(crate::Error::Decode)?;
    let mut wrapper = PrepacketizedHtj2kTileAccelerator {
        tile_data: Some(tile_body.tile_data),
    };
    let native_options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: tile_body.num_decomposition_levels,
        code_block_width_exp: tile_body.code_block_width_exp,
        code_block_height_exp: tile_body.code_block_height_exp,
        guard_bits: tile_body.guard_bits,
        use_ht_block_coding: true,
        progression_order: tile_body.progression_order,
        write_tlm: tile_body.write_tlm,
        use_mct: tile_body.used_fused_rct,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    };
    let codestream = j2k_native::encode_with_accelerator(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &native_options,
        &mut wrapper,
    )
    .map_err(|err| {
        crate::Error::Decode(j2k::J2kError::Backend(format!(
            "JPEG 2000 lossless encode failed: {err}"
        )))
    })?;
    Ok(EncodedJ2k {
        codestream,
        backend: BackendKind::Cpu,
        dispatch_report: j2k::adapter::encode_stage::J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

#[cfg(target_os = "macos")]
fn try_encode_lossless_tile_resident_ht_cpu_packetization_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<MetalLosslessEncodeOutcome>, crate::Error> {
    let encode_started = Instant::now();
    let Some(tile_body) = encode_resident_ht_tile_body_with_cpu_packetization(
        tile,
        options,
        session,
        staging,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
    )?
    else {
        return Ok(None);
    };
    let encoded = encode_cpu_codestream_from_prepacketized_ht_tile(tile_body, tile)?;
    let encode_duration = encode_started.elapsed();
    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        validate_lossless_roundtrip_on_metal_tile_with_session(
            tile,
            encoded.codestream.as_slice(),
            session,
        )?;
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };

    Ok(Some(MetalLosslessEncodeOutcome {
        encoded,
        input_copy_used: false,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: true,
            packetization_used: false,
            codestream_assembly_used: false,
        },
        input_copy_duration: Duration::ZERO,
        encode_duration,
        gpu_duration: None,
        validation_duration,
        host_readback_duration: Duration::ZERO,
    }))
}

#[cfg(target_os = "macos")]
fn try_encode_lossless_tile_device_resident_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<MetalLosslessEncodeOutcome>, crate::Error> {
    let Some(outcome) = try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
        tile, options, session, staging,
    )?
    else {
        return Ok(None);
    };
    host_outcome_from_buffer_outcome(outcome).map(Some)
}

#[cfg(target_os = "macos")]
fn try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Ok(None);
    }
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    let bytes_per_sample =
        u8::try_from(tile.format.bytes_per_sample()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal resident encode bytes per sample exceeds u8".to_string(),
        })?;
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    }
    let Some(plan) = lossless_device_encode_plan(
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        options,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
    )?
    else {
        return Ok(None);
    };

    let encode_started = Instant::now();
    let coefficient_count = lossless_device_coefficient_count(&plan.code_blocks)?;
    let prepared = compute::prepare_lossless_device_code_blocks(
        session,
        compute::J2kLosslessDevicePrepareJob {
            input: tile.buffer,
            input_byte_offset: tile.byte_offset,
            input_width: tile.width,
            input_height: tile.height,
            input_pitch_bytes: tile.pitch_bytes,
            output_width: tile.output_width,
            output_height: tile.output_height,
            components,
            bytes_per_sample,
            bit_depth,
            num_decomposition_levels: plan.num_decomposition_levels,
            coefficient_count,
        },
        plan.code_blocks.clone(),
    )?;
    let packetization_resolutions =
        resident_packetization_resolutions_from_lossless_device_plan(&plan)?;
    let packet_descriptors = packet_descriptors_for_lossless_device_order(
        plan.resolutions.len(),
        plan.components,
        plan.progression_order,
    )?;
    let packetization_job = compute::J2kResidentPacketizationEncodeJob {
        resolution_count: u32::try_from(plan.resolutions.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode resolution count exceeds u32".to_string(),
            }
        })?,
        num_layers: 1,
        num_components: plan.components,
        code_block_count: u32::try_from(plan.code_blocks.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode code-block count exceeds u32".to_string(),
            }
        })?,
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let assembly_job = compute::J2kLosslessCodestreamAssemblyJob {
        width: tile.output_width,
        height: tile.output_height,
        num_components: plan.components,
        bit_depth: plan.bit_depth,
        signed: false,
        num_decomposition_levels: plan.num_decomposition_levels,
        use_mct: plan.use_mct,
        guard_bits: plan.guard_bits,
        code_block_width_exp: plan.code_block_width_exp,
        code_block_height_exp: plan.code_block_height_exp,
        progression_order: plan.progression_order,
        write_tlm: plan.write_tlm,
        block_coding_mode: match plan.block_coding_mode {
            J2kBlockCodingMode::Classic => compute::J2kLosslessCodestreamBlockCodingMode::Classic,
            J2kBlockCodingMode::HighThroughput => {
                compute::J2kLosslessCodestreamBlockCodingMode::HighThroughput
            }
        },
    };
    let codestream = match plan.block_coding_mode {
        J2kBlockCodingMode::Classic => {
            let resident_tier1 =
                compute::encode_classic_tier1_prepared_device_code_blocks_resident(
                    session, prepared,
                )?;
            compute::encode_lossless_codestream_buffer_from_resident_tier1(
                session,
                &resident_tier1,
                packetization_job,
                assembly_job,
            )?
        }
        J2kBlockCodingMode::HighThroughput => {
            let resident_tier1 =
                compute::encode_ht_prepared_device_code_blocks_resident(session, prepared)?;
            compute::encode_lossless_codestream_buffer_from_resident_tier1(
                session,
                &resident_tier1,
                packetization_job,
                assembly_job,
            )?
        }
    };
    let encode_duration = encode_started.elapsed();

    let encoded = MetalEncodedJ2k {
        codestream_buffer: codestream.buffer,
        byte_offset: codestream.byte_offset,
        byte_len: codestream.byte_len,
        capacity: codestream.capacity,
        width: tile.output_width,
        height: tile.output_height,
        components,
        bit_depth,
        signed: false,
    };

    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
            validate_lossless_roundtrip_on_metal_tile_with_session(
                tile,
                encoded.codestream_bytes()?,
                session,
            )?;
        } else {
            validate_lossless_roundtrip_on_metal_region_with_session(
                tile,
                tile.output_width,
                tile.output_height,
                bytes_per_pixel,
                encoded.codestream_bytes()?,
                session,
            )?;
        }
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };

    Ok(Some(MetalLosslessBufferEncodeOutcome {
        encoded,
        input_copy_used: false,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: true,
            packetization_used: true,
            codestream_assembly_used: true,
        },
        input_copy_duration: Duration::ZERO,
        encode_duration,
        gpu_duration: codestream.gpu_duration,
        validation_duration,
    }))
}

#[cfg(target_os = "macos")]
fn encode_lossless_tile_to_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    validate_metal_encode_tile(tile)?;
    lossless_sample_shape(tile.format)?;
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal buffer output encode requires a device backend",
        });
    }
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    }
    if let Some(outcome) = try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
        tile, options, session, staging,
    )? {
        return Ok(outcome);
    }
    Err(crate::Error::UnsupportedMetalRequest {
        reason: "J2K Metal buffer output encode requires classic padded contiguous Gray/RGB lossless input with at most one DWT level",
    })
}

#[cfg(target_os = "macos")]
fn encode_lossless_tile_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    accelerator: &mut MetalEncodeStageAccelerator,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    validate_metal_encode_tile(tile)?;
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    if let Some(outcome) = try_encode_lossless_tile_resident_ht_cpu_packetization_with_report(
        tile, options, session, staging,
    )? {
        return Ok(outcome);
    }
    if should_try_resident_lossless_host_encode(options) {
        if let Some(outcome) =
            try_encode_lossless_tile_device_resident_with_report(tile, options, session, staging)?
        {
            return Ok(outcome);
        }
    }
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous)
        && options.backend == EncodeBackendPreference::RequireDevice
    {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode requires classic padded contiguous Gray/RGB lossless input with at most one DWT level",
        });
    }
    let mut input_copy_used = false;
    let mut input_copy_duration = Duration::ZERO;
    let mut staged_buffer = None;
    let mut source_byte_offset = tile.byte_offset;
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
        if tile.buffer.contents().is_null() {
            let copy_started = Instant::now();
            staged_buffer = Some(compute::copy_interleaved_padded_to_shared_buffer(
                tile.buffer,
                tile.byte_offset,
                tile.width,
                tile.height,
                tile.pitch_bytes,
                tile.output_width,
                tile.output_height,
                bytes_per_pixel,
                session,
            )?);
            input_copy_duration = copy_started.elapsed();
            input_copy_used = true;
            source_byte_offset = 0;
        }
    } else {
        let copy_started = Instant::now();
        staged_buffer = Some(compute::copy_interleaved_padded_to_shared_buffer(
            tile.buffer,
            tile.byte_offset,
            tile.width,
            tile.height,
            tile.pitch_bytes,
            tile.output_width,
            tile.output_height,
            bytes_per_pixel,
            session,
        )?);
        input_copy_duration = copy_started.elapsed();
        input_copy_used = true;
        source_byte_offset = 0;
    }
    let buffer = staged_buffer.as_ref().unwrap_or(tile.buffer);
    let len = tile.output_width as usize * tile.output_height as usize * bytes_per_pixel;
    let ptr = buffer.contents().cast::<u8>();
    if ptr.is_null() {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode input buffer is not host-visible",
        });
    }
    // SAFETY: Encoded Metal buffer views are bounds-checked before slice construction.
    let data = unsafe { core::slice::from_raw_parts(ptr.add(source_byte_offset), len) };
    let samples = J2kLosslessSamples::new(
        data,
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        false,
    )
    .map_err(crate::Error::Decode)?;

    let encode_options = host_output_encode_options(options);
    let encode_started = Instant::now();
    let encoded = j2k::encode_j2k_lossless_with_accelerator(
        samples,
        &encode_options,
        BackendKind::Metal,
        accelerator,
    )
    .map_err(crate::Error::Decode)?;
    let encode_duration = encode_started.elapsed();
    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        validate_lossless_roundtrip_on_metal_with_session(samples, &encoded.codestream, session)?;
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };
    Ok(MetalLosslessEncodeOutcome {
        encoded,
        input_copy_used,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: false,
            packetization_used: false,
            codestream_assembly_used: false,
        },
        input_copy_duration,
        encode_duration,
        gpu_duration: None,
        validation_duration,
        host_readback_duration: Duration::ZERO,
    })
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for submitted host-byte batch encode on non-macOS.
pub fn submit_lossless_batch(
    request: MetalLosslessEncodeBatchRequest<'_, '_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    let _ = (request, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for submitted Metal-buffer batch encode on non-macOS.
pub fn submit_lossless_batch_to_metal(
    request: MetalLosslessEncodeBatchRequest<'_, '_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalBufferEncodeBatch, crate::Error> {
    let _ = (request, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for reported batch encode on non-macOS.
pub fn encode_lossless_batch_with_report(
    request: MetalLosslessEncodeBatchRequest<'_, '_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    let _ = (request, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(test)]
mod tests;
