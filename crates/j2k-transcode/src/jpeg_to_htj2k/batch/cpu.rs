// SPDX-License-Identifier: MIT OR Apache-2.0

use rayon::prelude::*;

use super::super::workspace::JpegTranscodeWorkspace;
use super::individual::aggregate_tile_timings;
use super::{
    batch_output, encoded_transcode_retained_bytes, jpeg_tile_batch_to_htj2k_with_scratch,
    jpeg_to_htj2k_with_scratch, validate_batch_route, validate_jpeg_transcode_workspace,
    BatchTranscodeReport, CpuOnlyJ2kEncodeStageAccelerator, EncodedTranscode,
    EncodedTranscodeBatch, HostLiveBudget, JpegTileBatchInput, JpegToHtj2kError,
    JpegToHtj2kOptions, JpegToHtj2kScratch,
};
use crate::allocation::{checked_capacity_bytes, try_vec_with_capacity};
use crate::CpuOnlyDctToWaveletStageAccelerator;

type TileOutcome = Result<EncodedTranscode, JpegToHtj2kError>;

struct TilePlan {
    workspace: Result<usize, JpegToHtj2kError>,
    quota: usize,
}

struct WaveOutcome {
    tile: Option<TileOutcome>,
    quota_exhausted: bool,
}

pub(in super::super) fn jpeg_cpu_tile_batch_to_htj2k_with_scratch(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    let mut transform = CpuOnlyDctToWaveletStageAccelerator;
    let workers = rayon::current_num_threads();
    if tiles.len() < 2 || workers == 1 {
        return jpeg_tile_batch_to_htj2k_with_scratch(
            tiles,
            options,
            scratch,
            &mut transform,
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        );
    }
    // Preserve the existing all-tile admission policy even though waves can
    // reuse workspace. Invalid headers remain errors in their input slots.
    validate_batch_route(tiles, options, &transform)?;
    transcode_cpu_tiles(
        tiles,
        options,
        scratch.retained_bytes()?,
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        workers,
    )
}

fn transcode_cpu_tiles(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    retained_scratch_bytes: usize,
    max_host_bytes: usize,
    workers: usize,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    let wave_width = tiles.len().min(workers).max(1);
    let mut admission = HostLiveBudget::with_cap(max_host_bytes);
    admission.add_bytes(retained_scratch_bytes)?;
    admission.add_capacity::<TileOutcome>(tiles.len())?;
    admission.add_capacity::<WaveOutcome>(wave_width)?;
    admission.add_capacity::<TilePlan>(wave_width)?;
    let mut output = try_vec_with_capacity::<TileOutcome>(tiles.len())?;
    let mut wave_results = try_vec_with_capacity::<WaveOutcome>(wave_width)?;
    let mut retained = HostLiveBudget::with_cap(max_host_bytes);
    retained.add_bytes(retained_scratch_bytes)?;
    retained.add_capacity::<TileOutcome>(output.capacity())?;
    retained.add_capacity::<WaveOutcome>(wave_results.capacity())?;

    for wave in tiles.chunks(wave_width) {
        let mut plan_admission = retained;
        plan_admission.add_capacity::<TilePlan>(wave.len())?;
        let mut plans = try_vec_with_capacity::<TilePlan>(wave.len())?;
        let mut wave_budget = retained;
        wave_budget.add_capacity::<TilePlan>(plans.capacity())?;
        for tile in wave {
            plans.push(TilePlan {
                workspace: validate_jpeg_transcode_workspace(tile.bytes, options)
                    .map(JpegTranscodeWorkspace::peak_bytes),
                quota: 0,
            });
        }

        if !assign_wave_quotas(&mut plans, wave_budget.remaining_bytes()?) {
            transcode_serial_wave(wave, options, plans, &mut retained, &mut output)?;
            continue;
        }

        // Indexed collection into a buffer already reserved for the wave
        // cannot grow its capacity while workers hold their reservations.
        let reserved_capacity = wave_results.capacity();
        plans
            .into_par_iter()
            .zip(wave.par_iter())
            .map(|(plan, input)| {
                if let Err(error) = plan.workspace {
                    WaveOutcome {
                        tile: Some(Err(error)),
                        quota_exhausted: false,
                    }
                } else {
                    let tile = transcode_cpu_tile(input.bytes, options, plan.quota);
                    let quota_exhausted =
                        matches!(tile, Err(JpegToHtj2kError::MemoryCapExceeded { .. }));
                    WaveOutcome {
                        tile: Some(tile),
                        quota_exhausted,
                    }
                }
            })
            .collect_into_vec(&mut wave_results);
        assert_eq!(wave_results.capacity(), reserved_capacity);

        if wave_results.iter().any(|result| result.quota_exhausted) {
            retry_cpu_wave(wave, options, &mut wave_results, &mut retained, &mut output)?;
        } else {
            for result in wave_results.drain(..) {
                let tile = result.tile.ok_or(JpegToHtj2kError::InternalInvariant {
                    what: "CPU wave omitted a tile result",
                })?;
                retain_tile(tile, &mut retained, &mut output)?;
            }
        }
    }

    let timings = aggregate_tile_timings(&output);
    Ok(batch_output(
        output,
        BatchTranscodeReport {
            tile_count: tiles.len(),
            successful_tiles: 0,
            failed_tiles: 0,
            transformed_components: timings.component_count,
            reversible_dwt53_batches: 0,
            reversible_dwt53_batch_jobs: 0,
            extract_us: timings.jpeg_dct_extract_us,
            transform_us: timings.dct_to_wavelet_total_us,
            encode_us: timings.htj2k_encode_us,
            timings,
            coefficient_path: options.coefficient_path,
        },
    ))
}

fn assign_wave_quotas(plans: &mut [TilePlan], available_bytes: usize) -> bool {
    let mut workspace = HostLiveBudget::with_cap(available_bytes);
    let mut valid_tiles = 0;
    for plan in plans.iter() {
        if let Ok(bytes) = plan.workspace {
            if workspace.add_bytes(bytes).is_err() {
                return false;
            }
            valid_tiles += 1;
        }
    }
    let share = (available_bytes - workspace.live_bytes()) / valid_tiles.max(1);
    for plan in plans {
        if let Ok(bytes) = plan.workspace {
            plan.quota = bytes + share;
        }
    }
    true
}

fn transcode_cpu_tile(bytes: &[u8], options: &JpegToHtj2kOptions, quota: usize) -> TileOutcome {
    let external = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
        .checked_sub(quota)
        .ok_or(JpegToHtj2kError::MemoryCapExceeded {
            requested: quota,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    // A fresh scratch owner dies with the task, so retained buffers from an
    // earlier/larger tile cannot escape the current reservation.
    jpeg_to_htj2k_with_scratch(
        bytes,
        options,
        &mut JpegToHtj2kScratch::default(),
        &mut CpuOnlyDctToWaveletStageAccelerator,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
        external,
    )
}

fn transcode_serial_wave(
    wave: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    plans: Vec<TilePlan>,
    retained: &mut HostLiveBudget,
    output: &mut Vec<TileOutcome>,
) -> Result<(), JpegToHtj2kError> {
    let plan_bytes = checked_capacity_bytes::<TilePlan>(plans.capacity())?;
    for (tile, plan) in wave.iter().zip(plans) {
        let result = if let Err(error) = plan.workspace {
            Err(error)
        } else {
            let mut live = *retained;
            live.add_bytes(plan_bytes)?;
            transcode_cpu_tile(tile.bytes, options, live.remaining_bytes()?)
        };
        retain_tile(result, retained, output)?;
    }
    Ok(())
}

fn retry_cpu_wave(
    wave: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    wave_results: &mut Vec<WaveOutcome>,
    retained: &mut HostLiveBudget,
    output: &mut Vec<TileOutcome>,
) -> Result<(), JpegToHtj2kError> {
    // Release every speculative output before retrying valid tiles in input
    // order. Preserve original parse/codec errors without replaying them.
    for result in wave_results.iter_mut() {
        if result.quota_exhausted || matches!(result.tile, Some(Ok(_))) {
            result.tile = None;
        }
    }
    for (input, result) in wave.iter().zip(wave_results.drain(..)) {
        let tile = match result.tile {
            Some(error) => error,
            None => transcode_cpu_tile(input.bytes, options, retained.remaining_bytes()?),
        };
        retain_tile(tile, retained, output)?;
    }
    Ok(())
}

fn retain_tile(
    tile: TileOutcome,
    retained: &mut HostLiveBudget,
    output: &mut Vec<TileOutcome>,
) -> Result<(), JpegToHtj2kError> {
    if let Ok(encoded) = &tile {
        retained.add_bytes(encoded_transcode_retained_bytes(encoded)?)?;
    }
    output.push(tile);
    Ok(())
}

#[cfg(test)]
mod tests;
