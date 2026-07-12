// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    batch_output, encoded_transcode_retained_bytes, jpeg_to_htj2k_with_scratch,
    BatchTranscodeReport, DctToWaveletStageAccelerator, EncodedTranscode, EncodedTranscodeBatch,
    HostLiveBudget, Instant, J2kEncodeStageAccelerator, JpegTileBatchInput, JpegToHtj2kError,
    JpegToHtj2kOptions, JpegToHtj2kScratch, TranscodeTimingReport,
};
use crate::allocation::try_vec_with_capacity;

pub(super) fn transcode_tile_batch_individually<
    A: DctToWaveletStageAccelerator,
    E: J2kEncodeStageAccelerator,
>(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    encode_accelerator: &mut E,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    let start = Instant::now();
    let mut output_tiles = try_vec_with_capacity(tiles.len())?;
    let mut completed_outputs = 0usize;
    for tile in tiles {
        let mut external = HostLiveBudget::process_cap();
        external
            .add_capacity::<Result<EncodedTranscode, JpegToHtj2kError>>(output_tiles.capacity())?;
        external.add_bytes(completed_outputs)?;
        let encoded = jpeg_to_htj2k_with_scratch(
            tile.bytes,
            options,
            scratch,
            accelerator,
            encode_accelerator,
            external.live_bytes(),
        );
        if let Ok(encoded) = encoded.as_ref() {
            completed_outputs = completed_outputs
                .checked_add(encoded_transcode_retained_bytes(encoded)?)
                .ok_or(JpegToHtj2kError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                })?;
        }
        output_tiles.push(encoded);
    }
    let mut timings = aggregate_tile_timings(&output_tiles);
    timings.tile_count = output_tiles.iter().filter(|tile| tile.is_ok()).count();
    let elapsed_us = start.elapsed().as_micros();
    if timings.dct_to_wavelet_total_us == 0 {
        timings.dct_to_wavelet_total_us = elapsed_us
            .saturating_sub(timings.jpeg_dct_extract_us)
            .saturating_sub(timings.htj2k_encode_us);
    }
    Ok(batch_output(
        output_tiles,
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

fn aggregate_tile_timings(
    tiles: &[Result<EncodedTranscode, JpegToHtj2kError>],
) -> TranscodeTimingReport {
    let mut timings = TranscodeTimingReport::default();
    for tile in tiles.iter().filter_map(|tile| tile.as_ref().ok()) {
        timings.add_assign(tile.report.timings);
    }
    timings
}
