// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::build::Decomposition;
use super::super::codestream::WaveletTransform;
use super::super::decode::{DecompositionStorage, TileDecodeContext};
use super::direct::apply_level;
use super::interleave_i64::apply_level_i64;
use super::model::{IDWTInput, IDWTInputI64, IDWTTempOutput, InputSource};
use super::roi::apply_roi;
use crate::error::{bail, DecodingError};
use crate::j2c::Header;
use crate::{HtCodeBlockDecoder, Result};

/// Apply the inverse discrete wavelet transform (see Annex F). The output
/// will be transformed samples covering the rectangle of the smallest
/// decomposition level.
pub(crate) fn apply(
    storage: &DecompositionStorage<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_idx: usize,
    header: &Header<'_>,
    transform: WaveletTransform,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
) -> Result<()> {
    if storage.exact_integer_decode {
        return apply_i64(storage, tile_ctx, component_idx, header, transform);
    }

    let tile_decompositions = &storage.tile_decompositions[component_idx];
    if storage.roi_plan.is_some() {
        return apply_roi(storage, tile_ctx, component_idx, header, transform);
    }

    let mut decompositions = &storage.decompositions[tile_decompositions.decompositions.clone()];
    // If we requested a lower resolution level, we can skip some decompositions.
    decompositions = &decompositions[..decompositions
        .len()
        .saturating_sub(header.skipped_resolution_levels as usize)];
    let ll_sub_band = &storage.sub_bands[tile_decompositions.first_ll_sub_band];

    // To explain a bit why we have this scratch buffer and another coefficient
    // buffer: During IDWT, we need to continuously interleave the 4 sub-bands
    // into a new buffer, which is then either returned or used as the input
    // for the next decomposition, etc. It would be very inefficient if we
    // kept allocating new buffers each time. Therefore, we try to reuse them,
    // not only for all decompositions of a single tile, but all decompositions
    // of _all_ tiles.
    // Due to the fact that the output from the previous iteration might be
    // used as the input of the next, we need two separate buffers, which
    // are continuously swapped.
    let (scratch_buf, output) = (&mut tile_ctx.idwt_scratch_buffer, &mut tile_ctx.idwt_output);

    let estimate_buffer_size = |decomposition: &Decomposition| {
        let total_width = decomposition.rect.width() as usize;
        let total_height = decomposition.rect.height() as usize;

        let min = total_width * total_height;
        // Different sub-bands can have shifts by one, so add padding
        // for the maximum case.
        let max = (total_width + 1) * (total_height + 1);

        (min, max)
    };

    if decompositions.is_empty() {
        // Single decomposition, just copy the coefficients from the sub-band.
        output.coefficients.clear();
        output
            .coefficients
            .extend_from_slice(&storage.coefficients[ll_sub_band.coefficients.clone()]);

        output.rect = ll_sub_band.rect;
        tile_ctx.debug_counters.idwt_output_samples = tile_ctx
            .debug_counters
            .idwt_output_samples
            .saturating_add(output.coefficients.len());

        return Ok(());
    }

    // The coefficient array will always be the one that holds the coefficients
    // from the highest decomposition. Therefore, reserve as much.
    let (s_min, s_max) = estimate_buffer_size(decompositions.last().unwrap());
    if output.coefficients.len() < s_min {
        output
            .coefficients
            .reserve_exact(s_max - output.coefficients.len());
    }

    if decompositions.len() > 1 {
        // Due to the above, the intermediate buffer will never need more than
        // the second-highest decomposition.
        let (s_min, s_max) = estimate_buffer_size(&decompositions[decompositions.len() - 2]);

        if scratch_buf.len() < s_min {
            scratch_buf.reserve_exact(s_max - scratch_buf.len());
        }
    }

    // Determine which buffer we should use first, such that the `coefficients`
    // array will always hold the final values.
    let mut use_scratch = decompositions.len().is_multiple_of(2);
    let mut current_source = InputSource::SubBand;
    let mut current_rect = ll_sub_band.rect;
    let mut temp_output = IDWTTempOutput {
        rect: ll_sub_band.rect,
    };

    for decomposition in decompositions {
        temp_output = match (current_source, use_scratch) {
            (InputSource::SubBand, true) => apply_level(
                IDWTInput::from_sub_band(ll_sub_band, storage),
                scratch_buf,
                decomposition,
                transform,
                storage,
                backend,
            )?,
            (InputSource::SubBand, false) => apply_level(
                IDWTInput::from_sub_band(ll_sub_band, storage),
                &mut output.coefficients,
                decomposition,
                transform,
                storage,
                backend,
            )?,
            (InputSource::Scratch, false) => apply_level(
                IDWTInput::from_output(scratch_buf, current_rect),
                &mut output.coefficients,
                decomposition,
                transform,
                storage,
                backend,
            )?,
            (InputSource::Output, true) => apply_level(
                IDWTInput::from_output(&output.coefficients, current_rect),
                scratch_buf,
                decomposition,
                transform,
                storage,
                backend,
            )?,
            (InputSource::Scratch, true) | (InputSource::Output, false) => unreachable!(),
        };
        current_source = if use_scratch {
            InputSource::Scratch
        } else {
            InputSource::Output
        };
        current_rect = temp_output.rect;
        tile_ctx.debug_counters.idwt_output_samples = tile_ctx
            .debug_counters
            .idwt_output_samples
            .saturating_add(current_rect.width() as usize * current_rect.height() as usize);
        use_scratch = !use_scratch;
    }

    output.rect = temp_output.rect;
    Ok(())
}

fn apply_i64(
    storage: &DecompositionStorage<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_idx: usize,
    header: &Header<'_>,
    transform: WaveletTransform,
) -> Result<()> {
    if transform != WaveletTransform::Reversible53 {
        bail!(DecodingError::UnsupportedFeature(
            "25-38 bit integer IDWT currently requires reversible 5/3 coding",
        ));
    }
    if storage.roi_plan.is_some() {
        bail!(DecodingError::UnsupportedFeature(
            "25-38 bit region decode requires exact integer region IDWT support",
        ));
    }

    let tile_decompositions = &storage.tile_decompositions[component_idx];
    let mut decompositions = &storage.decompositions[tile_decompositions.decompositions.clone()];
    decompositions = &decompositions[..decompositions
        .len()
        .saturating_sub(header.skipped_resolution_levels as usize)];
    let ll_sub_band = &storage.sub_bands[tile_decompositions.first_ll_sub_band];
    let (scratch_buf, output) = (
        &mut tile_ctx.idwt_scratch_buffer_i64,
        &mut tile_ctx.idwt_output,
    );

    let estimate_buffer_size = |decomposition: &Decomposition| {
        let total_width = decomposition.rect.width() as usize;
        let total_height = decomposition.rect.height() as usize;
        let min = total_width * total_height;
        let max = (total_width + 1) * (total_height + 1);
        (min, max)
    };

    if decompositions.is_empty() {
        output.coefficients_i64.clear();
        output
            .coefficients_i64
            .extend_from_slice(&storage.coefficients_i64[ll_sub_band.coefficients.clone()]);
        output.rect = ll_sub_band.rect;
        tile_ctx.debug_counters.idwt_output_samples = tile_ctx
            .debug_counters
            .idwt_output_samples
            .saturating_add(output.coefficients_i64.len());
        return Ok(());
    }

    let (s_min, s_max) = estimate_buffer_size(decompositions.last().unwrap());
    if output.coefficients_i64.len() < s_min {
        output
            .coefficients_i64
            .reserve_exact(s_max - output.coefficients_i64.len());
    }

    if decompositions.len() > 1 {
        let (s_min, s_max) = estimate_buffer_size(&decompositions[decompositions.len() - 2]);
        if scratch_buf.len() < s_min {
            scratch_buf.reserve_exact(s_max - scratch_buf.len());
        }
    }

    let mut use_scratch = decompositions.len().is_multiple_of(2);
    let mut current_source = InputSource::SubBand;
    let mut current_rect = ll_sub_band.rect;
    let mut temp_output = IDWTTempOutput {
        rect: ll_sub_band.rect,
    };

    for decomposition in decompositions {
        temp_output = match (current_source, use_scratch) {
            (InputSource::SubBand, true) => apply_level_i64(
                IDWTInputI64::from_sub_band(ll_sub_band, storage),
                scratch_buf,
                decomposition,
                storage,
            ),
            (InputSource::SubBand, false) => apply_level_i64(
                IDWTInputI64::from_sub_band(ll_sub_band, storage),
                &mut output.coefficients_i64,
                decomposition,
                storage,
            ),
            (InputSource::Scratch, false) => apply_level_i64(
                IDWTInputI64::from_output(scratch_buf, current_rect),
                &mut output.coefficients_i64,
                decomposition,
                storage,
            ),
            (InputSource::Output, true) => apply_level_i64(
                IDWTInputI64::from_output(&output.coefficients_i64, current_rect),
                scratch_buf,
                decomposition,
                storage,
            ),
            (InputSource::Scratch, true) | (InputSource::Output, false) => unreachable!(),
        };
        current_source = if use_scratch {
            InputSource::Scratch
        } else {
            InputSource::Output
        };
        current_rect = temp_output.rect;
        tile_ctx.debug_counters.idwt_output_samples = tile_ctx
            .debug_counters
            .idwt_output_samples
            .saturating_add(current_rect.width() as usize * current_rect.height() as usize);
        use_scratch = !use_scratch;
    }

    output.rect = temp_output.rect;
    Ok(())
}
