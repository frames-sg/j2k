// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::codestream::WaveletTransform;
use super::super::decode::{DecompositionStorage, TileDecodeContext};
#[cfg(feature = "parallel")]
use super::banded::{filter_2d_banded_pair, pair_runs_on_pool};
use super::banded::{planned_band_scratch_elements, planned_pair_band_scratch_elements};
use super::direct::{apply_level, apply_level_into};
use super::interleave_i64::apply_level_i64;
use super::model::{
    idwt_buffer_size, IDWTInput, IDWTInputI64, IDWTTempOutput, IdwtPlacement, InputSource,
};
use super::roi::apply_roi;
use crate::error::{bail, DecodingError, ValidationError};
use crate::j2c::rect::IntRect;
use crate::j2c::Header;
use crate::{
    checked_decode_usize_product2, try_reserve_decode_elements, HtCodeBlockDecoder, Result,
};

/// Apply the inverse discrete wavelet transform (see Annex F). The output
/// will be transformed samples covering the rectangle of the smallest
/// decomposition level.
#[expect(
    clippy::too_many_lines,
    reason = "the level loop keeps both ping-pong buffers, band scratch, and the backend hook in one place"
)]
pub(crate) fn apply(
    storage: &DecompositionStorage<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_idx: usize,
    header: &Header<'_>,
    transform: WaveletTransform,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
    parallel: bool,
) -> Result<()> {
    tile_ctx.idwt_output.placement = IdwtPlacement::Coefficients;
    if storage.exact_integer_decode {
        return apply_i64(storage, tile_ctx, component_idx, header, transform);
    }

    let tile_decompositions = &storage.tile_decompositions[component_idx];
    if storage.roi_plan.is_some() {
        return apply_roi(storage, tile_ctx, component_idx, header, transform);
    }

    let mut decompositions = &storage.decompositions[tile_decompositions.decompositions.clone()];
    // If we requested a lower resolution level, we can skip some decompositions.
    // A last level left to the RGB8 pack stays in its sub-bands.
    decompositions = &decompositions[..decompositions.len().saturating_sub(
        header.skipped_resolution_levels as usize
            + usize::from(tile_ctx.fused_last_level.is_some()),
    )];
    let ll_sub_band = &storage.sub_bands[tile_decompositions.first_ll_sub_band];

    // To explain a bit why we have this scratch buffer and another coefficient
    // buffer: During IDWT, we need to continuously interleave the 4 sub-bands
    // into a new buffer, which is then either returned or used as the input
    // for the next decomposition, etc. It would be very inefficient if we
    // kept allocating a new buffer at each level. The coordinator drops prior
    // component/tile capacities before this call to keep the live-allocation
    // baseline exact; within this component, two buffers are continuously
    // swapped because one level's output feeds the next.
    let (scratch_buf, output, band_scratch) = (
        &mut tile_ctx.idwt_scratch_buffer,
        &mut tile_ctx.idwt_output,
        &mut tile_ctx.idwt_band_scratch,
    );

    // Earlier components have consumed every coefficient before their
    // finest high bands. Reconstruct into that dead prefix while preserving
    // component zero's LL in `output.coefficients` until the final pack.
    let mut prefix = if tile_ctx.fused_last_level.is_some() && component_idx > 0 {
        let rect = decompositions.last().unwrap().rect;
        let len = checked_decode_usize_product2(rect.width() as usize, rect.height() as usize)?;
        Some(&mut tile_ctx.channel_data[component_idx - 1].container[..len])
    } else {
        None
    };

    if decompositions.is_empty() {
        // Single decomposition, just copy the coefficients from the sub-band.
        let source = &storage.coefficients[ll_sub_band.coefficients.clone()];
        try_reserve_decode_elements(&mut output.coefficients, source.len())?;
        output.coefficients.clear();
        output.coefficients.extend_from_slice(source);

        output.rect = ll_sub_band.rect;
        tile_ctx.debug_counters.idwt_output_samples = tile_ctx
            .debug_counters
            .idwt_output_samples
            .saturating_add(output.coefficients.len());

        return Ok(());
    }

    // The two finest levels may run as one pass on the pool, which writes
    // only the finer level.
    #[cfg(feature = "parallel")]
    let pair = parallel
        && backend.is_none()
        && decompositions.len() >= 2
        && pair_runs_on_pool(
            decompositions[decompositions.len() - 2].rect,
            decompositions[decompositions.len() - 1].rect,
            parallel,
        );
    #[cfg(not(feature = "parallel"))]
    let pair = false;
    let writes = decompositions.len() - usize::from(pair);

    // The coefficient array will always be the one that holds the coefficients
    // from the highest decomposition. Therefore, reserve as much.
    let (s_min, s_max) = idwt_buffer_size(decompositions.last().unwrap().rect)?;
    if prefix.is_none() && output.coefficients.len() < s_min {
        try_reserve_decode_elements(&mut output.coefficients, s_max)?;
    }

    if writes > 1 {
        // Due to the above, the intermediate buffer will never need more than
        // the level written before the last: the second-highest
        // decomposition, or the third-highest below a paired pass.
        let (s_min, s_max) = idwt_buffer_size(decompositions[writes - 2].rect)?;

        if scratch_buf.len() < s_min {
            try_reserve_decode_elements(scratch_buf, s_max)?;
        }
    }

    // Determine which buffer we should use first, such that the `coefficients`
    // array will always hold the final values.
    let mut use_scratch = writes.is_multiple_of(2);
    let mut current_source = InputSource::SubBand;
    let mut current_rect = ll_sub_band.rect;
    // Level buffers only grow, so the previous level's samples are the first
    // `current_len` of whichever buffer holds them.
    let mut current_len = 0;
    let mut temp_output = IDWTTempOutput {
        rect: ll_sub_band.rect,
    };

    for (index, decomposition) in decompositions.iter().enumerate() {
        #[cfg(feature = "parallel")]
        if pair && index + 2 == decompositions.len() {
            let input = match current_source {
                InputSource::SubBand => IDWTInput::from_sub_band(ll_sub_band, storage),
                InputSource::Scratch => {
                    IDWTInput::from_output(&scratch_buf[..current_len], current_rect)
                }
                InputSource::Output => unreachable!(),
            };
            let outer = &decompositions[index + 1];
            let outer_len = checked_decode_usize_product2(
                outer.rect.width() as usize,
                outer.rect.height() as usize,
            )?;
            let target = if let Some(prefix) = prefix.as_deref_mut() {
                prefix
            } else {
                super::try_grow_level_buffer(&mut output.coefficients, outer_len)?;
                &mut output.coefficients[..]
            };
            temp_output = filter_2d_banded_pair(
                input,
                [decomposition, outer],
                target,
                band_scratch,
                transform,
                storage,
            )?;
            let inner_len = checked_decode_usize_product2(
                decomposition.rect.width() as usize,
                decomposition.rect.height() as usize,
            )?;
            tile_ctx.debug_counters.idwt_output_samples = tile_ctx
                .debug_counters
                .idwt_output_samples
                .saturating_add(inner_len)
                .saturating_add(outer_len);
            current_len = outer_len;
            break;
        }
        #[cfg(not(feature = "parallel"))]
        let _ = index;
        temp_output = match (current_source, use_scratch) {
            (InputSource::SubBand, true) => apply_level(
                IDWTInput::from_sub_band(ll_sub_band, storage),
                scratch_buf,
                band_scratch,
                decomposition,
                transform,
                storage,
                backend,
                parallel,
            )?,
            (InputSource::SubBand, false) if prefix.is_some() => apply_level_into(
                IDWTInput::from_sub_band(ll_sub_band, storage),
                prefix.as_deref_mut().unwrap(),
                band_scratch,
                decomposition,
                transform,
                storage,
                parallel,
            )?,
            (InputSource::Scratch, false) if prefix.is_some() => apply_level_into(
                IDWTInput::from_output(&scratch_buf[..current_len], current_rect),
                prefix.as_deref_mut().unwrap(),
                band_scratch,
                decomposition,
                transform,
                storage,
                parallel,
            )?,
            (InputSource::SubBand, false) => apply_level(
                IDWTInput::from_sub_band(ll_sub_band, storage),
                &mut output.coefficients,
                band_scratch,
                decomposition,
                transform,
                storage,
                backend,
                parallel,
            )?,
            (InputSource::Scratch, false) => apply_level(
                IDWTInput::from_output(&scratch_buf[..current_len], current_rect),
                &mut output.coefficients,
                band_scratch,
                decomposition,
                transform,
                storage,
                backend,
                parallel,
            )?,
            (InputSource::Output, true) => apply_level(
                IDWTInput::from_output(
                    &prefix.as_deref().unwrap_or(&output.coefficients)[..current_len],
                    current_rect,
                ),
                scratch_buf,
                band_scratch,
                decomposition,
                transform,
                storage,
                backend,
                parallel,
            )?,
            (InputSource::Scratch, true) | (InputSource::Output, false) => unreachable!(),
        };
        current_source = if use_scratch {
            InputSource::Scratch
        } else {
            InputSource::Output
        };
        current_rect = temp_output.rect;
        let output_samples = checked_decode_usize_product2(
            current_rect.width() as usize,
            current_rect.height() as usize,
        )?;
        tile_ctx.debug_counters.idwt_output_samples = tile_ctx
            .debug_counters
            .idwt_output_samples
            .saturating_add(output_samples);
        current_len = output_samples;
        use_scratch = !use_scratch;
    }

    // Keep the first fused LL intact when writing a prior plane. Otherwise
    // drop samples a larger earlier component left past this output rect.
    if prefix.is_none() {
        output.coefficients.truncate(current_len);
    }
    output.rect = temp_output.rect;
    Ok(())
}

/// The `f32` IDWT workspace [`apply`] holds at most for a component whose
/// finest active level is `highest`, above `second` and `third` when it has
/// them, planned for the pool: the finest level's output, the band scratch,
/// and the intermediate buffer of the level written before the last. Level
/// by level, that is `second` with the finest level's band scratch. When
/// the two finest levels pair (see [`planned_pair_band_scratch_elements`]),
/// their one pass writes only the finest level, so the buffer holds `third`
/// and the band scratch is the pair's. A serial decode or a code-block
/// decoder hook keeps the level passes, so a component whose levels pair is
/// planned for the larger of both.
pub(crate) fn planned_workspace_elements(
    highest: IntRect,
    [second, third]: [Option<IntRect>; 2],
    transform: WaveletTransform,
) -> Result<usize> {
    let padded = |rect: Option<IntRect>| {
        rect.map_or(Ok(0), |rect| {
            idwt_buffer_size(rect).map(|(_, maximum)| maximum)
        })
    };
    let total = |elements: [usize; 3]| {
        elements
            .into_iter()
            .try_fold(0_usize, usize::checked_add)
            .ok_or(ValidationError::ImageTooLarge)
    };
    let output = padded(Some(highest))?;
    let band_scratch =
        planned_band_scratch_elements(highest.width() as usize, highest.height() as usize)?;
    let levels = total([output, padded(second)?, band_scratch])?;
    let pair_band_scratch = match second {
        Some(inner) => planned_pair_band_scratch_elements(inner, highest, transform)?,
        None => None,
    };
    match pair_band_scratch {
        Some(pair_band_scratch) => {
            Ok(levels.max(total([output, padded(third)?, pair_band_scratch])?))
        }
        None => Ok(levels),
    }
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

    if decompositions.is_empty() {
        let source = &storage.coefficients_i64[ll_sub_band.coefficients.clone()];
        try_reserve_decode_elements(&mut output.coefficients_i64, source.len())?;
        output.coefficients_i64.clear();
        output.coefficients_i64.extend_from_slice(source);
        output.rect = ll_sub_band.rect;
        tile_ctx.debug_counters.idwt_output_samples = tile_ctx
            .debug_counters
            .idwt_output_samples
            .saturating_add(output.coefficients_i64.len());
        return Ok(());
    }

    let (s_min, s_max) = idwt_buffer_size(decompositions.last().unwrap().rect)?;
    if output.coefficients_i64.len() < s_min {
        try_reserve_decode_elements(&mut output.coefficients_i64, s_max)?;
    }

    if decompositions.len() > 1 {
        let (s_min, s_max) = idwt_buffer_size(decompositions[decompositions.len() - 2].rect)?;
        if scratch_buf.len() < s_min {
            try_reserve_decode_elements(scratch_buf, s_max)?;
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
            )?,
            (InputSource::SubBand, false) => apply_level_i64(
                IDWTInputI64::from_sub_band(ll_sub_band, storage),
                &mut output.coefficients_i64,
                decomposition,
                storage,
            )?,
            (InputSource::Scratch, false) => apply_level_i64(
                IDWTInputI64::from_output(scratch_buf, current_rect),
                &mut output.coefficients_i64,
                decomposition,
                storage,
            )?,
            (InputSource::Output, true) => apply_level_i64(
                IDWTInputI64::from_output(&output.coefficients_i64, current_rect),
                scratch_buf,
                decomposition,
                storage,
            )?,
            (InputSource::Scratch, true) | (InputSource::Output, false) => unreachable!(),
        };
        current_source = if use_scratch {
            InputSource::Scratch
        } else {
            InputSource::Output
        };
        current_rect = temp_output.rect;
        let output_samples = checked_decode_usize_product2(
            current_rect.width() as usize,
            current_rect.height() as usize,
        )?;
        tile_ctx.debug_counters.idwt_output_samples = tile_ctx
            .debug_counters
            .idwt_output_samples
            .saturating_add(output_samples);
        use_scratch = !use_scratch;
    }

    output.rect = temp_output.rect;
    Ok(())
}
