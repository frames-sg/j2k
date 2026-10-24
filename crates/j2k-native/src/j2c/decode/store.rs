// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::codestream::WaveletTransform;
use super::{
    bail, ComponentInfo, ComponentTile, DecodingError, DecompositionStorage, Header,
    HtCodeBlockDecoder, J2kStoreComponentJob, OutputRegion, ResolutionTile, Result, Tile,
    TileDecodeContext,
};
use crate::j2c::build::CoefficientLayout;
use crate::j2c::idwt::{IdwtPlacement, LastLevel};
use crate::j2c::rect::IntRect;
use crate::j2c::roi::{output_grid_offset, output_region_rect};
use crate::math::{round_ties_even_then_add, SIMD_WIDTH};

mod upsample;

pub(super) fn apply_sign_shift_after_mct(
    tile_ctx: &mut TileDecodeContext,
    component_infos: &[ComponentInfo],
) {
    let round_irreversible_output = tile_ctx.round_irreversible_output;
    for (channel_data, component_info) in tile_ctx
        .channel_data
        .iter_mut()
        .zip(component_infos)
        .skip(3)
    {
        if let Some(samples) = channel_data.integer_container.as_mut() {
            let addend = component_unsigned_level_shift_i64(component_info);
            for sample in samples {
                *sample += addend;
            }
        } else {
            let addend = component_unsigned_level_shift(component_info);
            apply_float_sign_shift(
                &mut channel_data.container,
                addend,
                component_info.wavelet_transform(),
                round_irreversible_output,
            );
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
pub(super) fn store<'a>(
    tile: &'a Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_info: &ComponentInfo,
    component_idx: usize,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
    parallel: bool,
) -> Result<()> {
    if tile_ctx.channel_data[component_idx]
        .integer_container
        .is_some()
    {
        return store_i64(tile, header, tile_ctx, component_info, component_idx);
    }

    let round_irreversible_output = tile_ctx.round_irreversible_output && !tile.mct;
    let channel_data = &mut tile_ctx.channel_data[component_idx];
    let idwt_output = &mut tile_ctx.idwt_output;

    let component_tile = ComponentTile::new(tile, component_info);
    let resolution_tile = ResolutionTile::new(
        component_tile,
        component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
    );

    let sign_shift = if tile.mct {
        0.0
    } else {
        component_unsigned_level_shift(component_info)
    };

    let (scale_x, scale_y) = (
        component_info.size_info.horizontal_resolution,
        component_info.size_info.vertical_resolution,
    );

    let (image_x_offset, image_y_offset) = output_grid_offset(&header.size_data);

    if let Some(output_region) = tile_ctx.output_region {
        store_region(
            tile,
            header,
            tile_ctx,
            component_info,
            component_idx,
            output_region,
            backend,
            sign_shift,
            round_irreversible_output,
        )?;
        return Ok(());
    }

    if scale_x == 1 && scale_y == 1 {
        let source_x = image_x_offset.saturating_sub(idwt_output.rect.x0);
        let source_y = image_y_offset.saturating_sub(idwt_output.rect.y0);
        let copy_width = resolution_tile
            .rect
            .width()
            .min(idwt_output.rect.width().saturating_sub(source_x));
        let copy_height = resolution_tile
            .rect
            .height()
            .min(idwt_output.rect.height().saturating_sub(source_y));
        let output_x = resolution_tile.rect.x0.saturating_sub(image_x_offset);
        let output_y = resolution_tile.rect.y0.saturating_sub(image_y_offset);
        // A tile narrower or shorter than the reduction factor can have no
        // samples at a reduced resolution.
        if copy_width == 0 || copy_height == 0 {
            return Ok(());
        }

        let handled = if !round_irreversible_output
            || component_info.wavelet_transform() != WaveletTransform::Irreversible97
        {
            if let Some(backend) = backend.as_deref_mut() {
                backend.decode_store_component(J2kStoreComponentJob {
                    input: &idwt_output.coefficients,
                    input_width: idwt_output.rect.width(),
                    source_x,
                    source_y,
                    copy_width,
                    copy_height,
                    output: &mut channel_data.container,
                    output_width: header.size_data.image_width(),
                    output_x,
                    output_y,
                    addend: sign_shift,
                })?
            } else {
                false
            }
        } else {
            false
        };

        if handled {
            return Ok(());
        }

        // If no sub-sampling, use a fast path where we copy rows of coefficients
        // at once.

        // The rect of the IDWT output corresponds to the rect of the highest
        // decomposition level of the tile, which is usually not 1:1 aligned
        // with the actual tile rectangle. We also need to account for the
        // offset of the reference grid.

        let skip_x = image_x_offset.saturating_sub(idwt_output.rect.x0) as usize;
        let skip_y = image_y_offset.saturating_sub(idwt_output.rect.y0) as usize;
        let output_x = resolution_tile.rect.x0.saturating_sub(image_x_offset) as usize;
        let output_y = resolution_tile.rect.y0.saturating_sub(image_y_offset) as usize;
        let input_width = idwt_output.rect.width() as usize;
        let input_rows = idwt_output.rect.height() as usize;
        let output_width = header.size_data.image_width() as usize;
        // The level shift is applied while copying: one pass instead of a
        // shift over the whole IDWT output followed by a copy.
        let shift = SampleShift::new(
            sign_shift,
            component_info.wavelet_transform(),
            round_irreversible_output,
        );
        // Each row returns the samples it wrote, so a reused plane's coverage
        // is counted from what was actually stored.
        let copy_row = |input_row: &[f32], output_row: &mut [f32]| {
            let input_row = &input_row[skip_x..];
            let output_row = &mut output_row[output_x..][..input_row.len()];
            shift.copy(input_row, output_row);
            output_row.len()
        };
        // An unshifted whole-plane copy is a buffer swap: the IDWT output
        // becomes the plane, and the plane's allocation the next IDWT output.
        if matches!(shift, SampleShift::Copy)
            && (skip_x, skip_y, output_x, output_y) == (0, 0, 0, 0)
            && input_width == output_width
            && channel_data
                .container
                .swap_samples(&mut idwt_output.coefficients)
        {
            tile_ctx.stored_plane_samples = tile_ctx
                .stored_plane_samples
                .saturating_add(channel_data.container.truncated().len());
            return Ok(());
        }

        #[cfg(feature = "parallel")]
        if parallel && input_width * input_rows >= PARALLEL_STORE_SAMPLES {
            use rayon::prelude::*;
            let stored = idwt_output
                .coefficients
                .par_chunks_exact(input_width)
                .skip(skip_y)
                .take(input_rows)
                .zip(
                    channel_data
                        .container
                        .par_chunks_exact_mut(output_width)
                        .skip(output_y),
                )
                .with_min_len(16)
                .map(|(input_row, output_row)| copy_row(input_row, output_row))
                .sum::<usize>();
            tile_ctx.stored_plane_samples = tile_ctx.stored_plane_samples.saturating_add(stored);
            return Ok(());
        }
        #[cfg(not(feature = "parallel"))]
        let _ = parallel;

        let stored = idwt_output
            .coefficients
            .chunks_exact(input_width)
            .skip(skip_y)
            .take(input_rows)
            .zip(
                channel_data
                    .container
                    .chunks_exact_mut(output_width)
                    .skip(output_y),
            )
            .map(|(input_row, output_row)| copy_row(input_row, output_row))
            .sum::<usize>();
        tile_ctx.stored_plane_samples = tile_ctx.stored_plane_samples.saturating_add(stored);
    } else {
        apply_float_sign_shift(
            &mut idwt_output.coefficients,
            sign_shift,
            component_info.wavelet_transform(),
            round_irreversible_output,
        );
        let image_width = header.size_data.image_width();
        let image_height = header.size_data.image_height();

        let x_shrink_factor = header.size_data.x_shrink_factor;
        let y_shrink_factor = header.size_data.y_shrink_factor;

        let (x_offset, y_offset) = output_grid_offset(&header.size_data);

        // Coefficients use the reconstructed resolution grid, which may be
        // reduced or clipped relative to the full component tile.
        let sample_rect = resolution_tile.rect.intersect(idwt_output.rect);
        if x_shrink_factor == 1 && y_shrink_factor == 1 {
            upsample::place_subsampled(
                &upsample::SubsampledPlacement {
                    input: &idwt_output.coefficients,
                    input_stride: idwt_output.rect.width() as usize,
                    component_origin: (idwt_output.rect.x0, idwt_output.rect.y0),
                    columns: sample_rect.x0..sample_rect.x1,
                    rows: sample_rect.y0..sample_rect.y1,
                    scale: (u32::from(scale_x), u32::from(scale_y)),
                    image_size: (image_width, image_height),
                    image_offset: (x_offset, y_offset),
                },
                &mut channel_data.container,
            );
            return Ok(());
        }

        for y in sample_rect.y0..sample_rect.y1 {
            let relative_y = (y - idwt_output.rect.y0) as usize;
            let reference_grid_y = (u32::from(scale_y) * y) / y_shrink_factor;

            for x in sample_rect.x0..sample_rect.x1 {
                let relative_x = (x - idwt_output.rect.x0) as usize;
                let reference_grid_x = (u32::from(scale_x) * x) / x_shrink_factor;

                let sample = idwt_output.coefficients
                    [relative_y * idwt_output.rect.width() as usize + relative_x];

                for x_position in u32::max(reference_grid_x, x_offset)
                    ..u32::min(
                        reference_grid_x + u32::from(scale_x),
                        image_width + x_offset,
                    )
                {
                    for y_position in u32::max(reference_grid_y, y_offset)
                        ..u32::min(
                            reference_grid_y + u32::from(scale_y),
                            image_height + y_offset,
                        )
                    {
                        let pos = (y_position - y_offset) as usize * image_width as usize
                            + (x_position - x_offset) as usize;

                        channel_data.container[pos] = sample;
                    }
                }
            }
        }
    }

    Ok(())
}

/// The last IDWT level a decode can leave to the RGB8 pack with its deferred
/// MCT, or `None` to store samples. The tile must cover the whole image with
/// three 8-bit components at full resolution that share one transform, each
/// ending in a banded level of at least two whose sub-bands are packed as
/// built, and each coefficient buffer must fit its plane without padding,
/// so that [`store_last_level_bands`] can swap one for the other.
pub(super) fn fused_last_level(
    tile: &Tile<'_>,
    header: &Header<'_>,
    tile_ctx: &TileDecodeContext,
    storage: &DecompositionStorage<'_>,
    hooked: bool,
) -> Option<LastLevel> {
    #[cfg(test)]
    if tile_ctx.staged_rgb8_pack {
        return None;
    }
    if !tile_ctx.defer_mct
        || hooked
        || !tile.mct
        || tile_ctx.output_region.is_some()
        || storage.exact_integer_decode
        || storage.roi_plan.is_some()
        || storage.coefficient_layout != CoefficientLayout::Component
        || header.skipped_resolution_levels != 0
        || header.component_infos.len() != 3
        || tile_ctx.channel_data.len() != 3
        || !storage.coefficients.len().is_multiple_of(SIMD_WIDTH)
    {
        return None;
    }
    let (x0, y0) = output_grid_offset(&header.size_data);
    let (width, height) = (
        header.size_data.image_width(),
        header.size_data.image_height(),
    );
    let image = IntRect::from_xywh(x0, y0, width, height);
    let transform = header.component_infos[0].wavelet_transform();
    let mut level = None;
    let components = header.component_infos.iter().zip(&tile_ctx.channel_data);
    for (idx, (info, channel)) in components.enumerate() {
        let decompositions = storage.tile_decompositions.get(idx)?.decompositions.clone();
        let decompositions = &storage.decompositions[decompositions];
        let last = decompositions
            .last()
            .filter(|_| decompositions.len() >= 2)?;
        let size = &info.size_info;
        let bands = last.sub_bands.map(|band| &storage.sub_bands[band]);
        let plane_len = match tile_ctx.deferred_plane {
            Some(samples) if idx == 2 => samples,
            _ => channel.container.truncated().len(),
        };
        if size.precision != 8
            || (size.horizontal_resolution, size.vertical_resolution) != (1, 1)
            || info.wavelet_transform() != transform
            || tile.component_infos.get(idx)?.wavelet_transform() != transform
            || channel.integer_container.is_some()
            || plane_len != storage.coefficients.len()
            || last.rect != image
            || bands.iter().any(|band| band.storage_rect != band.rect)
        {
            return None;
        }
        let bands = bands.map(|band| (band.rect, band.coefficients.clone()));
        let candidate = LastLevel::new(last.rect, bands, transform)?;
        if level.is_some_and(|level| level != candidate) {
            return None;
        }
        level = Some(candidate);
    }
    level
}

/// Store a component whose last IDWT level [`fused_last_level`] left to the
/// pack. The IDWT stopped at the level's LL band, in the component's
/// `fused_ll_bands` buffer, and the coefficient buffer, which holds the
/// level's other sub-bands, becomes the component's plane. The plane's
/// allocation becomes the next component's coefficient buffer.
pub(super) fn store_last_level_bands(
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    component_idx: usize,
    level: &LastLevel,
) -> Result<()> {
    let ll_rect = level.ll_rect();
    let ll_len = ll_rect.width() as usize * ll_rect.height() as usize;
    if tile_ctx.idwt_output.rect != ll_rect
        || tile_ctx.fused_ll_bands.get(component_idx).map(Vec::len) != Some(ll_len)
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let samples = storage.coefficients.len();
    let plane = &mut tile_ctx.channel_data[component_idx].container;
    if component_idx == 2 && tile_ctx.deferred_plane.take().is_some() {
        // The last plane was left unallocated: it takes the buffer, which
        // no later component needs.
        if !plane.adopt_samples(&mut storage.coefficients) {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
    } else {
        if !plane.swap_samples(&mut storage.coefficients) {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        storage.coefficients.truncate(samples);
        if storage.coefficients.len() != samples {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
    }
    tile_ctx.stored_plane_samples = tile_ctx.stored_plane_samples.saturating_add(samples);
    Ok(())
}

/// Stores with at least this many samples copy rows in parallel.
/// On x86, scheduling the rows of a 1 MiB plane costs more than its copy.
#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
const PARALLEL_STORE_SAMPLES: usize = 1024 * 1024;
#[cfg(all(feature = "parallel", not(target_arch = "x86_64")))]
const PARALLEL_STORE_SAMPLES: usize = 512 * 512;

/// One tile's slices of the plane rows `rect` covers, `rect.height()` per
/// component, component-major, for a tile worker to store into.
#[cfg_attr(
    not(feature = "parallel"),
    expect(
        dead_code,
        reason = "only tile workers split the planes into row slices"
    )
)]
pub(super) struct TileRows<'p> {
    /// Index of the tile in the parsed tile list.
    pub(super) index: usize,
    pub(super) rect: IntRect,
    pub(super) rows: Vec<&'p mut [f32]>,
}

/// The plane rectangle the 1x1 fast path of [`store`] writes for one
/// component of `tile`, or `None` when the decode skips more resolution
/// levels than the component has.
pub(super) fn store_rect(
    tile: &Tile<'_>,
    component_info: &ComponentInfo,
    header: &Header<'_>,
) -> Option<IntRect> {
    let resolution = component_info
        .num_resolution_levels()
        .checked_sub(1)?
        .checked_sub(header.skipped_resolution_levels)?;
    if resolution > component_info.num_decomposition_levels() {
        return None;
    }
    let rect = ResolutionTile::new(ComponentTile::new(tile, component_info), resolution).rect;
    let (x_offset, y_offset) = output_grid_offset(&header.size_data);
    Some(IntRect::from_ltrb(
        rect.x0.saturating_sub(x_offset),
        rect.y0.saturating_sub(y_offset),
        rect.x1.saturating_sub(x_offset),
        rect.y1.saturating_sub(y_offset),
    ))
}

/// Store one component of `tile` into `rows` as the 1x1 fast path of
/// [`store`] would. Fails before writing unless the IDWT output covers
/// exactly the planned rectangle.
pub(super) fn store_rows(
    rows: &mut TileRows<'_>,
    tile: &Tile<'_>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_info: &ComponentInfo,
    component_idx: usize,
    parallel: bool,
) -> Result<()> {
    let TileDecodeContext {
        idwt_output,
        round_irreversible_output,
        stored_plane_samples,
        ..
    } = tile_ctx;
    let (image_x_offset, image_y_offset) = output_grid_offset(&header.size_data);
    let skip_x = image_x_offset.saturating_sub(idwt_output.rect.x0) as usize;
    let skip_y = image_y_offset.saturating_sub(idwt_output.rect.y0) as usize;
    let input_width = idwt_output.rect.width() as usize;
    // The IDWT rows `store` copies: those below `skip_y`, within the rect.
    let input_rows = (idwt_output.coefficients.len().checked_div(input_width))
        .unwrap_or(0)
        .saturating_sub(skip_y)
        .min(idwt_output.rect.height() as usize);
    let (width, height) = (rows.rect.width() as usize, rows.rect.height() as usize);
    let planned = idwt_output.placement == IdwtPlacement::Coefficients
        && store_rect(tile, component_info, header)
            .is_some_and(|rect| (rect.x0, rect.y0) == (rows.rect.x0, rows.rect.y0))
        && input_width.checked_sub(skip_x) == Some(width)
        && input_rows == height;
    let first = component_idx * height;
    let output = rows
        .rows
        .get_mut(first..first + height)
        .filter(|_| planned)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    *stored_plane_samples = stored_plane_samples.saturating_add(width * height);

    let sign_shift = if tile.mct {
        0.0
    } else {
        component_unsigned_level_shift(component_info)
    };
    let shift = SampleShift::new(
        sign_shift,
        component_info.wavelet_transform(),
        *round_irreversible_output && !tile.mct,
    );
    let copy_row = |output_row: &mut [f32], input_row: &[f32]| {
        shift.copy(&input_row[skip_x..], output_row);
    };
    let input = &idwt_output.coefficients;
    #[cfg(feature = "parallel")]
    if parallel && width * height >= PARALLEL_STORE_SAMPLES {
        use rayon::prelude::*;
        output
            .par_iter_mut()
            .zip(input.par_chunks_exact(input_width).skip(skip_y))
            .with_min_len(16)
            .for_each(|(output_row, input_row)| copy_row(output_row, input_row));
        return Ok(());
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;
    output
        .iter_mut()
        .zip(input.chunks_exact(input_width).skip(skip_y))
        .for_each(|(output_row, input_row)| copy_row(output_row, input_row));
    Ok(())
}

/// The per-sample output transform of [`apply_float_sign_shift`].
#[derive(Clone, Copy)]
enum SampleShift {
    Copy,
    Add(f32),
    RoundThenAdd(f32),
}

impl SampleShift {
    fn new(sign_shift: f32, transform: WaveletTransform, round_irreversible_output: bool) -> Self {
        if round_irreversible_output && transform == WaveletTransform::Irreversible97 {
            Self::RoundThenAdd(sign_shift)
        } else if sign_shift != 0.0 {
            Self::Add(sign_shift)
        } else {
            Self::Copy
        }
    }

    /// Transform `samples` in place, as [`Self::copy`] would.
    fn apply(self, samples: &mut [f32]) {
        match self {
            Self::Copy => {}
            Self::Add(addend) => {
                for sample in samples {
                    *sample += addend;
                }
            }
            Self::RoundThenAdd(addend) => {
                for sample in samples {
                    *sample = round_ties_even_then_add(*sample, addend);
                }
            }
        }
    }

    fn copy(self, input: &[f32], output: &mut [f32]) {
        match self {
            Self::Copy => output.copy_from_slice(input),
            Self::Add(addend) => {
                for (output, &input) in output.iter_mut().zip(input) {
                    *output = input + addend;
                }
            }
            Self::RoundThenAdd(addend) => {
                for (output, &input) in output.iter_mut().zip(input) {
                    *output = round_ties_even_then_add(input, addend);
                }
            }
        }
    }
}

fn apply_float_sign_shift(
    samples: &mut [f32],
    sign_shift: f32,
    transform: WaveletTransform,
    round_irreversible_output: bool,
) {
    if round_irreversible_output && transform == WaveletTransform::Irreversible97 {
        for sample in samples {
            *sample = round_ties_even_then_add(*sample, sign_shift);
        }
    } else if sign_shift != 0.0 {
        for sample in samples {
            *sample += sign_shift;
        }
    }
}

fn store_i64<'a>(
    tile: &'a Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_info: &ComponentInfo,
    component_idx: usize,
) -> Result<()> {
    if tile_ctx.output_region.is_some() {
        bail!(DecodingError::UnsupportedFeature(
            "25-38 bit region decode requires exact integer region IDWT support"
        ));
    }

    let channel_data = &mut tile_ctx.channel_data[component_idx];
    let idwt_output = &mut tile_ctx.idwt_output;
    let output = channel_data
        .integer_container
        .as_mut()
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;

    let component_tile = ComponentTile::new(tile, component_info);
    let resolution_tile = ResolutionTile::new(
        component_tile,
        component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
    );

    let sign_shift = if tile.mct {
        0
    } else {
        component_unsigned_level_shift_i64(component_info)
    };

    let (scale_x, scale_y) = (
        component_info.size_info.horizontal_resolution,
        component_info.size_info.vertical_resolution,
    );

    let (image_x_offset, image_y_offset) = output_grid_offset(&header.size_data);

    if scale_x == 1 && scale_y == 1 {
        let source_x = image_x_offset.saturating_sub(idwt_output.rect.x0);
        let source_y = image_y_offset.saturating_sub(idwt_output.rect.y0);
        let copy_width = resolution_tile
            .rect
            .width()
            .min(idwt_output.rect.width().saturating_sub(source_x));
        let copy_height = resolution_tile
            .rect
            .height()
            .min(idwt_output.rect.height().saturating_sub(source_y));
        let output_x = resolution_tile.rect.x0.saturating_sub(image_x_offset);
        let output_y = resolution_tile.rect.y0.saturating_sub(image_y_offset);
        let input_width = idwt_output.rect.width() as usize;
        let image_width = header.size_data.image_width() as usize;
        let copy_width = copy_width as usize;

        for row in 0..copy_height as usize {
            let src_start = (source_y as usize + row)
                .checked_mul(input_width)
                .and_then(|offset| offset.checked_add(source_x as usize))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let dst_start = (output_y as usize + row)
                .checked_mul(image_width)
                .and_then(|offset| offset.checked_add(output_x as usize))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let src = &idwt_output.coefficients_i64[src_start..src_start + copy_width];
            let dst = &mut output[dst_start..dst_start + copy_width];
            if sign_shift == 0 {
                dst.copy_from_slice(src);
            } else {
                for (dst, src) in dst.iter_mut().zip(src.iter().copied()) {
                    *dst = src + sign_shift;
                }
            }
        }
    } else {
        let image_width = header.size_data.image_width();
        let image_height = header.size_data.image_height();

        let x_shrink_factor = header.size_data.x_shrink_factor;
        let y_shrink_factor = header.size_data.y_shrink_factor;

        let (x_offset, y_offset) = output_grid_offset(&header.size_data);

        let sample_rect = resolution_tile.rect.intersect(idwt_output.rect);
        for y in sample_rect.y0..sample_rect.y1 {
            let relative_y = (y - idwt_output.rect.y0) as usize;
            let reference_grid_y = (u32::from(scale_y) * y) / y_shrink_factor;

            for x in sample_rect.x0..sample_rect.x1 {
                let relative_x = (x - idwt_output.rect.x0) as usize;
                let reference_grid_x = (u32::from(scale_x) * x) / x_shrink_factor;

                let sample = idwt_output.coefficients_i64
                    [relative_y * idwt_output.rect.width() as usize + relative_x]
                    + sign_shift;

                for x_position in u32::max(reference_grid_x, x_offset)
                    ..u32::min(
                        reference_grid_x + u32::from(scale_x),
                        image_width + x_offset,
                    )
                {
                    for y_position in u32::max(reference_grid_y, y_offset)
                        ..u32::min(
                            reference_grid_y + u32::from(scale_y),
                            image_height + y_offset,
                        )
                    {
                        let pos = (y_position - y_offset) as usize * image_width as usize
                            + (x_position - x_offset) as usize;

                        output[pos] = sample;
                    }
                }
            }
        }
    }

    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the codec float domain intentionally receives bounded integer samples or metadata at this rounding boundary"
)]
pub(super) fn component_unsigned_level_shift(component_info: &ComponentInfo) -> f32 {
    if component_info.size_info.signed {
        0.0
    } else {
        (1_u64 << (component_info.size_info.precision - 1)) as f32
    }
}

fn component_unsigned_level_shift_i64(component_info: &ComponentInfo) -> i64 {
    if component_info.size_info.signed {
        0
    } else {
        1_i64 << (component_info.size_info.precision - 1)
    }
}

#[expect(
    clippy::similar_names,
    reason = "paired axis, subband, and marker names follow JPEG 2000 specification notation"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
fn store_region<'a>(
    tile: &'a Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_info: &ComponentInfo,
    component_idx: usize,
    output_region: OutputRegion,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
    sign_shift: f32,
    round_irreversible_output: bool,
) -> Result<()> {
    let channel_data = &mut tile_ctx.channel_data[component_idx];
    let idwt_output = &mut tile_ctx.idwt_output;

    let component_tile = ComponentTile::new(tile, component_info);
    let resolution_tile = ResolutionTile::new(
        component_tile,
        component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
    );

    let (scale_x, scale_y) = (
        component_info.size_info.horizontal_resolution,
        component_info.size_info.vertical_resolution,
    );
    let image_width = header.size_data.image_width();
    let image_height = header.size_data.image_height();
    let x_shrink_factor = header.size_data.x_shrink_factor;
    let y_shrink_factor = header.size_data.y_shrink_factor;
    let (x_offset, y_offset) = output_grid_offset(&header.size_data);
    let region_x1 = output_region.x + output_region.width;
    let region_y1 = output_region.y + output_region.height;
    let output_width = output_region.width as usize;

    if idwt_output.placement == IdwtPlacement::ChannelPadding {
        // The IDWT wrote its last level, the window `idwt_output.rect`, into
        // this plane's padding, and the tile covers the whole region.
        let region_rect = output_region_rect(&header.size_data, output_region);
        let offset = |region: u32, window: u32| {
            region
                .checked_sub(window)
                .ok_or(DecodingError::CodeBlockDecodeFailure)
        };
        let shift = SampleShift::new(
            sign_shift,
            component_info.wavelet_transform(),
            round_irreversible_output,
        );
        channel_data
            .container
            .compact_from_padding(
                idwt_output.rect.width() as usize,
                (
                    offset(region_rect.x0, idwt_output.rect.x0)? as usize,
                    offset(region_rect.y0, idwt_output.rect.y0)? as usize,
                ),
                (output_width, output_region.height as usize),
                |row| shift.apply(row),
            )
            .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
        idwt_output.placement = IdwtPlacement::Coefficients;
        return Ok(());
    }

    if scale_x == 1 && scale_y == 1 {
        let region_rect = output_region_rect(&header.size_data, output_region);
        let region_rect_x0 = region_rect.x0;
        let region_rect_y0 = region_rect.y0;
        let region_rect_x1 = region_rect.x1;
        let region_rect_y1 = region_rect.y1;
        let copy_x0 = idwt_output
            .rect
            .x0
            .max(resolution_tile.rect.x0)
            .max(region_rect_x0);
        let copy_y0 = idwt_output
            .rect
            .y0
            .max(resolution_tile.rect.y0)
            .max(region_rect_y0);
        let copy_x1 = idwt_output
            .rect
            .x1
            .min(resolution_tile.rect.x1)
            .min(region_rect_x1);
        let copy_y1 = idwt_output
            .rect
            .y1
            .min(resolution_tile.rect.y1)
            .min(region_rect_y1);

        if idwt_output.placement == IdwtPlacement::ChannelPlane {
            // The row-streamed IDWT already wrote this overlap into the plane.
            let shift = SampleShift::new(
                sign_shift,
                component_info.wavelet_transform(),
                round_irreversible_output,
            );
            if copy_x0 < copy_x1 && copy_y0 < copy_y1 {
                let copy_width = (copy_x1 - copy_x0) as usize;
                for y in copy_y0..copy_y1 {
                    let start = (y - region_rect_y0) as usize * output_width
                        + (copy_x0 - region_rect_x0) as usize;
                    shift.apply(&mut channel_data.container[start..start + copy_width]);
                }
            }
            idwt_output.placement = IdwtPlacement::Coefficients;
            return Ok(());
        }

        let handled = if !round_irreversible_output
            || component_info.wavelet_transform() != WaveletTransform::Irreversible97
        {
            if let Some(backend) = backend.as_deref_mut() {
                copy_x0 < copy_x1
                    && copy_y0 < copy_y1
                    && backend.decode_store_component(J2kStoreComponentJob {
                        input: &idwt_output.coefficients,
                        input_width: idwt_output.rect.width(),
                        source_x: copy_x0 - idwt_output.rect.x0,
                        source_y: copy_y0 - idwt_output.rect.y0,
                        copy_width: copy_x1 - copy_x0,
                        copy_height: copy_y1 - copy_y0,
                        output: &mut channel_data.container,
                        output_width: output_region.width,
                        output_x: copy_x0 - region_rect_x0,
                        output_y: copy_y0 - region_rect_y0,
                        addend: sign_shift,
                    })?
            } else {
                false
            }
        } else {
            false
        };

        if handled {
            return Ok(());
        }

        apply_float_sign_shift(
            &mut idwt_output.coefficients,
            sign_shift,
            component_info.wavelet_transform(),
            round_irreversible_output,
        );

        if copy_x0 < copy_x1 && copy_y0 < copy_y1 {
            let input_width = idwt_output.rect.width() as usize;
            let copy_width = (copy_x1 - copy_x0) as usize;
            for y in copy_y0..copy_y1 {
                let src_start = (y - idwt_output.rect.y0) as usize * input_width
                    + (copy_x0 - idwt_output.rect.x0) as usize;
                let dst_start = (y - region_rect_y0) as usize * output_width
                    + (copy_x0 - region_rect_x0) as usize;
                channel_data.container[dst_start..dst_start + copy_width]
                    .copy_from_slice(&idwt_output.coefficients[src_start..src_start + copy_width]);
            }
        }

        return Ok(());
    }

    apply_float_sign_shift(
        &mut idwt_output.coefficients,
        sign_shift,
        component_info.wavelet_transform(),
        round_irreversible_output,
    );

    let sample_rect = resolution_tile.rect.intersect(idwt_output.rect);
    for y in sample_rect.y0..sample_rect.y1 {
        let relative_y = (y - idwt_output.rect.y0) as usize;
        let reference_grid_y = (u32::from(scale_y) * y) / y_shrink_factor;

        for x in sample_rect.x0..sample_rect.x1 {
            let relative_x = (x - idwt_output.rect.x0) as usize;
            let reference_grid_x = (u32::from(scale_x) * x) / x_shrink_factor;

            let sample = idwt_output.coefficients
                [relative_y * idwt_output.rect.width() as usize + relative_x];

            for x_position in u32::max(reference_grid_x, x_offset)
                ..u32::min(
                    reference_grid_x + u32::from(scale_x),
                    image_width + x_offset,
                )
            {
                let image_x = x_position - x_offset;
                if image_x < output_region.x || image_x >= region_x1 {
                    continue;
                }

                for y_position in u32::max(reference_grid_y, y_offset)
                    ..u32::min(
                        reference_grid_y + u32::from(scale_y),
                        image_height + y_offset,
                    )
                {
                    let image_y = y_position - y_offset;
                    if image_y < output_region.y || image_y >= region_y1 {
                        continue;
                    }

                    let pos = (image_y - output_region.y) as usize * output_width
                        + (image_x - output_region.x) as usize;
                    channel_data.container[pos] = sample;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{apply_float_sign_shift, WaveletTransform};

    #[test]
    fn irreversible_level_shift_rounds_in_the_centered_domain() {
        let mut samples = [f32::from_bits(0xc117_fffc), -9.5, -8.5];
        apply_float_sign_shift(&mut samples, 128.0, WaveletTransform::Irreversible97, true);
        assert_eq!(
            samples.map(f32::to_bits),
            [119.0, 118.0, 120.0].map(f32::to_bits)
        );

        let mut signed_samples = [-9.5, -8.5];
        apply_float_sign_shift(
            &mut signed_samples,
            0.0,
            WaveletTransform::Irreversible97,
            true,
        );
        assert_eq!(
            signed_samples.map(f32::to_bits),
            [-10.0, -8.0].map(f32::to_bits)
        );
    }

    #[test]
    fn component_output_preserves_irreversible_fractional_samples() {
        let mut samples = [f32::from_bits(0xc117_fffc), -9.5, -8.5];
        apply_float_sign_shift(&mut samples, 128.0, WaveletTransform::Irreversible97, false);
        assert_eq!(
            samples.map(f32::to_bits),
            [118.5, 118.5, 119.5].map(f32::to_bits)
        );
    }

    #[test]
    fn reversible_level_shift_preserves_fractional_samples() {
        let mut samples = [-9.5, -8.5];
        apply_float_sign_shift(&mut samples, 128.0, WaveletTransform::Reversible53, true);
        assert_eq!(samples.map(f32::to_bits), [118.5, 119.5].map(f32::to_bits));
    }
}
