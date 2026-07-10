// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    bail, ComponentInfo, ComponentTile, DecodingError, DerefMut, Header, HtCodeBlockDecoder,
    J2kStoreComponentJob, OutputRegion, ResolutionTile, Result, Tile, TileDecodeContext,
};

pub(super) fn apply_sign_shift(
    tile_ctx: &mut TileDecodeContext,
    component_infos: &[ComponentInfo],
) {
    for (channel_data, component_info) in
        tile_ctx.channel_data.iter_mut().zip(component_infos.iter())
    {
        if let Some(samples) = channel_data.integer_container.as_mut() {
            let addend = component_unsigned_level_shift_i64(component_info);
            for sample in samples {
                *sample += addend;
            }
        } else {
            let addend = component_unsigned_level_shift(component_info);
            for sample in channel_data.container.deref_mut() {
                *sample += addend;
            }
        }
    }
}

pub(super) fn store<'a>(
    tile: &'a Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_info: &ComponentInfo,
    component_idx: usize,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
) -> Result<()> {
    if tile_ctx.channel_data[component_idx]
        .integer_container
        .is_some()
    {
        return store_i64(tile, header, tile_ctx, component_info, component_idx);
    }

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

    let (image_x_offset, image_y_offset) = (
        header.size_data.image_area_x_offset,
        header.size_data.image_area_y_offset,
    );

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

        let handled = if let Some(backend) = backend.as_deref_mut() {
            copy_width > 0
                && copy_height > 0
                && backend.decode_store_component(J2kStoreComponentJob {
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

        let skip_x = image_x_offset.saturating_sub(idwt_output.rect.x0);
        let skip_y = image_y_offset.saturating_sub(idwt_output.rect.y0);

        if sign_shift != 0.0 {
            for sample in idwt_output.coefficients.iter_mut() {
                *sample += sign_shift;
            }
        }

        let input_row_iter = idwt_output
            .coefficients
            .chunks_exact(idwt_output.rect.width() as usize)
            .skip(skip_y as usize)
            .take(idwt_output.rect.height() as usize);

        let output_row_iter = channel_data
            .container
            .chunks_exact_mut(header.size_data.image_width() as usize)
            .skip(resolution_tile.rect.y0.saturating_sub(image_y_offset) as usize);

        for (input_row, output_row) in input_row_iter.zip(output_row_iter) {
            let input_row = &input_row[skip_x as usize..];
            let output_row = &mut output_row
                [resolution_tile.rect.x0.saturating_sub(image_x_offset) as usize..]
                [..input_row.len()];

            output_row.copy_from_slice(input_row);
        }
    } else {
        if sign_shift != 0.0 {
            for sample in idwt_output.coefficients.iter_mut() {
                *sample += sign_shift;
            }
        }
        let image_width = header.size_data.image_width();
        let image_height = header.size_data.image_height();

        let x_shrink_factor = header.size_data.x_shrink_factor;
        let y_shrink_factor = header.size_data.y_shrink_factor;

        let x_offset = header
            .size_data
            .image_area_x_offset
            .div_ceil(x_shrink_factor);
        let y_offset = header
            .size_data
            .image_area_y_offset
            .div_ceil(y_shrink_factor);

        // Otherwise, copy sample by sample.
        for y in resolution_tile.rect.y0..resolution_tile.rect.y1 {
            let relative_y = (y - component_tile.rect.y0) as usize;
            let reference_grid_y = (scale_y as u32 * y) / y_shrink_factor;

            for x in resolution_tile.rect.x0..resolution_tile.rect.x1 {
                let relative_x = (x - component_tile.rect.x0) as usize;
                let reference_grid_x = (scale_x as u32 * x) / x_shrink_factor;

                let sample = idwt_output.coefficients
                    [relative_y * idwt_output.rect.width() as usize + relative_x];

                for x_position in u32::max(reference_grid_x, x_offset)
                    ..u32::min(reference_grid_x + scale_x as u32, image_width + x_offset)
                {
                    for y_position in u32::max(reference_grid_y, y_offset)
                        ..u32::min(reference_grid_y + scale_y as u32, image_height + y_offset)
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

    let (image_x_offset, image_y_offset) = (
        header.size_data.image_area_x_offset,
        header.size_data.image_area_y_offset,
    );

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

        let x_offset = header
            .size_data
            .image_area_x_offset
            .div_ceil(x_shrink_factor);
        let y_offset = header
            .size_data
            .image_area_y_offset
            .div_ceil(y_shrink_factor);

        for y in resolution_tile.rect.y0..resolution_tile.rect.y1 {
            let relative_y = (y - component_tile.rect.y0) as usize;
            let reference_grid_y = (scale_y as u32 * y) / y_shrink_factor;

            for x in resolution_tile.rect.x0..resolution_tile.rect.x1 {
                let relative_x = (x - component_tile.rect.x0) as usize;
                let reference_grid_x = (scale_x as u32 * x) / x_shrink_factor;

                let sample = idwt_output.coefficients_i64
                    [relative_y * idwt_output.rect.width() as usize + relative_x]
                    + sign_shift;

                for x_position in u32::max(reference_grid_x, x_offset)
                    ..u32::min(reference_grid_x + scale_x as u32, image_width + x_offset)
                {
                    for y_position in u32::max(reference_grid_y, y_offset)
                        ..u32::min(reference_grid_y + scale_y as u32, image_height + y_offset)
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

fn store_region<'a>(
    tile: &'a Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_info: &ComponentInfo,
    component_idx: usize,
    output_region: OutputRegion,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
    sign_shift: f32,
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
    let x_offset = header
        .size_data
        .image_area_x_offset
        .div_ceil(x_shrink_factor);
    let y_offset = header
        .size_data
        .image_area_y_offset
        .div_ceil(y_shrink_factor);
    let region_x1 = output_region.x + output_region.width;
    let region_y1 = output_region.y + output_region.height;
    let output_width = output_region.width as usize;

    if scale_x == 1 && scale_y == 1 {
        let region_rect_x0 = output_region.x + x_offset;
        let region_rect_y0 = output_region.y + y_offset;
        let region_rect_x1 = region_x1 + x_offset;
        let region_rect_y1 = region_y1 + y_offset;
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

        let handled = if let Some(backend) = backend.as_deref_mut() {
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
        };

        if handled {
            return Ok(());
        }

        if sign_shift != 0.0 {
            for sample in idwt_output.coefficients.iter_mut() {
                *sample += sign_shift;
            }
        }

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

    if sign_shift != 0.0 {
        for sample in idwt_output.coefficients.iter_mut() {
            *sample += sign_shift;
        }
    }

    for y in resolution_tile.rect.y0..resolution_tile.rect.y1 {
        let relative_y = (y - component_tile.rect.y0) as usize;
        let reference_grid_y = (scale_y as u32 * y) / y_shrink_factor;

        for x in resolution_tile.rect.x0..resolution_tile.rect.x1 {
            let relative_x = (x - component_tile.rect.x0) as usize;
            let reference_grid_x = (scale_x as u32 * x) / x_shrink_factor;

            let sample = idwt_output.coefficients
                [relative_y * idwt_output.rect.width() as usize + relative_x];

            for x_position in u32::max(reference_grid_x, x_offset)
                ..u32::min(reference_grid_x + scale_x as u32, image_width + x_offset)
            {
                let image_x = x_position - x_offset;
                if image_x < output_region.x || image_x >= region_x1 {
                    continue;
                }

                for y_position in u32::max(reference_grid_y, y_offset)
                    ..u32::min(reference_grid_y + scale_y as u32, image_height + y_offset)
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
