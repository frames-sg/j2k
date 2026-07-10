// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    adjust_component_step_sizes_for_guard_delta, adjust_reversible_step_sizes_for_guard_delta,
    codestream_write, component_sampling_for_options, component_step_sizes, encode_impl,
    max_decomposition_levels, max_total_bitplanes_for_components, precinct_exponents_for_options,
    quantize, raw_pixel_bytes_per_sample, reversible_guard_bits_for_marker_limit,
    roi_encode_plans_for_options, split_packetized_tile_into_tile_parts,
    validate_packet_header_marker_payloads, BlockCodingMode, EncodeComponentSampleInfo,
    EncodeOptions, EncodeParams, EncodeRoiRegion, J2kEncodeStageAccelerator, Vec,
    MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};

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
pub(super) fn encode_multitile_impl(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    roi_regions: &[EncodeRoiRegion],
    component_sample_info: &[EncodeComponentSampleInfo],
    accelerator: &mut impl J2kEncodeStageAccelerator,
    tile_width: u32,
    tile_height: u32,
) -> Result<Vec<u8>, &'static str> {
    let num_x_tiles = width.div_ceil(tile_width);
    let num_y_tiles = height.div_ceil(tile_height);
    let num_tiles = num_x_tiles
        .checked_mul(num_y_tiles)
        .ok_or("tile count overflow")?;
    if num_tiles > u32::from(u16::MAX) + 1 {
        return Err("multi-tile encode supports at most 65536 tiles");
    }

    let min_tile_width = if width.is_multiple_of(tile_width) {
        tile_width
    } else {
        width % tile_width
    };
    let min_tile_height = if height.is_multiple_of(tile_height) {
        tile_height
    } else {
        height % tile_height
    };
    let num_levels = options
        .num_decomposition_levels
        .min(max_decomposition_levels(min_tile_width, min_tile_height));
    let use_mct = options.use_mct && matches!(num_components, 3 | 4);
    let requested_guard_bits = if options.reversible {
        if use_mct {
            options.guard_bits.max(2)
        } else {
            options.guard_bits
        }
    } else {
        options.guard_bits.max(2)
    };
    let high_bit_exact = bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH;
    let guard_bits = if high_bit_exact && options.reversible {
        reversible_guard_bits_for_marker_limit(bit_depth, num_levels, requested_guard_bits)?
    } else {
        requested_guard_bits
    };
    let reversible_guard_delta = guard_bits.saturating_sub(requested_guard_bits);
    let mut step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        bit_depth,
        num_levels,
        options.reversible,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    if options.reversible && reversible_guard_delta != 0 {
        adjust_reversible_step_sizes_for_guard_delta(&mut step_sizes, reversible_guard_delta)?;
    }
    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let mut component_step_sizes = component_step_sizes(
        component_sample_info,
        num_levels,
        options.reversible,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    if options.reversible && reversible_guard_delta != 0 {
        adjust_component_step_sizes_for_guard_delta(
            &mut component_step_sizes,
            reversible_guard_delta,
        )?;
    }
    let component_quantization_step_sizes = component_step_sizes
        .iter()
        .map(|steps| {
            steps
                .iter()
                .map(|step| (step.exponent, step.mantissa))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut child_options = options.clone();
    child_options.num_decomposition_levels = num_levels;
    child_options.tile_size = None;
    child_options.write_tlm = false;
    child_options.write_plt = options.write_plt
        || options.write_plm
        || options.write_ppm
        || options.write_ppt
        || options.tile_part_packet_limit.is_some();
    child_options.write_plm = false;
    child_options.write_ppm = options.write_ppm || options.write_ppt;
    child_options.write_ppt = false;

    let mut tile_bodies = Vec::with_capacity(num_tiles as usize);
    for tile_y in 0..num_y_tiles {
        for tile_x in 0..num_x_tiles {
            let tile_index = tile_y
                .checked_mul(num_x_tiles)
                .and_then(|base| base.checked_add(tile_x))
                .ok_or("tile index overflow")?;
            let tile_index = u16::try_from(tile_index).map_err(|_| "tile index exceeds u16")?;
            let x0 = tile_x * tile_width;
            let y0 = tile_y * tile_height;
            let actual_width = (width - x0).min(tile_width);
            let actual_height = (height - y0).min(tile_height);
            let tile_pixels = extract_interleaved_tile(
                pixels,
                width,
                x0,
                y0,
                actual_width,
                actual_height,
                num_components,
                bit_depth,
            )?;
            let tile_roi_regions =
                roi_regions_for_tile(roi_regions, x0, y0, actual_width, actual_height)?;
            let tile_codestream = encode_impl(
                &tile_pixels,
                actual_width,
                actual_height,
                num_components,
                bit_depth,
                signed,
                &child_options,
                block_coding_mode,
                &tile_roi_regions,
                component_sample_info,
                accelerator,
            )?;
            let packet_lengths = if options.write_plt
                || options.write_plm
                || options.write_ppm
                || options.write_ppt
                || options.tile_part_packet_limit.is_some()
            {
                extract_single_tile_plt_packet_lengths(&tile_codestream)?
            } else {
                Vec::new()
            };
            let packet_headers = if options.write_ppm || options.write_ppt {
                extract_single_tile_ppm_packet_headers(&tile_codestream)?
            } else {
                Vec::new()
            };
            tile_bodies.extend(split_packetized_tile_into_tile_parts(
                tile_index,
                extract_single_tile_body(&tile_codestream)?,
                &packet_lengths,
                &packet_headers,
                options.tile_part_packet_limit,
            )?);
        }
    }

    let component_sampling = component_sampling_for_options(options, num_components)?;
    let roi_plans = roi_encode_plans_for_options(
        options,
        roi_regions,
        num_components,
        width,
        height,
        &component_sampling,
        max_total_bitplanes_for_components(&step_sizes, &component_step_sizes, guard_bits)?,
        block_coding_mode,
    )?;
    let precinct_exponents = precinct_exponents_for_options(options, num_levels)?;
    let params = EncodeParams {
        width,
        height,
        tile_width,
        tile_height,
        num_components,
        bit_depth,
        signed,
        component_sample_info: component_sample_info.to_vec(),
        component_quantization_step_sizes,
        num_decomposition_levels: num_levels,
        reversible: options.reversible,
        code_block_width_exp: options.code_block_width_exp,
        code_block_height_exp: options.code_block_height_exp,
        num_layers: options.num_layers,
        use_mct,
        guard_bits,
        block_coding_mode,
        progression_order: options.progression_order,
        write_tlm: options.write_tlm,
        write_plt: options.write_plt,
        write_plm: options.write_plm,
        write_ppm: options.write_ppm,
        write_ppt: options.write_ppt,
        write_sop: options.write_sop,
        write_eph: options.write_eph,
        terminate_coding_passes: block_coding_mode == BlockCodingMode::Classic
            && options.num_layers > 1,
        component_sampling,
        roi_component_shifts: roi_plans.iter().map(|plan| plan.shift).collect(),
        precinct_exponents,
    };
    let tile_packet_headers = tile_bodies
        .iter()
        .map(|tile| tile.packet_headers.as_slice())
        .collect::<Vec<_>>();
    validate_packet_header_marker_payloads(
        params.write_ppm,
        params.write_ppt,
        &tile_packet_headers,
    )?;
    let tile_parts = tile_bodies
        .iter()
        .map(|tile| codestream_write::TilePartData {
            tile_index: tile.tile_index,
            tile_part_index: tile.tile_part_index,
            num_tile_parts: tile.num_tile_parts,
            data: &tile.data,
            packet_lengths: &tile.packet_lengths,
            packet_headers: &tile.packet_headers,
        })
        .collect::<Vec<_>>();

    Ok(codestream_write::write_codestream_tiles(
        &params,
        &tile_parts,
        &quant_params,
    ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
fn extract_interleaved_tile(
    pixels: &[u8],
    image_width: u32,
    x0: u32,
    y0: u32,
    tile_width: u32,
    tile_height: u32,
    num_components: u16,
    bit_depth: u8,
) -> Result<Vec<u8>, &'static str> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)?;
    let bytes_per_pixel = usize::from(num_components)
        .checked_mul(bytes_per_sample)
        .ok_or("pixel stride overflow")?;
    let row_bytes = usize::try_from(tile_width)
        .map_err(|_| "tile width exceeds usize")?
        .checked_mul(bytes_per_pixel)
        .ok_or("tile row byte count overflow")?;
    let out_len = row_bytes
        .checked_mul(usize::try_from(tile_height).map_err(|_| "tile height exceeds usize")?)
        .ok_or("tile byte count overflow")?;
    let mut tile = Vec::with_capacity(out_len);
    let image_row_bytes = usize::try_from(image_width)
        .map_err(|_| "image width exceeds usize")?
        .checked_mul(bytes_per_pixel)
        .ok_or("image row byte count overflow")?;
    let x_byte_offset = usize::try_from(x0)
        .map_err(|_| "tile x offset exceeds usize")?
        .checked_mul(bytes_per_pixel)
        .ok_or("tile x byte offset overflow")?;

    for y in y0..y0 + tile_height {
        let row_start = usize::try_from(y)
            .map_err(|_| "tile y offset exceeds usize")?
            .checked_mul(image_row_bytes)
            .and_then(|offset| offset.checked_add(x_byte_offset))
            .ok_or("tile row offset overflow")?;
        let row_end = row_start
            .checked_add(row_bytes)
            .ok_or("tile row range overflow")?;
        tile.extend_from_slice(
            pixels
                .get(row_start..row_end)
                .ok_or("tile row range outside source pixels")?,
        );
    }

    Ok(tile)
}

pub(super) fn extract_component_plane_tile(
    data: &[u8],
    image_width: u32,
    x0: u32,
    y0: u32,
    tile_width: u32,
    tile_height: u32,
    bit_depth: u8,
) -> Result<Vec<u8>, &'static str> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)?;
    let row_bytes = usize::try_from(tile_width)
        .map_err(|_| "tile width exceeds usize")?
        .checked_mul(bytes_per_sample)
        .ok_or("tile row byte count overflow")?;
    let out_len = row_bytes
        .checked_mul(usize::try_from(tile_height).map_err(|_| "tile height exceeds usize")?)
        .ok_or("tile byte count overflow")?;
    let mut tile = Vec::with_capacity(out_len);
    let image_row_bytes = usize::try_from(image_width)
        .map_err(|_| "image width exceeds usize")?
        .checked_mul(bytes_per_sample)
        .ok_or("image row byte count overflow")?;
    let x_byte_offset = usize::try_from(x0)
        .map_err(|_| "tile x offset exceeds usize")?
        .checked_mul(bytes_per_sample)
        .ok_or("tile x byte offset overflow")?;

    for y in y0..y0 + tile_height {
        let row_start = usize::try_from(y)
            .map_err(|_| "tile y offset exceeds usize")?
            .checked_mul(image_row_bytes)
            .and_then(|offset| offset.checked_add(x_byte_offset))
            .ok_or("tile row offset overflow")?;
        let row_end = row_start
            .checked_add(row_bytes)
            .ok_or("tile row range overflow")?;
        tile.extend_from_slice(
            data.get(row_start..row_end)
                .ok_or("component plane tile row range outside source data")?,
        );
    }

    Ok(tile)
}

#[expect(
    clippy::similar_names,
    reason = "paired axis, subband, and marker names follow JPEG 2000 specification notation"
)]
fn roi_regions_for_tile(
    roi_regions: &[EncodeRoiRegion],
    tile_x: u32,
    tile_y: u32,
    tile_width: u32,
    tile_height: u32,
) -> Result<Vec<EncodeRoiRegion>, &'static str> {
    let tile_x1 = tile_x
        .checked_add(tile_width)
        .ok_or("tile ROI bounds overflow")?;
    let tile_y1 = tile_y
        .checked_add(tile_height)
        .ok_or("tile ROI bounds overflow")?;
    let mut clipped = Vec::new();

    for region in roi_regions {
        let region_x1 = region
            .x
            .checked_add(region.width)
            .ok_or("ROI region bounds overflow")?;
        let region_y1 = region
            .y
            .checked_add(region.height)
            .ok_or("ROI region bounds overflow")?;
        let x0 = region.x.max(tile_x);
        let y0 = region.y.max(tile_y);
        let x1 = region_x1.min(tile_x1);
        let y1 = region_y1.min(tile_y1);
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        clipped.push(EncodeRoiRegion {
            component: region.component,
            x: x0 - tile_x,
            y: y0 - tile_y,
            width: x1 - x0,
            height: y1 - y0,
            shift: region.shift,
        });
    }

    Ok(clipped)
}

fn extract_single_tile_body(codestream: &[u8]) -> Result<&[u8], &'static str> {
    let sod = codestream
        .windows(2)
        .position(|marker| marker == [0xFF, super::super::codestream::markers::SOD])
        .ok_or("encoded tile codestream missing SOD")?;
    let eoc = codestream
        .windows(2)
        .rposition(|marker| marker == [0xFF, super::super::codestream::markers::EOC])
        .ok_or("encoded tile codestream missing EOC")?;
    if eoc < sod + 2 {
        return Err("encoded tile codestream marker order invalid");
    }
    Ok(&codestream[sod + 2..eoc])
}

fn extract_single_tile_plt_packet_lengths(codestream: &[u8]) -> Result<Vec<u32>, &'static str> {
    let sod = codestream
        .windows(2)
        .position(|marker| marker == [0xFF, super::super::codestream::markers::SOD])
        .ok_or("encoded tile codestream missing SOD")?;
    let mut packet_lengths = Vec::new();
    let mut offset = 0usize;

    while offset + 4 <= sod {
        if codestream[offset] == 0xFF
            && codestream[offset + 1] == super::super::codestream::markers::PLT
        {
            let marker_len =
                u16::from_be_bytes([codestream[offset + 2], codestream[offset + 3]]) as usize;
            if marker_len < 3 {
                return Err("encoded tile codestream has invalid PLT length");
            }
            let marker_end = offset
                .checked_add(2)
                .and_then(|value| value.checked_add(marker_len))
                .ok_or("encoded tile codestream PLT length overflow")?;
            if marker_end > sod {
                return Err("encoded tile codestream PLT extends past SOD");
            }
            let length_bytes = codestream
                .get(offset + 5..marker_end)
                .ok_or("encoded tile codestream PLT payload out of range")?;
            packet_lengths.extend(
                super::super::codestream::decode_packet_lengths(length_bytes)
                    .ok_or("encoded tile codestream has invalid PLT packet lengths")?,
            );
            offset = marker_end;
        } else {
            offset += 1;
        }
    }

    if packet_lengths.is_empty() {
        return Err("encoded tile codestream missing PLT packet lengths");
    }

    Ok(packet_lengths)
}

fn extract_single_tile_ppm_packet_headers(codestream: &[u8]) -> Result<Vec<Vec<u8>>, &'static str> {
    let sot = codestream
        .windows(2)
        .position(|marker| marker == [0xFF, super::super::codestream::markers::SOT])
        .ok_or("encoded tile codestream missing SOT")?;
    let mut packet_headers = Vec::new();
    let mut offset = 0usize;

    while offset + 4 <= sot {
        if codestream[offset] == 0xFF
            && codestream[offset + 1] == super::super::codestream::markers::PPM
        {
            let marker_len =
                u16::from_be_bytes([codestream[offset + 2], codestream[offset + 3]]) as usize;
            if marker_len < 3 {
                return Err("encoded tile codestream has invalid PPM length");
            }
            let marker_end = offset
                .checked_add(2)
                .and_then(|value| value.checked_add(marker_len))
                .ok_or("encoded tile codestream PPM length overflow")?;
            if marker_end > sot {
                return Err("encoded tile codestream PPM extends past SOT");
            }
            let mut payload_offset = offset + 5;
            while payload_offset < marker_end {
                let header_len_end = payload_offset
                    .checked_add(2)
                    .ok_or("encoded tile codestream PPM payload overflow")?;
                let len_bytes = codestream
                    .get(payload_offset..header_len_end)
                    .ok_or("encoded tile codestream PPM packet length truncated")?;
                let header_len = u16::from_be_bytes([len_bytes[0], len_bytes[1]]) as usize;
                let header_start = header_len_end;
                let header_end = header_start
                    .checked_add(header_len)
                    .ok_or("encoded tile codestream PPM packet header overflow")?;
                let header = codestream
                    .get(header_start..header_end)
                    .ok_or("encoded tile codestream PPM packet header truncated")?;
                packet_headers.push(header.to_vec());
                payload_offset = header_end;
            }
            offset = marker_end;
        } else {
            offset += 1;
        }
    }

    if packet_headers.is_empty() {
        return Err("encoded tile codestream missing PPM packet headers");
    }

    Ok(packet_headers)
}
