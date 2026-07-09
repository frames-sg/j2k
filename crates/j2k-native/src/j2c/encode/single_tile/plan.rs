// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::*;

pub(super) enum ValidatedEncodeRoute {
    MultiTile { tile_width: u32, tile_height: u32 },
    SingleTile(ValidatedSingleTileInput),
}

pub(super) struct ValidatedSingleTileInput {
    pub(super) num_pixels: usize,
    pub(super) component_sampling: Vec<(u8, u8)>,
    pub(super) high_bit_exact: bool,
}

pub(super) struct SingleTilePlan {
    pub(super) num_pixels: usize,
    pub(super) high_bit_exact: bool,
    pub(super) use_mct: bool,
    pub(super) num_levels: u8,
    pub(super) guard_bits: u8,
    pub(super) step_sizes: Vec<QuantStepSize>,
    pub(super) quant_params: Vec<(u16, u16)>,
    pub(super) component_step_sizes: Vec<Vec<QuantStepSize>>,
    pub(super) roi_plans: Vec<ComponentRoiEncodePlan>,
    pub(super) roi_component_shifts: Vec<u8>,
    pub(super) cb_width: u32,
    pub(super) cb_height: u32,
    pub(super) ht_target_coding_passes: u8,
    pub(super) params: EncodeParams,
}

pub(super) fn validate_encode_request(
    pixels_len: usize,
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    component_sample_info: &[EncodeComponentSampleInfo],
) -> Result<ValidatedEncodeRoute, &'static str> {
    if width == 0 || height == 0 {
        return Err("invalid dimensions");
    }
    if num_components == 0 || num_components > MAX_J2K_SPEC_COMPONENTS {
        return Err("unsupported component count");
    }
    if bit_depth == 0 || bit_depth > MAX_PART1_SAMPLE_BIT_DEPTH {
        return Err("unsupported bit depth");
    }
    if options.num_layers == 0 || options.num_layers > 32 {
        return Err("unsupported quality layer count");
    }
    if options.write_ppm && options.write_ppt {
        return Err("PPM and PPT packet header markers are mutually exclusive");
    }
    if matches!(options.tile_part_packet_limit, Some(0)) {
        return Err("tile-part packet limit must be non-zero");
    }
    if !options.quality_layer_byte_targets.is_empty()
        && options.quality_layer_byte_targets.len() != usize::from(options.num_layers)
    {
        return Err("quality layer byte target count must match quality layer count");
    }
    if !options.reversible {
        validate_irreversible_quantization_profile(options)?;
    }
    validate_component_sample_info(component_sample_info, usize::from(num_components))?;

    let num_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or("image dimensions overflow")?;
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)?;
    let expected_len = num_pixels
        .checked_mul(num_components as usize)
        .and_then(|len| len.checked_mul(bytes_per_sample))
        .ok_or("image dimensions overflow")?;
    if pixels_len < expected_len {
        return Err("pixel data too short");
    }

    let component_sampling = component_sampling_for_options(options, num_components)?;
    let high_bit_exact = bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH;
    if high_bit_exact && options.reversible {
        validate_reversible_i64_encode_options(
            options,
            block_coding_mode,
            component_sample_info,
            &component_sampling,
        )?;
    }

    if let Some((tile_width, tile_height)) = options.tile_size {
        if tile_width == 0 || tile_height == 0 {
            return Err("invalid tile dimensions");
        }
        if component_sampling
            .iter()
            .any(|sampling| *sampling != (1, 1))
        {
            return Err("multi-tile encode with component sampling is not implemented");
        }
        if tile_width < width || tile_height < height {
            return Ok(ValidatedEncodeRoute::MultiTile {
                tile_width,
                tile_height,
            });
        }
    }

    Ok(ValidatedEncodeRoute::SingleTile(ValidatedSingleTileInput {
        num_pixels,
        component_sampling,
        high_bit_exact,
    }))
}

pub(super) fn build_single_tile_plan(
    validated: ValidatedSingleTileInput,
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    roi_regions: &[EncodeRoiRegion],
    component_sample_info: &[EncodeComponentSampleInfo],
) -> Result<SingleTilePlan, &'static str> {
    let ValidatedSingleTileInput {
        num_pixels,
        component_sampling,
        high_bit_exact,
    } = validated;
    let use_mct = options.use_mct && matches!(num_components, 3 | 4);
    let num_levels = options
        .num_decomposition_levels
        .min(max_decomposition_levels(width, height));
    let requested_guard_bits = if options.reversible {
        if use_mct {
            options.guard_bits.max(2)
        } else {
            options.guard_bits
        }
    } else {
        options.guard_bits.max(2)
    };
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
    let quant_params = step_sizes
        .iter()
        .map(|step| (step.exponent, step.mantissa))
        .collect::<Vec<_>>();
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
    let cb_width = 1_u32 << (options.code_block_width_exp + 2);
    let cb_height = 1_u32 << (options.code_block_height_exp + 2);
    let ht_target_coding_passes = ht_target_coding_passes_for_options(options);
    let precinct_exponents = precinct_exponents_for_options(options, num_levels)?;
    let max_base_bitplanes =
        max_total_bitplanes_for_components(&step_sizes, &component_step_sizes, guard_bits)?;
    let roi_plans = roi_encode_plans_for_options(
        options,
        roi_regions,
        num_components,
        width,
        height,
        &component_sampling,
        max_base_bitplanes,
        block_coding_mode,
    )?;
    let roi_component_shifts = roi_plans.iter().map(|plan| plan.shift).collect::<Vec<_>>();
    let params = EncodeParams {
        width,
        height,
        tile_width: options
            .tile_size
            .map_or(width, |(tile_width, _)| tile_width),
        tile_height: options
            .tile_size
            .map_or(height, |(_, tile_height)| tile_height),
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
        roi_component_shifts: roi_component_shifts.clone(),
        precinct_exponents,
    };

    Ok(SingleTilePlan {
        num_pixels,
        high_bit_exact,
        use_mct,
        num_levels,
        guard_bits,
        step_sizes,
        quant_params,
        component_step_sizes,
        roi_plans,
        roi_component_shifts,
        cb_width,
        cb_height,
        ht_target_coding_passes,
        params,
    })
}
