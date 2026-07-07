// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

pub(super) fn encode_impl(
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
) -> Result<Vec<u8>, &'static str> {
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
    if pixels.len() < expected_len {
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
            return encode_multitile_impl(
                pixels,
                width,
                height,
                num_components,
                bit_depth,
                signed,
                options,
                block_coding_mode,
                roi_regions,
                component_sample_info,
                accelerator,
                tile_width,
                tile_height,
            );
        }
    }

    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);

    let use_mct = options.use_mct && matches!(num_components, 3 | 4);
    let num_levels = options.num_decomposition_levels.min(
        // Don't decompose more than the image supports
        max_decomposition_levels(width, height),
    );
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
    let cb_width = 1u32 << (options.code_block_width_exp + 2);
    let cb_height = 1u32 << (options.code_block_height_exp + 2);
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
    let roi_component_shifts: Vec<u8> = roi_plans.iter().map(|plan| plan.shift).collect();
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

    if high_bit_exact && options.reversible {
        return encode_reversible_i64_single_tile_codestream(ReversibleI64SingleTileRequest {
            pixels,
            width,
            height,
            num_pixels,
            num_components,
            bit_depth,
            signed,
            options,
            params: &params,
            quant_params: &quant_params,
            step_sizes: &step_sizes,
            roi_plans: &roi_plans,
            use_mct,
            guard_bits,
            num_levels,
            cb_width,
            cb_height,
            ht_target_coding_passes,
            accelerator,
        });
    }

    let stage_start = profile::profile_now(profile_enabled);
    if block_coding_mode == BlockCodingMode::HighThroughput
        && component_sample_info.is_empty()
        && roi_component_shifts.iter().all(|shift| *shift == 0)
        && roi_regions.is_empty()
        && !(params.write_plt
            || params.write_plm
            || params.write_sop
            || params.write_eph
            || options.tile_part_packet_limit.is_some())
    {
        if let Some(tile_data) = accelerator.encode_htj2k_tile(J2kHtj2kTileEncodeJob {
            pixels,
            width,
            height,
            num_components,
            bit_depth,
            signed,
            num_decomposition_levels: num_levels,
            reversible: options.reversible,
            use_mct,
            guard_bits,
            code_block_width: cb_width,
            code_block_height: cb_height,
            progression_order: public_packetization_progression_order(options.progression_order),
            component_sampling: &params.component_sampling,
            quantization_steps: &quant_params,
        })? {
            let tile_body_us = profile::elapsed_us(stage_start);
            let stage_start = profile::profile_now(profile_enabled);
            let codestream = codestream_write::write_codestream(&params, &tile_data, &quant_params);
            let codestream_us = profile::elapsed_us(stage_start);
            if profile_enabled {
                profile::emit_profile_row(
                    "encode",
                    "accelerated",
                    &[
                        ("tile_body_us", tile_body_us),
                        ("codestream_us", codestream_us),
                        ("total_us", profile::elapsed_us(total_start)),
                    ],
                );
            }
            return Ok(codestream);
        }
    }

    // Step 1: Convert pixel bytes to f32 component arrays
    let stage_start = profile::profile_now(profile_enabled);
    let mut components = match accelerator.encode_deinterleave(J2kDeinterleaveToF32Job {
        pixels,
        num_pixels,
        num_components,
        bit_depth,
        signed,
    })? {
        Some(components) => {
            validate_deinterleaved_components(components, num_components, num_pixels)?
        }
        None => deinterleave_to_f32(pixels, num_pixels, num_components, bit_depth, signed),
    };
    let deinterleave_us = profile::elapsed_us(stage_start);

    // Step 2: Apply forward MCT if RGB with 3+ components
    let stage_start = profile::profile_now(profile_enabled);
    if use_mct {
        if options.reversible {
            if !try_encode_forward_rct(&mut components, accelerator)? {
                forward_mct::forward_rct(&mut components);
            }
        } else if !try_encode_forward_ict(&mut components, accelerator)? {
            forward_mct::forward_ict(&mut components);
        }
    }
    let mct_us = profile::elapsed_us(stage_start);

    // Step 3: Apply forward DWT to each component
    let stage_start = profile::profile_now(profile_enabled);
    let decompositions: Vec<DwtDecomposition> = components
        .iter()
        .map(|comp| {
            encode_forward_dwt(
                comp,
                width,
                height,
                num_levels,
                options.reversible,
                accelerator,
            )
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    validate_component_sampling_dwt_geometry(
        &decompositions,
        width,
        height,
        &params.component_sampling,
    )?;
    let dwt_us = profile::elapsed_us(stage_start);

    // Step 5: Quantize and encode code-blocks for each component
    let mut component_resolution_packets: Vec<Vec<PreparedResolutionPacket>> =
        Vec::with_capacity(num_components as usize);

    let stage_start = profile::profile_now(profile_enabled);
    for (component_idx, decomp) in decompositions
        .iter()
        .take(num_components as usize)
        .enumerate()
    {
        let component = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
        let component_bit_depth = component_sample_info
            .get(component_idx)
            .map_or(bit_depth, |info| info.bit_depth);
        let component_steps = component_step_sizes
            .get(component_idx)
            .map_or(step_sizes.as_slice(), Vec::as_slice);
        let roi_shift = roi_component_shifts
            .get(component_idx)
            .copied()
            .unwrap_or(0);
        let roi_plan = roi_plans
            .get(component_idx)
            .ok_or("ROI plan count does not match component count")?;
        let mut packets = Vec::with_capacity(num_levels as usize + 1);

        // LL subband (resolution 0)
        let ll_roi_scale = roi_subband_scale(num_levels, None)?;
        let ll_subband = prepare_subband(
            &decomp.ll,
            decomp.ll_width,
            decomp.ll_height,
            &component_steps[0],
            component_bit_depth,
            guard_bits,
            options.reversible,
            block_coding_mode,
            cb_width,
            cb_height,
            SubBandType::LowLow,
            roi_shift,
            &roi_plan.regions,
            ll_roi_scale,
            ht_target_coding_passes,
            accelerator,
        )?;
        packets.push(PreparedResolutionPacket {
            component,
            resolution: 0,
            precinct: 0,
            subbands: vec![ll_subband],
        });

        // Higher resolution levels
        for (level_idx, level) in decomp.levels.iter().enumerate() {
            let step_base = 1 + level_idx * 3;
            let level_roi_scale = roi_subband_scale(num_levels, Some(level_idx))?;

            // HL subband
            let hl_subband = prepare_subband(
                &level.hl,
                level.high_width,
                level.low_height,
                &component_steps[step_base],
                component_bit_depth,
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::HighLow,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                ht_target_coding_passes,
                accelerator,
            )?;

            // LH subband
            let lh_subband = prepare_subband(
                &level.lh,
                level.low_width,
                level.high_height,
                &component_steps[step_base + 1],
                component_bit_depth,
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::LowHigh,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                ht_target_coding_passes,
                accelerator,
            )?;

            // HH subband
            let hh_subband = prepare_subband(
                &level.hh,
                level.high_width,
                level.high_height,
                &component_steps[step_base + 2],
                component_bit_depth,
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::HighHigh,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                ht_target_coding_passes,
                accelerator,
            )?;

            packets.push(PreparedResolutionPacket {
                component,
                resolution: u32::try_from(level_idx + 1)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: vec![hl_subband, lh_subband, hh_subband],
            });
        }

        component_resolution_packets.push(packets);
    }
    let subband_prepare_us = profile::elapsed_us(stage_start);

    let component_resolution_packets = split_component_resolution_packets_by_precinct(
        component_resolution_packets,
        width,
        height,
        num_levels,
        &params.precinct_exponents,
    )?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, options)?;
    let stage_start = profile::profile_now(profile_enabled);
    let (resolution_packets, packet_descriptors, allow_packetization_accelerator) =
        if options.num_layers > 1 {
            let (resolution_packets, packet_descriptors) =
                encode_prepared_resolution_packets_layered(
                    prepared_resolution_packets,
                    options.num_layers,
                    options.progression_order,
                    &options.quality_layer_byte_targets,
                    accelerator,
                )?;
            (resolution_packets, packet_descriptors, false)
        } else {
            let packet_descriptors = packet_descriptors_for_order(
                &prepared_resolution_packets,
                1,
                options.progression_order,
            )?;
            let resolution_packets =
                encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
            (resolution_packets, packet_descriptors, true)
        };
    let block_encode_us = profile::elapsed_us(stage_start);

    // Step 6: Form tile bitstream (T2)
    let stage_start = profile::profile_now(profile_enabled);
    let mut resolution_packets = resolution_packets;
    let packetized_tile = packetize_resolution_packets_with_options(
        &mut resolution_packets,
        &packet_descriptors,
        options.num_layers,
        num_components,
        options.progression_order,
        packet_encode::PacketMarkerOptions {
            write_sop: params.write_sop,
            write_eph: params.write_eph,
            separate_packet_headers: params.write_ppm || params.write_ppt,
        },
        allow_packetization_accelerator,
        packetization_requires_scalar(&params, options.tile_part_packet_limit),
        accelerator,
    )?;
    let packetize_us = profile::elapsed_us(stage_start);

    // Step 7: Write codestream
    let stage_start = profile::profile_now(profile_enabled);
    let codestream = write_single_tile_packetized_codestream(
        &params,
        &packetized_tile,
        &quant_params,
        options.tile_part_packet_limit,
    )?;
    let codestream_us = profile::elapsed_us(stage_start);

    if profile_enabled {
        profile::emit_profile_row(
            "encode",
            "cpu",
            &[
                ("deinterleave_us", deinterleave_us),
                ("mct_us", mct_us),
                ("dwt_us", dwt_us),
                ("subband_prepare_us", subband_prepare_us),
                ("block_encode_us", block_encode_us),
                ("packetize_us", packetize_us),
                ("codestream_us", codestream_us),
                ("total_us", profile::elapsed_us(total_start)),
            ],
        );
    }

    Ok(codestream)
}

struct ReversibleI64SingleTileRequest<'a, A: J2kEncodeStageAccelerator> {
    pixels: &'a [u8],
    width: u32,
    height: u32,
    num_pixels: usize,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &'a EncodeOptions,
    params: &'a EncodeParams,
    quant_params: &'a [(u16, u16)],
    step_sizes: &'a [QuantStepSize],
    roi_plans: &'a [ComponentRoiEncodePlan],
    use_mct: bool,
    guard_bits: u8,
    num_levels: u8,
    cb_width: u32,
    cb_height: u32,
    ht_target_coding_passes: u8,
    accelerator: &'a mut A,
}

fn encode_reversible_i64_single_tile_codestream<A: J2kEncodeStageAccelerator>(
    request: ReversibleI64SingleTileRequest<'_, A>,
) -> Result<Vec<u8>, &'static str> {
    let ReversibleI64SingleTileRequest {
        pixels,
        width,
        height,
        num_pixels,
        num_components,
        bit_depth,
        signed,
        options,
        params,
        quant_params,
        step_sizes,
        roi_plans,
        use_mct,
        guard_bits,
        num_levels,
        cb_width,
        cb_height,
        ht_target_coding_passes,
        accelerator,
    } = request;
    let max_reversible_gain = if num_levels == 0 { 0 } else { 2 };
    if u16::from(bit_depth) + max_reversible_gain > MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES {
        return Err("25-38 bit reversible encode exceeds the current no-quantization guard/exponent signaling limit");
    }

    let mut components = deinterleave_to_i64(pixels, num_pixels, num_components, bit_depth, signed);
    if use_mct {
        forward_rct_i64(&mut components);
    }

    let decompositions = components
        .iter()
        .map(|component| fdwt::forward_dwt_i64(component, width, height, num_levels))
        .collect::<Vec<_>>();

    let mut component_resolution_packets = Vec::with_capacity(num_components as usize);
    for (component_idx, decomp) in decompositions
        .iter()
        .take(num_components as usize)
        .enumerate()
    {
        let component = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
        let roi_shift = params
            .roi_component_shifts
            .get(component_idx)
            .copied()
            .unwrap_or(0);
        let roi_plan = roi_plans
            .get(component_idx)
            .ok_or("ROI plan count does not match component count")?;
        let mut packets = Vec::with_capacity(num_levels as usize + 1);

        let ll_roi_scale = roi_subband_scale(num_levels, None)?;
        let base_subband_settings = I64SubbandEncodeSettings {
            guard_bits,
            cb_width,
            cb_height,
            roi_shift,
            roi_regions: &roi_plan.regions,
            roi_scale: ll_roi_scale,
            block_coding_mode: params.block_coding_mode,
            ht_target_coding_passes,
        };
        let ll_subband = prepare_subband_i64(
            &decomp.ll,
            decomp.ll_width,
            decomp.ll_height,
            step_sizes
                .first()
                .ok_or("reversible quantization step missing")?,
            SubBandType::LowLow,
            base_subband_settings,
        )?;
        packets.push(PreparedResolutionPacket {
            component,
            resolution: 0,
            precinct: 0,
            subbands: vec![ll_subband],
        });

        for (level_idx, level) in decomp.levels.iter().enumerate() {
            let step_base = 1 + level_idx * 3;
            let level_roi_scale = roi_subband_scale(num_levels, Some(level_idx))?;
            let hl_subband = prepare_subband_i64(
                &level.hl,
                level.high_width,
                level.low_height,
                step_sizes
                    .get(step_base)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::HighLow,
                I64SubbandEncodeSettings {
                    roi_scale: level_roi_scale,
                    ..base_subband_settings
                },
            )?;
            let lh_subband = prepare_subband_i64(
                &level.lh,
                level.low_width,
                level.high_height,
                step_sizes
                    .get(step_base + 1)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::LowHigh,
                I64SubbandEncodeSettings {
                    roi_scale: level_roi_scale,
                    ..base_subband_settings
                },
            )?;
            let hh_subband = prepare_subband_i64(
                &level.hh,
                level.high_width,
                level.high_height,
                step_sizes
                    .get(step_base + 2)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::HighHigh,
                I64SubbandEncodeSettings {
                    roi_scale: level_roi_scale,
                    ..base_subband_settings
                },
            )?;

            packets.push(PreparedResolutionPacket {
                component,
                resolution: u32::try_from(level_idx + 1)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: vec![hl_subband, lh_subband, hh_subband],
            });
        }
        component_resolution_packets.push(packets);
    }

    encode_i64_component_resolution_packets(
        component_resolution_packets,
        I64CodestreamPacketRequest {
            packetize: I64PacketizeRequest {
                width,
                height,
                num_components,
                num_levels,
                params,
                options,
                accelerator,
            },
            quant_params,
        },
    )
}
