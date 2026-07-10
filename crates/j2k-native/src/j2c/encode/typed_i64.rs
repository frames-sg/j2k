// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    adjust_component_step_sizes_for_guard_delta, adjust_reversible_step_sizes_for_guard_delta,
    block_coding_mode, codestream_write, component_step_sizes,
    encode_i64_component_resolution_packets, extract_component_plane_tile, fdwt,
    ht_target_coding_passes_for_options, max_decomposition_levels,
    packetize_i64_component_resolution_packets, precinct_exponents_for_options,
    prepare_subband_i64, quantize, raw_pixel_bytes_per_sample, read_le_sample_value,
    reversible_guard_bits_for_marker_limit, sign_extend_sample,
    split_packetized_tile_into_tile_parts, validate_packet_header_marker_payloads, vec,
    BlockCodingMode, CpuOnlyJ2kEncodeStageAccelerator, EncodeComponentSampleInfo, EncodeOptions,
    EncodeParams, EncodeTypedComponentPlane, I64CodestreamPacketRequest, I64PacketizeRequest,
    I64SubbandEncodeSettings, PreparedResolutionPacket, QuantStepSize, SubBandType, Vec,
};

struct TypedI64HighBitPlan {
    num_levels: u8,
    max_bit_depth: u8,
    guard_bits: u8,
    quant_params: Vec<(u16, u16)>,
    component_step_sizes: Vec<Vec<QuantStepSize>>,
    component_sample_info: Vec<EncodeComponentSampleInfo>,
    component_quantization_step_sizes: Vec<Vec<(u16, u16)>>,
    component_sampling: Vec<(u8, u8)>,
    block_coding_mode: BlockCodingMode,
}

impl TypedI64HighBitPlan {
    fn new(
        planes: &[EncodeTypedComponentPlane<'_>],
        options: &EncodeOptions,
        num_levels: u8,
    ) -> Result<Self, &'static str> {
        let max_bit_depth = planes
            .iter()
            .map(|plane| plane.bit_depth)
            .max()
            .ok_or("unsupported component count")?;
        let requested_guard_bits = options.guard_bits;
        let guard_bits = reversible_guard_bits_for_marker_limit(
            max_bit_depth,
            num_levels,
            requested_guard_bits,
        )?;
        let reversible_guard_delta = guard_bits.saturating_sub(requested_guard_bits);
        let mut step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
            max_bit_depth,
            num_levels,
            true,
            guard_bits,
            options.irreversible_quantization_scale,
            options.irreversible_quantization_subband_scales,
        );
        if reversible_guard_delta != 0 {
            adjust_reversible_step_sizes_for_guard_delta(&mut step_sizes, reversible_guard_delta)?;
        }
        let component_sample_info = planes
            .iter()
            .map(|plane| EncodeComponentSampleInfo {
                bit_depth: plane.bit_depth,
                signed: plane.signed,
            })
            .collect::<Vec<_>>();
        let mut component_step_sizes = component_step_sizes(
            &component_sample_info,
            num_levels,
            true,
            guard_bits,
            options.irreversible_quantization_scale,
            options.irreversible_quantization_subband_scales,
        );
        if reversible_guard_delta != 0 {
            adjust_component_step_sizes_for_guard_delta(
                &mut component_step_sizes,
                reversible_guard_delta,
            )?;
        }
        if step_sizes.iter().any(|step| step.exponent > 31)
            || component_step_sizes
                .iter()
                .flatten()
                .any(|step| step.exponent > 31)
        {
            return Err("25-38 bit typed component-plane encode exceeds the current no-quantization guard/exponent signaling limit");
        }

        let quant_params = step_sizes
            .iter()
            .map(|step| (step.exponent, step.mantissa))
            .collect();
        let component_quantization_step_sizes = component_step_sizes
            .iter()
            .map(|steps| {
                steps
                    .iter()
                    .map(|step| (step.exponent, step.mantissa))
                    .collect()
            })
            .collect();
        let component_sampling = planes
            .iter()
            .map(|plane| (plane.x_rsiz, plane.y_rsiz))
            .collect();

        Ok(Self {
            num_levels,
            max_bit_depth,
            guard_bits,
            quant_params,
            component_step_sizes,
            component_sample_info,
            component_quantization_step_sizes,
            component_sampling,
            block_coding_mode: block_coding_mode(options),
        })
    }

    fn encode_params(
        &self,
        dimensions: (u32, u32),
        tile_dimensions: (u32, u32),
        num_components: u16,
        options: &EncodeOptions,
        precinct_exponents: Vec<(u8, u8)>,
    ) -> EncodeParams {
        EncodeParams {
            width: dimensions.0,
            height: dimensions.1,
            tile_width: tile_dimensions.0,
            tile_height: tile_dimensions.1,
            num_components,
            bit_depth: self.max_bit_depth,
            signed: self.component_sample_info.iter().all(|info| info.signed),
            component_sample_info: self.component_sample_info.clone(),
            component_quantization_step_sizes: self.component_quantization_step_sizes.clone(),
            num_decomposition_levels: self.num_levels,
            reversible: true,
            code_block_width_exp: options.code_block_width_exp,
            code_block_height_exp: options.code_block_height_exp,
            num_layers: options.num_layers,
            use_mct: false,
            guard_bits: self.guard_bits,
            block_coding_mode: self.block_coding_mode,
            progression_order: options.progression_order,
            write_tlm: options.write_tlm,
            write_plt: options.write_plt,
            write_plm: options.write_plm,
            write_ppm: options.write_ppm,
            write_ppt: options.write_ppt,
            write_sop: options.write_sop,
            write_eph: options.write_eph,
            terminate_coding_passes: self.block_coding_mode == BlockCodingMode::Classic
                && options.num_layers > 1,
            component_sampling: self.component_sampling.clone(),
            roi_component_shifts: vec![0; usize::from(num_components)],
            precinct_exponents,
        }
    }

    fn subband_settings(&self, options: &EncodeOptions) -> I64SubbandEncodeSettings<'static> {
        I64SubbandEncodeSettings {
            guard_bits: self.guard_bits,
            cb_width: 1u32 << (options.code_block_width_exp + 2),
            cb_height: 1u32 << (options.code_block_height_exp + 2),
            roi_shift: 0,
            roi_regions: &[],
            roi_scale: 1,
            block_coding_mode: self.block_coding_mode,
            ht_target_coding_passes: ht_target_coding_passes_for_options(options),
        }
    }
}

pub(super) fn encode_typed_component_planes_53_i64(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
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
    if let Some((tile_width, tile_height)) = options.tile_size {
        if tile_width == 0 || tile_height == 0 {
            return Err("invalid tile dimensions");
        }
    }

    let num_components = u16::try_from(planes.len()).map_err(|_| "unsupported component count")?;
    if let Some((tile_width, tile_height)) = options.tile_size {
        if tile_width < width || tile_height < height {
            return encode_typed_component_planes_53_i64_multitile(
                planes,
                width,
                height,
                options,
                tile_width,
                tile_height,
                num_components,
            );
        }
    }
    let num_levels = planes
        .iter()
        .map(|plane| {
            let component_width = width.div_ceil(u32::from(plane.x_rsiz));
            let component_height = height.div_ceil(u32::from(plane.y_rsiz));
            max_decomposition_levels(component_width, component_height)
        })
        .min()
        .unwrap_or(0)
        .min(options.num_decomposition_levels);
    let plan = TypedI64HighBitPlan::new(planes, options, num_levels)?;
    let mut high_bit_options = options.clone();
    high_bit_options.reversible = true;
    high_bit_options.use_mct = false;
    high_bit_options.component_sampling = Some(plan.component_sampling.clone());
    let precinct_exponents = precinct_exponents_for_options(&high_bit_options, num_levels)?;
    let tile_dimensions = options.tile_size.unwrap_or((width, height));
    let params = plan.encode_params(
        (width, height),
        tile_dimensions,
        num_components,
        options,
        precinct_exponents,
    );
    let component_dimensions = planes
        .iter()
        .map(|plane| {
            (
                width.div_ceil(u32::from(plane.x_rsiz)),
                height.div_ceil(u32::from(plane.y_rsiz)),
            )
        })
        .collect::<Vec<_>>();
    let component_resolution_packets = prepare_typed_component_planes_i64_packets(
        planes,
        I64ComponentPlanePacketRequest {
            component_dimensions: &component_dimensions,
            component_step_sizes: &plan.component_step_sizes,
            num_levels,
            subband_settings: plan.subband_settings(options),
        },
    )?;

    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_i64_component_resolution_packets(
        component_resolution_packets,
        I64CodestreamPacketRequest {
            packetize: I64PacketizeRequest {
                width,
                height,
                num_components,
                num_levels,
                params: &params,
                options: &high_bit_options,
                accelerator: &mut accelerator,
            },
            quant_params: &plan.quant_params,
        },
    )
}

fn encode_typed_component_planes_53_i64_multitile(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    options: &EncodeOptions,
    tile_width: u32,
    tile_height: u32,
    num_components: u16,
) -> Result<Vec<u8>, &'static str> {
    let num_x_tiles = width.div_ceil(tile_width);
    let num_y_tiles = height.div_ceil(tile_height);
    let num_tiles = num_x_tiles
        .checked_mul(num_y_tiles)
        .ok_or("tile count overflow")?;
    if num_tiles > u32::from(u16::MAX) + 1 {
        return Err("multi-tile encode supports at most 65536 tiles");
    }

    let num_levels = min_sampled_tile_component_decomposition_levels(
        planes,
        width,
        height,
        tile_width,
        tile_height,
    )?
    .min(options.num_decomposition_levels);
    let plan = TypedI64HighBitPlan::new(planes, options, num_levels)?;
    let mut high_bit_options = options.clone();
    high_bit_options.num_decomposition_levels = num_levels;
    high_bit_options.reversible = true;
    high_bit_options.use_mct = false;
    high_bit_options.component_sampling = Some(plan.component_sampling.clone());
    let precinct_exponents = precinct_exponents_for_options(&high_bit_options, num_levels)?;

    let mut child_options = high_bit_options.clone();
    child_options.tile_size = None;
    child_options.write_tlm = false;
    child_options.write_plt = false;
    child_options.write_plm = false;
    child_options.write_ppm = false;
    child_options.write_ppt = false;

    let params = plan.encode_params(
        (width, height),
        (tile_width, tile_height),
        num_components,
        options,
        precinct_exponents,
    );

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
            let tile_plane_data = planes
                .iter()
                .map(|plane| {
                    let x_rsiz = u32::from(plane.x_rsiz);
                    let y_rsiz = u32::from(plane.y_rsiz);
                    let component_image_width = width.div_ceil(x_rsiz);
                    let component_image_height = height.div_ceil(y_rsiz);
                    let (component_x0, component_tile_width) = sampled_tile_component_axis(
                        x0,
                        actual_width,
                        x_rsiz,
                        component_image_width,
                    )?;
                    let (component_y0, component_tile_height) = sampled_tile_component_axis(
                        y0,
                        actual_height,
                        y_rsiz,
                        component_image_height,
                    )?;
                    let data = extract_component_plane_tile(
                        plane.data,
                        component_image_width,
                        component_x0,
                        component_y0,
                        component_tile_width,
                        component_tile_height,
                        plane.bit_depth,
                    )?;
                    Ok((data, component_tile_width, component_tile_height))
                })
                .collect::<Result<Vec<_>, &'static str>>()?;
            let tile_planes = planes
                .iter()
                .zip(tile_plane_data.iter())
                .map(|(plane, (data, _, _))| EncodeTypedComponentPlane {
                    data,
                    x_rsiz: plane.x_rsiz,
                    y_rsiz: plane.y_rsiz,
                    bit_depth: plane.bit_depth,
                    signed: plane.signed,
                })
                .collect::<Vec<_>>();
            let component_dimensions = tile_planes
                .iter()
                .zip(tile_plane_data.iter())
                .map(|(_, (_, component_width, component_height))| {
                    (*component_width, *component_height)
                })
                .collect::<Vec<_>>();
            let component_resolution_packets = prepare_typed_component_planes_i64_packets(
                &tile_planes,
                I64ComponentPlanePacketRequest {
                    component_dimensions: &component_dimensions,
                    component_step_sizes: &plan.component_step_sizes,
                    num_levels,
                    subband_settings: plan.subband_settings(options),
                },
            )?;
            let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
            let packetized_tile = packetize_i64_component_resolution_packets(
                component_resolution_packets,
                I64PacketizeRequest {
                    width: actual_width,
                    height: actual_height,
                    num_components,
                    num_levels,
                    params: &params,
                    options: &child_options,
                    accelerator: &mut accelerator,
                },
            )?;
            tile_bodies.extend(split_packetized_tile_into_tile_parts(
                tile_index,
                &packetized_tile.data,
                &packetized_tile.packet_lengths,
                &packetized_tile.packet_headers,
                options.tile_part_packet_limit,
            )?);
        }
    }

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
        &plan.quant_params,
    ))
}

fn min_sampled_tile_component_decomposition_levels(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
) -> Result<u8, &'static str> {
    let num_x_tiles = width.div_ceil(tile_width);
    let num_y_tiles = height.div_ceil(tile_height);
    let mut levels: Option<u8> = None;
    for tile_y in 0..num_y_tiles {
        for tile_x in 0..num_x_tiles {
            let x0 = tile_x * tile_width;
            let y0 = tile_y * tile_height;
            let actual_width = (width - x0).min(tile_width);
            let actual_height = (height - y0).min(tile_height);
            for plane in planes {
                let x_rsiz = u32::from(plane.x_rsiz);
                let y_rsiz = u32::from(plane.y_rsiz);
                let component_image_width = width.div_ceil(x_rsiz);
                let component_image_height = height.div_ceil(y_rsiz);
                let (_, component_tile_width) =
                    sampled_tile_component_axis(x0, actual_width, x_rsiz, component_image_width)?;
                let (_, component_tile_height) =
                    sampled_tile_component_axis(y0, actual_height, y_rsiz, component_image_height)?;
                let component_levels =
                    max_decomposition_levels(component_tile_width, component_tile_height);
                levels = Some(levels.map_or(component_levels, |min| min.min(component_levels)));
            }
        }
    }
    Ok(levels.unwrap_or(0))
}

fn sampled_tile_component_axis(
    tile_origin: u32,
    tile_extent: u32,
    sampling: u32,
    component_extent: u32,
) -> Result<(u32, u32), &'static str> {
    let tile_end = tile_origin
        .checked_add(tile_extent)
        .ok_or("tile component bounds overflow")?;
    let start = tile_origin.div_ceil(sampling).min(component_extent);
    let end = tile_end.div_ceil(sampling).min(component_extent);
    Ok((start, end.saturating_sub(start)))
}

struct I64ComponentPlanePacketRequest<'a> {
    component_dimensions: &'a [(u32, u32)],
    component_step_sizes: &'a [Vec<QuantStepSize>],
    num_levels: u8,
    subband_settings: I64SubbandEncodeSettings<'a>,
}

fn prepare_typed_component_planes_i64_packets(
    planes: &[EncodeTypedComponentPlane<'_>],
    request: I64ComponentPlanePacketRequest<'_>,
) -> Result<Vec<Vec<PreparedResolutionPacket>>, &'static str> {
    let I64ComponentPlanePacketRequest {
        component_dimensions,
        component_step_sizes,
        num_levels,
        subband_settings,
    } = request;
    if component_dimensions.len() != planes.len() {
        return Err("component dimensions count does not match component count");
    }
    let mut component_resolution_packets = Vec::with_capacity(planes.len());
    for (component_idx, (plane, &(component_width, component_height))) in
        planes.iter().zip(component_dimensions).enumerate()
    {
        let samples = typed_component_plane_to_i64(plane, component_width, component_height)?;
        let decomp = fdwt::forward_dwt_i64(&samples, component_width, component_height, num_levels);
        let steps = component_step_sizes
            .get(component_idx)
            .ok_or("component quantization step count mismatch")?;
        let component = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
        let mut packets = Vec::with_capacity(num_levels as usize + 1);

        let ll_subband = prepare_subband_i64(
            &decomp.ll,
            decomp.ll_width,
            decomp.ll_height,
            steps
                .first()
                .ok_or("reversible quantization step missing")?,
            SubBandType::LowLow,
            subband_settings,
        )?;
        packets.push(PreparedResolutionPacket {
            component,
            resolution: 0,
            precinct: 0,
            subbands: vec![ll_subband],
        });

        for (level_idx, level) in decomp.levels.iter().enumerate() {
            let step_base = 1 + level_idx * 3;
            let hl_subband = prepare_subband_i64(
                &level.hl,
                level.high_width,
                level.low_height,
                steps
                    .get(step_base)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::HighLow,
                subband_settings,
            )?;
            let lh_subband = prepare_subband_i64(
                &level.lh,
                level.low_width,
                level.high_height,
                steps
                    .get(step_base + 1)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::LowHigh,
                subband_settings,
            )?;
            let hh_subband = prepare_subband_i64(
                &level.hh,
                level.high_width,
                level.high_height,
                steps
                    .get(step_base + 2)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::HighHigh,
                subband_settings,
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

    Ok(component_resolution_packets)
}

fn typed_component_plane_to_i64(
    plane: &EncodeTypedComponentPlane<'_>,
    width: u32,
    height: u32,
) -> Result<Vec<i64>, &'static str> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(plane.bit_depth)?;
    let sample_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or("image dimensions overflow")?;
    let expected_len = sample_count
        .checked_mul(bytes_per_sample)
        .ok_or("image dimensions overflow")?;
    if plane.data.len() != expected_len {
        return Err("component plane data length mismatch");
    }
    let unsigned_offset = if plane.signed {
        0
    } else {
        1_i64 << (u32::from(plane.bit_depth) - 1)
    };
    Ok(plane
        .data
        .chunks_exact(bytes_per_sample)
        .map(|sample| {
            let raw = read_le_sample_value(sample, plane.bit_depth);
            if plane.signed {
                sign_extend_sample(raw, plane.bit_depth)
            } else {
                i64::try_from(raw).unwrap_or(i64::MAX) - unsigned_offset
            }
        })
        .collect())
}
