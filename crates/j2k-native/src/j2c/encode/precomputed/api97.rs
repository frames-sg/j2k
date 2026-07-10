// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    codestream_write, count_compact_code_blocks, encode_prepared_resolution_packets,
    encode_with_accelerator, ordered_prepared_compact_resolution_packets,
    ordered_prepared_resolution_packets, packet_descriptors_for_compact_order,
    packet_descriptors_for_order, packet_encode, packetize_resolution_packets_with_options,
    precinct_exponents_for_options, precomputed_97_level_count, preencoded_97_level_count,
    preencoded_compact_97_level_count,
    prepared_resolution_packets_from_preencoded_compact_component,
    prepared_resolution_packets_from_preencoded_component,
    prepared_resolution_packets_from_preencoded_component_owned,
    prepared_resolution_packets_from_prequantized_component, prequantized_97_level_count,
    public_packetization_progression_order, public_packetization_resolutions_from_compact,
    quantize, validate_irreversible_quantization_profile, validate_precomputed_dwt97_geometry,
    validate_preencoded_compact_htj2k97_image, validate_preencoded_htj2k97_image,
    validate_prequantized_htj2k97_image, vec, write_single_tile_packetized_codestream,
    zero_pixel_buffer, BlockCodingMode, CpuOnlyJ2kEncodeStageAccelerator, EncodeOptions,
    EncodeParams, J2kEncodeStageAccelerator, J2kPacketizationEncodeJob,
    PrecomputedDwt97Accelerator, PrecomputedHtj2k97Image, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97Image, PrequantizedHtj2k97Image, Vec, MAX_J2K_SPEC_COMPONENTS,
};

/// Encode precomputed irreversible 9/7 wavelet coefficients into an HTJ2K
/// codestream.
///
/// This experimental entry point is the lossy counterpart of
/// [`encode_precomputed_htj2k_53`]. It bypasses the encoder's forward 9/7 DWT
/// stage by supplying precomputed floating-point DWT output through the
/// internal stage hook. Coefficients are expected in the same sample domain as
/// the native irreversible FDWT input: unsigned components are already level
/// shifted by subtracting `2^(bit_depth - 1)`.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_97(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_htj2k_97_with_accelerator(image, options, &mut accelerator)
}

/// Encode precomputed irreversible 9/7 wavelet coefficients into an HTJ2K
/// codestream using optional block encode and packetization hooks.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_97_with_accelerator(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    validate_irreversible_quantization_profile(options)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }
    validate_precomputed_dwt97_geometry(image)?;

    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = precomputed_97_level_count(&image.components)?;
    let mut precomputed_options = options.clone();
    precomputed_options.num_decomposition_levels = num_levels;
    precomputed_options.reversible = false;
    precomputed_options.use_ht_block_coding = true;
    precomputed_options.use_mct = false;
    precomputed_options.validate_high_throughput_codestream = false;
    precomputed_options.component_sampling = Some(
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz))
            .collect(),
    );

    let dummy_pixels =
        zero_pixel_buffer(image.width, image.height, num_components, image.bit_depth)?;
    let mut precomputed_accelerator = PrecomputedDwt97Accelerator {
        outputs: image
            .components
            .iter()
            .map(|component| component.dwt.clone())
            .collect(),
        encode_accelerator: accelerator,
    };

    encode_with_accelerator(
        &dummy_pixels,
        image.width,
        image.height,
        num_components,
        image.bit_depth,
        image.signed,
        &precomputed_options,
        &mut precomputed_accelerator,
    )
}

/// Encode prequantized irreversible 9/7 code-block coefficients into an HTJ2K
/// codestream.
#[doc(hidden)]
pub fn encode_prequantized_htj2k_97(
    image: &PrequantizedHtj2k97Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_prequantized_htj2k_97_with_accelerator(image, options, &mut accelerator)
}

/// Encode prequantized irreversible 9/7 code-block coefficients into an HTJ2K
/// codestream using optional block encode and packetization hooks.
#[doc(hidden)]
pub fn encode_prequantized_htj2k_97_with_accelerator(
    image: &PrequantizedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    validate_irreversible_quantization_profile(options)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }

    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = prequantized_97_level_count(&image.components)?;
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        image.bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    validate_prequantized_htj2k97_image(image, guard_bits, &step_sizes)?;

    let mut prequantized_options = options.clone();
    prequantized_options.num_decomposition_levels = num_levels;
    prequantized_options.reversible = false;
    prequantized_options.use_ht_block_coding = true;
    prequantized_options.use_mct = false;
    prequantized_options.validate_high_throughput_codestream = false;
    prequantized_options.component_sampling = Some(
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz))
            .collect(),
    );

    let component_resolution_packets = image
        .components
        .iter()
        .enumerate()
        .map(|(component_idx, component)| {
            prepared_resolution_packets_from_prequantized_component(component_idx, component)
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, &prequantized_options)?;
    let packet_descriptors = packet_descriptors_for_order(
        &prepared_resolution_packets,
        1,
        prequantized_options.progression_order,
    )?;
    let mut resolution_packets =
        encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
    let packetized_tile = packetize_resolution_packets_with_options(
        &mut resolution_packets,
        &packet_descriptors,
        1,
        num_components,
        prequantized_options.progression_order,
        packet_encode::PacketMarkerOptions {
            write_sop: prequantized_options.write_sop,
            write_eph: prequantized_options.write_eph,
            separate_packet_headers: prequantized_options.write_ppm
                || prequantized_options.write_ppt,
        },
        true,
        prequantized_options.write_plt
            || prequantized_options.write_plm
            || prequantized_options.write_ppm
            || prequantized_options.write_ppt
            || prequantized_options.write_sop
            || prequantized_options.write_eph
            || prequantized_options.tile_part_packet_limit.is_some(),
        accelerator,
    )?;

    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let params = EncodeParams {
        width: image.width,
        height: image.height,
        tile_width: image.width,
        tile_height: image.height,
        num_components,
        bit_depth: image.bit_depth,
        signed: image.signed,
        component_sample_info: Vec::new(),
        component_quantization_step_sizes: Vec::new(),
        num_decomposition_levels: num_levels,
        reversible: false,
        code_block_width_exp: prequantized_options.code_block_width_exp,
        code_block_height_exp: prequantized_options.code_block_height_exp,
        num_layers: 1,
        use_mct: false,
        guard_bits,
        block_coding_mode: BlockCodingMode::HighThroughput,
        progression_order: prequantized_options.progression_order,
        write_tlm: prequantized_options.write_tlm,
        write_plt: prequantized_options.write_plt,
        write_plm: prequantized_options.write_plm,
        write_ppm: prequantized_options.write_ppm,
        write_ppt: prequantized_options.write_ppt,
        write_sop: prequantized_options.write_sop,
        write_eph: prequantized_options.write_eph,
        terminate_coding_passes: false,
        component_sampling: prequantized_options
            .component_sampling
            .clone()
            .ok_or("component sampling missing")?,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents: precinct_exponents_for_options(&prequantized_options, num_levels)?,
    };

    write_single_tile_packetized_codestream(
        &params,
        &packetized_tile,
        &quant_params,
        prequantized_options.tile_part_packet_limit,
    )
}

/// Encode preencoded irreversible 9/7 HTJ2K code-block payloads into a
/// codestream.
#[doc(hidden)]
pub fn encode_preencoded_htj2k_97(
    image: &PreencodedHtj2k97Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_preencoded_htj2k_97_with_accelerator(image, options, &mut accelerator)
}

/// Encode preencoded irreversible 9/7 HTJ2K code-block payloads into a
/// codestream using optional packetization hooks.
#[doc(hidden)]
pub fn encode_preencoded_htj2k_97_with_accelerator(
    image: &PreencodedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    validate_irreversible_quantization_profile(options)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }

    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = preencoded_97_level_count(&image.components)?;
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        image.bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    validate_preencoded_htj2k97_image(image, guard_bits, &step_sizes)?;

    let mut preencoded_options = options.clone();
    preencoded_options.num_decomposition_levels = num_levels;
    preencoded_options.reversible = false;
    preencoded_options.use_ht_block_coding = true;
    preencoded_options.use_mct = false;
    preencoded_options.validate_high_throughput_codestream = false;
    preencoded_options.component_sampling = Some(
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz))
            .collect(),
    );

    let component_resolution_packets = image
        .components
        .iter()
        .enumerate()
        .map(|(component_idx, component)| {
            prepared_resolution_packets_from_preencoded_component(component_idx, component)
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, &preencoded_options)?;
    let packet_descriptors = packet_descriptors_for_order(
        &prepared_resolution_packets,
        1,
        preencoded_options.progression_order,
    )?;
    let mut resolution_packets =
        encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
    let packetized_tile = packetize_resolution_packets_with_options(
        &mut resolution_packets,
        &packet_descriptors,
        1,
        num_components,
        preencoded_options.progression_order,
        packet_encode::PacketMarkerOptions {
            write_sop: preencoded_options.write_sop,
            write_eph: preencoded_options.write_eph,
            separate_packet_headers: preencoded_options.write_ppm || preencoded_options.write_ppt,
        },
        true,
        preencoded_options.write_plt
            || preencoded_options.write_plm
            || preencoded_options.write_ppm
            || preencoded_options.write_ppt
            || preencoded_options.write_sop
            || preencoded_options.write_eph
            || preencoded_options.tile_part_packet_limit.is_some(),
        accelerator,
    )?;

    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let params = EncodeParams {
        width: image.width,
        height: image.height,
        tile_width: image.width,
        tile_height: image.height,
        num_components,
        bit_depth: image.bit_depth,
        signed: image.signed,
        component_sample_info: Vec::new(),
        component_quantization_step_sizes: Vec::new(),
        num_decomposition_levels: num_levels,
        reversible: false,
        code_block_width_exp: preencoded_options.code_block_width_exp,
        code_block_height_exp: preencoded_options.code_block_height_exp,
        num_layers: 1,
        use_mct: false,
        guard_bits,
        block_coding_mode: BlockCodingMode::HighThroughput,
        progression_order: preencoded_options.progression_order,
        write_tlm: preencoded_options.write_tlm,
        write_plt: preencoded_options.write_plt,
        write_plm: preencoded_options.write_plm,
        write_ppm: preencoded_options.write_ppm,
        write_ppt: preencoded_options.write_ppt,
        write_sop: preencoded_options.write_sop,
        write_eph: preencoded_options.write_eph,
        terminate_coding_passes: false,
        component_sampling: preencoded_options
            .component_sampling
            .clone()
            .ok_or("component sampling missing")?,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents: precinct_exponents_for_options(&preencoded_options, num_levels)?,
    };

    write_single_tile_packetized_codestream(
        &params,
        &packetized_tile,
        &quant_params,
        preencoded_options.tile_part_packet_limit,
    )
}

/// Encode preencoded irreversible 9/7 HTJ2K code-block payloads into a
/// codestream, consuming the image so code-block payloads can move into packet
/// preparation without cloning.
#[doc(hidden)]
pub fn encode_preencoded_htj2k_97_owned_with_accelerator(
    image: PreencodedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    validate_irreversible_quantization_profile(options)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }

    let width = image.width;
    let height = image.height;
    let bit_depth = image.bit_depth;
    let signed = image.signed;
    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = preencoded_97_level_count(&image.components)?;
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    validate_preencoded_htj2k97_image(&image, guard_bits, &step_sizes)?;

    let component_sampling = image
        .components
        .iter()
        .map(|component| (component.x_rsiz, component.y_rsiz))
        .collect::<Vec<_>>();
    let mut preencoded_options = options.clone();
    preencoded_options.num_decomposition_levels = num_levels;
    preencoded_options.reversible = false;
    preencoded_options.use_ht_block_coding = true;
    preencoded_options.use_mct = false;
    preencoded_options.validate_high_throughput_codestream = false;
    preencoded_options.component_sampling = Some(component_sampling);

    let component_resolution_packets = image
        .components
        .into_iter()
        .enumerate()
        .map(|(component_idx, component)| {
            prepared_resolution_packets_from_preencoded_component_owned(component_idx, component)
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, &preencoded_options)?;
    let packet_descriptors = packet_descriptors_for_order(
        &prepared_resolution_packets,
        1,
        preencoded_options.progression_order,
    )?;
    let mut resolution_packets =
        encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
    let packetized_tile = packetize_resolution_packets_with_options(
        &mut resolution_packets,
        &packet_descriptors,
        1,
        num_components,
        preencoded_options.progression_order,
        packet_encode::PacketMarkerOptions {
            write_sop: preencoded_options.write_sop,
            write_eph: preencoded_options.write_eph,
            separate_packet_headers: preencoded_options.write_ppm || preencoded_options.write_ppt,
        },
        true,
        preencoded_options.write_plt
            || preencoded_options.write_plm
            || preencoded_options.write_ppm
            || preencoded_options.write_ppt
            || preencoded_options.write_sop
            || preencoded_options.write_eph
            || preencoded_options.tile_part_packet_limit.is_some(),
        accelerator,
    )?;

    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let params = EncodeParams {
        width,
        height,
        tile_width: width,
        tile_height: height,
        num_components,
        bit_depth,
        signed,
        component_sample_info: Vec::new(),
        component_quantization_step_sizes: Vec::new(),
        num_decomposition_levels: num_levels,
        reversible: false,
        code_block_width_exp: preencoded_options.code_block_width_exp,
        code_block_height_exp: preencoded_options.code_block_height_exp,
        num_layers: 1,
        use_mct: false,
        guard_bits,
        block_coding_mode: BlockCodingMode::HighThroughput,
        progression_order: preencoded_options.progression_order,
        write_tlm: preencoded_options.write_tlm,
        write_plt: preencoded_options.write_plt,
        write_plm: preencoded_options.write_plm,
        write_ppm: preencoded_options.write_ppm,
        write_ppt: preencoded_options.write_ppt,
        write_sop: preencoded_options.write_sop,
        write_eph: preencoded_options.write_eph,
        terminate_coding_passes: false,
        component_sampling: preencoded_options
            .component_sampling
            .clone()
            .ok_or("component sampling missing")?,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents: precinct_exponents_for_options(&preencoded_options, num_levels)?,
    };

    write_single_tile_packetized_codestream(
        &params,
        &packetized_tile,
        &quant_params,
        preencoded_options.tile_part_packet_limit,
    )
}

/// Encode compact preencoded irreversible 9/7 HTJ2K code-block payloads into a
/// codestream, borrowing code-block ranges from one image-level payload buffer
/// during packetization.
#[doc(hidden)]
pub fn encode_preencoded_htj2k_97_compact_owned_with_accelerator(
    image: PreencodedHtj2k97CompactImage,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    if options.write_plt
        || options.write_plm
        || options.write_sop
        || options.write_eph
        || options.tile_part_packet_limit.is_some()
    {
        return Err(
            "compact preencoded HTJ2K encode does not support packet marker or tile-part options",
        );
    }
    validate_irreversible_quantization_profile(options)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }

    let width = image.width;
    let height = image.height;
    let bit_depth = image.bit_depth;
    let signed = image.signed;
    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = preencoded_compact_97_level_count(&image.components)?;
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    validate_preencoded_compact_htj2k97_image(&image, guard_bits, &step_sizes)?;

    let component_sampling = image
        .components
        .iter()
        .map(|component| (component.x_rsiz, component.y_rsiz))
        .collect::<Vec<_>>();
    let mut preencoded_options = options.clone();
    preencoded_options.num_decomposition_levels = num_levels;
    preencoded_options.reversible = false;
    preencoded_options.use_ht_block_coding = true;
    preencoded_options.use_mct = false;
    preencoded_options.validate_high_throughput_codestream = false;
    preencoded_options.component_sampling = Some(component_sampling);

    let PreencodedHtj2k97CompactImage {
        payload,
        components,
        ..
    } = image;
    let component_resolution_packets = components
        .iter()
        .enumerate()
        .map(|(component_idx, component)| {
            prepared_resolution_packets_from_preencoded_compact_component(
                component_idx,
                component,
                &payload,
            )
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    let prepared_resolution_packets = ordered_prepared_compact_resolution_packets(
        component_resolution_packets,
        &preencoded_options,
    )?;
    let packet_descriptors = packet_descriptors_for_compact_order(
        &prepared_resolution_packets,
        1,
        preencoded_options.progression_order,
    )?;
    let packetization_resolutions =
        public_packetization_resolutions_from_compact(&prepared_resolution_packets);
    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: packetization_resolutions.len() as u32,
        num_layers: 1,
        num_components,
        code_block_count: count_compact_code_blocks(&prepared_resolution_packets)?,
        progression_order: public_packetization_progression_order(
            preencoded_options.progression_order,
        ),
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let tile_data = accelerator
        .encode_packetization(packetization_job)?
        .map_or_else(
            || crate::encode_j2k_packetization_scalar(packetization_job),
            Ok,
        )?;

    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let params = EncodeParams {
        width,
        height,
        tile_width: width,
        tile_height: height,
        num_components,
        bit_depth,
        signed,
        component_sample_info: Vec::new(),
        component_quantization_step_sizes: Vec::new(),
        num_decomposition_levels: num_levels,
        reversible: false,
        code_block_width_exp: preencoded_options.code_block_width_exp,
        code_block_height_exp: preencoded_options.code_block_height_exp,
        num_layers: 1,
        use_mct: false,
        guard_bits,
        block_coding_mode: BlockCodingMode::HighThroughput,
        progression_order: preencoded_options.progression_order,
        write_tlm: preencoded_options.write_tlm,
        write_plt: preencoded_options.write_plt,
        write_plm: preencoded_options.write_plm,
        write_ppm: preencoded_options.write_ppm,
        write_ppt: preencoded_options.write_ppt,
        write_sop: preencoded_options.write_sop,
        write_eph: preencoded_options.write_eph,
        terminate_coding_passes: false,
        component_sampling: preencoded_options
            .component_sampling
            .clone()
            .ok_or("component sampling missing")?,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents: precinct_exponents_for_options(&preencoded_options, num_levels)?,
    };

    Ok(codestream_write::write_codestream(
        &params,
        &tile_data,
        &quant_params,
    ))
}
