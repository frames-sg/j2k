// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// This mirrors [`encode_precomputed_htj2k_53`] while selecting classic EBCOT
/// block coding. It reuses the same quantization, packetization, and codestream
/// writer stages as the normal encoder and is primarily intended for fixtures
/// and coefficient-domain workflows that need JPEG-native component sampling.
#[doc(hidden)]
pub fn encode_precomputed_j2k_53(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_j2k_53_with_mct_and_accelerator(image, options, false, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into a classic
/// JPEG 2000 Part 1 codestream using optional block encode and packetization
/// hooks.
#[doc(hidden)]
pub fn encode_precomputed_j2k_53_with_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_j2k_53_with_mct_and_accelerator(image, options, false, accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into a classic
/// JPEG 2000 Part 1 codestream while controlling the output COD
/// multi-component transform flag.
#[doc(hidden)]
pub fn encode_precomputed_j2k_53_with_mct(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_j2k_53_with_mct_and_accelerator(image, options, use_mct, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into a classic
/// JPEG 2000 Part 1 codestream while controlling the output COD
/// multi-component transform flag and using optional encode stage hooks.
#[doc(hidden)]
pub fn encode_precomputed_j2k_53_with_mct_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_53_with_mct_and_accelerator(image, options, use_mct, false, accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into an HTJ2K
/// codestream.
///
/// This experimental entry point reuses the existing quantization, HT block
/// coding, packetization, and codestream writer stages. It bypasses the
/// encoder's forward DWT stage by supplying precomputed DWT output through the
/// internal stage hook. Coefficients are expected in the same sample domain as
/// the native encoder's FDWT input: unsigned components are already level
/// shifted by subtracting `2^(bit_depth - 1)`.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_53(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_htj2k_53_with_mct_and_accelerator(image, options, false, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into an HTJ2K
/// codestream using optional block encode and packetization hooks.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_53_with_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_htj2k_53_with_mct_and_accelerator(image, options, false, accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into an HTJ2K
/// codestream while controlling the output COD multi-component transform flag.
///
/// This is intended for coefficient-domain JPEG 2000 family recoding, where
/// source codestream components may already be reversible-color-transformed.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_53_with_mct(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_htj2k_53_with_mct_and_accelerator(image, options, use_mct, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients while controlling
/// the output COD multi-component transform flag and using optional encode
/// stage hooks.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_53_with_mct_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_53_with_mct_and_accelerator(image, options, use_mct, true, accelerator)
}

pub(super) fn encode_precomputed_53_with_mct_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    use_ht_block_coding: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_53_with_component_sample_info_and_accelerator(
        image,
        options,
        use_mct,
        use_ht_block_coding,
        &[],
        accelerator,
    )
}

pub(super) fn encode_precomputed_53_with_component_sample_info_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    use_ht_block_coding: bool,
    component_sample_info: &[EncodeComponentSampleInfo],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Err("unsupported bit depth");
    }
    validate_component_sample_info(component_sample_info, image.components.len())?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }
    validate_precomputed_dwt_geometry(image)?;

    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = precomputed_level_count(&image.components)?;
    let mut precomputed_options = options.clone();
    precomputed_options.num_decomposition_levels = num_levels;
    precomputed_options.reversible = true;
    precomputed_options.use_ht_block_coding = use_ht_block_coding;
    precomputed_options.use_mct = use_mct;
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
    let mut precomputed_accelerator = PrecomputedDwtAccelerator {
        outputs: image
            .components
            .iter()
            .map(|component| component.dwt.clone())
            .collect(),
        encode_accelerator: accelerator,
    };

    encode_with_accelerator_and_component_sample_info(
        &dummy_pixels,
        image.width,
        image.height,
        num_components,
        image.bit_depth,
        image.signed,
        &precomputed_options,
        component_sample_info,
        &mut precomputed_accelerator,
    )
}

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

/// Encode multiple precomputed irreversible 9/7 wavelet images while sharing
/// one HT code-block batch across all prepared tiles.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_97_batch_with_accelerator(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<Vec<u8>>, &'static str> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    if options.num_layers != 1 {
        return Err("batch precomputed 9/7 encode currently supports one quality layer");
    }

    let mut prepared_images = prepare_precomputed_htj2k97_images_for_batch(images, options)?;
    let mut all_packets = Vec::new();
    for prepared in &mut prepared_images {
        prepared.packet_count = prepared.prepared_packets.len();
        all_packets.append(&mut prepared.prepared_packets);
    }

    let mut encoded_packets =
        encode_prepared_resolution_packets(all_packets, accelerator)?.into_iter();
    let mut codestreams = Vec::with_capacity(prepared_images.len());
    for prepared in prepared_images {
        let mut resolution_packets = Vec::with_capacity(prepared.packet_count);
        for _ in 0..prepared.packet_count {
            resolution_packets.push(
                encoded_packets
                    .next()
                    .ok_or("encoded packet count mismatch")?,
            );
        }
        let scalar_packet_descriptors = scalar_packet_descriptors(&prepared.packet_descriptors);
        let packetized_tile =
            packet_encode::form_tile_bitstream_with_descriptors_lengths_and_markers(
                &mut resolution_packets,
                &scalar_packet_descriptors,
                packet_encode::PacketMarkerOptions {
                    write_sop: prepared.params.write_sop,
                    write_eph: prepared.params.write_eph,
                    separate_packet_headers: prepared.params.write_ppm || prepared.params.write_ppt,
                },
            )?;
        codestreams.push(write_single_tile_packetized_codestream(
            &prepared.params,
            &packetized_tile,
            &prepared.quant_params,
            options.tile_part_packet_limit,
        )?);
    }
    if encoded_packets.next().is_some() {
        return Err("encoded packet count mismatch");
    }

    Ok(codestreams)
}

#[cfg(feature = "parallel")]
pub(super) fn prepare_precomputed_htj2k97_images_for_batch(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
) -> Result<Vec<PreparedPrecomputedHtj2k97Image>, &'static str> {
    images
        .par_iter()
        .map(|image| prepare_precomputed_htj2k97_image_for_batch(image, options))
        .collect()
}

#[cfg(not(feature = "parallel"))]
pub(super) fn prepare_precomputed_htj2k97_images_for_batch(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
) -> Result<Vec<PreparedPrecomputedHtj2k97Image>, &'static str> {
    images
        .iter()
        .map(|image| prepare_precomputed_htj2k97_image_for_batch(image, options))
        .collect()
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

pub(super) fn validate_precomputed_dwt_geometry(
    image: &PrecomputedHtj2k53Image,
) -> Result<(), &'static str> {
    for component in &image.components {
        let component_width = image.width.div_ceil(u32::from(component.x_rsiz));
        let component_height = image.height.div_ceil(u32::from(component.y_rsiz));
        validate_precomputed_component_dwt_geometry(
            &component.dwt,
            component_width,
            component_height,
        )?;
    }

    Ok(())
}

pub(super) fn validate_precomputed_dwt97_geometry(
    image: &PrecomputedHtj2k97Image,
) -> Result<(), &'static str> {
    for component in &image.components {
        let component_width = image.width.div_ceil(u32::from(component.x_rsiz));
        let component_height = image.height.div_ceil(u32::from(component.y_rsiz));
        validate_precomputed_component_dwt_geometry(
            &component.dwt,
            component_width,
            component_height,
        )?;
    }

    Ok(())
}

pub(super) fn validate_precomputed_component_dwt_geometry(
    dwt: &impl PrecomputedDwtGeometryView,
    component_width: u32,
    component_height: u32,
) -> Result<(), &'static str> {
    if let Some(highest_level) = dwt.last_level_geometry() {
        if highest_level.width != component_width || highest_level.height != component_height {
            return Err("precomputed DWT component dimensions mismatch");
        }
    }

    let mut expected_width = component_width;
    let mut expected_height = component_height;
    for level_index in (0..dwt.level_count()).rev() {
        let level = dwt.level_geometry(level_index);
        let low_width = expected_width.div_ceil(2);
        let low_height = expected_height.div_ceil(2);
        let high_width = expected_width / 2;
        let high_height = expected_height / 2;

        if level.width != expected_width
            || level.height != expected_height
            || level.low_width != low_width
            || level.low_height != low_height
            || level.high_width != high_width
            || level.high_height != high_height
        {
            return Err("precomputed DWT recursive geometry mismatch");
        }
        validate_band_len(level.hl_len, high_width, low_height)?;
        validate_band_len(level.lh_len, low_width, high_height)?;
        validate_band_len(level.hh_len, high_width, high_height)?;

        expected_width = low_width;
        expected_height = low_height;
    }

    if dwt.ll_width() != expected_width || dwt.ll_height() != expected_height {
        return Err("precomputed DWT component dimensions mismatch");
    }
    validate_band_len(dwt.ll_len(), expected_width, expected_height)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PrecomputedDwtLevelGeometry {
    width: u32,
    height: u32,
    low_width: u32,
    low_height: u32,
    high_width: u32,
    high_height: u32,
    hl_len: usize,
    lh_len: usize,
    hh_len: usize,
}

pub(super) trait PrecomputedDwtGeometryView {
    fn ll_len(&self) -> usize;
    fn ll_width(&self) -> u32;
    fn ll_height(&self) -> u32;
    fn level_count(&self) -> usize;
    fn level_geometry(&self, index: usize) -> PrecomputedDwtLevelGeometry;

    fn last_level_geometry(&self) -> Option<PrecomputedDwtLevelGeometry> {
        self.level_count()
            .checked_sub(1)
            .map(|index| self.level_geometry(index))
    }
}

impl PrecomputedDwtGeometryView for J2kForwardDwt53Output {
    fn ll_len(&self) -> usize {
        self.ll.len()
    }

    fn ll_width(&self) -> u32 {
        self.ll_width
    }

    fn ll_height(&self) -> u32 {
        self.ll_height
    }

    fn level_count(&self) -> usize {
        self.levels.len()
    }

    fn level_geometry(&self, index: usize) -> PrecomputedDwtLevelGeometry {
        let level = &self.levels[index];
        PrecomputedDwtLevelGeometry {
            width: level.width,
            height: level.height,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
            hl_len: level.hl.len(),
            lh_len: level.lh.len(),
            hh_len: level.hh.len(),
        }
    }
}

impl PrecomputedDwtGeometryView for J2kForwardDwt97Output {
    fn ll_len(&self) -> usize {
        self.ll.len()
    }

    fn ll_width(&self) -> u32 {
        self.ll_width
    }

    fn ll_height(&self) -> u32 {
        self.ll_height
    }

    fn level_count(&self) -> usize {
        self.levels.len()
    }

    fn level_geometry(&self, index: usize) -> PrecomputedDwtLevelGeometry {
        let level = &self.levels[index];
        PrecomputedDwtLevelGeometry {
            width: level.width,
            height: level.height,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
            hl_len: level.hl.len(),
            lh_len: level.lh.len(),
            hh_len: level.hh.len(),
        }
    }
}

pub(super) fn uniform_level_count<T>(
    components: &[T],
    len_of: impl Fn(&T) -> usize,
    first_to_levels: impl Fn(usize) -> Result<usize, &'static str>,
    mismatch: &'static str,
) -> Result<u8, &'static str> {
    let first_len = len_of(components.first().ok_or("unsupported component count")?);
    let levels = first_to_levels(first_len)?;
    if components
        .iter()
        .any(|component| len_of(component) != first_len)
    {
        return Err(mismatch);
    }
    u8::try_from(levels).map_err(|_| "decomposition level count exceeds u8")
}

pub(super) fn dwt_levels_only(levels: usize) -> Result<usize, &'static str> {
    Ok(levels)
}

pub(super) fn precomputed_level_count(
    components: &[PrecomputedHtj2k53Component],
) -> Result<u8, &'static str> {
    uniform_level_count(
        components,
        |component| component.dwt.levels.len(),
        dwt_levels_only,
        "precomputed components must use the same decomposition level count",
    )
}

pub(super) fn precomputed_97_level_count(
    components: &[PrecomputedHtj2k97Component],
) -> Result<u8, &'static str> {
    uniform_level_count(
        components,
        |component| component.dwt.levels.len(),
        dwt_levels_only,
        "precomputed components must use the same decomposition level count",
    )
}

pub(super) fn prequantized_97_level_count(
    components: &[PrequantizedHtj2k97Component],
) -> Result<u8, &'static str> {
    uniform_level_count(
        components,
        |component| component.resolutions.len(),
        |len| {
            len.checked_sub(1)
                .ok_or("prequantized components must contain at least one decomposition level")
        },
        "prequantized components must use the same decomposition level count",
    )
}

pub(super) fn preencoded_97_level_count(
    components: &[PreencodedHtj2k97Component],
) -> Result<u8, &'static str> {
    uniform_level_count(
        components,
        |component| component.resolutions.len(),
        |len| {
            len.checked_sub(1)
                .ok_or("preencoded components must contain at least one decomposition level")
        },
        "preencoded components must use the same decomposition level count",
    )
}

pub(super) fn preencoded_compact_97_level_count(
    components: &[PreencodedHtj2k97CompactComponent],
) -> Result<u8, &'static str> {
    uniform_level_count(
        components,
        |component| component.resolutions.len(),
        |len| {
            len.checked_sub(1)
                .ok_or("preencoded components must contain at least one decomposition level")
        },
        "preencoded components must use the same decomposition level count",
    )
}

pub(super) fn validate_prequantized_htj2k97_image(
    image: &PrequantizedHtj2k97Image,
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    for component in &image.components {
        if component.resolutions.is_empty() {
            return Err("prequantized components must contain at least one resolution");
        }
        validate_prequantized_resolution(
            &component.resolutions[0],
            &[J2kSubBandType::LowLow],
            guard_bits,
            &step_sizes[0..1],
        )?;
        for (level_index, resolution) in component.resolutions.iter().enumerate().skip(1) {
            let step_base = 1 + (level_index - 1) * 3;
            validate_prequantized_resolution(
                resolution,
                &[
                    J2kSubBandType::HighLow,
                    J2kSubBandType::LowHigh,
                    J2kSubBandType::HighHigh,
                ],
                guard_bits,
                &step_sizes[step_base..step_base + 3],
            )?;
        }
    }

    Ok(())
}

pub(super) fn validate_preencoded_htj2k97_image(
    image: &PreencodedHtj2k97Image,
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    for component in &image.components {
        if component.resolutions.is_empty() {
            return Err("preencoded components must contain at least one resolution");
        }
        validate_preencoded_resolution(
            &component.resolutions[0],
            &[J2kSubBandType::LowLow],
            guard_bits,
            &step_sizes[0..1],
        )?;
        for (level_index, resolution) in component.resolutions.iter().enumerate().skip(1) {
            let step_base = 1 + (level_index - 1) * 3;
            validate_preencoded_resolution(
                resolution,
                &[
                    J2kSubBandType::HighLow,
                    J2kSubBandType::LowHigh,
                    J2kSubBandType::HighHigh,
                ],
                guard_bits,
                &step_sizes[step_base..step_base + 3],
            )?;
        }
    }

    Ok(())
}

pub(super) fn validate_preencoded_compact_htj2k97_image(
    image: &PreencodedHtj2k97CompactImage,
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    for component in &image.components {
        if component.resolutions.is_empty() {
            return Err("preencoded components must contain at least one resolution");
        }
        validate_preencoded_compact_resolution(
            &component.resolutions[0],
            &[J2kSubBandType::LowLow],
            guard_bits,
            &step_sizes[0..1],
            image.payload.len(),
        )?;
        for (level_index, resolution) in component.resolutions.iter().enumerate().skip(1) {
            let step_base = 1 + (level_index - 1) * 3;
            validate_preencoded_compact_resolution(
                resolution,
                &[
                    J2kSubBandType::HighLow,
                    J2kSubBandType::LowHigh,
                    J2kSubBandType::HighHigh,
                ],
                guard_bits,
                &step_sizes[step_base..step_base + 3],
                image.payload.len(),
            )?;
        }
    }

    Ok(())
}

pub(super) fn validate_prequantized_resolution(
    resolution: &PrequantizedHtj2k97Resolution,
    expected_subbands: &[J2kSubBandType],
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    if resolution.subbands.len() != expected_subbands.len() {
        return Err("prequantized resolution subband count mismatch");
    }
    for ((subband, expected_subband), step_size) in resolution
        .subbands
        .iter()
        .zip(expected_subbands)
        .zip(step_sizes)
    {
        if subband.sub_band_type != *expected_subband {
            return Err("prequantized resolution subband order mismatch");
        }
        let expected_blocks = subband
            .num_cbs_x
            .checked_mul(subband.num_cbs_y)
            .ok_or("prequantized code-block count overflow")?;
        if expected_blocks == 0 {
            if subband.total_bitplanes != 0 || !subband.code_blocks.is_empty() {
                return Err("empty prequantized subbands must not contain code-block data");
            }
            continue;
        }
        debug_assert!(step_size.exponent <= u16::from(u8::MAX));
        let expected_total_bitplanes = guard_bits
            .saturating_add(step_size.exponent as u8)
            .saturating_sub(1);
        if subband.total_bitplanes != expected_total_bitplanes {
            return Err("prequantized subband bitplane count mismatch");
        }
        if usize::try_from(expected_blocks).map_err(|_| "prequantized code-block count overflow")?
            != subband.code_blocks.len()
        {
            return Err("prequantized code-block count mismatch");
        }
        for block in &subband.code_blocks {
            if block.width == 0 || block.height == 0 {
                return Err("prequantized code-block dimensions must be non-zero");
            }
            validate_band_len(block.coefficients.len(), block.width, block.height)?;
        }
    }

    Ok(())
}

pub(super) fn validate_preencoded_resolution(
    resolution: &PreencodedHtj2k97Resolution,
    expected_subbands: &[J2kSubBandType],
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    if resolution.subbands.len() != expected_subbands.len() {
        return Err("preencoded resolution subband count mismatch");
    }
    for ((subband, expected_subband), step_size) in resolution
        .subbands
        .iter()
        .zip(expected_subbands)
        .zip(step_sizes)
    {
        if subband.sub_band_type != *expected_subband {
            return Err("preencoded resolution subband order mismatch");
        }
        let expected_blocks = subband
            .num_cbs_x
            .checked_mul(subband.num_cbs_y)
            .ok_or("preencoded code-block count overflow")?;
        if expected_blocks == 0 {
            if subband.total_bitplanes != 0 || !subband.code_blocks.is_empty() {
                return Err("empty preencoded subbands must not contain code-block data");
            }
            continue;
        }
        debug_assert!(step_size.exponent <= u16::from(u8::MAX));
        let expected_total_bitplanes = guard_bits
            .saturating_add(step_size.exponent as u8)
            .saturating_sub(1);
        if subband.total_bitplanes != expected_total_bitplanes {
            return Err("preencoded subband bitplane count mismatch");
        }
        if usize::try_from(expected_blocks).map_err(|_| "preencoded code-block count overflow")?
            != subband.code_blocks.len()
        {
            return Err("preencoded code-block count mismatch");
        }
        for block in &subband.code_blocks {
            if block.width == 0 || block.height == 0 {
                return Err("preencoded code-block dimensions must be non-zero");
            }
            validate_preencoded_code_block_payload(&block.encoded, subband.total_bitplanes)?;
        }
    }

    Ok(())
}

pub(super) fn validate_preencoded_compact_resolution(
    resolution: &PreencodedHtj2k97CompactResolution,
    expected_subbands: &[J2kSubBandType],
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
    payload_len: usize,
) -> Result<(), &'static str> {
    if resolution.subbands.len() != expected_subbands.len() {
        return Err("preencoded resolution subband count mismatch");
    }
    for ((subband, expected_subband), step_size) in resolution
        .subbands
        .iter()
        .zip(expected_subbands)
        .zip(step_sizes)
    {
        if subband.sub_band_type != *expected_subband {
            return Err("preencoded resolution subband order mismatch");
        }
        let expected_blocks = subband
            .num_cbs_x
            .checked_mul(subband.num_cbs_y)
            .ok_or("preencoded code-block count overflow")?;
        if expected_blocks == 0 {
            if subband.total_bitplanes != 0 || !subband.code_blocks.is_empty() {
                return Err("empty preencoded subbands must not contain code-block data");
            }
            continue;
        }
        debug_assert!(step_size.exponent <= u16::from(u8::MAX));
        let expected_total_bitplanes = guard_bits
            .saturating_add(step_size.exponent as u8)
            .saturating_sub(1);
        if subband.total_bitplanes != expected_total_bitplanes {
            return Err("preencoded subband bitplane count mismatch");
        }
        if usize::try_from(expected_blocks).map_err(|_| "preencoded code-block count overflow")?
            != subband.code_blocks.len()
        {
            return Err("preencoded code-block count mismatch");
        }
        for block in &subband.code_blocks {
            if block.width == 0 || block.height == 0 {
                return Err("preencoded code-block dimensions must be non-zero");
            }
            validate_preencoded_compact_code_block_payload(
                block,
                payload_len,
                subband.total_bitplanes,
            )?;
        }
    }

    Ok(())
}

pub(super) fn validate_preencoded_code_block_payload(
    block: &EncodedHtJ2kCodeBlock,
    total_bitplanes: u8,
) -> Result<(), &'static str> {
    let data_len = u32::try_from(block.data.len()).map_err(|_| "HTJ2K payload too large")?;
    if block.num_coding_passes == 0 {
        if data_len != 0 || block.cleanup_length != 0 || block.refinement_length != 0 {
            return Err("empty HTJ2K code-block payload metadata mismatch");
        }
        if block.num_zero_bitplanes != total_bitplanes {
            return Err("empty HTJ2K code-block zero-bitplane count mismatch");
        }
        return Ok(());
    }
    if block.num_coding_passes > 164 {
        return Err("HTJ2K code-block coding pass count out of range");
    }
    if block.num_zero_bitplanes >= total_bitplanes {
        return Err("HTJ2K code-block zero-bitplane count out of range");
    }
    let segment_len = block
        .cleanup_length
        .checked_add(block.refinement_length)
        .ok_or("HTJ2K payload segment length overflow")?;
    if segment_len != data_len {
        return Err("HTJ2K payload segment length mismatch");
    }
    Ok(())
}

pub(super) fn validate_preencoded_compact_code_block_payload(
    block: &PreencodedHtj2k97CompactCodeBlock,
    payload_len: usize,
    total_bitplanes: u8,
) -> Result<(), &'static str> {
    if block.payload_range.start > block.payload_range.end || block.payload_range.end > payload_len
    {
        return Err("HTJ2K payload range out of bounds");
    }
    let data_len = u32::try_from(block.payload_range.end - block.payload_range.start)
        .map_err(|_| "HTJ2K payload too large")?;
    if block.num_coding_passes == 0 {
        if data_len != 0 || block.cleanup_length != 0 || block.refinement_length != 0 {
            return Err("empty HTJ2K code-block payload metadata mismatch");
        }
        if block.num_zero_bitplanes != total_bitplanes {
            return Err("empty HTJ2K code-block zero-bitplane count mismatch");
        }
        return Ok(());
    }
    if block.num_coding_passes > 164 {
        return Err("HTJ2K code-block coding pass count out of range");
    }
    if block.num_zero_bitplanes >= total_bitplanes {
        return Err("HTJ2K code-block zero-bitplane count out of range");
    }
    let segment_len = block
        .cleanup_length
        .checked_add(block.refinement_length)
        .ok_or("HTJ2K payload segment length overflow")?;
    if segment_len != data_len {
        return Err("HTJ2K payload segment length mismatch");
    }
    Ok(())
}

pub(super) fn prepared_resolution_packets_from_prequantized_component(
    component_idx: usize,
    component: &PrequantizedHtj2k97Component,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .iter()
                    .map(prepared_subband_from_prequantized)
                    .collect::<Result<Vec<_>, &'static str>>()?,
            })
        })
        .collect()
}

pub(super) fn prepared_subband_from_prequantized(
    subband: &PrequantizedHtj2k97Subband,
) -> Result<PreparedEncodeSubband, &'static str> {
    Ok(PreparedEncodeSubband {
        code_blocks: subband
            .code_blocks
            .iter()
            .map(|block| PreparedEncodeCodeBlock {
                coefficients: block.coefficients.iter().copied().map(i64::from).collect(),
                width: block.width,
                height: block.height,
            })
            .collect(),
        preencoded_ht_code_blocks: None,
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
        code_block_width: subband
            .code_blocks
            .iter()
            .map(|block| block.width)
            .max()
            .unwrap_or(0),
        code_block_height: subband
            .code_blocks
            .iter()
            .map(|block| block.height)
            .max()
            .unwrap_or(0),
        width: precomputed_subband_width(
            subband.num_cbs_x,
            subband.code_blocks.iter().map(|block| block.width),
        ),
        height: precomputed_subband_height(
            subband.num_cbs_x,
            subband.num_cbs_y,
            subband.code_blocks.iter().map(|block| block.height),
        ),
        sub_band_type: internal_sub_band_type(subband.sub_band_type),
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 1,
    })
}

pub(super) fn precomputed_subband_width(
    width_in_blocks: u32,
    widths: impl Iterator<Item = u32>,
) -> u32 {
    if width_in_blocks == 0 {
        return 0;
    }

    widths.take(width_in_blocks as usize).sum()
}

pub(super) fn precomputed_subband_height(
    width_in_blocks: u32,
    height_in_blocks: u32,
    heights: impl Iterator<Item = u32>,
) -> u32 {
    if width_in_blocks == 0 || height_in_blocks == 0 {
        return 0;
    }

    heights
        .step_by(width_in_blocks as usize)
        .take(height_in_blocks as usize)
        .sum()
}

pub(super) fn prepared_resolution_packets_from_preencoded_component(
    component_idx: usize,
    component: &PreencodedHtj2k97Component,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .iter()
                    .map(prepared_subband_from_preencoded)
                    .collect::<Result<Vec<_>, &'static str>>()?,
            })
        })
        .collect()
}

pub(super) fn prepared_resolution_packets_from_preencoded_component_owned(
    component_idx: usize,
    component: PreencodedHtj2k97Component,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .into_iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .into_iter()
                    .map(prepared_subband_from_preencoded_owned)
                    .collect::<Result<Vec<_>, &'static str>>()?,
            })
        })
        .collect()
}

pub(super) fn prepared_resolution_packets_from_preencoded_compact_component<'a>(
    component_idx: usize,
    component: &'a PreencodedHtj2k97CompactComponent,
    payload: &'a [u8],
) -> Result<Vec<PreparedCompactResolutionPacket<'a>>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedCompactResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .iter()
                    .map(|subband| prepared_subband_from_preencoded_compact(subband, payload))
                    .collect::<Result<Vec<_>, &'static str>>()?,
            })
        })
        .collect()
}

pub(super) fn prepared_subband_from_preencoded(
    subband: &PreencodedHtj2k97Subband,
) -> Result<PreparedEncodeSubband, &'static str> {
    Ok(PreparedEncodeSubband {
        code_blocks: subband
            .code_blocks
            .iter()
            .map(|block| PreparedEncodeCodeBlock {
                coefficients: Vec::new(),
                width: block.width,
                height: block.height,
            })
            .collect(),
        preencoded_ht_code_blocks: Some(
            subband
                .code_blocks
                .iter()
                .map(|block| block.encoded.clone())
                .collect(),
        ),
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
        code_block_width: subband
            .code_blocks
            .iter()
            .map(|block| block.width)
            .max()
            .unwrap_or(0),
        code_block_height: subband
            .code_blocks
            .iter()
            .map(|block| block.height)
            .max()
            .unwrap_or(0),
        width: precomputed_subband_width(
            subband.num_cbs_x,
            subband.code_blocks.iter().map(|block| block.width),
        ),
        height: precomputed_subband_height(
            subband.num_cbs_x,
            subband.num_cbs_y,
            subband.code_blocks.iter().map(|block| block.height),
        ),
        sub_band_type: internal_sub_band_type(subband.sub_band_type),
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 1,
    })
}

pub(super) fn prepared_subband_from_preencoded_owned(
    subband: PreencodedHtj2k97Subband,
) -> Result<PreparedEncodeSubband, &'static str> {
    let code_block_width = subband
        .code_blocks
        .iter()
        .map(|block| block.width)
        .max()
        .unwrap_or(0);
    let code_block_height = subband
        .code_blocks
        .iter()
        .map(|block| block.height)
        .max()
        .unwrap_or(0);
    let width = precomputed_subband_width(
        subband.num_cbs_x,
        subband.code_blocks.iter().map(|block| block.width),
    );
    let height = precomputed_subband_height(
        subband.num_cbs_x,
        subband.num_cbs_y,
        subband.code_blocks.iter().map(|block| block.height),
    );
    let code_blocks = subband
        .code_blocks
        .into_iter()
        .map(|block| {
            let PreencodedHtj2k97CodeBlock {
                width,
                height,
                encoded,
            } = block;
            (
                PreparedEncodeCodeBlock {
                    coefficients: Vec::new(),
                    width,
                    height,
                },
                encoded,
            )
        })
        .collect::<Vec<_>>();
    let (code_blocks, preencoded_ht_code_blocks): (Vec<_>, Vec<_>) =
        code_blocks.into_iter().unzip();

    Ok(PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks: Some(preencoded_ht_code_blocks),
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
        code_block_width,
        code_block_height,
        width,
        height,
        sub_band_type: internal_sub_band_type(subband.sub_band_type),
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 1,
    })
}

pub(super) fn prepared_subband_from_preencoded_compact<'a>(
    subband: &'a PreencodedHtj2k97CompactSubband,
    payload: &'a [u8],
) -> Result<PreparedCompactSubband<'a>, &'static str> {
    let code_blocks = subband
        .code_blocks
        .iter()
        .map(|block| {
            Ok(PreparedCompactCodeBlock {
                data: compact_payload_slice(payload, &block.payload_range)?,
                cleanup_length: block.cleanup_length,
                refinement_length: block.refinement_length,
                num_coding_passes: block.num_coding_passes,
                num_zero_bitplanes: block.num_zero_bitplanes,
            })
        })
        .collect::<Result<Vec<_>, &'static str>>()?;

    Ok(PreparedCompactSubband {
        code_blocks,
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
    })
}

pub(super) fn compact_payload_slice<'a>(
    payload: &'a [u8],
    range: &Range<usize>,
) -> Result<&'a [u8], &'static str> {
    if range.start > range.end || range.end > payload.len() {
        return Err("HTJ2K payload range out of bounds");
    }
    Ok(&payload[range.clone()])
}

pub(super) fn zero_pixel_buffer(
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
) -> Result<Vec<u8>, &'static str> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)?;
    let len = width as usize;
    let len = len
        .checked_mul(height as usize)
        .and_then(|value| value.checked_mul(usize::from(num_components)))
        .and_then(|value| value.checked_mul(bytes_per_sample))
        .ok_or("pixel buffer dimensions overflow")?;
    Ok(vec![0; len])
}

pub(super) struct PrecomputedDwtAccelerator<'a, A: J2kEncodeStageAccelerator> {
    outputs: Vec<J2kForwardDwt53Output>,
    encode_accelerator: &'a mut A,
}

pub(super) struct PrecomputedDwt97Accelerator<'a, A: J2kEncodeStageAccelerator> {
    outputs: Vec<J2kForwardDwt97Output>,
    encode_accelerator: &'a mut A,
}

// These wrappers only replace the forward-DWT stage with caller-supplied
// coefficients. Earlier sample/color hooks and whole-subband/tile HTJ2K hooks
// keep the trait defaults so precomputed-DWT APIs cannot intercept unrelated
// encode stages.
macro_rules! forward_precomputed_encode_stage_hooks {
    () => {
        fn dispatch_report(&self) -> crate::J2kEncodeDispatchReport {
            self.encode_accelerator.dispatch_report()
        }

        fn encode_quantize_subband(
            &mut self,
            job: J2kQuantizeSubbandJob<'_>,
        ) -> Result<Option<Vec<i32>>, &'static str> {
            self.encode_accelerator.encode_quantize_subband(job)
        }

        fn encode_tier1_code_block(
            &mut self,
            job: J2kTier1CodeBlockEncodeJob<'_>,
        ) -> Result<Option<EncodedJ2kCodeBlock>, &'static str> {
            self.encode_accelerator.encode_tier1_code_block(job)
        }

        fn encode_tier1_code_blocks(
            &mut self,
            jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
        ) -> Result<Option<Vec<EncodedJ2kCodeBlock>>, &'static str> {
            self.encode_accelerator.encode_tier1_code_blocks(jobs)
        }

        fn encode_ht_code_block(
            &mut self,
            job: crate::J2kHtCodeBlockEncodeJob<'_>,
        ) -> Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
            self.encode_accelerator.encode_ht_code_block(job)
        }

        fn encode_ht_code_blocks(
            &mut self,
            jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
        ) -> Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
            self.encode_accelerator.encode_ht_code_blocks(jobs)
        }

        fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
            self.encode_accelerator
                .prefer_parallel_cpu_code_block_fallback()
        }

        fn prefer_parallel_cpu_tile_encode(&self) -> bool {
            self.encode_accelerator.prefer_parallel_cpu_tile_encode()
        }

        fn encode_packetization(
            &mut self,
            job: J2kPacketizationEncodeJob<'_>,
        ) -> Result<Option<Vec<u8>>, &'static str> {
            self.encode_accelerator.encode_packetization(job)
        }
    };
}

impl<A: J2kEncodeStageAccelerator> J2kEncodeStageAccelerator for PrecomputedDwtAccelerator<'_, A> {
    fn encode_forward_dwt53(
        &mut self,
        _job: J2kForwardDwt53Job<'_>,
    ) -> Result<Option<J2kForwardDwt53Output>, &'static str> {
        if self.outputs.is_empty() {
            return Err("precomputed DWT output exhausted");
        }

        Ok(Some(self.outputs.remove(0)))
    }

    forward_precomputed_encode_stage_hooks!();
}

impl<A: J2kEncodeStageAccelerator> J2kEncodeStageAccelerator
    for PrecomputedDwt97Accelerator<'_, A>
{
    fn encode_forward_dwt97(
        &mut self,
        _job: J2kForwardDwt97Job<'_>,
    ) -> Result<Option<J2kForwardDwt97Output>, &'static str> {
        if self.outputs.is_empty() {
            return Err("precomputed DWT output exhausted");
        }

        Ok(Some(self.outputs.remove(0)))
    }

    forward_precomputed_encode_stage_hooks!();
}
