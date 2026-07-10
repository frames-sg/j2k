// SPDX-License-Identifier: MIT OR Apache-2.0

use super::precomputed::{precomputed_97_level_count, validate_precomputed_dwt97_geometry};
use super::{
    ordered_prepared_resolution_packets, packet_descriptors_for_order,
    precinct_exponents_for_options, prepare_subband_cpu_quantized, quantize,
    split_component_resolution_packets_by_precinct, validate_irreversible_quantization_profile,
    vec, BlockCodingMode, EncodeOptions, EncodeParams, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image, PreparedPrecomputedHtj2k97Image, PreparedResolutionPacket,
    QuantStepSize, SubBandType, Vec, MAX_J2K_SPEC_COMPONENTS,
};

pub(super) fn prepare_precomputed_htj2k97_image_for_batch(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
) -> Result<PreparedPrecomputedHtj2k97Image, &'static str> {
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
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        image.bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let cb_width = 1u32 << (options.code_block_width_exp + 2);
    let cb_height = 1u32 << (options.code_block_height_exp + 2);
    let component_sampling = image
        .components
        .iter()
        .map(|component| (component.x_rsiz, component.y_rsiz))
        .collect::<Vec<_>>();
    let mut precomputed_options = options.clone();
    precomputed_options.num_decomposition_levels = num_levels;
    precomputed_options.reversible = false;
    precomputed_options.use_ht_block_coding = true;
    precomputed_options.use_mct = false;
    precomputed_options.validate_high_throughput_codestream = false;
    precomputed_options.component_sampling = Some(component_sampling.clone());
    let precinct_exponents = precinct_exponents_for_options(&precomputed_options, num_levels)?;
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
        code_block_width_exp: precomputed_options.code_block_width_exp,
        code_block_height_exp: precomputed_options.code_block_height_exp,
        num_layers: 1,
        use_mct: false,
        guard_bits,
        block_coding_mode: BlockCodingMode::HighThroughput,
        progression_order: precomputed_options.progression_order,
        write_tlm: precomputed_options.write_tlm,
        write_plt: precomputed_options.write_plt,
        write_plm: precomputed_options.write_plm,
        write_ppm: precomputed_options.write_ppm,
        write_ppt: precomputed_options.write_ppt,
        write_sop: precomputed_options.write_sop,
        write_eph: precomputed_options.write_eph,
        terminate_coding_passes: false,
        component_sampling,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents,
    };

    let component_resolution_packets = image
        .components
        .iter()
        .enumerate()
        .map(|(component_idx, component)| {
            prepared_resolution_packets_from_precomputed_97_component(
                component_idx,
                component,
                &step_sizes,
                image.bit_depth,
                guard_bits,
                cb_width,
                cb_height,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let component_resolution_packets = split_component_resolution_packets_by_precinct(
        component_resolution_packets,
        image.width,
        image.height,
        num_levels,
        &params.precinct_exponents,
    )?;
    let prepared_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, &precomputed_options)?;
    let packet_descriptors =
        packet_descriptors_for_order(&prepared_packets, 1, precomputed_options.progression_order)?;

    Ok(PreparedPrecomputedHtj2k97Image {
        params,
        quant_params,
        packet_descriptors,
        packet_count: 0,
        prepared_packets,
    })
}

pub(super) fn prepared_resolution_packets_from_precomputed_97_component(
    component_idx: usize,
    component: &PrecomputedHtj2k97Component,
    step_sizes: &[QuantStepSize],
    bit_depth: u8,
    guard_bits: u8,
    cb_width: u32,
    cb_height: u32,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    let mut packets = Vec::with_capacity(component.dwt.levels.len() + 1);
    packets.push(PreparedResolutionPacket {
        component: component_idx,
        resolution: 0,
        precinct: 0,
        subbands: vec![prepare_subband_cpu_quantized(
            &component.dwt.ll,
            component.dwt.ll_width,
            component.dwt.ll_height,
            step_sizes
                .first()
                .ok_or("irreversible quantization step missing")?,
            bit_depth,
            guard_bits,
            false,
            BlockCodingMode::HighThroughput,
            cb_width,
            cb_height,
            SubBandType::LowLow,
        )?],
    });

    for (level_idx, level) in component.dwt.levels.iter().enumerate() {
        let step_base = 1 + level_idx * 3;
        packets.push(PreparedResolutionPacket {
            component: component_idx,
            resolution: u32::try_from(level_idx + 1).map_err(|_| "resolution index exceeds u32")?,
            precinct: 0,
            subbands: vec![
                prepare_subband_cpu_quantized(
                    &level.hl,
                    level.high_width,
                    level.low_height,
                    step_sizes
                        .get(step_base)
                        .ok_or("irreversible quantization step missing")?,
                    bit_depth,
                    guard_bits,
                    false,
                    BlockCodingMode::HighThroughput,
                    cb_width,
                    cb_height,
                    SubBandType::HighLow,
                )?,
                prepare_subband_cpu_quantized(
                    &level.lh,
                    level.low_width,
                    level.high_height,
                    step_sizes
                        .get(step_base + 1)
                        .ok_or("irreversible quantization step missing")?,
                    bit_depth,
                    guard_bits,
                    false,
                    BlockCodingMode::HighThroughput,
                    cb_width,
                    cb_height,
                    SubBandType::LowHigh,
                )?,
                prepare_subband_cpu_quantized(
                    &level.hh,
                    level.high_width,
                    level.high_height,
                    step_sizes
                        .get(step_base + 2)
                        .ok_or("irreversible quantization step missing")?,
                    bit_depth,
                    guard_bits,
                    false,
                    BlockCodingMode::HighThroughput,
                    cb_width,
                    cb_height,
                    SubBandType::HighHigh,
                )?,
            ],
        });
    }

    Ok(packets)
}

pub(super) fn copy_code_block_coefficients(
    quantized: &[i32],
    width: usize,
    x0: usize,
    y0: usize,
    cbw: usize,
    cbh: usize,
) -> Vec<i32> {
    let len = cbw * cbh;
    let start = y0 * width + x0;
    if cbw == width {
        return quantized[start..start + len].to_vec();
    }

    let mut coefficients = Vec::with_capacity(len);
    for y in 0..cbh {
        let row_start = (y0 + y) * width + x0;
        coefficients.extend_from_slice(&quantized[row_start..row_start + cbw]);
    }
    coefficients
}

pub(super) fn copy_code_block_coefficients_i64(
    quantized: &[i64],
    width: usize,
    x0: usize,
    y0: usize,
    cbw: usize,
    cbh: usize,
) -> Vec<i64> {
    let len = cbw * cbh;
    let start = y0 * width + x0;
    if cbw == width {
        return quantized[start..start + len].to_vec();
    }

    let mut coefficients = Vec::with_capacity(len);
    for y in 0..cbh {
        let row_start = (y0 + y) * width + x0;
        coefficients.extend_from_slice(&quantized[row_start..row_start + cbw]);
    }
    coefficients
}

pub(super) fn coefficients_fit_i32(coefficients: &[i64]) -> bool {
    coefficients
        .iter()
        .all(|&coefficient| i32::try_from(coefficient).is_ok())
}

pub(super) fn downcast_i64_coefficients_to_i32(
    coefficients: &[i64],
) -> Result<Vec<i32>, &'static str> {
    coefficients
        .iter()
        .map(|&coefficient| {
            i32::try_from(coefficient).map_err(|_| {
                "HTJ2K/accelerated code-block encode does not support i64 coefficients"
            })
        })
        .collect()
}
