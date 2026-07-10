// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    fdwt, quantize, raw_pixel_bytes_per_sample, read_le_sample_value, sign_extend_sample,
    DwtDecomposition, EncodeComponentSampleInfo, IrreversibleQuantizationSubbandScales,
    J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardDwt53Level, J2kForwardDwt53Output,
    J2kForwardDwt97Job, J2kForwardDwt97Level, J2kForwardDwt97Output, J2kForwardIctJob,
    J2kForwardRctJob, QuantStepSize, Vec, MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES,
    MAX_RAW_PIXEL_ENCODE_BIT_DEPTH, MAX_REVERSIBLE_NO_QUANT_EXPONENT,
    MAX_REVERSIBLE_NO_QUANT_GUARD_BITS,
};

pub(super) fn try_encode_forward_rct(
    components: &mut [Vec<f32>],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bool, &'static str> {
    debug_assert!(components.len() >= 3);
    let (plane0, rest) = components.split_at_mut(1);
    let (plane1, plane2) = rest.split_at_mut(1);
    accelerator.encode_forward_rct(J2kForwardRctJob {
        plane0: &mut plane0[0],
        plane1: &mut plane1[0],
        plane2: &mut plane2[0],
    })
}

pub(super) fn try_encode_forward_ict(
    components: &mut [Vec<f32>],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bool, &'static str> {
    debug_assert!(components.len() >= 3);
    let (plane0, rest) = components.split_at_mut(1);
    let (plane1, plane2) = rest.split_at_mut(1);
    accelerator.encode_forward_ict(J2kForwardIctJob {
        plane0: &mut plane0[0],
        plane1: &mut plane1[0],
        plane2: &mut plane2[0],
    })
}

pub(super) fn encode_forward_dwt(
    component: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
    reversible: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<DwtDecomposition, &'static str> {
    if reversible {
        if let Some(output) = accelerator.encode_forward_dwt53(J2kForwardDwt53Job {
            samples: component,
            width,
            height,
            num_levels,
        })? {
            return convert_forward_dwt53_output(output);
        }
    } else if let Some(output) = accelerator.encode_forward_dwt97(J2kForwardDwt97Job {
        samples: component,
        width,
        height,
        num_levels,
    })? {
        return convert_forward_dwt97_output(output);
    }

    Ok(fdwt::forward_dwt(
        component, width, height, num_levels, reversible,
    ))
}

pub(super) fn convert_forward_dwt53_output(
    output: J2kForwardDwt53Output,
) -> Result<DwtDecomposition, &'static str> {
    validate_band_len(output.ll.len(), output.ll_width, output.ll_height)?;
    let mut levels = Vec::with_capacity(output.levels.len());
    for level in output.levels {
        validate_dwt53_level(&level)?;
        levels.push(fdwt::DwtLevel {
            hl: level.hl,
            lh: level.lh,
            hh: level.hh,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
        });
    }
    Ok(DwtDecomposition {
        ll: output.ll,
        ll_width: output.ll_width,
        ll_height: output.ll_height,
        levels,
    })
}

pub(super) fn convert_forward_dwt97_output(
    output: J2kForwardDwt97Output,
) -> Result<DwtDecomposition, &'static str> {
    validate_band_len(output.ll.len(), output.ll_width, output.ll_height)?;
    let mut levels = Vec::with_capacity(output.levels.len());
    for level in output.levels {
        validate_dwt97_level(&level)?;
        levels.push(fdwt::DwtLevel {
            hl: level.hl,
            lh: level.lh,
            hh: level.hh,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
        });
    }
    Ok(DwtDecomposition {
        ll: output.ll,
        ll_width: output.ll_width,
        ll_height: output.ll_height,
        levels,
    })
}

pub(super) fn validate_dwt53_level(level: &J2kForwardDwt53Level) -> Result<(), &'static str> {
    validate_band_len(level.hl.len(), level.high_width, level.low_height)?;
    validate_band_len(level.lh.len(), level.low_width, level.high_height)?;
    validate_band_len(level.hh.len(), level.high_width, level.high_height)?;
    Ok(())
}

pub(super) fn validate_dwt97_level(level: &J2kForwardDwt97Level) -> Result<(), &'static str> {
    validate_band_len(level.hl.len(), level.high_width, level.low_height)?;
    validate_band_len(level.lh.len(), level.low_width, level.high_height)?;
    validate_band_len(level.hh.len(), level.high_width, level.high_height)?;
    Ok(())
}

pub(super) fn validate_band_len(
    actual: usize,
    width: u32,
    height: u32,
) -> Result<(), &'static str> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or("accelerated DWT output dimensions overflow")?;
    if actual != expected {
        return Err("accelerated DWT output length mismatch");
    }
    Ok(())
}

pub(super) fn validate_deinterleaved_components(
    components: Vec<Vec<f32>>,
    num_components: u16,
    num_pixels: usize,
) -> Result<Vec<Vec<f32>>, &'static str> {
    if components.len() != usize::from(num_components) {
        return Err("accelerated deinterleave component count mismatch");
    }
    if components
        .iter()
        .any(|component| component.len() != num_pixels)
    {
        return Err("accelerated deinterleave component length mismatch");
    }
    Ok(components)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the codec float domain intentionally receives bounded integer samples or metadata at this rounding boundary"
)]
pub(super) fn component_plane_to_f32(
    data: &[u8],
    width: u32,
    height: u32,
    bit_depth: u8,
    signed: bool,
) -> Result<Vec<f32>, &'static str> {
    let sample_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or("image dimensions overflow")?;
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)?;
    let expected_len = sample_count
        .checked_mul(bytes_per_sample)
        .ok_or("image dimensions overflow")?;
    if data.len() != expected_len {
        return Err("component plane data length mismatch");
    }

    let unsigned_offset = if signed {
        0.0
    } else {
        (1_u64 << (u32::from(bit_depth) - 1)) as f32
    };
    Ok(data
        .chunks_exact(bytes_per_sample)
        .map(|sample| {
            let raw = read_le_sample_value(sample, bit_depth);
            if signed {
                sign_extend_sample(raw, bit_depth) as f32
            } else {
                raw as f32 - unsigned_offset
            }
        })
        .collect())
}

pub(super) fn forward_dwt53_output_from_decomposition(
    decomposition: DwtDecomposition,
) -> J2kForwardDwt53Output {
    J2kForwardDwt53Output {
        ll: decomposition.ll,
        ll_width: decomposition.ll_width,
        ll_height: decomposition.ll_height,
        levels: decomposition
            .levels
            .into_iter()
            .map(|level| {
                let width = level.low_width + level.high_width;
                let height = level.low_height + level.high_height;
                J2kForwardDwt53Level {
                    hl: level.hl,
                    lh: level.lh,
                    hh: level.hh,
                    width,
                    height,
                    low_width: level.low_width,
                    low_height: level.low_height,
                    high_width: level.high_width,
                    high_height: level.high_height,
                }
            })
            .collect(),
    }
}

pub(super) fn validate_component_sampling_dwt_geometry(
    decompositions: &[DwtDecomposition],
    reference_width: u32,
    reference_height: u32,
    component_sampling: &[(u8, u8)],
) -> Result<(), &'static str> {
    if decompositions.len() != component_sampling.len() {
        return Err("component sampling count does not match component count");
    }
    for (decomposition, &(x_rsiz, y_rsiz)) in decompositions.iter().zip(component_sampling) {
        let expected_width = reference_width.div_ceil(u32::from(x_rsiz.max(1)));
        let expected_height = reference_height.div_ceil(u32::from(y_rsiz.max(1)));
        if dwt_decomposition_dimensions(decomposition) != (expected_width, expected_height) {
            return Err("component sampling requires component-sized DWT geometry");
        }
    }
    Ok(())
}

pub(super) fn dwt_decomposition_dimensions(decomposition: &DwtDecomposition) -> (u32, u32) {
    decomposition
        .levels
        .last()
        .map_or((decomposition.ll_width, decomposition.ll_height), |level| {
            (
                level.low_width + level.high_width,
                level.low_height + level.high_height,
            )
        })
}

pub(super) fn validate_component_sample_info(
    component_sample_info: &[EncodeComponentSampleInfo],
    num_components: usize,
) -> Result<(), &'static str> {
    if component_sample_info.is_empty() {
        return Ok(());
    }
    if component_sample_info.len() != num_components {
        return Err("component sample metadata count does not match component count");
    }
    if component_sample_info
        .iter()
        .any(|info| info.bit_depth == 0 || info.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH)
    {
        return Err("unsupported bit depth");
    }
    Ok(())
}

pub(super) fn component_step_sizes(
    component_sample_info: &[EncodeComponentSampleInfo],
    num_levels: u8,
    reversible: bool,
    guard_bits: u8,
    quantization_scale: f32,
    subband_scales: IrreversibleQuantizationSubbandScales,
) -> Vec<Vec<QuantStepSize>> {
    component_sample_info
        .iter()
        .map(|info| {
            quantize::compute_step_sizes_with_irreversible_profile(
                info.bit_depth,
                num_levels,
                reversible,
                guard_bits,
                quantization_scale,
                subband_scales,
            )
        })
        .collect()
}

pub(super) fn reversible_guard_bits_for_marker_limit(
    bit_depth: u8,
    num_levels: u8,
    requested_guard_bits: u8,
) -> Result<u8, &'static str> {
    if requested_guard_bits > MAX_REVERSIBLE_NO_QUANT_GUARD_BITS {
        return Err("reversible guard bits exceed the Part 1 marker field");
    }
    let max_reversible_gain = if num_levels == 0 { 0 } else { 2 };
    let requested_bitplanes = u16::from(requested_guard_bits)
        .checked_add(u16::from(bit_depth))
        .and_then(|value| value.checked_add(max_reversible_gain))
        .and_then(|value| value.checked_sub(1))
        .ok_or("reversible no-quantization bitplane count underflows")?;
    if requested_bitplanes > MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES {
        return Err("25-38 bit reversible encode exceeds the current no-quantization guard/exponent signaling limit");
    }
    let min_guard_bits = requested_bitplanes.saturating_sub(MAX_REVERSIBLE_NO_QUANT_EXPONENT - 1);
    let guard_bits = requested_guard_bits
        .max(u8::try_from(min_guard_bits).map_err(|_| "reversible guard bits exceed u8")?);
    if guard_bits > MAX_REVERSIBLE_NO_QUANT_GUARD_BITS {
        return Err("reversible guard bits exceed the Part 1 marker field");
    }
    Ok(guard_bits)
}

pub(super) fn adjust_component_step_sizes_for_guard_delta(
    component_step_sizes: &mut [Vec<QuantStepSize>],
    guard_delta: u8,
) -> Result<(), &'static str> {
    for step_sizes in component_step_sizes {
        adjust_reversible_step_sizes_for_guard_delta(step_sizes, guard_delta)?;
    }
    Ok(())
}

pub(super) fn adjust_reversible_step_sizes_for_guard_delta(
    step_sizes: &mut [QuantStepSize],
    guard_delta: u8,
) -> Result<(), &'static str> {
    let guard_delta = u16::from(guard_delta);
    for step in step_sizes {
        step.exponent = step
            .exponent
            .checked_sub(guard_delta)
            .ok_or("reversible no-quantization exponent underflows guard-bit adjustment")?;
        if step.exponent > MAX_REVERSIBLE_NO_QUANT_EXPONENT {
            return Err("reversible no-quantization exponent exceeds the Part 1 marker field");
        }
    }
    Ok(())
}
