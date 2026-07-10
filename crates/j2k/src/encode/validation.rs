// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{format, string::ToString, vec::Vec};

use j2k_native::{DecodeSettings, Image};

use super::contracts::{J2kEncodeValidation, MAX_RAW_PIXEL_ENCODE_BIT_DEPTH};
use super::lossy::usize_to_f64;
use super::samples::{
    raw_pixel_bytes_per_sample, J2kLosslessComponentPlane, J2kLosslessComponentSamples,
    J2kLosslessSamples, J2kLosslessTypedComponentPlane, J2kLosslessTypedComponentSamples,
    J2kLossySamples,
};
use crate::J2kError;

pub(super) fn validate_lossy_roundtrip(
    samples: J2kLossySamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<Option<f64>, J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(None);
    }

    let decoded = Image::new(codestream, &DecodeSettings::default())
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?
        .decode_native()
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;

    if decoded.width != samples.width
        || decoded.height != samples.height
        || decoded.num_components != samples.components
        || decoded.bit_depth != samples.bit_depth
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossy encode failed round-trip geometry validation".to_string(),
        });
    }

    Ok(Some(psnr_from_decoded(samples, &decoded.data)?))
}

pub(super) fn decoded_psnr(
    samples: J2kLossySamples<'_>,
    codestream: &[u8],
) -> Result<f64, J2kError> {
    let decoded = Image::new(codestream, &DecodeSettings::default())
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?
        .decode_native()
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;
    psnr_from_decoded(samples, &decoded.data)
}

#[allow(clippy::cast_precision_loss)]
pub(super) fn psnr_from_decoded(
    samples: J2kLossySamples<'_>,
    decoded: &[u8],
) -> Result<f64, J2kError> {
    if decoded.len() != samples.data.len() {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "JPEG 2000 lossy encode validation length mismatch: expected {} bytes, got {} bytes",
                samples.data.len(),
                decoded.len()
            ),
        });
    }
    let bytes_per_sample = raw_pixel_bytes_per_sample(samples.bit_depth);
    let sample_count = samples.data.len() / bytes_per_sample;
    let mut squared_error = 0.0f64;
    for sample_idx in 0..sample_count {
        let original = sample_value(samples.data, sample_idx, samples.bit_depth, samples.signed);
        let decoded = sample_value(decoded, sample_idx, samples.bit_depth, samples.signed);
        let error = original - decoded;
        squared_error += error * error;
    }
    if squared_error == 0.0 {
        return Ok(f64::INFINITY);
    }
    let mse = squared_error / usize_to_f64(sample_count);
    let peak = ((1_u64 << u32::from(samples.bit_depth)) - 1) as f64;
    Ok(10.0 * ((peak * peak) / mse).log10())
}

#[allow(clippy::cast_precision_loss)]
pub(super) fn sample_value(data: &[u8], sample_idx: usize, bit_depth: u8, signed: bool) -> f64 {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth);
    let byte_idx = sample_idx * bytes_per_sample;
    let raw = read_le_sample_value(&data[byte_idx..byte_idx + bytes_per_sample], bit_depth);
    if signed {
        sign_extend_sample(raw, bit_depth) as f64
    } else {
        raw as f64
    }
}

pub(super) fn read_le_sample_value(bytes: &[u8], bit_depth: u8) -> u64 {
    let mut raw = 0_u64;
    for (shift, byte) in bytes.iter().enumerate() {
        raw |= u64::from(*byte) << (shift * 8);
    }
    let mask = (1_u64 << bit_depth) - 1;
    raw & mask
}

pub(super) fn sign_extend_sample(raw: u64, bit_depth: u8) -> i64 {
    let shift = 64 - u32::from(bit_depth);
    ((raw << shift) as i64) >> shift
}

pub(super) fn validate_lossless_roundtrip(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }
    if samples.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        let header = j2k_native::inspect_j2k_codestream_header(codestream).map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;
        if header.dimensions != (samples.width, samples.height)
            || header.components != samples.components
            || header.bit_depth != samples.bit_depth
            || header
                .component_info
                .iter()
                .any(|component| component.signed != samples.signed)
        {
            return Err(J2kError::InvalidSamples {
                what: "JPEG 2000 high-bit lossless encode failed metadata validation".to_string(),
            });
        }
        return Ok(());
    }

    let decoded = Image::new(codestream, &DecodeSettings::default())
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?
        .decode_native()
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;

    if decoded.width != samples.width
        || decoded.height != samples.height
        || decoded.num_components != samples.components
        || decoded.bit_depth != samples.bit_depth
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossless encode failed round-trip geometry validation".to_string(),
        });
    }
    if decoded.data != samples.data {
        let mismatch = decoded
            .data
            .iter()
            .zip(samples.data.iter())
            .position(|(actual, expected)| actual != expected);
        return Err(J2kError::InvalidSamples {
            what: match mismatch {
            Some(index) => format!(
                "JPEG 2000 lossless encode failed round-trip validation at byte {index}: expected {}, got {}",
                samples.data[index], decoded.data[index]
            ),
            None => format!(
                "JPEG 2000 lossless encode failed round-trip validation: expected {} bytes, got {} bytes",
                samples.data.len(),
                decoded.data.len()
            ),
        }});
    }
    Ok(())
}

pub(super) fn validate_lossless_component_roundtrip(
    samples: J2kLosslessComponentSamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }

    let image = Image::new(codestream, &DecodeSettings::default()).map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;
    let mut context = j2k_native::DecoderContext::default();
    let decoded = image
        .decode_components_with_context(&mut context)
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;

    if decoded.dimensions() != (samples.width, samples.height)
        || decoded.planes().len() != samples.planes.len()
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossless component-plane encode failed round-trip geometry validation"
                .to_string(),
        });
    }

    for (index, (expected, actual)) in samples
        .planes
        .iter()
        .zip(decoded.planes().iter())
        .enumerate()
    {
        let expected_sampling = (expected.x_rsiz, expected.y_rsiz);
        if actual.bit_depth() != samples.bit_depth
            || actual.signed() != samples.signed
            || actual.sampling() != expected_sampling
        {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless component-plane encode failed metadata validation for component {index}"
                ),
            });
        }
        if expected_sampling == (1, 1) {
            validate_full_resolution_component_samples(samples, expected, actual.samples(), index)?;
        }
    }
    Ok(())
}

pub(super) fn validate_lossless_high_bit_component_roundtrip(
    samples: J2kLosslessComponentSamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }

    let image = Image::new(codestream, &DecodeSettings::default()).map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;
    let decoded = image.decode_native_components().map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;

    if decoded.dimensions() != (samples.width, samples.height)
        || decoded.planes().len() != samples.planes.len()
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossless high-bit component-plane encode failed round-trip geometry validation"
                .to_string(),
        });
    }

    for (index, (expected, actual)) in samples
        .planes
        .iter()
        .zip(decoded.planes().iter())
        .enumerate()
    {
        if actual.bit_depth() != samples.bit_depth
            || actual.signed() != samples.signed
            || actual.sampling() != (expected.x_rsiz, expected.y_rsiz)
            || actual.data() != expected.data
        {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless high-bit component-plane encode failed validation for component {index}"
                ),
            });
        }
    }
    Ok(())
}

pub(super) fn validate_lossless_typed_component_roundtrip(
    samples: J2kLosslessTypedComponentSamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }
    if samples.max_bit_depth() > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return validate_lossless_high_bit_typed_component_roundtrip(
            samples, codestream, validation,
        );
    }

    let image = Image::new(codestream, &DecodeSettings::default()).map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;
    let mut context = j2k_native::DecoderContext::default();
    let decoded = image
        .decode_components_with_context(&mut context)
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;

    if decoded.dimensions() != (samples.width, samples.height)
        || decoded.planes().len() != samples.planes.len()
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossless typed component-plane encode failed round-trip geometry validation"
                .to_string(),
        });
    }

    for (index, (expected, actual)) in samples
        .planes
        .iter()
        .zip(decoded.planes().iter())
        .enumerate()
    {
        let expected_sampling = (expected.x_rsiz, expected.y_rsiz);
        if actual.bit_depth() != expected.bit_depth
            || actual.signed() != expected.signed
            || actual.sampling() != expected_sampling
        {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless typed component-plane encode failed metadata validation for component {index}"
                ),
            });
        }
        if expected_sampling == (1, 1) {
            validate_full_resolution_typed_component_samples(expected, actual.samples(), index)?;
        }
    }
    Ok(())
}

pub(super) fn validate_lossless_high_bit_typed_component_roundtrip(
    samples: J2kLosslessTypedComponentSamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }

    let image = Image::new(codestream, &DecodeSettings::default()).map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;
    let decoded = image.decode_native_components().map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;

    if decoded.dimensions() != (samples.width, samples.height)
        || decoded.planes().len() != samples.planes.len()
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossless high-bit typed component-plane encode failed round-trip geometry validation"
                .to_string(),
        });
    }

    for (index, (expected, actual)) in samples
        .planes
        .iter()
        .zip(decoded.planes().iter())
        .enumerate()
    {
        let expected_data = canonical_native_typed_component_bytes_for_reference_grid(
            expected,
            samples.width,
            samples.height,
        )?;
        if actual.bit_depth() != expected.bit_depth
            || actual.signed() != expected.signed
            || actual.sampling() != (expected.x_rsiz, expected.y_rsiz)
            || actual.data() != expected_data.as_slice()
        {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless high-bit typed component-plane encode failed validation for component {index}"
                ),
            });
        }
    }
    Ok(())
}

pub(super) fn canonical_native_typed_component_bytes_for_reference_grid(
    plane: &J2kLosslessTypedComponentPlane<'_>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, J2kError> {
    let component_bytes = canonical_native_typed_component_bytes(plane)?;
    if (plane.x_rsiz, plane.y_rsiz) == (1, 1) {
        return Ok(component_bytes);
    }

    let bytes_per_sample = raw_pixel_bytes_per_sample(plane.bit_depth);
    let component_width = width.div_ceil(u32::from(plane.x_rsiz)) as usize;
    let component_height = height.div_ceil(u32::from(plane.y_rsiz)) as usize;
    let output_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|sample_count| sample_count.checked_mul(bytes_per_sample))
        .ok_or(J2kError::DimensionOverflow { width, height })?;
    let mut out = Vec::with_capacity(output_len);

    for y in 0..height as usize {
        let component_y = (y / usize::from(plane.y_rsiz)).min(component_height.saturating_sub(1));
        for x in 0..width as usize {
            let component_x =
                (x / usize::from(plane.x_rsiz)).min(component_width.saturating_sub(1));
            let component_idx = component_y
                .checked_mul(component_width)
                .and_then(|offset| offset.checked_add(component_x))
                .ok_or(J2kError::DimensionOverflow { width, height })?;
            let start = component_idx
                .checked_mul(bytes_per_sample)
                .ok_or(J2kError::DimensionOverflow { width, height })?;
            let end = start
                .checked_add(bytes_per_sample)
                .ok_or(J2kError::DimensionOverflow { width, height })?;
            out.extend_from_slice(component_bytes.get(start..end).ok_or_else(|| {
                J2kError::InvalidSamples {
                    what: "JPEG 2000 typed component-plane canonicalization length mismatch"
                        .to_string(),
                }
            })?);
        }
    }

    Ok(out)
}

pub(super) fn canonical_native_typed_component_bytes(
    plane: &J2kLosslessTypedComponentPlane<'_>,
) -> Result<Vec<u8>, J2kError> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(plane.bit_depth);
    let mut out = Vec::with_capacity(plane.data.len());
    for sample in plane.data.chunks_exact(bytes_per_sample) {
        let raw = read_le_sample_value(sample, plane.bit_depth);
        if plane.signed {
            let value = sign_extend_sample(raw, plane.bit_depth);
            if plane.bit_depth <= 8 {
                out.push((value as i8) as u8);
            } else if plane.bit_depth <= 16 {
                out.extend_from_slice(&(value as i16).to_le_bytes());
            } else {
                let bytes = value.to_le_bytes();
                out.extend_from_slice(&bytes[..bytes_per_sample]);
            }
        } else if plane.bit_depth <= 8 {
            out.push(raw as u8);
        } else if plane.bit_depth <= 16 {
            out.extend_from_slice(&(raw as u16).to_le_bytes());
        } else {
            let bytes = raw.to_le_bytes();
            out.extend_from_slice(&bytes[..bytes_per_sample]);
        }
    }
    if out.len() != plane.data.len() {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 typed component-plane canonicalization length mismatch".to_string(),
        });
    }
    Ok(out)
}

pub(super) fn validate_full_resolution_component_samples(
    samples: J2kLosslessComponentSamples<'_>,
    expected: &J2kLosslessComponentPlane<'_>,
    actual: &[f32],
    component_index: usize,
) -> Result<(), J2kError> {
    let expected_samples = (samples.width as usize)
        .checked_mul(samples.height as usize)
        .ok_or(J2kError::DimensionOverflow {
            width: samples.width,
            height: samples.height,
        })?;
    if actual.len() < expected_samples {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "JPEG 2000 lossless component-plane encode failed validation for component {component_index}: expected {expected_samples} samples, got {}",
                actual.len()
            ),
        });
    }
    for (sample_index, actual_sample) in actual.iter().take(expected_samples).enumerate() {
        let expected_sample = sample_value(
            expected.data,
            sample_index,
            samples.bit_depth,
            samples.signed,
        );
        if (f64::from(actual_sample.round()) - expected_sample).abs() > f64::EPSILON {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless component-plane encode failed validation for component {component_index} sample {sample_index}: expected {expected_sample}, got {}",
                    actual_sample.round()
                ),
            });
        }
    }
    Ok(())
}

pub(super) fn validate_full_resolution_typed_component_samples(
    expected: &J2kLosslessTypedComponentPlane<'_>,
    actual: &[f32],
    component_index: usize,
) -> Result<(), J2kError> {
    let expected_samples = expected.data.len() / raw_pixel_bytes_per_sample(expected.bit_depth);
    if actual.len() < expected_samples {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "JPEG 2000 lossless typed component-plane encode failed validation for component {component_index}: expected {expected_samples} samples, got {}",
                actual.len()
            ),
        });
    }
    for (sample_index, actual_sample) in actual.iter().take(expected_samples).enumerate() {
        let expected_sample = sample_value(
            expected.data,
            sample_index,
            expected.bit_depth,
            expected.signed,
        );
        if (f64::from(actual_sample.round()) - expected_sample).abs() > f64::EPSILON {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless typed component-plane encode failed validation for component {component_index} sample {sample_index}: expected {expected_sample}, got {}",
                    actual_sample.round()
                ),
            });
        }
    }
    Ok(())
}
