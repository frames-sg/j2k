// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{format, vec::Vec};

use j2k_native::RawBitmap;

use super::contracts::J2kEncodeValidation;
use super::lossy::usize_to_f64;
use super::samples::{raw_pixel_bytes_per_sample, J2kLosslessSamples, J2kLossySamples};
use crate::J2kError;

mod component;
pub(super) use component::{
    validate_lossless_component_roundtrip, validate_lossless_high_bit_component_roundtrip,
    validate_lossless_typed_component_roundtrip,
};
mod decode;
use self::decode::{
    decode_native_components_for_validation, decode_native_for_validation, output_validation_error,
    raw_bitmap_metadata_matches,
};
#[cfg(test)]
use self::decode::{validation_decode_error, validation_retained_capacity};

// This intentionally receives the `Vec`: validation budgets its retained
// allocation capacity, not only the initialized codestream bytes.
pub(super) fn validate_lossy_roundtrip(
    samples: J2kLossySamples<'_>,
    codestream: &Vec<u8>,
    validation: J2kEncodeValidation,
) -> Result<Option<f64>, J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(None);
    }

    let decoded =
        decode_native_for_validation(codestream, "encoded JPEG 2000 lossy validation failed")?;
    if !raw_bitmap_metadata_matches(
        &decoded,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
    ) {
        return Err(output_validation_error(
            "JPEG 2000 lossy encode failed round-trip geometry validation",
        ));
    }

    Ok(Some(psnr_from_decoded(samples, &decoded.data)?))
}

// See `validate_lossy_roundtrip` for the `Vec` ownership contract.
pub(super) fn decoded_psnr(
    samples: J2kLossySamples<'_>,
    codestream: &Vec<u8>,
) -> Result<f64, J2kError> {
    let decoded =
        decode_native_for_validation(codestream, "encoded JPEG 2000 PSNR validation failed")?;
    psnr_from_validated_bitmap(samples, &decoded)
}

fn psnr_from_validated_bitmap(
    samples: J2kLossySamples<'_>,
    decoded: &RawBitmap,
) -> Result<f64, J2kError> {
    if !raw_bitmap_metadata_matches(
        decoded,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
    ) {
        return Err(output_validation_error(
            "JPEG 2000 PSNR validation metadata mismatch",
        ));
    }
    psnr_from_decoded(samples, &decoded.data)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "PSNR is an approximate f64 validation metric over bounded sample counts"
)]
pub(super) fn psnr_from_decoded(
    samples: J2kLossySamples<'_>,
    decoded: &[u8],
) -> Result<f64, J2kError> {
    if decoded.len() != samples.data.len() {
        return Err(output_validation_error(format!(
            "JPEG 2000 lossy encode validation length mismatch: expected {} bytes, got {} bytes",
            samples.data.len(),
            decoded.len()
        )));
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

#[expect(
    clippy::cast_precision_loss,
    reason = "sample values are intentionally represented as f64 for PSNR arithmetic"
)]
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

#[expect(
    clippy::cast_possible_wrap,
    reason = "the cast deliberately reinterprets the shifted sign bit before arithmetic extension"
)]
pub(super) fn sign_extend_sample(raw: u64, bit_depth: u8) -> i64 {
    let shift = 64 - u32::from(bit_depth);
    ((raw << shift) as i64) >> shift
}

// See `validate_lossy_roundtrip` for the `Vec` ownership contract.
pub(super) fn validate_lossless_roundtrip(
    samples: J2kLosslessSamples<'_>,
    codestream: &Vec<u8>,
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }
    let decoded =
        decode_native_for_validation(codestream, "encoded JPEG 2000 lossless validation failed")?;
    if !raw_bitmap_metadata_matches(
        &decoded,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
    ) {
        return Err(output_validation_error(
            "JPEG 2000 lossless encode failed round-trip geometry validation",
        ));
    }
    if let Some(mismatch) = first_native_sample_mismatch(
        samples.data,
        &decoded.data,
        samples.bit_depth,
        samples.signed,
    ) {
        return Err(output_validation_error(format!(
            "JPEG 2000 lossless encode failed round-trip validation at sample {mismatch}"
        )));
    }
    Ok(())
}

fn first_native_sample_mismatch(
    expected: &[u8],
    actual: &[u8],
    bit_depth: u8,
    signed: bool,
) -> Option<usize> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth);
    if expected.len() != actual.len()
        || !expected.len().is_multiple_of(bytes_per_sample)
        || !actual.len().is_multiple_of(bytes_per_sample)
    {
        return Some(expected.len().min(actual.len()) / bytes_per_sample);
    }
    expected
        .chunks_exact(bytes_per_sample)
        .zip(actual.chunks_exact(bytes_per_sample))
        .position(|(expected, actual)| {
            let canonical = canonical_native_sample_bytes(expected, bit_depth, signed);
            actual != &canonical[..bytes_per_sample]
        })
}

pub(super) fn canonical_native_sample_bytes(sample: &[u8], bit_depth: u8, signed: bool) -> [u8; 8] {
    let raw = read_le_sample_value(sample, bit_depth);
    if signed {
        sign_extend_sample(raw, bit_depth).to_le_bytes()
    } else {
        raw.to_le_bytes()
    }
}

#[cfg(test)]
mod tests;
