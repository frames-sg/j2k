// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact component-plane round-trip validation without reference-image copies.

use alloc::{format, vec::Vec};

use j2k_native::NativeComponentPlane;

use super::{
    canonical_native_sample_bytes, decode_native_components_for_validation, output_validation_error,
};
use crate::encode::samples::{
    raw_pixel_bytes_per_sample, J2kLosslessComponentSamples, J2kLosslessTypedComponentPlane,
    J2kLosslessTypedComponentSamples,
};
use crate::encode::J2kEncodeValidation;
use crate::J2kError;

// This intentionally receives the `Vec`: validation budgets its retained
// allocation capacity, not only the initialized codestream bytes.
pub(in crate::encode) fn validate_lossless_component_roundtrip(
    samples: J2kLosslessComponentSamples<'_>,
    codestream: &Vec<u8>,
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    validate_component_roundtrip(
        samples.width,
        samples.height,
        samples.planes.len(),
        codestream,
        validation,
        "JPEG 2000 lossless component-plane encode",
        |index| {
            let plane = &samples.planes[index];
            ExpectedPlane {
                data: plane.data,
                sampling: (plane.x_rsiz, plane.y_rsiz),
                bit_depth: samples.bit_depth,
                signed: samples.signed,
            }
        },
    )
}

// See `validate_lossless_component_roundtrip` for the `Vec` ownership contract.
pub(in crate::encode) fn validate_lossless_high_bit_component_roundtrip(
    samples: J2kLosslessComponentSamples<'_>,
    codestream: &Vec<u8>,
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    validate_component_roundtrip(
        samples.width,
        samples.height,
        samples.planes.len(),
        codestream,
        validation,
        "JPEG 2000 lossless high-bit component-plane encode",
        |index| {
            let plane = &samples.planes[index];
            ExpectedPlane {
                data: plane.data,
                sampling: (plane.x_rsiz, plane.y_rsiz),
                bit_depth: samples.bit_depth,
                signed: samples.signed,
            }
        },
    )
}

// See `validate_lossless_component_roundtrip` for the `Vec` ownership contract.
pub(in crate::encode) fn validate_lossless_typed_component_roundtrip(
    samples: J2kLosslessTypedComponentSamples<'_>,
    codestream: &Vec<u8>,
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    validate_component_roundtrip(
        samples.width,
        samples.height,
        samples.planes.len(),
        codestream,
        validation,
        "JPEG 2000 lossless typed component-plane encode",
        |index| expected_typed_plane(&samples.planes[index]),
    )
}

// See `validate_lossless_component_roundtrip` for the `Vec` ownership contract.
fn validate_component_roundtrip<'a>(
    width: u32,
    height: u32,
    component_count: usize,
    codestream: &Vec<u8>,
    validation: J2kEncodeValidation,
    operation: &'static str,
    expected_plane: impl Fn(usize) -> ExpectedPlane<'a>,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }

    let decoded = decode_native_components_for_validation(
        codestream,
        "encoded JPEG 2000 component validation failed",
    )?;
    if decoded.dimensions() != (width, height) || decoded.planes().len() != component_count {
        return Err(output_validation_error(format!(
            "{operation} failed round-trip geometry validation"
        )));
    }

    for (index, actual) in decoded.planes().iter().enumerate() {
        validate_component_plane(
            expected_plane(index),
            actual,
            (width, height),
            index,
            operation,
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ExpectedPlane<'a> {
    data: &'a [u8],
    sampling: (u8, u8),
    bit_depth: u8,
    signed: bool,
}

fn expected_typed_plane<'a>(plane: &J2kLosslessTypedComponentPlane<'a>) -> ExpectedPlane<'a> {
    ExpectedPlane {
        data: plane.data,
        sampling: (plane.x_rsiz, plane.y_rsiz),
        bit_depth: plane.bit_depth,
        signed: plane.signed,
    }
}

fn validate_component_plane(
    expected: ExpectedPlane<'_>,
    actual: &NativeComponentPlane,
    reference_dimensions: (u32, u32),
    component_index: usize,
    operation: &'static str,
) -> Result<(), J2kError> {
    let layout = validate_component_layout(
        expected,
        actual,
        reference_dimensions,
        component_index,
        operation,
    )?;
    validate_component_samples(expected, actual, layout, component_index, operation)
}

#[derive(Clone, Copy)]
struct ValidatedPlaneLayout {
    component_dimensions: (u32, u32),
    actual_dimensions: (u32, u32),
    bytes_per_sample: usize,
    reference_dimensions: (u32, u32),
}

fn validate_component_layout(
    expected: ExpectedPlane<'_>,
    actual: &NativeComponentPlane,
    reference_dimensions: (u32, u32),
    component_index: usize,
    operation: &'static str,
) -> Result<ValidatedPlaneLayout, J2kError> {
    if expected.sampling.0 == 0 || expected.sampling.1 == 0 {
        return Err(J2kError::InvalidSamples {
            what: format!("{operation} received zero sampling for component {component_index}"),
        });
    }
    if actual.bit_depth() != expected.bit_depth
        || actual.signed() != expected.signed
        || actual.sampling() != expected.sampling
    {
        return Err(output_validation_error(format!(
            "{operation} failed metadata validation for component {component_index}"
        )));
    }

    let bytes_per_sample = raw_pixel_bytes_per_sample(expected.bit_depth);
    if usize::from(actual.bytes_per_sample()) != bytes_per_sample {
        return Err(output_validation_error(format!(
            "{operation} returned an invalid native sample width for component {component_index}"
        )));
    }

    let component_dimensions = (
        reference_dimensions
            .0
            .div_ceil(u32::from(expected.sampling.0)),
        reference_dimensions
            .1
            .div_ceil(u32::from(expected.sampling.1)),
    );
    let expected_len =
        plane_byte_len(component_dimensions, bytes_per_sample, reference_dimensions)?;
    if expected.data.len() != expected_len {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "{operation} received invalid source data length for component {component_index}: expected {expected_len}, got {}",
                expected.data.len()
            ),
        });
    }
    let actual_dimensions = actual.dimensions();
    if actual_dimensions != component_dimensions && actual_dimensions != reference_dimensions {
        return Err(output_validation_error(format!(
            "{operation} returned invalid dimensions {actual_dimensions:?} for component {component_index}"
        )));
    }

    let actual_len = plane_byte_len(actual_dimensions, bytes_per_sample, reference_dimensions)?;
    if actual.data().len() != actual_len {
        return Err(output_validation_error(format!(
                "{operation} returned invalid data length for component {component_index}: expected {actual_len}, got {}",
                actual.data().len()
            )));
    }

    Ok(ValidatedPlaneLayout {
        component_dimensions,
        actual_dimensions,
        bytes_per_sample,
        reference_dimensions,
    })
}

fn plane_byte_len(
    dimensions: (u32, u32),
    bytes_per_sample: usize,
    reference_dimensions: (u32, u32),
) -> Result<usize, J2kError> {
    (dimensions.0 as usize)
        .checked_mul(dimensions.1 as usize)
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or(J2kError::DimensionOverflow {
            width: reference_dimensions.0,
            height: reference_dimensions.1,
        })
}

fn validate_component_samples(
    expected: ExpectedPlane<'_>,
    actual: &NativeComponentPlane,
    layout: ValidatedPlaneLayout,
    component_index: usize,
    operation: &'static str,
) -> Result<(), J2kError> {
    let ValidatedPlaneLayout {
        component_dimensions,
        actual_dimensions,
        bytes_per_sample,
        reference_dimensions,
    } = layout;

    let component_width = component_dimensions.0 as usize;
    let component_height = component_dimensions.1 as usize;
    for y in 0..actual_dimensions.1 as usize {
        let expected_y = if actual_dimensions == component_dimensions {
            y
        } else {
            (y / usize::from(expected.sampling.1)).min(component_height.saturating_sub(1))
        };
        for x in 0..actual_dimensions.0 as usize {
            let expected_x = if actual_dimensions == component_dimensions {
                x
            } else {
                (x / usize::from(expected.sampling.0)).min(component_width.saturating_sub(1))
            };
            let expected_index = expected_y
                .checked_mul(component_width)
                .and_then(|row| row.checked_add(expected_x))
                .ok_or(J2kError::DimensionOverflow {
                    width: reference_dimensions.0,
                    height: reference_dimensions.1,
                })?;
            let actual_index = y
                .checked_mul(actual_dimensions.0 as usize)
                .and_then(|row| row.checked_add(x))
                .ok_or(J2kError::DimensionOverflow {
                    width: reference_dimensions.0,
                    height: reference_dimensions.1,
                })?;
            if !native_sample_matches(
                expected,
                expected_index,
                actual.data(),
                actual_index,
                bytes_per_sample,
            ) {
                return Err(output_validation_error(format!(
                        "{operation} failed round-trip validation for component {component_index} sample {actual_index}"
                    )));
            }
        }
    }
    Ok(())
}

fn native_sample_matches(
    expected: ExpectedPlane<'_>,
    expected_index: usize,
    actual: &[u8],
    actual_index: usize,
    bytes_per_sample: usize,
) -> bool {
    let Some(expected_start) = expected_index.checked_mul(bytes_per_sample) else {
        return false;
    };
    let Some(actual_start) = actual_index.checked_mul(bytes_per_sample) else {
        return false;
    };
    let Some(expected_end) = expected_start.checked_add(bytes_per_sample) else {
        return false;
    };
    let Some(actual_end) = actual_start.checked_add(bytes_per_sample) else {
        return false;
    };
    let Some(expected_sample) = expected.data.get(expected_start..expected_end) else {
        return false;
    };
    let Some(actual_sample) = actual.get(actual_start..actual_end) else {
        return false;
    };
    let canonical =
        canonical_native_sample_bytes(expected_sample, expected.bit_depth, expected.signed);
    actual_sample == &canonical[..bytes_per_sample]
}

#[cfg(test)]
mod tests {
    use super::{native_sample_matches, ExpectedPlane};

    #[test]
    fn native_sample_comparison_masks_unused_unsigned_bits() {
        let expected = ExpectedPlane {
            data: &[0xff, 0xff],
            sampling: (1, 1),
            bit_depth: 12,
            signed: false,
        };
        assert!(native_sample_matches(expected, 0, &[0xff, 0x0f], 0, 2));
    }

    #[test]
    fn native_sample_comparison_sign_extends_non_byte_aligned_values() {
        let expected = ExpectedPlane {
            data: &[0x00, 0x08],
            sampling: (1, 1),
            bit_depth: 12,
            signed: true,
        };
        assert!(native_sample_matches(expected, 0, &[0x00, 0xf8], 0, 2));
    }
}
