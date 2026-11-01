// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    DecodeError, DecodingError, J2kCodeBlockSegment, J2kCodeBlockStyle, J2kOwnedCodeBlockBatchJob,
    J2kSubBandType,
};

use super::validate_classic_job;
use crate::Error;

/// An 8 x 8 block of a sub-band with four bit-planes, at most ten passes.
fn job(
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
    strict: bool,
) -> J2kOwnedCodeBlockBatchJob {
    J2kOwnedCodeBlockBatchJob {
        output_x: 0,
        output_y: 0,
        data: vec![0; 4],
        segments: vec![J2kCodeBlockSegment {
            data_offset: 0,
            data_length: 4,
            start_coding_pass: 0,
            end_coding_pass: number_of_coding_passes,
            use_arithmetic: true,
        }],
        width: 8,
        height: 8,
        output_stride: 8,
        missing_bit_planes,
        number_of_coding_passes,
        total_bitplanes: 4,
        roi_shift: 0,
        sub_band_type: J2kSubBandType::LowLow,
        style: J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        },
        strict,
        dequantization_step: 1.0,
    }
}

/// The native decode error behind `error`, if any.
fn native(error: &Error) -> Option<&DecodeError> {
    let mut next: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(current) = next {
        if let Some(native) = current.downcast_ref::<DecodeError>() {
            return Some(native);
        }
        next = current.source();
    }
    None
}

#[test]
fn strict_code_blocks_with_codestream_errors_fail_with_the_native_error() {
    assert!(validate_classic_job(&job(0, 10, true), 4).is_ok());
    for (missing_bit_planes, passes, expected) in [
        (0, 11, DecodingError::TooManyCodingPasses),
        (3, 2, DecodingError::TooManyCodingPasses),
        (4, 1, DecodingError::TooManyCodingPasses),
        (5, 1, DecodingError::InvalidBitplaneCount),
    ] {
        let error = validate_classic_job(&job(missing_bit_planes, passes, true), 4)
            .expect_err("a codestream error");
        assert_eq!(
            native(&error),
            Some(&DecodeError::Decoding(expected)),
            "{missing_bit_planes} missing bit-planes, {passes} passes"
        );
    }
}

/// A sub-band with no bit-planes (guard bits 0, exponent 1) lies outside
/// what the device decodes, yet strict decodes still report the
/// codestream error the CPU reports.
#[test]
fn strict_code_blocks_of_a_sub_band_without_bit_planes_fail_with_the_native_error() {
    for (missing_bit_planes, expected) in [
        (0, DecodingError::TooManyCodingPasses),
        (1, DecodingError::InvalidBitplaneCount),
    ] {
        let job = J2kOwnedCodeBlockBatchJob {
            total_bitplanes: 0,
            ..job(missing_bit_planes, 1, true)
        };
        let error = validate_classic_job(&job, 4).expect_err("a codestream error");
        assert_eq!(
            native(&error),
            Some(&DecodeError::Decoding(expected)),
            "{missing_bit_planes} missing bit-planes"
        );
    }
    let lenient = J2kOwnedCodeBlockBatchJob {
        total_bitplanes: 0,
        ..job(0, 1, false)
    };
    let error = validate_classic_job(&lenient, 4).expect_err("unsupported on the device");
    assert!(
        matches!(error, Error::UnsupportedCudaRequest { .. }),
        "{error:?}"
    );
}

#[test]
fn lenient_code_blocks_past_their_bit_planes_stay_unsupported() {
    for (missing_bit_planes, passes) in [(0, 11), (4, 1), (5, 1)] {
        let error = validate_classic_job(&job(missing_bit_planes, passes, false), 4)
            .expect_err("unsupported on the device");
        assert!(
            matches!(error, Error::UnsupportedCudaRequest { .. }),
            "{error:?}"
        );
    }
}
