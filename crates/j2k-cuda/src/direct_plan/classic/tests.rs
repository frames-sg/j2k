// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    decode_j2k_code_block_scalar, DecodeError, DecodingError, J2kCodeBlockDecodeJob,
    J2kCodeBlockSegment, J2kCodeBlockStyle, J2kOwnedCodeBlockBatchJob, J2kSubBandType,
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

fn native_scalar_error(job: &J2kOwnedCodeBlockBatchJob) -> Option<DecodeError> {
    let mut output = vec![0.0; job.output_stride * job.height as usize];
    decode_j2k_code_block_scalar(
        J2kCodeBlockDecodeJob {
            data: &job.data,
            segments: &job.segments,
            width: job.width,
            height: job.height,
            output_stride: job.output_stride,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            total_bitplanes: job.total_bitplanes,
            roi_shift: job.roi_shift,
            sub_band_type: job.sub_band_type,
            style: job.style,
            strict: job.strict,
            dequantization_step: job.dequantization_step,
        },
        &mut output,
    )
    .err()
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

/// Native classic decode accepts at most 63 coded bit-planes, while the
/// device accepts 31. Strict planning keeps those two rejections distinct.
#[test]
fn strict_roi_shift_past_the_native_bitplane_limit_fails_with_the_native_error() {
    let native_limit = J2kOwnedCodeBlockBatchJob {
        total_bitplanes: 31,
        roi_shift: 32,
        ..job(0, 1, true)
    };
    assert!(!matches!(
        native_scalar_error(&native_limit),
        Some(DecodeError::Decoding(DecodingError::TooManyBitplanes))
    ));
    let error = validate_classic_job(&native_limit, 4).expect_err("past the device limit");
    assert!(
        matches!(error, Error::UnsupportedCudaRequest { .. }),
        "{error:?}"
    );

    for (roi_shift, missing_bit_planes) in [(33, 70), (u8::MAX, 0)] {
        let job = J2kOwnedCodeBlockBatchJob {
            total_bitplanes: 31,
            roi_shift,
            ..job(missing_bit_planes, 1, true)
        };
        assert_eq!(
            native_scalar_error(&job),
            Some(DecodeError::Decoding(DecodingError::TooManyBitplanes)),
            "native scalar, 31 bit-planes plus ROI shift {roi_shift}"
        );
        let error = validate_classic_job(&job, 4).expect_err("too many coded bit-planes");
        assert_eq!(
            native(&error),
            Some(&DecodeError::Decoding(DecodingError::TooManyBitplanes)),
            "31 bit-planes plus ROI shift {roi_shift}"
        );
    }

    let device_limit = J2kOwnedCodeBlockBatchJob {
        total_bitplanes: 31,
        roi_shift: 1,
        ..job(0, 1, true)
    };
    let error = validate_classic_job(&device_limit, 4).expect_err("past the device limit");
    assert!(
        matches!(error, Error::UnsupportedCudaRequest { .. }),
        "{error:?}"
    );

    let lenient = J2kOwnedCodeBlockBatchJob {
        strict: false,
        roi_shift: u8::MAX,
        ..device_limit
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
