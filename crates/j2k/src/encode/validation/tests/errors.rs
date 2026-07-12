// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;

use super::super::{
    psnr_from_validated_bitmap, raw_bitmap_metadata_matches, validation_decode_error,
    validation_retained_capacity,
};
use crate::encode::J2kLossySamples;
use crate::{BackendErrorKind, J2kError};
use j2k_native::{DecodeError, DecodingError, RawBitmap, DEFAULT_MAX_DECODE_BYTES};

#[test]
fn oversized_validation_capacity_is_a_typed_resource_error() {
    let error =
        validation_retained_capacity(DEFAULT_MAX_DECODE_BYTES + 1, "validation fixture failed")
            .expect_err("codestream capacity exceeds the decode cap");
    assert!(matches!(
        error,
        J2kError::NativeValidation {
            context: "validation fixture failed",
            source,
        } if source == crate::NativeBackendError::decode(DecodeError::AllocationTooLarge {
            what: "facade encode validation retained codestreams",
            requested: DEFAULT_MAX_DECODE_BYTES + 1,
            cap: DEFAULT_MAX_DECODE_BYTES,
        })
    ));
}

#[test]
fn interleaved_validation_requires_uniform_signedness_metadata() {
    let mut decoded = RawBitmap {
        data: vec![7],
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        component_signed: vec![false],
        num_components: 1,
        bytes_per_sample: 1,
    };
    assert!(raw_bitmap_metadata_matches(&decoded, 1, 1, 1, 8, false));

    decoded.signed = true;
    assert!(!raw_bitmap_metadata_matches(&decoded, 1, 1, 1, 8, false));
    decoded.signed = false;
    decoded.component_signed[0] = true;
    assert!(!raw_bitmap_metadata_matches(&decoded, 1, 1, 1, 8, false));
}

#[test]
fn generated_codestream_failures_keep_validation_context() {
    for source in [
        DecodeError::Decoding(DecodingError::UnexpectedEof),
        DecodeError::Decoding(DecodingError::UnsupportedFeature("fixture feature")),
    ] {
        let error = validation_decode_error(source, "generated codestream validation failed");
        assert!(matches!(
            error,
            J2kError::NativeValidation {
                context: "generated codestream validation failed",
                source: stored,
            } if stored == crate::NativeBackendError::decode(source)
        ));
    }
}

#[test]
fn psnr_validation_rejects_matching_bytes_with_wrong_metadata() {
    let samples = J2kLossySamples::new(&[7], 1, 1, 1, 8, false).expect("fixture samples");
    let decoded = RawBitmap {
        data: vec![7],
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: true,
        component_signed: vec![true],
        num_components: 1,
        bytes_per_sample: 1,
    };

    assert!(matches!(
        psnr_from_validated_bitmap(samples, &decoded),
        Err(J2kError::Backend(error))
            if error.kind() == BackendErrorKind::Validation
                && error.message() == "JPEG 2000 PSNR validation metadata mismatch"
    ));
}
