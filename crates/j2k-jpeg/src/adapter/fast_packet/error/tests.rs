// SPDX-License-Identifier: MIT OR Apache-2.0

use core::error::Error as _;

use j2k_core::CodecError;

use super::{FastPacketError, TableKind};
use crate::error::{JpegError, UnsupportedReason};
use crate::info::{ColorSpace, Rect, SofKind};

#[derive(Clone, Copy)]
enum Classification {
    None,
    Capability,
    Truncated,
    NotImplemented,
    Unsupported,
    Buffer,
}

fn assert_classification(error: &FastPacketError, expected: Classification) {
    assert_eq!(
        error.is_truncated(),
        matches!(expected, Classification::Truncated),
        "{error:?}"
    );
    assert_eq!(
        error.is_not_implemented(),
        matches!(
            expected,
            Classification::Capability | Classification::NotImplemented
        ),
        "{error:?}"
    );
    assert_eq!(
        error.is_unsupported(),
        matches!(
            expected,
            Classification::Capability | Classification::Unsupported
        ),
        "{error:?}"
    );
    assert_eq!(
        error.is_buffer_error(),
        matches!(expected, Classification::Buffer),
        "{error:?}"
    );
}

#[test]
fn direct_classification_distinguishes_capability_truncation_and_input_errors() {
    let capability_errors = [
        FastPacketError::UnsupportedSof(SofKind::Progressive8),
        FastPacketError::UnsupportedColorSpace(ColorSpace::Cmyk),
        FastPacketError::UnsupportedSampling,
        FastPacketError::UnsupportedComponentOrder,
        FastPacketError::EntropyMarkerUnsupported { marker: 0xD0 },
    ];
    for error in capability_errors {
        assert!(error.is_capability_mismatch());
        assert_classification(&error, Classification::Capability);
    }

    assert_classification(
        &FastPacketError::TruncatedEntropy,
        Classification::Truncated,
    );
    assert_classification(&FastPacketError::MissingScan, Classification::None);
}

#[test]
fn decode_classification_delegates_to_each_typed_jpeg_category() {
    assert_classification(
        &FastPacketError::Decode(JpegError::Truncated {
            offset: 12,
            expected: 3,
        }),
        Classification::Truncated,
    );
    assert_classification(
        &FastPacketError::Decode(JpegError::NotImplemented {
            sof: SofKind::Progressive8,
        }),
        Classification::NotImplemented,
    );
    assert_classification(
        &FastPacketError::Decode(JpegError::UnsupportedSof {
            marker: 0xC9,
            reason: UnsupportedReason::ArithmeticCoding,
        }),
        Classification::Unsupported,
    );

    for error in [
        JpegError::OutputBufferTooSmall {
            required: 96,
            provided: 48,
        },
        JpegError::InvalidStride { stride: 8, row: 12 },
        JpegError::RectOutOfBounds {
            rect: Rect {
                x: 3,
                y: 4,
                w: 5,
                h: 6,
            },
            width: 7,
            height: 8,
        },
    ] {
        assert_classification(&FastPacketError::Decode(error), Classification::Buffer);
    }

    assert_classification(
        &FastPacketError::Decode(JpegError::InternalInvariant {
            reason: "classification negative control",
        }),
        Classification::None,
    );
}

#[test]
fn decode_conversion_preserves_display_and_typed_source() {
    let jpeg_error = JpegError::Truncated {
        offset: 12,
        expected: 3,
    };
    let error = FastPacketError::from(jpeg_error.clone());

    assert_eq!(error.to_string(), jpeg_error.to_string());
    assert_eq!(
        error
            .source()
            .and_then(|source| source.downcast_ref::<JpegError>()),
        Some(&jpeg_error)
    );

    let direct = FastPacketError::MissingHuffmanTable {
        kind: TableKind::Ac,
        slot: 3,
    };
    assert_eq!(
        direct.to_string(),
        "JPEG fast packet input is missing Ac Huffman table 3"
    );
    assert!(direct.source().is_none());
}
