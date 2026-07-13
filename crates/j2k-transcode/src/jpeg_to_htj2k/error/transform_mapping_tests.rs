// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{dct53_transform_error, dct97_transform_error, JpegToHtj2kError};
use crate::{DctGridError, DctTransformError};

type TransformMapper = fn(DctTransformError) -> JpegToHtj2kError;

#[derive(Clone, Copy)]
enum TransformOperation {
    Dct53,
    Dct97,
}

impl TransformOperation {
    const fn display_prefix(self) -> &'static str {
        match self {
            Self::Dct53 => "5/3 DCT transform failed:",
            Self::Dct97 => "9/7 DCT transform failed:",
        }
    }
}

fn mappers() -> [(TransformMapper, TransformOperation); 2] {
    [
        (dct53_transform_error, TransformOperation::Dct53),
        (dct97_transform_error, TransformOperation::Dct97),
    ]
}

fn semantic_transform_errors() -> [DctTransformError; 4] {
    let grid = crate::dct_grid::validate_dct_block_grid(1, 2, 2, 16, 16)
        .expect_err("fixture grid must be invalid");
    [
        DctTransformError::Grid(grid),
        DctTransformError::InvalidSamplePlaneDimensions {
            width: 0,
            height: 7,
        },
        DctTransformError::SamplePlaneLengthMismatch {
            sample_count: 13,
            width: 4,
            height: 4,
        },
        DctTransformError::SymbolicWeightIndexOutOfRange {
            sample_len: 8,
            output_index: 5,
            high_pass: true,
        },
    ]
}

fn mapped_transform_source(
    mapped: &JpegToHtj2kError,
    operation: TransformOperation,
) -> &DctTransformError {
    match (mapped, operation) {
        (JpegToHtj2kError::Dct53(source), TransformOperation::Dct53)
        | (JpegToHtj2kError::Dct97(source), TransformOperation::Dct97) => source,
        _ => panic!("transform error changed operation classification: {mapped}"),
    }
}

#[test]
fn transform_resource_failures_lift_without_losing_exact_limits() {
    for (map, _) in mappers() {
        let cap = map(DctTransformError::MemoryCapExceeded {
            requested: 65,
            cap: 64,
        });
        assert!(matches!(
            cap,
            JpegToHtj2kError::MemoryCapExceeded {
                requested: 65,
                cap: 64,
            }
        ));
        assert!(std::error::Error::source(&cap).is_none());

        let allocation = map(DctTransformError::HostAllocationFailed { bytes: 4096 });
        assert!(matches!(
            allocation,
            JpegToHtj2kError::HostAllocationFailed { bytes: 4096 }
        ));
        assert!(std::error::Error::source(&allocation).is_none());
    }
}

#[test]
fn semantic_transform_failures_preserve_operation_and_concrete_error_sources() {
    for (map, operation) in mappers() {
        for expected in semantic_transform_errors() {
            let mapped = map(expected.clone());
            let transform_source = mapped_transform_source(&mapped, operation);

            assert_eq!(transform_source, &expected);
            assert_eq!(
                std::error::Error::source(&mapped)
                    .and_then(|source| source.downcast_ref::<DctTransformError>()),
                Some(transform_source)
            );
            assert!(mapped.to_string().starts_with(operation.display_prefix()));

            let nested_source = std::error::Error::source(transform_source);
            if matches!(expected, DctTransformError::Grid(_)) {
                assert!(nested_source
                    .and_then(|source| source.downcast_ref::<DctGridError>())
                    .is_some());
            } else {
                assert!(nested_source.is_none());
            }
        }
    }
}
