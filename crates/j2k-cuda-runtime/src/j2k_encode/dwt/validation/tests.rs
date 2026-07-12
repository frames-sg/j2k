// SPDX-License-Identifier: MIT OR Apache-2.0

mod launch_geometry;

use j2k_codec_math::dwt::max_decomposition_levels;

use crate::error::CudaError;

use super::{
    validate_forward_dwt_request, FORWARD_DWT_LEVELS_EXCEED_GEOMETRY,
    FORWARD_DWT_SAMPLES_EXCEED_INDEX_ABI,
};

#[test]
fn maximum_levels_match_native_minimum_axis_contract() {
    for (width, height, expected) in [
        (0, 8, 0),
        (1, 8, 0),
        (1, 7, 0),
        (8, 1, 0),
        (7, 1, 0),
        (2, 8, 1),
        (8, 2, 1),
        (3, 9, 1),
        (4, 8, 2),
        (7, 9, 2),
        (8, 8, 3),
        (u32::MAX, u32::MAX, 31),
    ] {
        assert_eq!(
            max_decomposition_levels(width, height),
            expected,
            "unexpected maximum for {width}x{height}"
        );
    }
}

#[test]
fn zero_and_boundary_level_requests_are_valid() {
    for (width, height) in [(1, 7), (1, 8), (2, 8), (8, 2), (7, 9), (8, 8)] {
        let maximum = max_decomposition_levels(width, height);
        assert!(validate_forward_dwt_request(width, height, 0).is_ok());
        assert!(validate_forward_dwt_request(width, height, maximum).is_ok());
    }
}

#[test]
fn one_axis_only_and_excess_level_requests_are_rejected() {
    for (width, height, requested, maximum) in [
        (2, 8, 2, 1),
        (8, 2, 2, 1),
        (1, 8, 1, 0),
        (1, 7, 1, 0),
        (8, 1, 1, 0),
        (7, 1, 1, 0),
        (8, 8, 4, 3),
    ] {
        let error = validate_forward_dwt_request(width, height, requested)
            .expect_err("excess DWT levels must be rejected");
        match error {
            CudaError::InvalidArgument { message } => assert_eq!(
                message,
                format!(
                    "{FORWARD_DWT_LEVELS_EXCEED_GEOMETRY}: requested {requested}, maximum {maximum} for {width}x{height}"
                )
            ),
            other => panic!("expected invalid DWT geometry, got {other}"),
        }
    }
}

#[test]
fn linear_index_validation_accepts_the_exact_u32_boundary_and_zero_level_passthrough() {
    assert!(validate_forward_dwt_request(65_536, 65_536, 1).is_ok());
    assert!(validate_forward_dwt_request(65_536, 65_537, 0).is_ok());
}

#[test]
fn linear_index_overflow_precedes_the_level_ceiling() {
    for (width, height, levels) in [
        (641, 6_700_417, 1),
        (65_536, 65_537, 1),
        (65_537, 65_536, 1),
        (u32::MAX, u32::MAX, u8::MAX),
    ] {
        let samples = u64::from(width) * u64::from(height);
        let error = validate_forward_dwt_request(width, height, levels)
            .expect_err("forward DWT linear index must fit u32");
        match error {
            CudaError::InvalidArgument { message } => assert_eq!(
                message,
                format!(
                    "{FORWARD_DWT_SAMPLES_EXCEED_INDEX_ABI}: {samples} samples for {width}x{height}"
                )
            ),
            other => panic!("expected invalid DWT indexing geometry, got {other}"),
        }
    }
}
