// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{num::NonZeroUsize, sync::OnceLock};

use j2k_core::PixelFormat;

use super::{
    decode_batch, decode_external_batch, decode_external_mixed_batch, decode_j2k_batch,
    decode_j2k_mixed_batch, decode_j2k_single_case,
};
use crate::fixture_compare::{
    fixtures::fixture_cases, BatchInputs, BenchmarkMode, DecoderKind, MixedFixtureBatch, Operation,
    OperationClass,
};

fn generated_operation_cases() -> Vec<crate::fixture_compare::FixtureCase> {
    static CASES: OnceLock<Vec<crate::fixture_compare::FixtureCase>> = OnceLock::new();
    CASES
        .get_or_init(|| {
            let cases = fixture_cases().expect("generated fixture catalog");
            [
                "classic_jp2_rgb8_128_full",
                "classic_jp2_rgb8_128_roi64",
                "classic_jp2_rgb8_128_q4",
                "classic_jp2_rgb8_128_roi64_q4",
            ]
            .into_iter()
            .map(|name| {
                cases
                    .iter()
                    .find(|case| case.name == name)
                    .unwrap_or_else(|| panic!("generated fixture {name} must exist"))
                    .clone()
            })
            .collect()
        })
        .clone()
}

#[test]
fn single_and_homogeneous_batch_routes_match_for_every_operation() {
    for case in generated_operation_cases() {
        let expected = decode_j2k_single_case(&case, &case.bytes).expect("serial decode");
        let single_input = BatchInputs::new(&case.bytes, 1, 1);
        assert_eq!(
            decode_batch(
                BenchmarkMode::Capability,
                &case,
                DecoderKind::J2k,
                &single_input,
                None,
            )
            .expect("single dispatch"),
            expected
        );

        let batch_inputs = BatchInputs::new(&case.bytes, 3, 2);
        let actual = decode_j2k_batch(&case, &batch_inputs, NonZeroUsize::new(2))
            .expect("homogeneous batch decode");
        assert_eq!(actual, expected.repeat(3));
    }
}

#[test]
fn mixed_batch_routes_preserve_rotation_and_operation_outputs() {
    for case in generated_operation_cases() {
        let mut second = case.clone();
        second.name.push_str("_second");
        let mixed = MixedFixtureBatch {
            name: format!("mixed_{}", case.operation.label()),
            cases: vec![case.clone(), second],
            format: case.format,
            operation_class: case.operation.class(),
        };
        let expected_one = decode_j2k_single_case(&case, &case.bytes).expect("serial decode");

        assert_eq!(
            decode_j2k_mixed_batch(&mixed, 1, None).expect("single mixed dispatch"),
            expected_one
        );
        assert_eq!(
            decode_j2k_mixed_batch(&mixed, 3, NonZeroUsize::new(2)).expect("mixed batch decode"),
            expected_one.repeat(3)
        );
    }
}

#[test]
fn external_batch_routes_return_typed_context_without_process_launches() {
    let mut case = generated_operation_cases().remove(0);
    case.format = PixelFormat::Rgba8;
    case.operation = Operation::Full;
    let inputs = BatchInputs::new(&case.bytes, 3, 2);

    let direct_error = decode_external_batch(
        BenchmarkMode::Capability,
        &case,
        DecoderKind::OpenJpeg,
        &inputs,
        NonZeroUsize::new(2),
    )
    .expect_err("unsupported format must fail before the external decoder");
    assert!(direct_error.contains("openjpeg"));
    assert!(direct_error.contains("does not support Rgba8"));

    let dispatch_error = decode_batch(
        BenchmarkMode::Capability,
        &case,
        DecoderKind::OpenJpeg,
        &inputs,
        NonZeroUsize::new(2),
    )
    .expect_err("external dispatch must preserve the same failure");
    assert_eq!(dispatch_error, direct_error);

    let mixed = MixedFixtureBatch {
        name: "unsupported_external".to_string(),
        cases: vec![case],
        format: PixelFormat::Rgba8,
        operation_class: OperationClass::Full,
    };
    let mixed_error = decode_external_mixed_batch(
        BenchmarkMode::Capability,
        &mixed,
        DecoderKind::OpenJpeg,
        3,
        NonZeroUsize::new(2),
    )
    .expect_err("unsupported mixed format must fail before the external decoder");
    assert!(mixed_error.contains("openjpeg"));
    assert!(mixed_error.contains("does not support Rgba8"));
}
