// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{num::NonZeroUsize, sync::OnceLock};

use j2k_core::{Downscale, PixelFormat, Rect};

use super::{
    decode_batch, decode_external_batch, decode_external_mixed_batch, decode_external_once,
    decode_external_region_scaled_emulated_once, decode_j2k_batch, decode_j2k_mixed_batch,
    decode_j2k_single_case, decode_method_label, decode_mixed_batch, should_emulate_region_scaled,
};
use crate::fixture_compare::{
    fixtures::fixture_cases, BatchInputs, BenchmarkMode, DecoderKind, FixtureCase,
    MixedFixtureBatch, Operation, OperationClass,
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

fn generated_gray_operation_cases() -> Vec<FixtureCase> {
    let mut full = fixture_cases()
        .expect("generated fixture catalog")
        .into_iter()
        .find(|case| case.name == "classic_raw_gray8_128_full")
        .expect("generated gray fixture");
    let roi = Rect {
        x: 32,
        y: 32,
        w: 64,
        h: 64,
    };
    [
        ("gray_full", Operation::Full),
        ("gray_roi", Operation::Region(roi)),
        ("gray_scaled", Operation::Scaled(Downscale::Quarter)),
        (
            "gray_roi_scaled",
            Operation::RegionScaled {
                roi,
                scale: Downscale::Quarter,
            },
        ),
    ]
    .into_iter()
    .map(|(name, operation)| {
        full.name = name.to_string();
        full.operation = operation;
        full.clone()
    })
    .collect()
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

#[test]
fn j2k_decode_failures_preserve_operation_and_batch_context() {
    for (case, (serial_context, batch_context, mixed_context)) in
        generated_operation_cases().into_iter().zip([
            (
                "j2k serial full decode failed",
                "j2k full decode failed",
                "j2k mixed full decode failed",
            ),
            (
                "j2k serial ROI decode failed",
                "j2k ROI decode failed",
                "j2k mixed ROI decode failed",
            ),
            (
                "j2k serial scaled decode failed",
                "j2k scaled decode failed",
                "j2k mixed scaled decode failed",
            ),
            (
                "j2k serial ROI+scaled decode failed",
                "j2k ROI+scaled decode failed",
                "j2k mixed ROI+scaled decode failed",
            ),
        ])
    {
        let mut invalid = case;
        invalid.bytes = b"not a JPEG 2000 codestream".to_vec();

        let error = decode_j2k_single_case(&invalid, &invalid.bytes)
            .expect_err("invalid single input must fail");
        assert!(
            error.contains(serial_context),
            "missing serial context {serial_context:?}: {error}"
        );

        let batch_inputs = BatchInputs::new(&invalid.bytes, 3, 2);
        let error = decode_j2k_batch(&invalid, &batch_inputs, NonZeroUsize::new(2))
            .expect_err("invalid homogeneous batch must fail");
        assert!(
            error.contains(batch_context),
            "missing batch context {batch_context:?}: {error}"
        );

        let operation_class = invalid.operation.class();
        let mixed = MixedFixtureBatch {
            name: format!("invalid_{}", invalid.operation.label()),
            cases: vec![invalid.clone(), invalid],
            format: PixelFormat::Rgb8,
            operation_class,
        };
        let error = decode_j2k_mixed_batch(&mixed, 3, NonZeroUsize::new(2))
            .expect_err("invalid mixed batch must fail");
        assert!(
            error.contains(mixed_context),
            "missing mixed context {mixed_context:?}: {error}"
        );
    }
}

#[test]
fn openjpeg_router_matches_j2k_for_gray_and_rgb_operations() {
    let cases = generated_gray_operation_cases()
        .into_iter()
        .chain(generated_operation_cases());
    for case in cases {
        let expected = decode_j2k_single_case(&case, &case.bytes).expect("j2k reference decode");
        let actual = decode_external_once(
            BenchmarkMode::Capability,
            &case,
            DecoderKind::OpenJpeg,
            &case.bytes,
        )
        .expect("OpenJPEG routed decode");
        assert_eq!(actual, expected, "operation {}", case.operation.label());
    }
}

#[test]
fn external_batch_dispatch_preserves_order_for_homogeneous_and_mixed_inputs() {
    let case = generated_operation_cases().remove(1);
    let expected = decode_j2k_single_case(&case, &case.bytes).expect("j2k reference decode");
    let batch_inputs = BatchInputs::new(&case.bytes, 3, 2);

    let direct = decode_external_batch(
        BenchmarkMode::Capability,
        &case,
        DecoderKind::OpenJpeg,
        &batch_inputs,
        NonZeroUsize::new(2),
    )
    .expect("direct external batch");
    assert_eq!(direct, expected.repeat(3));
    assert_eq!(
        decode_batch(
            BenchmarkMode::Capability,
            &case,
            DecoderKind::OpenJpeg,
            &batch_inputs,
            NonZeroUsize::new(2),
        )
        .expect("dispatched external batch"),
        direct
    );

    let mut second = case.clone();
    second.name.push_str("_second");
    let mixed = MixedFixtureBatch {
        name: "external_order".to_string(),
        cases: vec![case.clone(), second],
        format: case.format,
        operation_class: case.operation.class(),
    };
    let direct = decode_external_mixed_batch(
        BenchmarkMode::Capability,
        &mixed,
        DecoderKind::OpenJpeg,
        3,
        NonZeroUsize::new(2),
    )
    .expect("direct external mixed batch");
    assert_eq!(direct, expected.repeat(3));
    assert_eq!(
        decode_mixed_batch(
            BenchmarkMode::Capability,
            &mixed,
            DecoderKind::OpenJpeg,
            3,
            NonZeroUsize::new(2),
        )
        .expect("dispatched external mixed batch"),
        direct
    );
}

#[test]
fn portable_emulation_crops_full_scaled_openjpeg_output() {
    let mut case = generated_gray_operation_cases()
        .pop()
        .expect("gray ROI+scaled case");
    case.input_source = "external:unit-fixture".to_string();

    assert!(should_emulate_region_scaled(
        BenchmarkMode::PortableEmulated,
        DecoderKind::OpenJpeg,
        &case
    ));
    assert!(!should_emulate_region_scaled(
        BenchmarkMode::Capability,
        DecoderKind::OpenJpeg,
        &case
    ));
    assert_eq!(
        decode_method_label(
            BenchmarkMode::PortableEmulated,
            DecoderKind::OpenJpeg,
            &case
        ),
        "emulated-full-scaled-crop"
    );

    let expected = decode_j2k_single_case(&case, &case.bytes).expect("j2k reference decode");
    let actual = decode_external_once(
        BenchmarkMode::PortableEmulated,
        &case,
        DecoderKind::OpenJpeg,
        &case.bytes,
    )
    .expect("emulated OpenJPEG decode");
    assert_eq!(actual, expected);

    let error =
        decode_external_region_scaled_emulated_once(&case, DecoderKind::Kakadu, &case.bytes)
            .expect_err("unsupported emulator must fail before process launch");
    assert!(error.contains("kakadu"));
    assert!(error.contains("does not support emulated Gray8"));
}

#[test]
fn external_router_rejects_decoded_length_mismatches() {
    let mut case = generated_operation_cases().remove(0);
    case.dimensions.0 -= 1;

    let error = decode_external_once(
        BenchmarkMode::Capability,
        &case,
        DecoderKind::OpenJpeg,
        &case.bytes,
    )
    .expect_err("metadata length mismatch must fail");
    assert!(error.contains("openjpeg"));
    assert!(error.contains("decoded length"));
    assert!(error.contains("!= expected"));
}
